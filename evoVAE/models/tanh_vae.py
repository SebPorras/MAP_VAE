import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class tanhVAE(nn.Module):

    def __init__(
        self,
        dim_latent_vars: int,
        dim_msa_vars: int,
        num_hidden_units: List[int],
        num_aa_type: int = 21,
    ):
        super(tanhVAE, self).__init__()

        self.num_aa_type = num_aa_type
        self.dim_latent_vars = dim_latent_vars
        self.dim_msa_vars = dim_msa_vars  # assume that input is linear
        self.num_hidden_units = num_hidden_units

        # encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(dim_msa_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(
                nn.Linear(num_hidden_units[i - 1], num_hidden_units[i])
            )
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias=True)
        self.encoder_logsigma = nn.Linear(
            num_hidden_units[-1], dim_latent_vars, bias=True
        )

        # decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(
                nn.Linear(num_hidden_units[i - 1], num_hidden_units[i])
            )
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], dim_msa_vars))

    def encoder(self, x):
        """
        encoder transforms x into latent space z
        """

        h = x.to(torch.float32)
        for T in self.encoder_linears:
            h = T(h)
            h = torch.tanh(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        """
        decoder transforms latent space z into p, which is the log probability  of x being 1.
        """

        h = z.to(torch.float32)
        for i in range(len(self.decoder_linears) - 1):
            h = self.decoder_linears[i](h)
            h = torch.tanh(h)
        h = self.decoder_linears[-1](h)

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(fixed_shape + (-1, self.num_aa_type))

        log_p = F.log_softmax(h, dim=-1)
        log_p = log_p.view(fixed_shape + (-1,))

        return log_p

    def compute_weighted_elbo(self, x, weight, anneal_schedule, epoch, c_fx_x=2):

        x = torch.flatten(x, start_dim=1)
        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x * log_p, -1)

        # Set parameter for training
        # loss = (x - f(x)) - (1/C * KL(qZ, pZ)
        #      reconstruction    normalization
        #         parameter        parameter
        # The bigger C is more accurate the reconstruction will be
        # default value is 2.0
        c = 1 / c_fx_x

        # compute elbo
        elbo = log_PxGz - anneal_schedule[epoch] * torch.sum(
            c * (sigma**2 + mu**2 - 2 * torch.log(sigma) - 1), -1
        )
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo * weight)

        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples):

        with torch.no_grad():

            x = x.expand(num_samples, -1, -1)
            x = torch.flatten(x, start_dim=1)
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(
                -0.5 * z**2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1
            )
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(
                -0.5 * (eps) ** 2
                - 0.5 * torch.log(2 * z.new_tensor(np.pi))
                - torch.log(sigma),
                -1,
            )
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max

            return elbo
    

    def configure_optimiser(
        self, learning_rate: float = 1e-2, weight_decay: float = 0.0
    ):
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
