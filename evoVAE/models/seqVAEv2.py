import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from evoVAE.models.types_ import *


class SeqVAE(nn.Module):

    def __init__(
        self,
        dim_latent_vars: int,
        dim_msa_vars: int,
        num_hidden_units: List[int],
        settings: Dict,
        num_aa_type: int = 21,
    ):
        super(SeqVAE, self).__init__()

        self.num_aa_type = num_aa_type
        self.dim_latent_vars = dim_latent_vars
        self.dim_msa_vars = dim_msa_vars  # assume that input is linear
        self.num_hidden_units = num_hidden_units
        self.settings = settings

        # encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_batch_norm = nn.ModuleList()
        self.encoder_dropout = nn.ModuleList()
        # connect linear MSA to first layer of hidden units
        self.encoder_layers.append(nn.Linear(dim_msa_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_layers.append(
                nn.Linear(self.num_hidden_units[i - 1], self.num_hidden_units[i])
            )
            self.encoder_batch_norm.append(nn.BatchNorm1d(self.num_hidden_units[i]))
            self.encoder_dropout.append(nn.Dropout(settings["dropout"]))

        # latent layers
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias=True)
        self.encoder_logsigma = nn.Linear(
            num_hidden_units[-1], dim_latent_vars, bias=True
        )

        # decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_batch_norm = nn.ModuleList()
        self.decoder_dropout = nn.ModuleList()

        self.decoder_layers.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_layers.append(
                nn.Linear(num_hidden_units[i - 1], num_hidden_units[i])
            )
            self.decoder_batch_norm.append(nn.BatchNorm1d(self.num_hidden_units[i]))
            self.decoder_dropout.append(nn.Dropout(settings["dropout"]))

        # final layer to output P(x)
        self.final_layer = nn.Linear(num_hidden_units[-1], dim_msa_vars)

    def encoder(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x is the sequence in the shape (batch_size, columns, num_aa_type).
        Applies batch norm, ReLU activation and finally a dropout layer.

        Returns:
        mu: mean
        sigma: standard deviation

        source: https://github.com/loschmidt/vae-dehalogenases/
        """

        h = x.to(torch.float32)
        for T, batch_norm, dropout in zip(
            self.encoder_layers, self.encoder_batch_norm, self.encoder_dropout
        ):
            # feed forward
            h = T(h)
            h = batch_norm(h)
            h = torch.relu(h)
            h = dropout(h)

        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))

        return mu, sigma

    def decoder(self, z: Tensor) -> Tensor:
        """
        decoder transforms latent space z into p, which is the log probability  of x being 1.

        Applies batch norm, relu activation followed by a dropout layer.

        Returns:
        log_p

        source:
        https://github.com/loschmidt/vae-dehalogenases/
        """

        h = z.to(torch.float32)
        for i in range(len(self.decoder_layers) - 1):
            h = self.decoder_layers[i](h)
            h = self.decoder_batch_norm[i](h)
            h = torch.relu(h)
            h = self.decoder_dropout[i](h)

        # final layer output
        h = self.final_layer(h)

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        # reshape into (batch_size, columns, 21)
        h = h.view(fixed_shape + (-1, self.num_aa_type))

        # apply softmax over each layer to make distribtuion
        log_p = F.log_softmax(h, dim=-1)
        # reflatten
        log_p = log_p.view(fixed_shape + (-1,))

        return log_p

    def compute_weighted_elbo(
        self, x: Tensor, weight: Tensor, anneal_schedule: np.ndarray, epoch: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Returns:
        elbo, log_PxGz, KLD
        """

        # make the input linear
        x = torch.flatten(x, start_dim=1)

        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x * log_p, -1)

        # cyclic annealing to help accuracy
        kld = anneal_schedule[epoch] * torch.sum(
            0.5 * (sigma**2 + mu**2 - 2 * torch.log(sigma) - 1), -1
        )

        # compute elbo.
        elbo = log_PxGz - kld
        weight = weight / torch.sum(weight)

        # compute for the batch
        elbo = torch.sum(elbo * weight)
        log_PxGz = torch.sum(log_PxGz * weight)
        kld = torch.sum(kld * weight)

        return (elbo, log_PxGz, kld)

    def compute_elbo_with_multiple_samples(self, x: Tensor, num_samples: int = 10):
        """
        Approximation of marginal probability of sequence using importance sampling.

        Described by Ding et al., https://www.nature.com/articles/s41467-019-13633-0

        Return:
        log_elbo

        source:
        https://github.com/loschmidt/vae-dehalogenases/
        """

        with torch.no_grad():

            # duplicate the x sample
            x = x.expand(num_samples, -1, -1)
            x = torch.flatten(x, start_dim=1)

            # estimate log_p using prob density function for a Gaussian normal
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(
                -0.5 * z**2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1
            )
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)

            # find the joint distribution P(x,z)
            log_Pxz = log_Pz + log_PxGz

            # now estimate q(z|x) using prob density function for a Gaussian normal
            log_QzGx = torch.sum(
                -0.5 * (eps) ** 2
                - 0.5 * torch.log(2 * z.new_tensor(np.pi))
                - torch.log(sigma),
                -1,
            )

            # P(x,z) / Q(z|x) to find importance
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()

            # applying normalisation for numerical stability
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)

            # note this is log(elbo)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max

            return elbo

    def configure_optimiser(
        self, learning_rate: float = 1e-2, weight_decay: float = 0.0
    ):
        """
        Setup the Adam optimiser.
        """

        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
