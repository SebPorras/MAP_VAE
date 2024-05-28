from evoVAE.models.types_ import *
import torch
from evoVAE.models.base import BaseVAE
from torch import nn
from evoVAE.loss.standard_loss import KL_divergence, sequence_likelihood
import torch.nn.functional as F
from typing import Dict
import numpy as np


class SeqVAE(BaseVAE):
    """
    Most basic sequence VAE.
    """

    def __init__(
        self, input_dims: int, latent_dims: int, hidden_dims: List[int], config: Dict
    ) -> None:
        super(SeqVAE, self).__init__()

        # will transform to an initial SD of 1 for gaussian likelihood
        self.logStandardDeviation = nn.Parameter(torch.tensor([0.0]))
        self.AA_COUNT = 21  # standard AA and gap

        self.latent_dims = latent_dims
        self.encoded_seq_len = input_dims  # save this to reconstruct seqs

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        ### ENCODER ###

        encoder_modules = []
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(input_dims, h_dim),
                    nn.LeakyReLU(),
                )
            )
            input_dims = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        ### LATENT SPACE ###

        # extract Mu and logVar (log variance)
        self.z_mu_sampler = nn.Linear(hidden_dims[-1], latent_dims)
        self.z_logvar_sampler = nn.Linear(hidden_dims[-1], latent_dims)

        # restructure latent sample to be passed to decoder
        self.upscale_z = nn.Linear(latent_dims, hidden_dims[-1])

        ### DECODER ###

        hidden_dims.reverse()
        decoder_modules = []
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        # add a final layer to get back to length of seq * AA_Count
        decoder_modules.append(nn.Linear(hidden_dims[-1], self.encoded_seq_len))

        self.decoder = nn.Sequential(*decoder_modules)

    def reparameterise(self, z_mu: Tensor, z_logvar: Tensor) -> Tensor:
        """
        Construct a Gaussian distribution from learnt values of
        Mu and sigma and sample a vector Z from this distribution

        Returns:
        Z: Tensor
        """

        std = z_logvar.mul(0.5).exp()
        q_phi = torch.distributions.Normal(z_mu, std)

        return q_phi.rsample()

    def encode(self, raw_input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Take the rawInput and encode into a bottle neck dimension
        specified by the encoder. Transform this into Mu and log variance
        values to construct a distribution. i.e. P_phi(Z|X).

        Returns:
        z_mu, z_logvar
        """

        encoder_output = self.encoder(raw_input)
        z_mu = self.z_mu_sampler(encoder_output)
        z_logvar = self.z_logvar_sampler(encoder_output)

        return z_mu, z_logvar

    def decode(self, z_sample: Tensor) -> Tensor:
        """
        Sample a latent vector, Z, and
        learn the parameters from which reconstructions
        can be sampled from. I.e. we are learning P_sigma(X|Z)
        """

        # upscale from latent dims to the decoder
        z_upscaled = self.upscale_z(z_sample)

        xHat = self.decoder(z_upscaled)

        return xHat

    def forward(self, raw_input: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Complete the forward pass.

        Return:
        log_p, z_sample, z_mu, z_logvar
        """

        flat_input = torch.flatten(raw_input, start_dim=1)

        # encode for q_phi mu and log variance
        z_mu, z_logvar = self.encode(flat_input)

        # sample from the distribution
        z_sample = self.reparameterise(z_mu, z_logvar)

        x_hat = self.decode(z_sample)

        # record input shape
        input_shape = tuple(x_hat.shape[0:-1])

        # add on extra dimension
        x_hat = torch.unsqueeze(x_hat, -1)

        # reshape back to original one-hot input size
        x_hat = x_hat.view(input_shape + (-1, self.AA_COUNT))

        # apply the softmax over last dim, i.e the 21 amino acids
        log_p = F.log_softmax(x_hat, dim=-1)
        # log_p = F.softmax(x_hat, dim=-1)

        # reflatten our probability distribution
        log_p = log_p.view(input_shape + (-1,))

        return log_p, z_sample, z_mu, z_logvar

    def loss_function(
        self,
        modelOutputs: Tuple[Tensor, Tensor, Tensor, Tensor],
        x: Tensor,
        seq_weight: Tensor,
        epoch: int,
        anneal_schedule: np.ndarray,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """The standard ELBO loss is used in a StandardVAE. Also passes
        in seq_weight which is a weighting based on how the sequences cluster
        together.

        Returns:
        elbo, kld, recon_loss
        """

        xHat, zSample, zMu, zLogvar = modelOutputs

        # KLD across whole all dimensions for each x
        kld = KL_divergence(zMu, zLogvar, zSample, seq_weight)

        # Recon loss: estimate likelihood of each input sequence
        log_PxGz = sequence_likelihood(x, xHat)

        # remove KLD and then use normalised sequence weights
        elbo = log_PxGz - (kld * anneal_schedule[epoch])
        norm_weight = seq_weight / torch.sum(seq_weight)

        # reweight
        elbo = (-1) * torch.sum(elbo * norm_weight)
        recon_weighted = torch.sum(log_PxGz * norm_weight)
        reg_weighted = torch.sum(kld * norm_weight)

        return elbo, reg_weighted.detach(), recon_weighted.detach()

    def generate(self, x: Tensor) -> Tensor:
        """Return the reconstructed input

        Returns:
        xHat
        """

        xHat, _, _, _ = self.forward(x)

        return xHat

    def configure_optimiser(
        self, learning_rate: float = 1e-2, weight_decay: float = 1e-4
    ):
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
