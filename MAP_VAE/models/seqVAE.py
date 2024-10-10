import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from MAP_VAE.models.types_ import *
from MAP_VAE.utils.seq_tools import GAPPY_ALPHABET_LEN


class SeqVAE(nn.Module):
    """
    Initializes the SeqVAE model.

    Args:
        dim_latent_vars (int): Dimension of the latent space.
        dim_msa_vars (int): Length of a flattened MSA sequence. i.e. seq_len * alphabet_size.
        num_hidden_units (List[int]): A list of integers representing the number of hidden units in each layer of the encoder and decoder.
        settings (Dict): A dictionary containing additional settings for the model. Refer to dummy_config.yaml for more info.
        num_aa_type (int, optional): The number of amino acid types. Defaults to 21.
    """

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

        # connect linear MSA to first layer of hidden units
        self.encoder_layers.append(nn.Linear(dim_msa_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_layers.append(
                nn.Linear(self.num_hidden_units[i - 1], self.num_hidden_units[i])
            )

        # latent layers
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias=True)
        self.encoder_logsigma = nn.Linear(
            num_hidden_units[-1], dim_latent_vars, bias=True
        )

        # decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_layers.append(
                nn.Linear(num_hidden_units[i - 1], num_hidden_units[i])
            )

        # final layer to output P(x)
        self.decoder_layers.append(nn.Linear(num_hidden_units[-1], dim_msa_vars))

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
        # feed forward
        for T in self.encoder_layers:
            h = T(h)
            h = torch.tanh(h)

        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))

        return mu, sigma

    def decoder(self, z: Tensor) -> Tensor:
        """
        decoder transforms latent space z into p, which is the log probability  of x being 1.

        Returns:
        log_p

        source:
        https://github.com/loschmidt/vae-dehalogenases/
        """

        h = z.to(torch.float32)
        for i in range(len(self.decoder_layers) - 1):
            h = self.decoder_layers[i](h)
            h = torch.tanh(h)

        # final layer output
        h = self.decoder_layers[-1](h)

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

    def compute_elbo_with_multiple_samples(self, x: Tensor, num_samples: int = 500):
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

            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            # estimate log_Pz using prob density function for a Gaussian normal
            # remember that our distribution is isotropic (mean=0, std=1), so many terms cancel out
            log_Pz = torch.sum(
                -0.5 * z**2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1
            )
            log_p = self.decoder(z)
            # get the conditional distribution P(x|z)
            log_PxGz = torch.sum(x * log_p, -1)

            # find the joint distribution P(x,z)= P(x|z)P(z)
            log_Pxz = log_Pz + log_PxGz

            # now estimate q(z|x) using prob density function for a Gaussian normal
            # log PDF of a Gaussian normal is given by:
            # ln(1/(sigma*(2pi)^0.5))) - 0.5*((z - mu)^2 / sigma^2)
            # ln(1/(sigma*(2pi)^0.5))) - 0.5*((mu + sigma * eps - mu)^2 / sigma^2)
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

            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max

            return elbo

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Reconstructs the input tensor using the VAE model.

        Args:
            x (Tensor): The input tensor of shape (batch, columns, num_aa_type).

        Returns:
            Tensor: The reconstructed tensor of shape (batch, columns).
        """
        orig_shape = tuple(x.shape[1:])
        x = torch.flatten(x, start_dim=1)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        log_p = self.decoder(z)

        x_hat = torch.unsqueeze(log_p, -1)
        x_hat = x_hat.view(-1, orig_shape[0], orig_shape[1])
        indices = x_hat.argmax(dim=-1)

        return indices

    @torch.no_grad()
    def get_log_p(self, x: Tensor) -> np.ndarray:
        """
        Calculates the log probability of the input tensor x.

        Args:
            x (Tensor): The input tensor.

        Returns:
            np.ndarray: The log probability tensor.

        """
        orig_shape = x.shape[1:]

        x = torch.flatten(x, start_dim=1)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps

        log_p = self.decoder(z)
        log_p = log_p.view(-1, orig_shape[0], orig_shape[1])

        return log_p.cpu().numpy()

    @torch.no_grad()
    def latent_to_log_p(
        self, zs: Tensor, seq_len: int, alphabet_size: int = GAPPY_ALPHABET_LEN
    ) -> np.ndarray:
        """
        Converts the latent space to the log probability.

        Args:
            zs: Tensor: The latent space coordinates.
            seq_len: int: The sequence length.
            alphabet_size: int: The sequence alphabet size.

        Returns:
            np.ndarray: The log probability tensor.

        """
        orig_shape = tuple([seq_len, alphabet_size])
        log_p = self.decoder(zs)
        log_p = log_p.view(-1, orig_shape[0], orig_shape[1])

        return log_p.cpu().numpy()

    def configure_optimiser(
        self, learning_rate: float = 1e-2, weight_decay: float = 0.0
    ):
        """
        Setup the Adam optimiser.
        """

        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
