from evoVAE.models.types_ import *
import torch
from evoVAE.models.seqVAE import SeqVAE
from torch import nn
from evoVAE.loss.standard_loss import KL_divergence, gaussian_likelihood
import torch.nn.functional as F
from typing import Dict


class SeqVAETest(SeqVAE):
    """
    Most basic sequence VAE. Uses a regular dictionary
    rather than the WandDB config dict so that I can test
    in notebooks without having to start an entire job.
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
                    nn.Linear(h_dim, h_dim),
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
                    nn.Linear(hidden_dims[i + 1], hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        # add a final layer to get back to length of seq * AA_Count
        decoder_modules.append(nn.Linear(hidden_dims[-1], self.encoded_seq_len))

        self.decoder = nn.Sequential(*decoder_modules)

    def loss_function(
        self,
        modelOutputs: Tuple[Tensor, Tensor, Tensor, Tensor],
        input: Tensor,
        seq_weight: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """The standard ELBO loss is used in a StandardVAE. Also passes
        in seq_weight which is a weighting based on how the sequences cluster
        together.
        """

        xHat, zSample, zMu, zLogvar = modelOutputs

        # average KL across whole batch
        kl = KL_divergence(zMu, zLogvar, zSample, seq_weight)

        # averaged across the whole batch
        likelihood = gaussian_likelihood(
            xHat,
            self.logStandardDeviation,
            torch.flatten(input, start_dim=1),
            seq_weight,
        )

        elbo = kl - likelihood

        return elbo, kl.detach(), likelihood.detach()
