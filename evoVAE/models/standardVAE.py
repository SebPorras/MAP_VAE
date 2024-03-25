from evoVAE.models.types_ import *
import torch
from evoVAE.models.base import BaseVAE
from torch import nn
from evoVAE.loss.standard_loss import KL_divergence, gaussian_likelihood


class StandardVAE(BaseVAE):
    """A standard VAE implementatio. Nothing fancy,
    just regualr ELBO used for loss. Global standard deviation
    used for estimating the gaussian likelihood."""

    def __init__(
        self,
        inputDims: int,
        bottleNeckDim: int,
        latentDim: int,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:

        super(StandardVAE, self).__init__()

        # hyper-parameters
        self.latent_dim = latentDim
        self.bottleNeckDim = bottleNeckDim

        self.encoder = encoder

        # extract Mu and logVar (log variance)
        self.zMuSampler = nn.Linear(bottleNeckDim, latentDim)
        self.zLogvarSampler = nn.Linear(bottleNeckDim, latentDim)

        # restructure latent sample to be passed to decoder
        self.latentUpscaler = nn.Linear(latentDim, bottleNeckDim)

        self.decoder = decoder

        # will transform to an initial SD of 1 for gaussian likelihood
        self.logStandardDeviation = nn.Parameter(torch.tensor([0.0]))

    def reparameterise(self, zMu: Tensor, zLogvar: Tensor) -> Tensor:
        """Construct a Gaussian distribution from learnt values of
        Mu and sigma and sample a vector Z from this distribution"""

        # convert Log(variance) to standard deviation
        std = zLogvar.mul(0.5).exp()
        qPhi = torch.distributions.Normal(zMu, std)

        return qPhi.rsample()

    def encode(self, rawInput: Tensor) -> Tensor:
        """Take the rawInput and encode into a bottle neck dimension
        specified by the encoder. Transform this into Mu and log variance
        values to construct a distribution. i.e. P_phi(Z|X)
        """

        # find qPhi encoding of the raw input
        encoderOutput = self.encoder(rawInput)

        # sample out the gaussian parameters
        zMu = self.zMuSampler(encoderOutput)
        zLogvar = self.zLogvarSampler(encoderOutput)

        return zMu, zLogvar

    def decode(self, zSample: Tensor) -> Tensor:
        """Sample a latent vector, Z, and
        learn the parameters from which reconstructions
        can be sampled from. I.e. we are learning P_sigma(X|Z)
        """
        upscaledZ = self.latentUpscaler(zSample)
        xHat = self.decoder(upscaledZ)

        return xHat

    def forward(self, rawInput: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        zMu, zLogvar = self.encode(rawInput=rawInput)

        # construct a gaussian distribution and sample
        zSample = self.reparameterise(zMu, zLogvar)

        # learn parameters for q_theta P(x|z)
        xHat = self.decode(zSample)

        return xHat, zSample, zMu, zLogvar

    def loss_function(
        self, modelOutputs: Tuple[Tensor, Tensor, Tensor, Tensor], input: Tensor
    ) -> Tensor:
        """The standard ELBO loss is used in a StandardVAE"""

        xHat, zSample, zMu, zLogvar = modelOutputs

        kl = KL_divergence(zMu, zLogvar, zSample)

        likelihood = gaussian_likelihood(xHat, self.logStandardDeviation, input)

        elbo = kl - likelihood

        return elbo, kl.detach(), likelihood.detach()

    def generate(self, x: Tensor) -> Tensor:
        """Return the reconstructed input"""

        xHat, _, _, _ = self.forward(x)

        return xHat

    def configure_optimiser(self, learningRate: float = 1e-4):
        return torch.optim.Adam(self.parameters(), lr=learningRate)
