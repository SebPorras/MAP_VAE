from json import encoder
from .types_ import *
import torch
from .base import BaseVAE
from torch import max_pool2d, nn
from ..loss.standard_loss import KL_divergence, gaussian_likelihood


class ConvoVAE(BaseVAE):
    """A VAE implementation using convolutions for encoder
    and decoder. Based off https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    just regualr ELBO used for loss. Global standard deviation
    used for estimating the gaussian likelihood. Rather than input size,
    in_channels is used to specify the number of colour channels in the image. Assumes that
    you're working with a (1, 1, 28, 28) tensor."""

    def __init__(
        self, in_channels: int, latentDims: int, hiddenDims: List[int] = None
    ) -> None:

        super(ConvoVAE, self).__init__()

        # hyper-parameters
        self.latentDims = latentDims

        if hiddenDims is None:
            hiddenDims = [32, 64, 128, 256, 512]

        ### ENCODER ###

        # starts with 28 x 28 size image
        start_size = 28
        output_size = None

        encoderModules = []
        for h_dim in hiddenDims:
            # use the formula (Weights - Kernal - 2Padding)/ Stride + 1
            output_size = start_size - 3 + 1
            encoderModules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=1),
                    nn.Dropout(0.3),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(h_dim),
                )
            )
            in_channels = h_dim
            start_size = output_size

        self.encoder = nn.Sequential(*encoderModules)

        # extract Mu and logVar (log variance)
        self.zMuSampler = nn.Linear(
            hiddenDims[-1] * output_size * output_size, latentDims
        )
        self.zLogvarSampler = nn.Linear(
            hiddenDims[-1] * output_size * output_size, latentDims
        )

        # restructure latent sample to be passed to decoder
        self.decoder_input = nn.Linear(
            latentDims, hiddenDims[-1] * output_size * output_size
        )

        ### DECODER ###
        hiddenDims.reverse()

        decoderModules = []
        for i in range(len(hiddenDims) - 1):
            # use the formula (Weights - Kernal - 2Padding)/ Stride + 1
            output_size = start_size - 3 + 1
            decoderModules.append(
                nn.Sequential(
                    nn.Conv2d(
                        hiddenDims[i],
                        out_channels=hiddenDims[i + 1],
                        kernel_size=3,
                        stride=1,
                    ),
                    nn.Dropout(0.3),
                    nn.BatchNorm2d(hiddenDims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*decoderModules)

        ### FINAL LAYER ###

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hiddenDims[-1],
                hiddenDims[-1],
                kernel_size=3,
                stride=3,
                padding=1,
                output_padding=1,
            ),
            # (Hin −1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+
            nn.BatchNorm2d(hiddenDims[-1]),
            nn.LeakyReLU(),
            # use the formula 28 = (20 - Kernal - 2Padding)/ Stride + 1
            nn.Conv2d(hiddenDims[-1], out_channels=1, kernel_size=2),
            nn.Tanh(),
        )

        # will transform to an initial SD of 1 for gaussian likelihood
        self.logStandardDeviation = nn.Parameter(torch.tensor([0.0]))

    def reparameterise(self, zMu: Tensor, zLogvar: Tensor) -> Tensor:
        """Construct a Gaussian distribution from learnt values of
        Mu and sigma and sample a vector Z from this distribution"""

        # convert Log(variance) to standard deviation
        std = zLogvar.mul(0.5).exp()
        qPhi = torch.distributions.Normal(zMu, std)

        return qPhi.rsample()

    def encode(self, rawInput: Tensor) -> Tuple[Tensor, Tensor, torch.Size]:
        """Take the rawInput and encode into a bottle neck dimension
        specified by the encoder. Transform this into Mu and log variance
        values to construct a distribution. i.e. P_phi(Z|X)
        """

        # find qPhi encoding of the raw input
        encoderOutput = self.encoder(rawInput)
        # save this for later so that we can resize out flattened tensor
        shape = encoderOutput.shape

        encoderOutput = torch.flatten(encoderOutput, start_dim=1)

        # sample out the gaussian parameters
        zMu = self.zMuSampler(encoderOutput)
        zLogvar = self.zLogvarSampler(encoderOutput)

        return zMu, zLogvar, shape

    def decode(self, upscaledZ: Tensor) -> Tensor:
        """Sample a latent vector, Z, and
        learn the parameters from which reconstructions
        can be sampled from. I.e. we are learning P_sigma(X|Z)
        """

        xHat = self.decoder(upscaledZ)

        xHat = self.final_layer(xHat)

        return xHat

    def forward(self, rawInput: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        zMu, zLogvar, shape = self.encode(rawInput=rawInput)

        # construct a gaussian distribution and sample
        zSample = self.reparameterise(zMu, zLogvar)

        upscaledZ = self.decoder_input(zSample)

        upscaledZ = upscaledZ.view(shape)

        # learn parameters for q_theta P(x|z)
        xHat = self.decode(upscaledZ)

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
