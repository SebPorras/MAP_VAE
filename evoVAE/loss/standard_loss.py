from evoVAE.models.types_ import *
import torch


def KL_divergence(
    zMu: Tensor, zLogvar: Tensor, zSample: Tensor, seq_weights: Tensor
) -> Tensor:
    """Based off a Monte Carlo estimation of KL divergence.
    Above batch sizes of 128 it is generally fairly accurate."""

    # Define the distribution we are regularising the encoder with
    p_theta = torch.distributions.Normal(torch.zeros_like(zMu), torch.ones_like(zMu))

    # now define a distribution from our learned parameters
    zStd = zLogvar.mul(0.5).exp()

    q_phi = torch.distributions.Normal(zMu, zStd)

    # find the probability of our sample Z under each distribution
    log_qzx = q_phi.log_prob(zSample)
    log_pz = p_theta.log_prob(zSample)

    kl = log_qzx - log_pz

    # sum up across all dims except the first
    # (https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed)
    kl = kl.sum(dim=tuple(range(1, kl.ndim)))

    # reweight seqs
    kl = kl * seq_weights

    # get the average
    return kl.mean(dim=0)


def gaussian_likelihood(
    xHat: Tensor, globalLogSD: Tensor, x: Tensor, seq_weights: Tensor
) -> Tensor:
    """Build a distribution from parameters estimated from the decoder and the global log variance
    which is another model parameter. Calculate the log probability of the original input x under this
    distribution and return the average across all samples."""

    globalStd = globalLogSD.exp()

    qPhi = torch.distributions.Normal(loc=xHat, scale=globalStd)

    log_pxz = qPhi.log_prob(x)

    # sum up across all dims except the first
    # (https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed)
    log_pxz = log_pxz.sum(dim=tuple(range(1, log_pxz.ndim)))

    log_pxz = log_pxz * seq_weights

    return log_pxz.mean(dim=0)
