from evoVAE.models.types_ import Tensor
import torch


def KL_divergence(zMu: Tensor, zLogvar: Tensor, zSample: Tensor) -> Tensor:
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

    # sum over all dimensions get the average
    return kl.sum(dim=tuple(range(1, kl.ndim))).mean(dim=0)


def gaussian_likelihood(xHat: Tensor, globalLogSD: Tensor, x: Tensor) -> Tensor:
    """Build a distribution from parameters estimated from the decoder and the global log variance
    which is another model parameter. Calculate the log probability of the original input x under this
    distribution and return the average across all samples."""

    globalStd = globalLogSD.exp()
    qPhi = torch.distributions.Normal(xHat, globalStd)

    log_pxz = qPhi.log_prob(x)

    # sum up across all dims and then average
    return log_pxz.sum(dim=tuple(range(1, log_pxz.ndim))).mean(dim=0)
