from evoVAE.models.types_ import *
import torch
import numpy as np


def KL_divergence(
    zMu: Tensor, zLogvar: Tensor, zSample: Tensor, seq_weights: Tensor
) -> Tensor:
    """
    KLD calculation when there is a gaussian normal distribution
    https://github.com/loschmidt/vae-dehalogenases/blob/main/scripts/VAE_model.py
    """

    zStd = zLogvar.mul(0.5).exp()
    KLD = torch.sum(0.5 * (zStd**2 + zMu**2 - 2 * torch.log(zStd) - 1), -1)

    return KLD


def KL_divergence_monte_carlo(
    zMu: Tensor, zLogvar: Tensor, zSample: Tensor, seq_weights: Tensor
) -> Tensor:
    """Based off a Monte Carlo estimation of KL divergence.
    Above batch sizes of 128 it is generally fairly accurate.
    https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    """

    zStd = zLogvar.mul(0.5).exp()
    p = torch.distributions.Normal(torch.zeros_like(zMu), torch.ones_like(zStd))
    q = torch.distributions.Normal(zMu, zStd)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(zSample)
    log_pz = p.log_prob(zSample)

    # kl
    kl = log_qzx - log_pz
    kl = kl.sum(-1)
    return kl


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


def sequence_likelihood(x_one_hot, x_hat):
    """
    X_hat is the distribution of possible residues,
    calculate the likelihood of observing the original
    input under this distribution.
    """

    flat_input = torch.flatten(x_one_hot, start_dim=1)
    log_PxGz = torch.sum(flat_input * x_hat, -1)

    return log_PxGz


def elbo_importance_sampling(
    model, one_hot_encoding: torch.Tensor, n_samples: int = 500
) -> Tensor:
    """
    Approximation of marginal probability of sequence using importance sample.

    Described by Ding et al., https://www.nature.com/articles/s41467-019-13633-0

    Return:
    log_elbo: Tensor

    source:
    https://github.com/loschmidt/vae-dehalogenases/
    """

    one_hot_encoding = one_hot_encoding.expand(n_samples, -1, -1)

    log_p, z, z_mu, z_logvar = model.forward(one_hot_encoding)
    z_sigma = torch.sqrt(torch.exp(z_logvar))
    # find the random noise used for this sample
    eps = (z - z_mu) / z_sigma

    # P(x|z)
    log_PxGz = sequence_likelihood(one_hot_encoding, log_p)
    # P(z) - log probability density function for Gausian distribution
    log_Pz = torch.sum(-0.5 * z**2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)
    # joint probability P(x,z)
    log_Pxz = log_PxGz + log_Pz

    # now estimate q(z|x)
    log_QzGx = torch.sum(
        -0.5 * (eps) ** 2
        - 0.5 * torch.log(2 * z.new_tensor(np.pi))
        - torch.log(z_sigma),
        -1,
    )

    # P(x,z) / Q(z|x)
    log_weight = (log_Pxz - log_QzGx).detach().data
    log_weight = log_weight.double()
    # find the max to allow for normalisation
    log_weight_max = torch.max(log_weight, 0)[0]
    # now normalise, prevents overflow when we exponentiate
    log_weight = log_weight - log_weight_max
    weight = torch.exp(log_weight)
    # average the weightings and correct the normalisation
    log_elbo = torch.log(torch.mean(weight, 0)) + log_weight_max

    return log_elbo


def frange_cycle_linear(
    n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5
) -> np.ndarray:
    """
    #https://github.com/haofuml/cyclical_annealing

    Paper from Microsoft Research describes how to prevent the
    issue of vanishing KLD.
    """

    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1

    return L
