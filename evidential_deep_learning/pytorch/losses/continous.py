import torch
from torch.distributions import Normal
from torch import nn
import numpy as np

MSE = nn.MSELoss(reduction='mean')


def reduce(val, reduction):
    if reduction == 'mean':
        val = val.mean()
    elif reduction == 'sum':
        val = val.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f"Invalid reduction argument: {reduction}")
    return val


def RMSE(y, y_):
    return MSE(y, y_).sqrt()


def Gaussian_NLL(y, mu, sigma, reduction='mean'):
    dist = Normal(loc=mu, scale=sigma)
    # TODO: refactor to mirror TF implementation due to numerical instability
    logprob = -1. * dist.log_prob(y)
    return reduce(logprob, reduction=reduction)


def NIG_NLL(y: torch.Tensor,
            gamma: torch.Tensor,
            nu: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor, reduction='mean'):
    inter = 2 * beta * (1 + nu)

    nll = 0.5 * (np.pi / nu).log() \
          - alpha * inter.log() \
          + (alpha + 0.5) * (nu * (y - gamma) ** 2 + inter).log() \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)
    return reduce(nll, reduction=reduction)


def NIG_Reg(y, gamma, nu, alpha, reduction='mean'):
    error = (y - gamma).abs()
    evidence = 2. * nu + alpha
    return reduce(error * evidence, reduction=reduction)


def EvidentialRegression(y: torch.Tensor, evidential_output: torch.Tensor, lmbda=1.):
    gamma, nu, alpha, beta = evidential_output
    loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
    loss_reg = NIG_Reg(y, gamma, nu, alpha)
    return loss_nll, lmbda * loss_reg
