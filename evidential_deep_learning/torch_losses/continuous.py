import torch
import torch.nn as nn
from torch import log as Log
import numpy as np

"""
nn.MSELoss()
# in pytorch; reduction is 'none' by default
reduction='none': returns column vector
reduction='mean' or 'sum': scalar
"""

"""
CHECK TORCH AXIS
"""


def MSE(y, y_, reduction='mean'):
    loss_fn = nn.MSELoss(reduction=reduction)
    return loss_fn(y, y_)


def RMSE(y, y_):
    loss_fn = nn.MSELoss()
    return torch.sqrt(loss_fn(y, y_))


def Gaussian_NLL(y, mu, sigma, reduce=True):
    neglogprob = Log(sigma) + 0.5*Log(2 * np.pi) - 0.5*((y-mu)/sigma)**2
    return torch.mean(-neglogprob) if reduce else -neglogprob


def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    log_liklihood = 0.5*(-torch.exp(-logvar)*(mu-y) **
                         2-Log(2*torch.Tensor(np.pi)))
    loss = torch.mean(-log_liklihood)
    return torch.mean(loss) if reduce else loss


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*Log(np.pi/v) \
        - alpha*Log(twoBlambda) \
        + (alpha + 0.5) * Log(v*(y-gamma)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    return torch.mean(nll) if reduce else nll


def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*((a1-1)/b1 * (v2*torch.square(mu2-mu1))
              + v2/v1
              - Log(b1/b2) - 1) \
        + a2*Log(b1/b2) \
        - (torch.lgamma(a1) - torch.lgamma(a2)) \
        + (a1 - a2)*torch.digamma(a1) \
        - (b1-b2)*a1/b1
    return KL


def NIG_Reg(y, gamma, v, alpha, beta, omega=1e-2, reduce=True, kl=False):
    error = torch.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v + (alpha)
        reg = error*evi

    return torch.mean(reg) if reduce else reg


def EvidentialRegression(y_true, evidential_output, coeff=1.):
    gamma, v, alpha, beta = torch.tensor_split(evidential_output, 4, dim=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg
