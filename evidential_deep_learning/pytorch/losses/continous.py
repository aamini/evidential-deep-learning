import torch
from torch.distributions import StudentT
from torch import nn

MSE = nn.MSELoss(reduction='mean')


def reduce(val, reduction):
    if reduction == 'mean':
        val = val.mean()
    if reduction == 'sum':
        val = val.sum()
    if reduction == 'none':
        pass
    else:
        raise ValueError(f"Invalid reduction argument: {reduction}")
    return val


def RMSE(y, y_):
    return MSE(y, y_).sqrt()


def NIG_NLL(y: torch.Tensor,
            gamma: torch.Tensor,
            nu: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor, reduction='mean'):
    student_var = beta * (1. + alpha) / (nu * alpha)
    dist = StudentT(loc=gamma, scale=student_var, df=2*alpha)
    nll = -1. * dist.log_prob(y)
    return reduce(nll, reduction=reduction)


def NIG_Reg(y, gamma, nu, alpha, reduction='mean'):
    error = (y - gamma).abs()
    evidence = 2. * nu + alpha
    return reduce(error * evidence, reduction=reduction)


def EvidentialRegression(y: torch.Tensor, evidential_output: torch.Tensor, lmbda=1.):
    gamma, nu, alpha, beta = evidential_output
    loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
    loss_reg = NIG_Reg(y, gamma, nu, alpha)
    return loss_nll + lmbda * loss_reg
