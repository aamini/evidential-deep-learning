import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
axis problem for every loss not done: TO TRAIN ON BATCH
There is no axis argument in torch losses
"""


def MSE(y, y_, reduction='mean'):
    """
    nn.MSELoss()
    reduction='none': returns column vector
    reduction='mean' or 'sum': scalar
    """
    loss_fn = nn.MSELoss(reduction=reduction)
    # loss = mse_function(y,y_)
    # return loss
    return loss_fn(y, y_)


def RMSE(y, y_):
    """# reduction is 'none' by default"""
    loss_fn = nn.MSELoss()
    return torch.sqrt(loss_fn(y, y_))


def Gaussian_NLL(y, mu, sigma, reduce=True):
    neglogprob = torch.log(sigma) + 0.5*torch.log(2 *
                                                  np.pi) - 0.5*((y-mu)/sigma)**2
    return torch.mean(-neglogprob) if reduce else -neglogprob


def Gaussian_NLL_logvar():
    pass


def NIG_NLL():
    pass


def KL_NIG():
    pass


def NIG_Reg():
    pass


def EvidentialRegression():
    pass
