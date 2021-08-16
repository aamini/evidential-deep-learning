import torch
import torch.nn
import numpy as np

"""
# in pytorch; reduction is 'none' by default
reduction='none': returns column vector
reduction='mean' or 'sum': scalar
"""
# CHECK AXIS & SHAPE AGAIN


def Dirichlet_SOS(y, alpha, t):
    def KL(alpha):
        # check syntax of dtype in torch
        beta = torch.Tensor(torch.ones((1, alpha.shape[1])), dtype=float32)
        S_alpha = torch.sum(alpha)
        S_beta = torch.sum(alpha)
        lnB = torch.sum(torch.lgamma(beta))
        lnB_uni = torch.sum(torch.lgamma(beta))

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        return torch.sum((alpha-beta)*(dig1-dig0)) + lnB + lnB_uni

    S = torch.sum(alpha)
    """
    # not use in the tensorflow version discrete.py
    evidence = alpha - 1
    """
    m = alpha / S

    A = torch.sum((y-m)**2)
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)))

    alpha_hat = y + (1-y)*alpha
    C = KL(alpha_hat)

    C = torch.mean(C)
    return torch.mean(A+B+C)


def Sigmoid_CE(y, y_logits):
    loss = torch.nn.CrossEntropyLoss(y, y_logits)
    return torch.mean(loss)
