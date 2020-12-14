import torch
import warnings

BCELoss = torch.nn.BCEWithLogitsLoss()


def _KL_(alpha):
    beta = torch.ones((1, alpha.shape[1]), dtype=torch.float)
    S_alpha = alpha.sum(dim=1)
    S_beta = beta.sum(dim=1)
    lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1)
    lnB_uni = torch.lgamma(beta).sum(dim=1) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = ((alpha - beta) * (dg1 - dg0)).sum(dim=1) + lnB + lnB_uni
    return kl


def Dirichlet_SOS(y, alpha):
    # TODO: Validate this port, too many reductions in tf to track
    warnings.warn(f"This function is not validated")
    S = alpha.sum(dim=1)
    m = alpha / S

    A = (y-m).square().sum(axis=1)
    B = (alpha * (S-alpha)) / (S.square() * (S+1))
    B = B.sum(axis=1)

    alpha_hat = y + (1-y)*alpha
    C = _KL_(alpha_hat)
    C = C.mean(dim=1)
    return (A + B + C).mean()


def Sigmoid_CE(y, y_logits):
    return BCELoss(y_logits, y)