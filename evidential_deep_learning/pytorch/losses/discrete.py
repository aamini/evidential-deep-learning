import torch

BCELoss = torch.nn.BCEWithLogitsLoss()


def Dirichlet_SOS(y, alpha, t):
    raise NotImplementedError()


def Sigmoid_CE(y, y_logits):
    return BCELoss(y_logits, y)