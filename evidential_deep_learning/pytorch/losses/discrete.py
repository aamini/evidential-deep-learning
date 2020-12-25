import torch
import torch.nn.functional as F

BCELoss = torch.nn.BCEWithLogitsLoss()


def Dirichlet_SOS(y, outputs, device=None):
    return edl_log_loss(outputs, y, device=device if device else outputs.device)


def Dirichlet_Evidence(outputs):
    """Calculate ReLU evidence"""
    return relu_evidence(outputs)


def Dirichlet_Matches(predictions, labels):
    """Calculate the number of matches from index predictions"""
    assert predictions.shape == labels.shape, f"Dimension mismatch between predictions " \
                                              f"({predictions.shape}) and labels ({labels.shape})"
    return torch.reshape(torch.eq(predictions, labels).float(), (-1, 1))


def Dirichlet_Predictions(outputs):
    """Calculate predictions from logits"""
    return torch.argmax(outputs, dim=1)


def Dirichlet_Uncertainty(outputs):
    """Calculate uncertainty from logits"""
    alpha = relu_evidence(outputs) + 1
    return alpha.size(1) / torch.sum(alpha, dim=1, keepdim=True)


def Sigmoid_CE(y, y_logits, device=None):
    return BCELoss(y_logits, y, device=device if device else y_logits.device)


# MIT License
#
# Copyright (c) 2019 Douglas Brion
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
          torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl


def edl_loss(func, y, alpha, device=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = kl_divergence(kl_alpha, y.shape[1], device=device)
    return A + kl_div


def edl_log_loss(output, target, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha, device=device))
    assert loss is not None
    return loss
