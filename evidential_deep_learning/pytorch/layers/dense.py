import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module


class DenseNormalGamma(Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(DenseNormalGamma, self).__init__()
        self.n_in = n_input
        self.n_out = 4 * n_out_tasks
        self.l1 = nn.Linear(self.n_in, self.n_out)

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            mu, lognu, logalpha, logbeta = x[0::4], x[1::4], x[2::4], x[3::4]
        else:
            mu, lognu, logalpha, logbeta = x[:, 0::4], x[:, 1::4], x[:, 2::4], x[:, 3::4]

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)

        return torch.stack([mu, nu, alpha, beta]).to(x.device)
