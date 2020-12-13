import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

# TODO: Find a way to efficiently handle batch dimension


class DenseNormal(Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(DenseNormal, self).__init__()
        self.n_in = n_input
        self.n_out = 2 * n_out_tasks
        self.n_tasks = n_out_tasks
        self.l1 = nn.Linear(self.n_in, self.n_out)

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            mu, logsigma = torch.split(x, self.n_tasks, dim=0)
        else:
            mu, logsigma = torch.split(x, self.n_tasks, dim=1)

        sigma = F.softplus(logsigma) + 1e-6
        return torch.stack(mu, sigma)


class DenseNormalGamma(Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(DenseNormalGamma, self).__init__()
        self.n_in = n_input
        self.n_out = 4 * n_out_tasks
        self.n_tasks = n_out_tasks
        self.l1 = nn.Linear(self.n_in, self.n_out)

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=1)

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)

        return torch.stack([gamma, nu, alpha, beta]).to(x.device)
