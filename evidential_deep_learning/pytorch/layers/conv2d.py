import torch
from torch.nn import Module, Conv2d
import torch.nn.functional as F


# TODO: efficiently handle batch dimension


class Conv2DNormal(Module):
    def __init__(self, in_channels, out_tasks, kernel_size, **kwargs):
        super(Conv2DNormal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * out_tasks
        self.n_tasks = out_tasks
        self.conv = Conv2d(self.in_channels, self.out_channels, kernel_size, **kwargs)

    def forward(self, x):
        output = self.conv(x)
        if len(x.shape) == 3:
            mu, logsigma = torch.split(output, self.n_tasks, dim=0)
        else:
            mu, logsigma = torch.split(output, self.n_tasks, dim=1)

        sigma = F.softplus(logsigma) + 1e-6

        return torch.stack([mu, sigma])


class Conv2DNormalGamma(Module):
    def __init__(self, in_channels, out_tasks, kernel_size, **kwargs):
        super(Conv2DNormalGamma, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_tasks
        self.conv = Conv2d(in_channels, 4 * out_tasks, kernel_size, **kwargs)

    def forward(self, x):
        output = self.conv(x)

        if len(x.shape) == 3:
            gamma, lognu, logalpha, logbeta = torch.split(output, self.out_channels, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(output, self.out_channels, dim=1)

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)
        return torch.stack([gamma, nu, alpha, beta])

