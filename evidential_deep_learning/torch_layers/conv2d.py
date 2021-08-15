import torch
import torch.nn as nn


class Conv2DNormal(nn.Module):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(Conv2DNormal, self).__init__()
        self.conv2d = torch.nn.Conv2d(kernel_size=kernel_size)

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass

    # search for get config in torch


class Conv2DNormalGamma(nn.Module):
    def __init__(self, filters, kernel_size, **kwargs):
        pass

    def evidence(self, x):
        pass

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass

    # search for get config in torch
