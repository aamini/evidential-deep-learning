import torch
import torch.nn as nn

# search for get config in torch

"""Tensorflow api"""

"""
class Conv2DNormal(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(Conv2DNormal, self).__init__()
        self.conv = Conv2D(2 * filters, kernel_size, **kwargs)

    def call(self, x):
        output = self.conv(x)
        mu, logsigma = tf.split(output, 2, axis=-1)
        sigma = tf.nn.softplus(logsigma) + 1e-6
        # return [mu, sigma]
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        base_config = super(Conv2DNormal, self).get_config()
        base_config['filters'] = self.filters
        base_config['kernel_size'] = self.kernel_size
        return base_config
"""
"""
from pytorch doc:

 torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.

 For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.

 If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
"""

class Conv2DNormal(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        super(Conv2DNormal, self).__init__()
        """self.conv = Conv2D(2 * filters, kernel_size, **kwargs)"""
        """in torch, input channel of conv layer has to be specified"""
        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channel, out_channels=2 * out_channel, kernel_size=kernel_size, **kwargs)

    def call(self, x):
        output = self.conv2d(x)
        mu, logsigma = torch.split(output, 2, dim=-1)
        softplus = nn.Softplus()
        sigma = softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


"""
class Conv2DNormalGamma(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(Conv2DNormalGamma, self).__init__()
        self.conv = Conv2D(4 * filters, kernel_size, **kwargs)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.conv(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        base_config = super(Conv2DNormalGamma, self).get_config()
        base_config['filters'] = self.filters
        base_config['kernel_size'] = self.kernel_size
        return base_config


"""


class Conv2DNormalGamma(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        super(Conv2DNormal, self).__init__()
        """self.conv = Conv2D(4 * filters, kernel_size, **kwargs)"""
        """in torch, input channel of conv layer has to be specified"""
        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channel, out_channels=4 * out_channel, kernel_size=kernel_size, **kwargs)

    def evidence(self, x):
        softplus = nn.Softplus()
        return softplus(x)

    def call(self, x):
        output = self.conv2d(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)
