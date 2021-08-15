import torch
import torch.nn as nn


class DenseNormal(nn.Module):
    def __init__(self, units):
        super(DenseNormal, self).__init__()
        self.units = units  # check type
        self.dense = '''TODO'''

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass

    # search for get config in torch


class DenseNormalGamma(nn.Module):
    def __init__(self, units):
        super(DenseNormalGamma, self).__init__()
        self.units = units  # check type
        self.dense = '''TODO'''

    def evidence(self, x):
        pass

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass

    # get config


class DenseDirichlet(nn.Module):
    def __init__(self, units):
        super(DenseDirichlet, self).__init__()
        self.units = units  # check type
        self.dense = '''TODO'''

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass


class DenseSigmoid(nn.Module):
    def __init__(self, units):
        super(DenseSigmoid, self).__init__()
        self.units = units  # check type
        self.dense = '''TODO'''

    def call(self, x):
        pass

    def compute_output_shape(self, input_shape):
        pass
