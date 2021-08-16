import torch
import torch.nn as nn


class DenseNormal(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseNormal, self).__init__()
        # check type
        self.in_features = in_features
        self.out_features = out_features
        # Dense in tf equivalent to Linear in torch
        self.dense = nn.Linear(in_features, out_features)

    def call(self, x):
        output = self.dense(x)
        mu, logsigma = torch.split(output, 2, dim=-1)
        softplus = nn.Softplus()
        sigma = softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.out_features)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        return base_config


class DenseNormalGamma(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseNormalGamma, self).__init__()
        # check data type
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dense = nn.Linear(self.in_features, 4*self.out_features)

    def evidence(self, x):
        return nn.Softplus(x)
        softplus = nn.Softplus()
        return softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(output, 4, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4*self.out_features)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config


class DenseDirichlet(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseDirichlet, self).__init__()
        # check dtype
        self.in_features = in_features
        self.out_features = out_features
        self.dense = nn.Linear(in_features, out_features)

    def call(self, x):
        output = self.dense(x)
        evidence = torch.exp(output)
        alpha = evidence + 1
        """# check keepsdim argument in tf"""
        prob = alpha / torch.sum(alpha, 1)
        return torch.cat([alpha, prob], dim=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2*self.out_features)


class DenseSigmoid(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseSigmoid, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dense = nn.Linear(in_features, out_features)

    def call(self, x):
        logits = self.dense(x)
        prob = torch.sigmoid(logits)
        return [logits, prob]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)
