import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D


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


# Conv2DNormalGamma(32, (5,5))
