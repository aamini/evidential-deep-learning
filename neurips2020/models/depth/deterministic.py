import tensorflow as tf

from . import dropout


def create(input_shape, activation=tf.nn.relu, num_class=1):
    opts = locals().copy()
    model, opts = dropout.create(input_shape, drop_prob=0.0, sigma=False, activation=activation, num_class=num_class)
    return model, opts
