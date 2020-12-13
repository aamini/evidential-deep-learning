import tensorflow as tf

from . import dropout


# tf.enable_eager_execution()


def create(input_shape, activation=tf.nn.relu, num_class=1):
    opts = locals().copy()
    model, dropout_options = dropout.create(input_shape, drop_prob=0.0, sigma=True, activation=activation,
                                            num_class=num_class)
    return model, dropout_options
