import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate, ZeroPadding2D

import functools
from evidential_deep_learning.layers import Conv2DNormal

def create(input_shape, activation=tf.nn.relu, num_class=1):
    opts = locals().copy()
    model, opts = dropout.create(input_shape, drop_prob=0.0, sigma=False, activation=activation, num_class=num_class)
    return model, opts
