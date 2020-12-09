import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate, ZeroPadding2D, SpatialDropout2D
import functools

from evidential_deep_learning.layers import Conv2DNormal
from . import dropout

def create(input_shape, activation=tf.nn.relu, num_class=1):
    opts = locals().copy()
    model, dropout_options = dropout.create(input_shape, drop_prob=0.0, sigma=True, activation=activation, num_class=num_class)
    return model, dropout_options
