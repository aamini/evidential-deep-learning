import tensorflow as tf
import numpy as np

def normalize(x, crop=True):
    if len(x.shape) == 4:
        x = x[:, 10:-10,5:-5]
    else:
        x = x[10:-10,5:-5]

    min = tf.reduce_min(x, axis=(-1,-2,-3), keepdims=True)
    max = tf.reduce_max(x, axis=(-1,-2,-3), keepdims=True)
    return (x - min)/(max-min)

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result
