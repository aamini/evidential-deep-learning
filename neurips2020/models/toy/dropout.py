import tensorflow as tf
from tensorflow.keras.regularizers import l2
import functools

def create(
    input_shape,
    num_neurons=50,
    num_layers=1,
    activation=tf.nn.relu,
    drop_prob=0.05,
    lam=1e-3,
    l=1e-2,
    sigma=False
    ):

    options = locals().copy()

    Dense = functools.partial(tf.keras.layers.Dense, kernel_regularizer=l2(lam), bias_regularizer=l2(lam), activation=activation)
    Dropout = functools.partial(tf.keras.layers.Dropout, drop_prob)
    n_out = 2 if sigma else 1

    layers = []
    for _ in range(num_layers):
        layers.append(Dense(num_neurons))
        layers.append(Dropout())
    layers.append(Dense(n_out, activation=tf.identity))

    model = tf.keras.models.Sequential(layers)

    return model, options
