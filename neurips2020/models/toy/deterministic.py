import tensorflow as tf
import functools

def create(
    input_shape,
    num_neurons=100,
    num_layers=2,
    activation=tf.nn.relu,
    ):

    options = locals().copy()

    Dense = functools.partial(tf.keras.layers.Dense, activation=activation)

    layers = []
    for _ in range(num_layers):
        layers.append(Dense(num_neurons))
    layers.append(Dense(1, activation=tf.identity))

    model = tf.keras.models.Sequential(layers)

    return model, options
