import tensorflow as tf
import tensorflow_probability as tfp
import functools
import evidential_deep_learning as edl

def create(
    input_shape,
    num_neurons=50,
    num_layers=1,
    activation=tf.nn.relu,
    ):

    options = locals().copy()

    inputs = tf.keras.Input(input_shape)
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    output = edl.layers.DenseNormal(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model, options
