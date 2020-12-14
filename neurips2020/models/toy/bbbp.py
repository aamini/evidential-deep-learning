import tensorflow as tf
import tensorflow_probability as tfp
import functools

def create(
    input_shape,
    num_neurons=100,
    num_layers=2,
    activation=tf.nn.relu,
    ):

    options = locals().copy()

    DenseReparameterization = functools.partial(tfp.layers.DenseReparameterization, activation=activation)
    layers = []
    for _ in range(num_layers):
        layers.append(DenseReparameterization(num_neurons))
    layers.append(DenseReparameterization(1, activation=tf.identity))

    model = tf.keras.models.Sequential(layers)

    return model, options
