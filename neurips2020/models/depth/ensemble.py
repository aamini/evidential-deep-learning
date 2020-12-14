import tensorflow as tf
from . import dropout

def create(input_shape, num_ensembles=5, sigma=True, activation=tf.nn.relu, num_class=1):
    opts = locals().copy()

    def create_single_model():
        model, dropout_options = dropout.create(input_shape, drop_prob=0.0, sigma=sigma, activation=activation, num_class=num_class)
        return model

    models = [create_single_model() for _ in range(num_ensembles)]
    return models, opts
