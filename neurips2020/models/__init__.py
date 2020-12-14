from .toy import bbbp
from .toy import dropout
from .toy import ensemble
from .toy import evidential
from .toy import gaussian
from .toy import deterministic
from .toy.h_params import h_params

from .depth import bbbp
from .depth import dropout
from .depth import ensemble
from .depth import evidential
from .depth import gaussian
from .depth import deterministic


def get_correct_model(dataset, trainer):
    """ Hacky helper function to grab the right model for a given dataset and trainer. """
    dataset_loader = globals()[dataset]
    trainer_lookup = trainer.__name__.lower()
    model_pointer = dataset_loader.__dict__[trainer_lookup]
    return model_pointer

def load_depth_model(path, compile=False):
    import glob
    import tensorflow as tf
    import edl

    model_paths = glob.glob(path)
    if model_paths == []:
        model_paths = [path]

    custom_objects ={'Conv2DNormal': edl.layers.Conv2DNormal,
        'Conv2DNormalGamma': edl.layers.Conv2DNormalGamma}

    models = [tf.keras.models.load_model(model_path, custom_objects, compile=compile) for model_path in model_paths]
    if len(models) == 1:
        models = models[0]

    return models
