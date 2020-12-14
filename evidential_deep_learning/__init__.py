# TODO: This is pretty hacky namespace manipulation but it works
import sys

self = sys.modules[__name__]

default_backend = 'tf'

self.torch_avail = False
try:
    import torch

    self.torch_avail = True
    self.backend = 'torch'
except ImportError:
    pass

self.tf_avail = False
try:
    import tensorflow as tf

    self.tf_avail = True
    self.backend = 'tf'
except ImportError:
    pass

if not (self.torch_avail or self.tf_avail):
    raise ImportError("Must have either PyTorch or Tensorflow available")

if self.torch_avail and self.tf_avail:
    self.backend = default_backend


def set_backend(backend):
    if backend == 'tf':
        if not self.tf_avail:
            raise ImportError(f"Cannot use backend 'tf' if tensorflow is not installed")
        from .tf import layers as layers
        from .tf import losses as losses
        self.layers = layers
        self.losses = losses
    elif backend == 'torch':
        if not self.torch_avail:
            raise ImportError(f"Cannot use backend 'torch' if pytorch is not installed")
        from .pytorch import layers as layers
        from .pytorch import losses as losses
        self.layers = layers
        self.losses = losses
    else:
        raise ValueError(f"Invalid choice of backend: {backend}, options are 'tf' or 'torch'")


def get_backend():
    return self.backend


self.get_backend = get_backend
self.set_backend = set_backend
self.set_backend(self.backend)
