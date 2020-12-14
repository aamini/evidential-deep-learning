import argparse
import cv2
import h5py
import numpy as np
import os
import time
import tensorflow as tf

import edl
import data_loader
import models
import trainers


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="evidential", type=str,
                    choices=["evidential", "dropout", "ensemble"])
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--iters", default=60000, type=int)
parser.add_argument("--learning-rate", default=5e-5, type=float)
args = parser.parse_args()

### Try to limit GPU memory to fit ensembles on RTX 2080Ti
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=9000)])
    except RuntimeError as e:
        print(e)

### Load the data
(x_train, y_train), (x_test, y_test) = data_loader.load_depth()

### Create the trainer
if args.model == "evidential":
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="depth", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=x_train.shape[1:])
    trainer = trainer_obj(model, opts, args.learning_rate, lam=2e-1, epsilon=0., maxi_rate=0.)

elif args.model == "dropout":
    trainer_obj = trainers.Dropout
    model_generator = models.get_correct_model(dataset="depth", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=x_train.shape[1:], sigma=False)
    trainer = trainer_obj(model, opts, args.learning_rate)

elif args.model == "ensemble":
    trainer_obj = trainers.Ensemble
    model_generator = models.get_correct_model(dataset="depth", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=x_train.shape[1:], sigma=False)
    trainer = trainer_obj(model, opts, args.learning_rate)


### Train the model
model, rmse, nll = trainer.train(x_train, y_train, x_test, y_test, np.array([[1.]]), iters=args.iters, batch_size=args.batch_size, verbose=True)
tf.keras.backend.clear_session()

print("Done training!")
