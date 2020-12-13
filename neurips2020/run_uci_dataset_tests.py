import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from scipy import stats

import evidential_deep_learning as edl
import data_loader
import trainers
import models
from models.toy.h_params import h_params


parser = argparse.ArgumentParser()
parser.add_argument("--num-trials", default=20, type=int,
                    help="Number of trials to repreat training for \
                    statistically significant results.")
parser.add_argument("--num-epochs", default=40, type=int)
parser.add_argument('--datasets', nargs='+', default=["yacht"],
                    choices=['boston', 'concrete', 'energy-efficiency',
                            'kin8nm', 'naval', 'power-plant', 'protein',
                            'wine', 'yacht'])
args = parser.parse_args()

"""" ================================================"""
training_schemes = [trainers.Evidential]
datasets = args.datasets
num_trials = args.num_trials
num_epochs = args.num_epochs
dev = "/cpu:0" # for small datasets/models cpu is faster than gpu
"""" ================================================"""

RMSE = np.zeros((len(datasets), len(training_schemes), num_trials))
NLL = np.zeros((len(datasets), len(training_schemes), num_trials))
for di, dataset in enumerate(datasets):
    for ti, trainer_obj in enumerate(training_schemes):
        for n in range(num_trials):
            (x_train, y_train), (x_test, y_test), y_scale = data_loader.load_dataset(dataset, return_as_tensor=False)
            batch_size = h_params[dataset]["batch_size"]
            num_iterations = num_epochs * x_train.shape[0]//batch_size
            done = False
            while not done:
                with tf.device(dev):
                    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
                    model, opts = model_generator.create(input_shape=x_train.shape[1:])
                    trainer = trainer_obj(model, opts, dataset, learning_rate=h_params[dataset]["learning_rate"])
                    model, rmse, nll = trainer.train(x_train, y_train, x_test, y_test, y_scale, iters=num_iterations, batch_size=batch_size, verbose=True)
                    del model
                    tf.keras.backend.clear_session()
                    done = False if np.isinf(nll) or np.isnan(nll) else True
            print("saving {} {}".format(rmse, nll))
            RMSE[di, ti, n] = rmse
            NLL[di, ti, n] = nll

RESULTS = np.hstack((RMSE, NLL))
mu = RESULTS.mean(axis=-1)
error = np.std(RESULTS, axis=-1)

print("==========================")
print("[{}]: {} pm {}".format(dataset, mu, error))
print("==========================")

print("TRAINERS: {}\nDATASETS: {}".format([trainer.__name__ for trainer in training_schemes], datasets))
print("MEAN: \n{}".format(mu))
print("ERROR: \n{}".format(error))

import pdb; pdb.set_trace()
