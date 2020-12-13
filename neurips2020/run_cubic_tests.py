import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
import os

import evidential_deep_learning
import data_loader
import trainers
import models

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

save_fig_dir = "./figs/toy"
batch_size = 128
iterations = 5000
show = True

noise_changing = False
train_bounds = [[-4, 4]]
x_train = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in train_bounds]).reshape(-1,1)
y_train, sigma_train = data_loader.generate_cubic(x_train, noise=True)

test_bounds = [[-7,+7]]
x_test = np.concatenate([np.linspace(xmin, xmax, 1000) for (xmin, xmax) in test_bounds]).reshape(-1,1)
y_test, sigma_test = data_loader.generate_cubic(x_test, noise=False)

### Plotting helper functions ###
def plot_scatter_with_var(mu, var, path, n_stds=3):
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(x_test[:,0], (mu-k*var)[:,0], (mu+k*var)[:,0], alpha=0.3, edgecolor=None, facecolor='#00aeef', linewidth=0, antialiased=True, zorder=1)

    plt.plot(x_test, y_test, 'r--', zorder=2)
    plt.plot(x_test, mu, color='#007cab', zorder=3)
    plt.gca().set_xlim(*test_bounds)
    plt.gca().set_ylim(-150,150)
    plt.title(path)
    plt.savefig(path, transparent=True)
    if show:
        plt.show()
    plt.clf()

def plot_ng(model, save="ng", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    outputs = model(x_test_input)
    mu, v, alpha, beta = tf.split(outputs, 4, axis=1)

    epistemic = np.sqrt(beta/(v*(alpha-1)))
    epistemic = np.minimum(epistemic, 1e3) # clip the unc for vis
    plot_scatter_with_var(mu, epistemic, path=save+ext, n_stds=3)

def plot_ensemble(models, save="ensemble", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = tf.stack([model(x_test_input, training=False) for model in models], axis=0) #forward pass
    mus, sigmas = tf.split(preds, 2, axis=-1)

    mean_mu = tf.reduce_mean(mus, axis=0)
    epistemic = tf.math.reduce_std(mus, axis=0) + tf.reduce_mean(sigmas, axis=0)
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)

def plot_dropout(model, save="dropout", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = tf.stack([model(x_test_input, training=True) for _ in range(15)], axis=0) #forward pass
    mus, logvar = tf.split(preds, 2, axis=-1)
    var = tf.exp(logvar)

    mean_mu = tf.reduce_mean(mus, axis=0)
    epistemic = tf.math.reduce_std(mus, axis=0) + tf.reduce_mean(var**0.5, axis=0)
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)

def plot_bbbp(model, save="bbbp", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = tf.stack([model(x_test_input, training=True) for _ in range(15)], axis=0) #forward pass

    mean_mu = tf.reduce_mean(preds, axis=0)
    epistemic = tf.math.reduce_std(preds, axis=0)
    plot_scatter_with_var(mean_mu, epistemic, path=save+ext, n_stds=3)

def plot_gaussian(model, save="gaussian", ext=".pdf"):
    x_test_input = tf.convert_to_tensor(x_test, tf.float32)
    preds = model(x_test_input, training=False) #forward pass
    mu, sigma = tf.split(preds, 2, axis=-1)
    plot_scatter_with_var(mu, sigma, path=save+ext, n_stds=3)



#### Different toy configurations to train and plot
def evidence_reg_2_layers_50_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=50, num_layers=2)
    trainer = trainer_obj(model, opts, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_2_layer_50_neurons"))

def evidence_reg_2_layers_100_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=2)
    trainer = trainer_obj(model, opts, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_2_layers_100_neurons"))

def evidence_reg_4_layers_50_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=50, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_4_layers_50_neurons"))

def evidence_reg_4_layers_100_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=5e-3, lam=1e-2, maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_reg_4_layers_100_neurons"))

def evidence_noreg_4_layers_50_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=50, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=5e-3, lam=0., maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_noreg_4_layers_50_neurons"))

def evidence_noreg_4_layers_100_neurons():
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=5e-3, lam=0., maxi_rate=0.)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ng(model, os.path.join(save_fig_dir,"evidence_noreg_4_layers_100_neurons"))

def ensemble_4_layers_100_neurons():
    trainer_obj = trainers.Ensemble
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=5e-3)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_ensemble(model, os.path.join(save_fig_dir,"ensemble_4_layers_100_neurons"))

def gaussian_4_layers_100_neurons():
    trainer_obj = trainers.Gaussian
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=5e-3)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_gaussian(model, os.path.join(save_fig_dir,"gaussian_4_layers_100_neurons"))

def dropout_4_layers_100_neurons():
    trainer_obj = trainers.Dropout
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4, sigma=True)
    trainer = trainer_obj(model, opts, learning_rate=5e-3)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_dropout(model, os.path.join(save_fig_dir,"dropout_4_layers_100_neurons"))

def bbbp_4_layers_100_neurons():
    trainer_obj = trainers.BBBP
    model_generator = models.get_correct_model(dataset="toy", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=1, num_neurons=100, num_layers=4)
    trainer = trainer_obj(model, opts, learning_rate=1e-3)
    model, rmse, nll = trainer.train(x_train, y_train, x_train, y_train, np.array([[1.]]), iters=iterations, batch_size=batch_size, verbose=True)
    plot_bbbp(model, os.path.join(save_fig_dir,"bbbp_4_layers_100_neurons"))


### Main file to run the different methods and compare results ###
if __name__ == "__main__":
    Path(save_fig_dir).mkdir(parents=True, exist_ok=True)

    evidence_reg_4_layers_100_neurons()
    # evidence_noreg_4_layers_100_neurons()

    # ensemble_4_layers_100_neurons()
    # gaussian_4_layers_100_neurons()
    # dropout_4_layers_100_neurons()
    # bbbp_4_layers_100_neurons()

    print(f"Done! Figures saved to {save_fig_dir}")
