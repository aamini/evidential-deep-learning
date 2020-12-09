# This file contains the hyperparmeters for reproducing the benchmark UCI
# dataset regression tasks. The hyperparmeters, which included the learning_rate
# and batch_size were optimized using grid search on an 80-20 train-test split
# of the dataset with the optimal resulting hyperparmeters saved in this file
# for quick reloading.

h_params = {
    'yacht': {'learning_rate': 5e-4, 'batch_size': 1},
    'naval': {'learning_rate': 5e-4, 'batch_size': 1},
    'concrete': {'learning_rate': 5e-3, 'batch_size': 1},
    'energy-efficiency': {'learning_rate': 2e-3, 'batch_size': 1},
    'kin8nm': {'learning_rate': 1e-3, 'batch_size': 1},
    'power-plant': {'learning_rate': 1e-3, 'batch_size': 2},
    'boston': {'learning_rate': 1e-3, 'batch_size': 8},
    'wine': {'learning_rate': 1e-4, 'batch_size': 32},
    'protein': {'learning_rate': 1e-3, 'batch_size': 64},
}
