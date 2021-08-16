import functools
import numpy as np
import matplotlib.pyplot as plt
#import evidential_deep_learning as edl
import torch_ver as edl
import torch
import torch.nn as nn

"""
in torch, you may have to assign data type and the device
dtype = torch.float
device = torch.device('cpu')
device = torch.device('cuda:0')
"""


def main():
    # Create some training and testing data
    x_train, y_train = my_data(-4, 4, 1000)
    x_test, y_test = my_data(-7, 7, 1000, train=False)

    # Define our model with an evidential output
    """model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        edl.layers.DenseNormalGamma(1),
    ])"""
    model = nn.Sequential(
        # in_features missing
        nn.Linear(in_features=1,out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64,out_features=64),
        nn.ReLU(),
        edl.torch_layers.DenseNormalGamma(64,1),
    )

    # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(true, pred):
        """return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)"""
        return edl.torch_losses.EvidentialRegression(true, pred, coeff=1e-2)

    # Compile and fit the model!
    """model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=EvidentialRegressionLoss)
    model.fit(x_train, y_train, batch_size=100, epochs=500)"""

    # Predict and plot using the trained model
    """y_pred = model(x_test)
    plot_predictions(x_train, y_train, x_test, y_test, y_pred)"""

    """pytorch uses loss.backward() to backprop gradients"""
    learning_rate = 1e-4
    for i in range(500):
        # check if we need to unsqueeze it
        # input must be a tensor, not a numpy.ndarray
        #y_hat = model(x_train)
        x = torch.from_numpy(x_train)
        #x = torch.unsqueeze(x,0)
        y_hat = model(x)

        loss = EvidentialRegressionLoss(y_train, y_hat)

        if i % 50 == 49:
            print(t, loss.item())

        # clear gradient before backprop
        model.zero_grad()

        # calculate and pass the gradient
        loss.backward()

        # gradient descent (what optimizer does in tensorflow)
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate*param.grad

    # prediction
    #y_pred = model(x_test)
    #plot_predictions(x_train, y_train, x_test, y_test, y_pred)

    # Done!!


"""helper functions below"""


def my_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + np.random.normal(0, sigma).astype(np.float32)

    return x, y


def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = torch.split(y_pred, 4, dim=-1)
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
    plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred")
    plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    plt.gca().set_ylim(-150, 150)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
