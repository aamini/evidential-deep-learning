import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch_Evidential as edl


def main():
    x_train, y_train = mydata(-4, 4, 1000)
    x_test, y_test = mydata(-4, 4, 1000, train=False)
    # transform to tensor
    x_t = torch.from_numpy(x_train)
    x_te = torch.from_numpy(x_test)
    y_t = torch.from_numpy(y_train)
    y_te = torch.from_numpy(y_test)
    # model
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        edl.torch_layers.DenseNormalGamma(64, 1),

    )

    def loss_fn(true, pred):
        return edl.torch_losses.continuous.EvidentialRegression(true, pred, coeff=1e-2)

    lr = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # training loop
    for t in range(50):

        logits = model(x_t)

        loss = loss_fn(y_t, logits)

        print(t+1, loss.item())

        # model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for param in model.parameters():
                param -= lr*param.grad

    # prediction
    y_pred = model(x_te)
    assert y_pred.shape == (1000, 4)
    plot_predictions(x_t, y_t, x_te, y_te, y_pred)


def mydata(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 3*np.ones_like(x) if train else np.zeros_like(x)
    y = x**3 + np.random.normal(0, sigma).astype(np.float32)

    return x, y


def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, k=0):
    x_test = x_test[:, 0]

    mu, v, alpha, beta = torch.split(y_pred, [1, 1, 1, 1], dim=-1)
    assert mu.shape == (1000, 1)
    assert v.shape == (1000, 1)
    assert alpha.shape == (1000, 1)
    assert beta.shape == (1000, 1)
    mu = mu[:, 0]
    """# torch.Tensor.detach(): to change the tensor's mode so that it doesn't require grad (require_grad=False)"""
    mu = mu.detach().numpy()
    v = v.detach().numpy()
    alpha = alpha.detach().numpy()
    beta = beta.detach().numpy()
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    plt.plot(x_test, y_test, 'r--', zorder=2, label="True")
    plt.plot(x_test, mu,
             color='#007cab', zorder=3, label="Pred")
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
