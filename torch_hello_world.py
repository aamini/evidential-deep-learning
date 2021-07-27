from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


class LinearNormalGamma(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_chanels, out_channels*4)

    def evidence(self, x):
        return torch.log(torch.exp(x) + 1)

    def forward(self, x):
        pred = self.linear(x).view(x.shape[0], -1, 4)
        mu, logv, logalpha, logbeta = [w.squeeze(-1) for w in torch.split(pred, 1, dim=-1)]
        return mu, self.evidence(logv), self.evidence(logalpha) + 1, self.evidence(logbeta)


def nig_nll(y, gamma, v, alpha, beta):
    two_blambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(two_blambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)

    return nll


def nig_reg(y, gamma, v, alpha, beta):
    error = F.l1_loss(y, gamma, reduction="none")
    evi = 2 * v + alpha
    return error * evi


def evidential_regresssion_loss(y, pred, coeff=1.0):
    gamma, v, alpha, beta = pred
    loss_nll = nig_nll(y, gamma, v, alpha, beta)
    loss_reg = nig_reg(y, gamma, v, alpha, beta)
    return loss_nll.mean() + coeff * loss_reg.mean()


def main():
    x_train, y_train = my_data(-4, 4, 1000)
    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)
    train_iter = iter(train_loader)
    x_test, y_test = my_data(-7, 7, 1000, train=False)

    model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            LinearNormalGamma(64, 1))
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for t in tqdm(range(1500)):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        loss = evidential_regresssion_loss(y, model(x), 1e-2)
        if t % 10 == 9:
            print(t, loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    y_pred = torch.cat(model(x_test), dim=-1)
    plot_predictions(*[v.detach().numpy() for v in [x_train, y_train, x_test, y_test, y_pred]])


def my_data(x_min, x_max, n, train=True):
    x = np.linspace(x_min, x_max, n)
    x = np.expand_dims(x, -1).astype(np.float32)

    sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    y = x ** 3 + np.random.normal(0, sigma).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)
  

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = np.split(y_pred, 4, axis=-1)
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
