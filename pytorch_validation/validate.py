from datamodules import MNISTDataModule
from models import LeNet
from torchvision import transforms
from evidential_deep_learning.pytorch.losses import *
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from torchvision.datasets.mnist import MNIST


if __name__ == '__main__':
    model: pl.LightningModule = LeNet(dropout=True)
    dm = MNISTDataModule(batch_size=32, num_workers=4)

    logger = pl_loggers.TensorBoardLogger('logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=50, logger=logger, accumulate_grad_batches=16, callbacks=[lr_monitor])
    trainer.fit(model, datamodule=dm)

    model.eval()
    model.freeze()

    num_classes = 10
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    classifications = []

    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))

    data_val = MNIST("./data/mnist",
                     train=False,
                     download=True,
                     transform=transforms.Compose([
                         transforms.Resize((28, 28)),
                         transforms.ToTensor()]))

    def rotate_img(x, deg):
        return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()

    img, _ = data_val[5]
    with torch.no_grad():
        for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
            nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

            nimg = np.clip(a=nimg, a_min=0, a_max=1)

            rimgs[:, i*28:(i+1)*28] = nimg
            trans = transforms.ToTensor()
            img_tensor = trans(nimg)
            img_tensor.unsqueeze_(0)

            outputs = model(img_tensor)
            uncertainty = Dirichlet_Uncertainty(outputs)
            alpha = Dirichlet_Evidence(outputs) + 1
            preds = Dirichlet_Predictions(outputs)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            output = outputs.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())
            lu.append(uncertainty.mean())

            scores += prob.detach().cpu().numpy() >= 0.5
            ldeg.append(deg)
            lp.append(prob.tolist())

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"]*2
    labels = labels.tolist()
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 1, 12]})

    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    labels += ["uncertainty"]
    axs[2].plot(ldeg, lu, marker="<", c="red")

    print(classifications)

    axs[0].set_title("Rotated \"1\" Digit Classifications")
    axs[0].imshow(1 - rimgs, cmap="gray")
    axs[0].axis("off")

    empty_lst = [classifications]
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")

    axs[2].legend(labels, loc='best')
    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")

    plt.savefig('pytorch_discrete_validation.png')
