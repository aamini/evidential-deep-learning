import pytorch_lightning as pl
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1):
        super(MNISTDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        full_mnist = MNIST("./data/mnist",
                               download=True,
                               train=True,
                               transform=transforms.Compose([
                                   transforms.Resize((28, 28)),
                                   transforms.ToTensor()]))
        test_size = int(0.95 * len(full_mnist))
        train_size = len(full_mnist) - test_size
        self.data_train, self.data_val = random_split(full_mnist, [test_size, train_size])

        self.data_test = MNIST("./data/mnist",
                              train=False,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((28, 28)),
                                  transforms.ToTensor()]))

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True,
                          num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False,
                          num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False,
                          num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=False)