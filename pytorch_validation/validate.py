from datamodules import MNISTDataModule
from models import LeNet
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor


if __name__ == '__main__':
    model = LeNet(dropout=True)
    dm = MNISTDataModule(batch_size=32, num_workers=4)

    logger = pl_loggers.TensorBoardLogger('logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=50, logger=logger, accumulate_grad_batches=16, callbacks=[lr_monitor])
    trainer.fit(model, datamodule=dm)
