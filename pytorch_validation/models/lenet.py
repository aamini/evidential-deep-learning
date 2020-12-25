# MIT License
#
# Copyright (c) 2019 Douglas Brion
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from evidential_deep_learning.pytorch.losses import Dirichlet_SOS, Dirichlet_Predictions, Dirichlet_Uncertainty


class LeNet(pl.LightningModule):
    def __init__(self, dropout=False):
        super(LeNet, self).__init__()
        self.use_dropout = dropout
        self.model = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(1),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20000, 500),
            nn.ReLU(),
            nn.Dropout(0.2) if dropout else nn.Identity(),
            nn.Linear(500, 10)
        )
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def shared_inference_step(self, batch):
        inputs, labels = batch
        y = torch.eye(10, device=self.device)[labels]
        outputs = self(inputs)
        loss = Dirichlet_SOS(y, outputs, device=self.device)
        return loss, outputs
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_inference_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.shared_inference_step(batch)

        # Logging
        self.log('val_loss', loss)
        predictions = Dirichlet_Predictions(outputs)
        u = Dirichlet_Uncertainty(outputs)
        self.log('accuracy', self.accuracy(predictions, batch[1]), on_step=True, on_epoch=True, logger=True)
        self.log('mean_uncertainty', u.mean())
        tb = self.logger.experiment
        images = batch[0][:9]
        grid = torchvision.utils.make_grid(images, nrow=3)
        tb.add_image('images', grid, self.global_step)
        tb.add_graph(self, images)

        if batch_idx == 0:
            for name, param in self.named_parameters():
                tb.add_histogram(name, param.clone().cpu().data.numpy(), self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = {
            'scheduler': CosineAnnealingWarmRestarts(optimizer, T_0=3),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
