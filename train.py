import os
from typing import Any

import torch.cuda
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import optim, nn, utils, tensor
from torch.utils.data import DataLoader
from cifar10 import load_cifar10_datasets
from model import AlexNet

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LitModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        #self.model_device = device
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inp, target = batch

        out = self.model(inp)
        loss = self.criterion(out, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, target = batch

        out = self.model(inp)
        loss = self.criterion(out, target)
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.001)
        return optimiser


def main():
    model = AlexNet(32, 10)
    litmodel = LitModule(model)
    train_dset, test_dset = load_cifar10_datasets()
    train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=64)

    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    trainer = L.Trainer(max_epochs=30,
                        logger=logger,
                        accelerator="gpu")

    trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
