import os
from typing import Any

import torch.cuda
from torchmetrics.classification import Accuracy
from torch import optim, nn, utils, tensor
from torch.utils.data import DataLoader
from cifar10 import load_cifar10_datasets
from model import choose_model

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

from fire import Fire

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LitModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        #self.model_device = device
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        inp, target = batch

        out = self.model(inp)
        loss = self.criterion(out, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, target = batch

        out = self.model(inp)
        preds = torch.argmax(out, dim=1)
        self.accuracy.update(preds, target)
        loss = self.criterion(out, target)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=4e-5, weight_decay=0.001)
        return optimiser


def main(original_model_version: bool):
    print(original_model_version)
    torch.manual_seed(0) # for reproducibility.
    model = choose_model(original_model_version)

    litmodel = LitModule(model)
    train_dset, test_dset = load_cifar10_datasets(original_model_version)
    train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=64)

    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    early_stopping = EarlyStopping("val_loss", patience=6, mode="min")

    trainer = L.Trainer(max_epochs=35,
                        logger=logger,
                        accelerator="gpu",
                        callbacks=[early_stopping])

    trainer.fit(model=litmodel, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    Fire(main)
