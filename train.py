import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Chatbot
from dataset import ChatbotDataset

def define_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--freeze_bert',
        action='store_true',
    )
    p.add_argument(
        '--dropout_p',
        type=float,
        default=0.2,
    )
    p.add_argument(
        '--n_label',
        type=int,
        default=4,
    )
    p.add_argument(
        '--lr',
        type=float,
        default=0.1
    )
    p.add_argument(
        '--n_epoch',
        type=int,
        default=20
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=64
    )

    return p.parse_args(args=[])


class Trainer(pl.LightningModule):
    def __init__(self, model, config):
        super(Trainer, self).__init__()
        self.model = model
        self.config = config
        self.loss_func = self.configure_loss_function()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x,y = batch

        y_hat = self.model(x)
        train_loss = self.loss_func(y_hat, y)

        return train_loss

    def validation_step(self, batch, batch_idx):
        print(batch)
        x, y = batch

        y_hat = self.model(x)
        val_loss = self.loss_func(y_hat, y)
        self.log('val_loss', val_loss)

        return val_loss

    def validation_epoch_end(self, loss):
        pass
        # torch.save({'model':self.model.state_dict(), 'optim':self.optimizers.state_dict()} ,model_fn)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def configure_loss_function(self):
        return  nn.NLLLoss(
            weight = torch.ones(self.config.n_label),
        )


if __name__ == '__main__':
    config = define_args()

    chatbot = Trainer(Chatbot(config), config)
    dataloader = ChatbotDataset('chatbot/temi_data.xlsx', config.batch_size)

    model_fn = '{epoch:02d}-{val_loss:.2f}.pth'
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="./chatbot/model_ckpt",
        filename="temi-model-"+model_fn,
    )

    trainer = pl.Trainer(accelerator="mps", max_epochs=config.n_epoch, callbacks=checkpoint_callback)
    trainer.fit(chatbot, dataloader)
