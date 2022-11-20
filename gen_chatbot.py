import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import genChatbotDataset

import hyperparameters as hps

from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")

class AbstractiveChatbot(pl.LightningModule):

    def __init__(self, model, config):
        super(AbstractiveChatbot, self).__init__()
        self.model = model
        self.config = config
        # self.loss_func = self.configure_loss_function()

        if config.freeze_encoder:
            for param in self.model.model.encdoer.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch[0:2], batch[2]

        y_hat = self.model(x, labels=y)
        # train_loss = self.loss_func(y_hat, y)
        train_loss = y_hat.loss

        return train_loss

    def validation_step(self, batch, batch_idx):
        print(batch)
        x, y = batch[0:2], batch[2]

        y_hat = self.model(x, labels=y)
        # val_loss = self.loss_func(y_hat, y)
        val_loss = y_hat.loss
        self.log('val_loss', val_loss)

        return val_loss

    def validation_epoch_end(self, loss):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    # def configure_loss_function(self):
    #     return nn.NLLLoss(
    #         weight=torch.ones(self.config.n_label),
    #     )

if __name__ == '__main__':
    config = hps.define_args()

    dataloader = genChatbotDataset('개인및관계df', config.batch_size)
    model.resize_token_embeddings(len(dataloader.tokenizer))
    chatbot = AbstractiveChatbot(model, config)

    model_fn = '{epoch:02d}-{val_loss:.2f}.pth'
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="./chatbot/model_ckpt",
        filename="temi-genChatbotModel-" + model_fn,
    )

    trainer = pl.Trainer(accelerator="mps", devices=1, max_epochs=config.n_epoch, callbacks=checkpoint_callback)
    trainer.fit(chatbot, dataloader)


