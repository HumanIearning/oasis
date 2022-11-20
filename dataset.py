import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import BertTokenizer, MBart50TokenizerFast

class ChatbotDataset(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super(ChatbotDataset, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def setup(self, stage=None):
        self.train_data = pd.read_excel(self.data_dir)
        self.train_data = [[self.tokenizer.encode_plus(x, return_tensors='pt', max_length=32, pad_to_max_length=True), y] for x, y in
             zip(self.train_data['x'], self.train_data['y'])]

        self.test_data = pd.read_excel(self.data_dir)
        self.test_data = [[self.tokenizer.encode_plus(x, return_tensors='pt', max_length=32, pad_to_max_length=True), y] for x, y in
                      zip(self.test_data['x'], self.test_data['y'])]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class genChatbotDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super(genChatbotDataset, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = MBart50TokenizerFast.from_pretrained("./tokenizer/")

    def prepare_data(self):
        self.train_data = pd.read_csv('data/Validation/'+self.data_dir, sep='\t', names=['x', 'y'])
        self.test_data = pd.read_csv('data/Training/'+self.data_dir, sep='\t', names=['x', 'y'])

    def setup(self, stage=None):
        self.train_data = [self.tokenizer(list(self.train_data['x'].values), return_tensors='pt', truncation=True, padding='max_length', max_length=200),
                           self.tokenizer(list(self.train_data['y'].values), return_tensors='pt', truncation=True, padding='max_length', max_length=200, return_attention_mask=False)]
        self.test_data = [self.tokenizer(list(self.test_data['x'].values), return_tensors='pt', truncation=True, padding='max_length', max_length=200),
                          self.tokenizer(list(self.test_data['y'].values), return_tensors='pt', truncation=True, padding='max_length', max_length=200, return_attention_mask=False)]

    def train_dataloader(self):
        return DataLoader([self.train_data[0].input_ids, self.train_data[0].attention_mask, self.train_data[1].input_ids], batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader([self.test_data[0].input_ids, self.test_data[0].attention_mask, self.test_data[1].input_ids], batch_size=self.batch_size, num_workers=8)
