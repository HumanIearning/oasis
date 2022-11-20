import numpy as np

import torch
import torch.nn as nn

from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, dropout_p, freeze_bert):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.d_model = self.bert.config.hidden_size



        self.fc = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, x, attention_mask):
        x = self.bert(x, attention_mask)
        # |x| = [last_hidden_state=(bs, max_length, d_model), pooler_output=(bs, d_model)]
        h = self.fc(x[1])

        return h

class Chatbot(nn.Module):
    def __init__(self, config):
        super(Chatbot, self).__init__()

        self.classifier = BertClassifier(config.dropout_p, config.freeze_bert)
        self.bert_config = self.classifier.bert.config

        self.generator = nn.Sequential(
            nn.LayerNorm(self.bert_config.hidden_size),
            nn.Linear(self.bert_config.hidden_size, config.n_label),
            nn.LogSoftmax(dim=-1),
        )


    def forward(self, x):
        return self.generator(self.classifier(x['input_ids'].squeeze(1), x['attention_mask'].squeeze(1)))

