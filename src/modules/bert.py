# -*- coding: utf-8 -*-

import torch.nn as nn
from pytorch_pretrained_bert import BertModel

from .scalar_mix import ScalarMix


class BertEmbedding(nn.Module):

    def __init__(self, path, n_layers, n_out, freeze=True):
        super(BertEmbedding, self).__init__()

        self.model = BertModel.from_pretrained(path)
        self.n_layers = n_layers
        self.n_out = n_out
        self.freeze = freeze
        self.hidden_size = self.model.config.hidden_size

        self.scalar_mix = ScalarMix(n_layers)
        self.projection = nn.Linear(self.hidden_size, n_out, False)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}"
        if self.freeze:
            s += f", freeze={self.freeze}"
        s += ')'

        return s

    def forward(self, subwords):
        mask = subwords.ge(0).long()
        bert, _ = self.model(subwords, attention_mask=mask)
        bert = bert[-self.n_layers:]
        bert = self.scalar_mix(bert)[:, 1:-1]
        bert = self.projection(bert)

        return bert
