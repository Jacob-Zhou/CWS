# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from src.common import pad_index
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWSModel(nn.Module):

    def __init__(self, name, conf, use_cuda):
        super(CWSModel, self).__init__()

        self._conf = conf
        self._use_cuda = use_cuda
        self._name = name

        self.emb_char = None
        self.emb_bichar = None
        self.emb_drop_layer = None
        self.lstm_layer = None
        self.ffn = None
        self.criterion = None

    @property
    def name(self):
        return self._name

    # create and init all the models needed according to config
    def init_models(self, char_dict, bichar_dict, label_dict):
        self.emb_char = nn.Embedding(num_embeddings=len(char_dict),
                                     embedding_dim=self._conf.char_emb_dim)
        self.emb_bichar = nn.Embedding(num_embeddings=len(bichar_dict),
                                       embedding_dim=self._conf.char_emb_dim)
        self.emb_drop_layer = nn.Dropout(self._conf.emb_dropout)

        self.lstm_layer = nn.LSTM(input_size=self._conf.char_emb_dim*2,
                                  hidden_size=self._conf.lstm_hidden_dim//2,
                                  num_layers=self._conf.num_lstm_layers,
                                  batch_first=True,
                                  dropout=self._conf.lstm_dropout,
                                  bidirectional=True)

        self.ffn = nn.Linear(in_features=self._conf.lstm_hidden_dim,
                             out_features=len(label_dict))
        self.criterion = nn.CrossEntropyLoss()
        print('init models done')

    def forward(self, chars, bichars):
        mask = chars.ne(pad_index)
        sen_lens = mask.sum(1)

        emb_ch = self.emb_char(chars)
        emb_bichar = self.emb_bichar(bichars)
        x = self.emb_drop_layer(torch.cat((emb_ch, emb_bichar), -1))

        sorted_lens, sorted_indices = torch.sort(sen_lens, descending=True)
        inverse_indices = sorted_indices.argsort()
        x = pack_padded_sequence(x[sorted_indices], sorted_lens, True)
        x, _ = self.lstm_layer(x)
        x, _ = pad_packed_sequence(x, True)
        x = x[inverse_indices]

        x = self.ffn(x)

        return x

    def get_loss(self, out, target, mask):
        return self.criterion(out[mask], target[mask])

    def load_model(self, path, eval_num):
        path = os.path.join(path, 'models.%s.%d' % (self.name, eval_num))
        print(path)
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print('Load model %s done.' % path)

    def save_model(self, path, eval_num):
        path = os.path.join(path, 'models.%s.%d' % (self.name, eval_num))
        torch.save(self.state_dict(), path)
        print('Save model %s done.' % path)
