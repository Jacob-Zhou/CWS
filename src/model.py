# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from src.common import pad_index
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWSModel(nn.Module):

    def __init__(self, name, conf):
        super(CWSModel, self).__init__()

        self._conf = conf
        self._name = name

        self.emb_char = None
        self.emb_bichar = None
        self.embed_dropout = None
        self.lstm = None
        self.ffn = None
        self.criterion = None

    @property
    def name(self):
        return self._name

    # create and init all the models needed according to config
    def init_models(self, char_dict, bichar_dict, label_dict):
        self.emb_char = nn.Embedding(num_embeddings=len(char_dict),
                                     embedding_dim=self._conf.n_char_embed)
        self.emb_bichar = nn.Embedding(num_embeddings=len(bichar_dict),
                                       embedding_dim=self._conf.n_char_embed)
        self.embed_dropout = nn.Dropout(self._conf.embed_dropout)

        self.lstm = nn.LSTM(input_size=self._conf.n_char_embed*2,
                            hidden_size=self._conf.n_lstm_hidden,
                            num_layers=self._conf.n_lstm_layers,
                            batch_first=True,
                            dropout=self._conf.lstm_dropout,
                            bidirectional=True)

        self.ffn = nn.Linear(in_features=self._conf.n_lstm_hidden*2,
                             out_features=len(label_dict))
        self.ffn_aux = nn.Linear(in_features=self._conf.n_lstm_hidden*2,
                                 out_features=len(label_dict))
        self.criterion = nn.CrossEntropyLoss()
        print('init models done')

    def forward(self, chars, bichars, aux=False):
        mask = chars.ne(pad_index)
        lens = mask.sum(1)
        batch_size, seq_len = chars.shape

        emb_char = self.emb_char(chars)
        emb_bichar = self.emb_bichar(bichars)
        x = self.embed_dropout(torch.cat((emb_char, emb_bichar), -1))

        sorted_lens, sorted_indices = torch.sort(lens, descending=True)
        inverse_indices = sorted_indices.argsort()
        x = pack_padded_sequence(x[sorted_indices], sorted_lens, True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = x[inverse_indices]

        x = self.ffn_aux(x) if aux else self.ffn(x)

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
