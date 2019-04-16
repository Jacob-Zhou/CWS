# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from src.common import pad_index, unk_index
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CWSModel(nn.Module):

    def __init__(self, name, conf, use_cuda):
        super(CWSModel, self).__init__()

        self._conf = conf
        self._use_cuda = use_cuda
        self._name = name

        self.emb_char = None
        self.emb_bichar = None
        self.emb_subword = None
        self.pretrained = None
        self.emb_dropout = None
        self.lstm_layer = None
        self.ffn = None
        self.criterion = None

    @property
    def name(self):
        return self._name

    # create and init all the models needed according to config
    def init_models(self, char_dict, bichar_dict, subword_dict, label_dict):
        self.emb_char = nn.Embedding(num_embeddings=len(char_dict),
                                     embedding_dim=self._conf.n_char_emb)
        self.emb_bichar = nn.Embedding(num_embeddings=len(bichar_dict),
                                       embedding_dim=self._conf.n_char_emb)
        self.emb_subword = nn.Embedding(num_embeddings=subword_dict.init_num,
                                        embedding_dim=self._conf.n_subword_emb)
        self.pretrained = nn.Embedding.from_pretrained(subword_dict.embed)
        self.emb_dropout = nn.Dropout(self._conf.emb_dropout)

        self.lstm_layer = nn.LSTM(input_size=self._conf.n_char_emb*3,
                                  hidden_size=self._conf.n_lstm_hidden//2,
                                  num_layers=self._conf.n_lstm_layers,
                                  batch_first=True,
                                  dropout=self._conf.lstm_dropout,
                                  bidirectional=True)

        self.ffn = nn.Linear(in_features=self._conf.n_lstm_hidden,
                             out_features=len(label_dict))
        self.criterion = nn.CrossEntropyLoss()
        self.reset_parameters()
        print('init models done')

    def reset_parameters(self):
        nn.init.zeros_(self.emb_subword.weight)

    def forward(self, chars, bichars, subwords):
        mask = chars.ne(pad_index)
        ext_mask = subwords.ge(self.emb_subword.num_embeddings)
        batch_size, seq_len, max_len = subwords.shape
        lens = mask.sum(1)

        emb_ch = self.emb_char(chars)
        emb_bich = self.emb_bichar(bichars)
        emb_sub = self.pretrained(subwords)
        # set indices larger than num_embeddings to unk_index, that is,
        # make all subwords not in emb_subword but in pretrained to unk
        emb_sub += self.emb_subword(subwords.masked_fill_(ext_mask, unk_index))
        emb_sub = emb_sub.masked_fill_(subwords.eq(pad_index).unsqueeze(-1), 0)
        emb_sub = emb_sub.mean(dim=-2)
        x = self.emb_dropout(torch.cat((emb_ch, emb_bich, emb_sub), -1))

        sorted_lens, sorted_indices = torch.sort(lens, descending=True)
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
