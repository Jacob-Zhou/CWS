# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from src.common import pad_index, unk_index
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class CWSModel(nn.Module):

    def __init__(self, name, conf):
        super(CWSModel, self).__init__()

        self._conf = conf
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
                                  hidden_size=self._conf.n_lstm_hidden,
                                  num_layers=self._conf.n_lstm_layers,
                                  batch_first=True,
                                  dropout=self._conf.lstm_dropout,
                                  bidirectional=True)

        self.ffn_char = nn.Linear(in_features=self._conf.n_lstm_hidden*6,
                                  out_features=len(label_dict) - 1)
        self.ffn_subword = nn.Linear(in_features=self._conf.n_lstm_hidden*6,
                                     out_features=len(label_dict))
        self.criterion = nn.CrossEntropyLoss()
        self.reset_parameters()
        print('init models done')

    def reset_parameters(self):
        nn.init.zeros_(self.emb_subword.weight)

    def forward(self, chars, bichars, subwords):
        mask = chars.ne(pad_index)
        subword_mask = subwords.ne(pad_index)
        # set indices larger than num_embeddings to unk_index, that is,
        # make all subwords not in emb_subword but in pretrained to unk
        ext_mask = subwords.ge(self.emb_subword.num_embeddings)
        ext_subwords = subwords.masked_fill(ext_mask, unk_index)
        batch_size, seq_len, max_len = subwords.shape
        lens = mask.sum(1)

        emb_char = self.emb_char(chars)
        emb_bichar = self.emb_bichar(bichars)
        emb_subword = self.pretrained(subwords)
        emb_subword += self.emb_subword(ext_subwords)
        emb_subword = emb_subword.masked_fill_(~subword_mask.unsqueeze(-1), 0)
        emb_subword = emb_subword.mean(dim=-2)
        emb = torch.cat((emb_char, emb_bichar, emb_subword), dim=-1)
        x = self.emb_dropout(emb)

        sorted_lens, sorted_indices = torch.sort(lens, descending=True)
        inverse_indices = sorted_indices.argsort()
        x = pack_padded_sequence(x[sorted_indices], sorted_lens, True)
        x, _ = self.lstm_layer(x)
        x, _ = pad_packed_sequence(x, True)
        x = x[inverse_indices].transpose(0, 1)

        f_x, b_x = x.chunk(2, dim=-1)
        h_sub = pad_sequence([x[i:i+max_len] for i in range(1, seq_len - 1)])
        f_sub = pad_sequence([f_x[i:i+max_len] for i in range(1, seq_len - 1)])
        b_sub = pad_sequence([b_x[i:i+max_len] for i in range(2, seq_len)])

        h_x = x[1:-1].expand_as(h_sub)
        f_span = f_sub - f_x[0:-2]
        b_span = b_x[1:-1] - b_sub

        x = torch.cat((h_x, h_sub, f_span, b_span), dim=-1)
        x_char, x_subword = x.split([1, max_len - 1])
        x_char = self.ffn_char(x_char).transpose(0, 2)
        x_subword = self.ffn_subword(x_subword).transpose(0, 2)

        return x_char, x_subword

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
