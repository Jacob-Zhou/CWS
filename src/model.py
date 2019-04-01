# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from src.common import pad_index, unk_index
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class CWSModel(nn.Module):

    def __init__(self, name, conf, use_cuda):
        super(CWSModel, self).__init__()

        self._conf = conf
        self._use_cuda = use_cuda
        self._name = name

        self.emb_chars = None
        self.emb_bichars = None
        self.emb_subwords = None
        self.pretrained = None
        self.emb_drop_layer = None
        self.lstm_layer = None
        self.ffn = None
        self.criterion = None

    @property
    def name(self):
        return self._name

    # create and init all the models needed according to config
    def init_models(self, char_dict_size, bichar_dict_size,
                    subword_dict_size, label_dict_size):
        assert char_dict_size > 0
        assert bichar_dict_size > 0
        assert subword_dict_size > 0
        assert label_dict_size > 0

        self.emb_chars = nn.Embedding(num_embeddings=char_dict_size,
                                      embedding_dim=self._conf.char_emb_dim)
        self.emb_bichars = nn.Embedding(num_embeddings=bichar_dict_size,
                                        embedding_dim=self._conf.char_emb_dim)
        self.emb_subwords = nn.Embedding(num_embeddings=subword_dict_size,
                                         embedding_dim=self._conf.subword_emb_dim)
        self.emb_drop_layer = nn.Dropout(self._conf.emb_dropout)

        self.lstm_layer = nn.LSTM(input_size=self._conf.char_emb_dim*2,
                                  hidden_size=self._conf.lstm_hidden_dim//2,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers=self._conf.lstm_layer_num,
                                  dropout=self._conf.lstm_dropout)

        self.ffn = nn.Linear(self._conf.lstm_hidden_dim + self._conf.subword_emb_dim,
                             label_dict_size)
        self.criterion = nn.CrossEntropyLoss()
        print('init models done')

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ffn.weight)

    def load_pretrained(self, embeddings):
        self.pretrained = nn.Embedding.from_pretrained(embeddings=embeddings,
                                                       freeze=False)
        nn.init.zeros_(self.emb_subwords.weight)

    def forward(self, chars, bichars, subwords):
        mask = chars.ne(pad_index)
        batch_size, seq_len, word_length = subwords.shape
        lens = mask.sum(1)

        emb_ch = self.emb_chars(chars)
        emb_bich = self.emb_bichars(bichars)
        # emb_subword = self.pretrained(subwords)
        # # set indices larger than num_embeddings to unk_index, that is
        # # make all subwords not in emb_subword but in pretrained to unk
        # emb_subword += self.emb_subwords(
        #     subwords.masked_fill_(subwords.ge(self.emb_subwords.num_embeddings),
        #                           unk_index)
        # )
        emb_subword = self.emb_subwords(
            subwords.masked_fill_(subwords.ge(self.emb_subwords.num_embeddings),
                                  unk_index)
        )
        x = self.emb_drop_layer(torch.cat((emb_ch, emb_bich), -1))

        sorted_lens, sorted_indices = torch.sort(lens, descending=True)
        inverse_indices = sorted_indices.argsort()
        x = pack_padded_sequence(x[sorted_indices], sorted_lens, True)
        x, _ = self.lstm_layer(x)
        x, _ = pad_packed_sequence(x, True)
        x = x[inverse_indices].transpose(0, 1)

        x = x.unsqueeze(1) - x
        x_f, x_b = x.chunk(2, dim=-1)
        x_f = x_f[1:-1, :-2].permute(2, 1, 0, 3)
        x_b = x_b[1:-1, 2:].permute(2, 0, 1, 3)
        x_span = torch.cat([x_f, x_b], dim=-1)
        x_span = pad_sequence([
            x_span.diagonal(offset=i, dim1=1, dim2=2).permute(2, 0, 1)
            for i in range(word_length)
        ], True)
        x_span = x_span.transpose(0, 2)

        x = torch.cat([x_span, emb_subword], dim=-1)
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
