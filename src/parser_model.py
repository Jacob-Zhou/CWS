import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common import *


class ParserModel(nn.Module):
    def __init__(self, name, conf, use_cuda):
        super(ParserModel, self).__init__()
        self._conf = conf
        self._use_cuda = use_cuda
        self._name = name

        self.emb_chars = None
        self.emb_bichars = None
        self.emb_drop_layer = None
        self.lstm_layer = None
        self.mlp_layer = None
        self.loss_func = None

    @property
    def name(self):
        return self._name

    # create and init all the models needed according to config
    def init_models(self, char_dict_size, bichar_dict_size, label_dict_size):
        assert char_dict_size > 0
        assert bichar_dict_size > 0
        assert label_dict_size > 0

        self.emb_chars = nn.Embedding(num_embeddings=char_dict_size,
                                      embedding_dim=self._conf.char_emb_dim,
                                      padding_idx=padding_id)
        self.emb_bichars = nn.Embedding(num_embeddings=bichar_dict_size,
                                        embedding_dim=self._conf.char_emb_dim,
                                        padding_idx=padding_id)
        self.emb_drop_layer = nn.Dropout(self._conf.emb_dropout_ratio)

        self.lstm_layer = nn.LSTM(
            input_size=self._conf.char_emb_dim * 2,
            hidden_size=self._conf.lstm_hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=self._conf.lstm_layer_num,
            dropout=self._conf.lstm_dropout_ratio
        )

        self.mlp_layer = nn.Linear(self._conf.lstm_hidden_dim, label_dict_size)
        self.loss_func = nn.CrossEntropyLoss(reduction='sum',
                                             ignore_index=ignore_label_id)
        print('init models done')

    def reset_parameters(self):
        nn.init.normal_(self.emb_chars.weight, 0,
                        1 / self._conf.char_emb_dim ** 0.5)
        nn.init.normal_(self.emb_bichars.weight, 0,
                        1 / self._conf.char_emb_dim ** 0.5)
        nn.init.xavier_uniform_(self.mlp_layer.weight)
        # init.xavier_uniform_(self.hidden.weight)
        # bias = (3.0 / self.embedding.weight.size(1)) ** 0.5
        # init.uniform_(self.embedding.weight, -bias, bias)

    def put_models_on_gpu_if_need(self):
        if not self._use_cuda:
            return
        self.cuda()

    def forward(self, chars, bichars):
        mask = chars.gt(0)
        sen_lens = mask.sum(1)

        emb_ch, emb_bich = self.emb_chars(chars), self.emb_bichars(bichars)
        input_x = self.emb_drop_layer(torch.cat((emb_ch, emb_bich), -1))

        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        unsorted_idx = sorted_idx.argsort()
        input_x_sorted = input_x[sorted_idx]
        input_packed = pack_padded_sequence(input_x_sorted, sorted_lens, True)

        lstm_out, state = self.lstm_layer(input_packed, None)
        lstm_out_pad, _ = pad_packed_sequence(lstm_out, True)
        lstm_out_pad_unsorted = lstm_out_pad[unsorted_idx]
        # out = torch.tanh(self.hidden(out))
        mlp_out = self.mlp_layer(lstm_out_pad_unsorted)
        return mlp_out

    def get_loss(self, mlp_out, target):
        length, batch_size, _ = mlp_out.size()
        loss = self.loss_func(mlp_out.view(
            length * batch_size, -1), target.contiguous().view(-1))
        return loss

    def load_model(self, path, eval_num):
        path = os.path.join(path, 'models.%s.%d' % (self.name, eval_num))
        print(path)
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print('Load model %s done.' % path)

    def save_model(self, path, eval_num):
        path = os.path.join(path, 'models.%s.%d' % (self.name, eval_num))
        torch.save(self.state_dict(), path)
        print('Save model %s done.' % path)
