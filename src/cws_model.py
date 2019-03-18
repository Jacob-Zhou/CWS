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
                                      embedding_dim=self._conf.char_emb_dim)
        self.emb_bichars = nn.Embedding(num_embeddings=bichar_dict_size,
                                        embedding_dim=self._conf.char_emb_dim)
        self.emb_drop_layer = nn.Dropout(self._conf.emb_dropout)

        self.lstm_layer = nn.LSTM(input_size=self._conf.char_emb_dim*2,
                                  hidden_size=self._conf.lstm_hidden_dim//2,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers=self._conf.lstm_layer_num,
                                  dropout=self._conf.lstm_dropout)

        self.mlp_layer = nn.Linear(self._conf.lstm_hidden_dim, label_dict_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.loss_func = nn.NLLLoss()
        print('init models done')

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp_layer.weight)

    def put_models_on_gpu_if_need(self):
        if not self._use_cuda:
            return
        self.cuda()

    def forward(self, chars, bichars):
        mask = chars.ne(pad_index)
        sen_lens = mask.sum(1)

        emb_ch = self.emb_chars(chars)
        emb_bich = self.emb_bichars(bichars)
        x = self.emb_drop_layer(torch.cat((emb_ch, emb_bich), -1))

        sorted_lens, sorted_indices = torch.sort(sen_lens, descending=True)
        inverse_indices = sorted_indices.argsort()
        x = pack_padded_sequence(x[sorted_indices], sorted_lens, True)

        x, _ = self.lstm_layer(x)
        x, _ = pad_packed_sequence(x, True)
        x = x[inverse_indices]
        x = self.mlp_layer(x)
        x = self.log_softmax(x)

        return x

    def get_loss(self, mlp_out, target, mask):
        return self.loss_func(mlp_out[mask], target[mask])

    def load_model(self, path, eval_num):
        path = os.path.join(path, 'models.%s.%d' % (self.name, eval_num))
        print(path)
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print('Load model %s done.' % path)

    def save_model(self, path, eval_num):
        path = os.path.join(path, 'models.%s.%d' % (self.name, eval_num))
        torch.save(self.state_dict(), path)
        print('Save model %s done.' % path)
