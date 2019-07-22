# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from src.common import pad_index
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .modules import BertEmbedding


class CWSModel(nn.Module):

    def __init__(self, name, conf):
        super(CWSModel, self).__init__()

        self._conf = conf
        self._name = name

        self.emb_char = None
        self.emb_bichar = None
        self.emb_bert = None
        self.emb_dict = None
        self.dict_embed_dropout = None
        self.embed_dropout = None
        self.lstm = None
        self.ffn = None
        self.criterion = None

    @property
    def name(self):
        return self._name

    # create and init all the models needed according to config
    def init_models(self, char_dict, bichar_dict, label_dict, extra_dicts):
        self.emb_char = nn.Embedding(num_embeddings=len(char_dict),
                                     embedding_dim=self._conf.n_char_embed)
        self.emb_bichar = nn.Embedding(num_embeddings=len(bichar_dict),
                                       embedding_dim=self._conf.n_char_embed)

        if self._conf.with_bert:
            self.emb_bert = BertEmbedding(self._conf.bert_path,
                                        self._conf.n_bert_layers,
                                        self._conf.n_bert_embed)
            n_bert_embed = self._conf.n_bert_embed
        else:
            n_bert_embed = 0

        self.embed_dropout = nn.Dropout(self._conf.embed_dropout)

        pre_dict_dim = 0
        post_dict_dim = 0
        self.dft = self._conf.dict_feature_type
        self.dct = self._conf.dict_concat_type
        self.with_dict = self._conf.with_extra_dictionarys

        if self._conf.with_extra_dictionarys:
            if self.dft == "ours":
                self._conf.with_dict_emb = True
                self.emb_dict = nn.ModuleList([
                    nn.Embedding(num_embeddings=8, embedding_dim=self._conf.n_dict_embed[dict_k])
                    for dict_k in range(len(extra_dicts))
                ])
                # self.dict_embed_dropout = nn.Dropout(self._conf.embed_dropout)
            else:
                if self._conf.with_dict_emb:
                    self.emb_dict = nn.ModuleList([
                        nn.Embedding(num_embeddings=2, embedding_dim=self._conf.n_dict_embed[dict_k])
                        for dict_k in range(len(extra_dicts))
                    ])

            if self._conf.with_dict_emb:
                n_dict_embed = 0
                for n_e, max_l, min_l in zip(self._conf.n_dict_embed, self._conf.max_word_len, self._conf.min_word_len):
                    n_dict_embed += (max_l - min_l + 1) * n_e
                if self.dft != "ours":
                    n_dict_embed *= 2
            else:
                n_dict_embed = 0
                for max_l, min_l in zip(self._conf.max_word_len, self._conf.min_word_len):
                    n_dict_embed += (max_l - min_l + 1) * 2

            if self.dct:
                if self.dct == "post":
                    post_dict_dim = self._conf.n_lstm_hidden*2
                    self.dict_lstm = nn.LSTM(input_size=n_dict_embed,
                    hidden_size=self._conf.n_lstm_hidden,
                    num_layers=self._conf.n_lstm_layers,
                    batch_first=True,
                    dropout=self._conf.lstm_dropout,
                    bidirectional=True)
                elif self.dct == "pre":
                    pre_dict_dim = n_dict_embed
            else:
                self.dict_weight = nn.Linear(in_features=n_dict_embed, out_features=self._conf.n_char_embed*2+n_bert_embed)

                self.dict_bias = nn.Linear(in_features=n_dict_embed, out_features=self._conf.n_char_embed*2+n_bert_embed)

        self.lstm = nn.LSTM(input_size=self._conf.n_char_embed*2+n_bert_embed + pre_dict_dim,
                            hidden_size=self._conf.n_lstm_hidden,
                            num_layers=self._conf.n_lstm_layers,
                            batch_first=True,
                            dropout=self._conf.lstm_dropout,
                            bidirectional=True)

        self.ffns = nn.ModuleList([
            nn.Linear(in_features=self._conf.n_lstm_hidden*2 + post_dict_dim,
                      out_features=len(label_dict))
            for _ in range(len(self._conf.train_files))
        ])
        self.criterion = nn.CrossEntropyLoss()
        print('init models done')

    def forward(self, subwords, chars, bichars, dict_feats, index=0):
        mask = chars.ne(pad_index)
        lens = mask.sum(1)
        batch_size, seq_len = chars.shape

        emb_char = self.emb_char(chars)
        emb_bichar = self.emb_bichar(bichars)
        
        if self.emb_bert:
            emb_bert = self.emb_bert(subwords)
            x = self.embed_dropout(torch.cat((emb_char, emb_bichar, emb_bert), -1))
        else:
            x = self.embed_dropout(torch.cat((emb_char, emb_bichar), -1))

        if self.with_dict:
            if self._conf.with_dict_emb:
                emb_dict = [None] * len(self.emb_dict)
                for i, emb_dict_layer in enumerate(self.emb_dict):
                    emb_this = emb_dict_layer(dict_feats[i])
                    emb_dict[i] = emb_this.view((batch_size, seq_len, -1))
            else:
                emb_dict = [dict_feat.float() for dict_feat in dict_feats]

            emb_dict = torch.cat(tuple(emb_dict), -1)

            # emb_dict = self.dict_embed_dropout(emb_dict)

        sorted_lens, sorted_indices = torch.sort(lens, descending=True)
        inverse_indices = sorted_indices.argsort()

        if self.with_dict:
            if self.dct == "pre":
                x = torch.cat((x, emb_dict), -1)
            elif self.dct == "post":
                emb_dict = pack_padded_sequence(emb_dict[sorted_indices], sorted_lens, True)
                emb_dict, _ = self.dict_lstm(emb_dict)
                emb_dict, _ = pad_packed_sequence(emb_dict, True, total_length=seq_len)
                emb_dict = emb_dict[inverse_indices]
            else:
                attn_w = self.dict_weight(emb_dict) # (B, L, len(x))
                attn_b = self.dict_bias(emb_dict) # (B, L, len(x))
                x = torch.addcmul(attn_b, attn_w, x)

        x = pack_padded_sequence(x[sorted_indices], sorted_lens, True)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = x[inverse_indices]

        if self.with_dict:
            if self.dct == "post":
                x = torch.cat((x, emb_dict), -1)
        x = self.ffns[index](x)

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
