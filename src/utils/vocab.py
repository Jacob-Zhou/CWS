# -*- coding: utf-8 -*-

from collections import Counter

import torch
from src.common import bos, eos, pad, unk


class VocabDict(object):

    def __init__(self, name):
        self._name = name
        self._counter = Counter()
        self._str2id = {}
        self._id2str = []
        self._pad_index = -1
        self._unk_index = -1
        self._bos_index = -1
        self._eos_index = -1
        self._embed = None

    def __len__(self):
        return len(self._str2id)

    @property
    def name(self):
        return self._name

    @property
    def init_num(self):
        # initial num of tokens, that is, num of tokens in train
        # before extending the vocab with pretrained files
        return self._init_num

    @property
    def pad_index(self):
        if self._pad_index < 0:
            raise AttributeError
        else:
            return self._pad_index

    @property
    def unk_index(self):
        return self._unk_index

    @property
    def bos_index(self):
        if self._bos_index < 0:
            raise AttributeError
        else:
            return self._bos_index

    @property
    def eos_index(self):
        if self._eos_index < 0:
            raise AttributeError
        else:
            return self._eos_index

    @property
    def tokens(self):
        return self._id2str

    @property
    def embed(self):
        return self._embed

    def count(self, k):
        self._counter[k] += 1

    def get_id(self, key):
        return self._str2id.get(key, self._unk_index)

    def get_str(self, i):
        return self._id2str[i]

    def read_embeddings(self, embed, init=None, smooth=False):
        tokens = embed.tokens
        # if the UNK token has existed in pretrained vocab,
        # then replace it with a self-defined one
        if embed.unk:
            tokens[tokens.index(unk)] = self.get_str(self.unk_index)
        # add unknown tokens to vocab and update the dict
        self._id2str += sorted(token for token in tokens
                               if token not in self._str2id)
        self._str2id = {token: i for i, token in enumerate(self._id2str)}

        self._embed = torch.empty(len(self), embed.dim)
        if init:
            self._embed = init(self._embed)
        else:
            scale = (3 / embed.dim) ** 0.5
            self._embed = self._embed.uniform_(-scale, scale)

        indices = [self._str2id[token] for token in tokens]
        self._embed[indices] = embed.vectors
        if smooth:
            self._embed /= torch.std(self._embed)
        print('Reading embeddings %s done: %d keys in file; %d keys in total' %
              (embed.name, len(embed), self.embed.size(0)))

    def save(self, filename):
        assert len(self._counter) > 0
        total_num = len(self._counter)
        with open(filename, mode='w', encoding='utf-8') as f:
            f.write("total-num=%d\n" % len(self._counter))
            for s, cnt in self._counter.most_common():
                f.write("%s\t%d\n" % (s, cnt))
        print("\tSaved %d vocab into %s\n" % (total_num, filename))
        self._counter.clear()

    def load(self, filename, cutoff_freq=0, default_keys=[]):
        assert len(self._counter) == 0
        assert len(self._id2str) == 0

        with open(filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            total_num = int(lines[0].split('=')[-1])
            assert len(lines) > 0
            assert total_num == len(lines) - 1
        tokens, freqs = zip(*[line.split('\t') for line in lines[1:]])
        # sort the tokens to avoid randomness, especially for labels.
        # all labels must be sorted to correspond to their transition scores.
        tokens = sorted(token for token, freq in zip(tokens, freqs)
                        if int(freq) > cutoff_freq)
        self._id2str = list(default_keys) + tokens
        self._str2id = {token: i for i, token in enumerate(self._id2str)}
        self._init_num = len(self)
        self._pad_index = self._str2id.get(pad, -1)
        self._unk_index = self._str2id.get(unk, -1)
        self._bos_index = self._str2id.get(bos, -1)
        self._eos_index = self._str2id.get(eos, -1)
        print('Loading dict %s done: %d keys; unk_index=%d' %
              (self.name, len(self), self._unk_index))
