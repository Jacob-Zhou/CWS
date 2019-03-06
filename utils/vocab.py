# -*- coding: utf-8 -*-

import collections
from itertools import chain

import torch
import torch.nn.init as init


class Vocab(object):
    def collect(self, corpus, min_freq=1):
        labels = sorted(set(chain(*corpus.label_seqs)))
        chars = list(chain(*corpus.char_seqs))
        bichars = list(chain(*corpus.bichar_seqs))

        chars_freq = collections.Counter(chars)
        chars = [c for c, f in chars_freq.items() if f > min_freq]
        bichars_freq = collections.Counter(bichars)
        bichars = [w for w, f in bichars_freq.items() if f > min_freq]

        return chars, bichars, labels

    def __init__(self, train_corpus, min_freq=1):
        chars, bichars, labels = self.collect(train_corpus, min_freq)
        #  ensure the <PAD> index is 0
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self._chars = [self.PAD] + chars + [self.UNK]
        self._bichars = [self.PAD] + bichars + [self.UNK]
        self._labels = labels

        self._char2id = {w: i for i, w in enumerate(self._chars)}
        self._bichar2id = {c: i for i, c in enumerate(self._bichars)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}

        self.num_chars = len(self._chars)
        self.num_bichars = len(self._bichars)
        self.num_labels = len(self._labels)

        self.UNK_char_index = self._char2id[self.UNK]
        self.UNK_bichar_index = self._bichar2id[self.UNK]
        self.PAD_char_index = self._char2id[self.PAD]
        self.PAD_bichar_index = self._bichar2id[self.PAD]

    def read_embedding(self, embedding_file, unk_in_pretrain=None):
        #  ensure the <PAD> index is 0
        with open(embedding_file, 'r') as f:
            lines = f.readlines()
        splits = [line.split() for line in lines]
        # read pretrained embedding file
        words, vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])
        if isinstance(unk_in_pretrain, str):
            assert unk_in_pretrain in words
            words = list(words)
            words[words.index(unk_in_pretrain)] = self.UNK

        pretrained = {w: torch.tensor(v) for w, v in zip(words, vectors)}
        out_train_chars = [w for w in words if w not in self._char2id]
        out_train_bichars = [c for c in ''.join(
            out_train_words) if c not in self._bichar2id]

        # extend words and chars
        # ensure the <PAD> token at the first position
        self._chars = [self.PAD] + \
            sorted(set(self._chars + out_train_chars) - {self.PAD})
        self._bichars = [self.PAD] + \
            sorted(set(self._bichars + out_train_bichars) - {self.PAD})

        # update the words,chars dictionary
        self._char2id = {w: i for i, w in enumerate(self._chars)}
        self._bichar2id = {c: i for i, c in enumerate(self._bichars)}
        self.UNK_char_index = self._char2id[self.UNK]
        self.UNK_bichar_index = self._bichar2id[self.UNK]
        self.PAD_char_index = self._char2id[self.PAD]
        self.PAD_bichar_index = self._bichar2id[self.PAD]

        # update the numbers of words and chars
        self.num_chars = len(self._chars)
        self.num_bichars = len(self._bichars)

        # initial the extended embedding table
        embdim = len(vectors[0])
        extended_embed = torch.randn(self.num_chars, embdim)
        init.normal_(extended_embed, 0, 1 / embdim ** 0.5)

        # the word in pretrained file use pretrained vector
        # the word not in pretrained file but in training data use random initialized vector
        for i, w in enumerate(self._chars):
            if w in pretrained:
                extended_embed[i] = pretrained[w]
        return extended_embed

    def char2id(self, char):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self.UNK_char_index)
        elif isinstance(char, list):
            return [self._char2id.get(w, self.UNK_char_index) for w in char]

    def label2id(self, label):
        assert (isinstance(label, str) or isinstance(label, list))
        if isinstance(label, str):
            # if label not in training data, index to 0 ?
            return self._label2id.get(label, 0)
        elif isinstance(label, list):
            return [self._label2id.get(l, 0) for l in label]

    def bichar2id(self, bichar):
        assert (isinstance(bichar, str) or isinstance(bichar, list))
        if isinstance(bichar, str):
            return self._bichar2id.get(bichar, self.UNK_bichar_index)
        elif isinstance(bichar, list):
            return [self._bichar2id.get(c, self.UNK_bichar_index) for c in bichar]

    def id2label(self, id):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            assert (id >= self.num_labels)
            return self._labels[id]
        elif isinstance(id, list):
            return [self._labels[i] for i in id]
