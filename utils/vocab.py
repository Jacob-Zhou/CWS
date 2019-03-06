# -*- coding: utf-8 -*-

from collections import Counter

import torch
import torch.nn as nn


class Vocab(object):

    def __init__(self, train_corpus, min_freq=1):
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'

        self.pad_index = 0
        self.unk_index = 1
        chars, bichars, labels = self.collect(train_corpus, min_freq)
        self._chars = [self.PAD, self.UNK] + chars
        self._bichars = [self.PAD, self.UNK] + bichars
        self._labels = labels

        self._char2id = {w: i for i, w in enumerate(self._chars)}
        self._bichar2id = {c: i for i, c in enumerate(self._bichars)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}

        self.num_chars = len(self._chars)
        self.num_bichars = len(self._bichars)
        self.num_labels = len(self._labels)

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
        out_train_chars = {char for char in ''.join(words)
                           if char not in self._char2id}
        out_train_bichars = {fchar+bchar
                             for word in words
                             for fchar, bchar in zip(word[:-1], word[1:])
                             if fchar+bchar not in self._bichar2id}

        self._chars += sorted(out_train_chars)
        self._bichars += sorted(out_train_bichars)

        # update the words,chars dictionary
        self._char2id = {w: i for i, w in enumerate(self._chars)}
        self._bichar2id = {c: i for i, c in enumerate(self._bichars)}
        # update the numbers of words and chars
        self.num_chars = len(self._chars)
        self.num_bichars = len(self._bichars)

        # initial the extended embedding table
        embdim = len(vectors[0])
        extended_embed = torch.randn(self.num_chars, embdim)
        nn.init.normal_(extended_embed, 0, 1 / embdim ** 0.5)

        # the word in pretrained file use pretrained vector
        # the word not in pretrained file but in training data use random initialized vector
        for i, w in enumerate(self._chars):
            if w in pretrained:
                extended_embed[i] = pretrained[w]
        return extended_embed

    def char2id(self, char):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self.unk_index)
        elif isinstance(char, list):
            return [self._char2id.get(w, self.unk_index) for w in char]

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
            return self._bichar2id.get(bichar, self.unk_index)
        elif isinstance(bichar, list):
            return [self._bichar2id.get(c, self.unk_index) for c in bichar]

    def id2label(self, id):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            assert (id >= self.num_labels)
            return self._labels[id]
        elif isinstance(id, list):
            return [self._labels[i] for i in id]

    def collect(self, corpus, min_freq=1):
        labels = sorted(set(label for seq in corpus.label_seqs
                            for label in seq))
        chars = list(set(char for seq in corpus.char_seqs
                         for char in seq))
        bichars = list(set(bichar for seq in corpus.bichar_seqs
                           for bichar in seq))

        chars_freq = Counter(chars)
        chars = [c for c, f in chars_freq.items() if f > min_freq]
        bichars_freq = Counter(bichars)
        bichars = [w for w, f in bichars_freq.items() if f > min_freq]

        return chars, bichars, labels
