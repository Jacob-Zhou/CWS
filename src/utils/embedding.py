# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, subwords, vectors, unk=None):
        super(Embedding, self).__init__()

        self.subwords = subwords
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(subwords, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.subwords)

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return torch.tensor(self.pretrained[word])

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname, unk=None):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        subwords = [line.split()[0] for line in lines]
        vectors = torch.zeros(len(subwords), 50).normal_().tolist()
        # splits = [line.split() for line in lines]
        # reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        # subwords, vectors = map(list, zip(*reprs))
        embedding = cls(subwords, vectors, unk=unk)

        return embedding
