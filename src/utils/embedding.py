# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, name, tokens, vectors, unk=None):
        super(Embedding, self).__init__()

        self.name = name
        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    @property
    def dim(self):
        return self.vectors.size(1)

    @classmethod
    def load(cls, filename, unk=None):
        with open(filename, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                for s in splits])
        embedding = cls(filename, tokens, vectors, unk=unk)

        return embedding
