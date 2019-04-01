# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, words, vectors, unk=None):
        super(Embedding, self).__init__()

        self.words = words
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(words, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return torch.tensor(self.pretrained[word])

    @property
    def dim(self):
        return len(self.vectors[0])

    def get_embeddings(self, words, smooth=True):
        embeddings = torch.zeros(len(words), self.dim)
        for i, word in enumerate(words):
            if word in self:
                embeddings[i] = self[word]
        if smooth:
            embeddings /= torch.std(embeddings)
        return embeddings

    @classmethod
    def load(cls, fname, unk=None):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        words = [line.split()[0] for line in lines]
        vectors = torch.zeros(len(words), 50).normal_().tolist()
        # splits = [line.split() for line in lines]
        # reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        # words, vectors = map(list, zip(*reprs))
        embedding = cls(words, vectors, unk=unk)

        return embedding
