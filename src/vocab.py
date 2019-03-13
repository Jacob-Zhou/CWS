# -*- coding: utf-8 -*-

from collections import Counter

from common import unknown_str


class VocabDict(object):
    def __init__(self, name):
        self._name = name
        self._counter = Counter()
        self._str2id = {}
        self._id2str = []
        self._unknown_index = -1

    def __len__(self):
        return len(self._str2id)

    @property
    def name(self):
        return self._name

    #  ------ _counter ------
    def add_key_into_counter(self, k):
        self._counter[k] += 1

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
        self._unknown_index = self._get_id(unknown_str)
        print('Loading dict %s done: %d keys; unknown-id=%d' % (self.name, len(self), self._unknown_index),
              flush=True)

    def _size(self):
        return len(self._str2id)

    def size(self):
        return len(self)

    def _get_id(self, key):
        return self._str2id.get(key, -1)

    def get_id(self, key):
        i = self._get_id(key)
        if -1 == i:
            i = self._unknown_index
            # print('%s, unk: %s %d' % (self.name, key, i))
        return i

    def get_str(self, i):
        return self._id2str[i]
