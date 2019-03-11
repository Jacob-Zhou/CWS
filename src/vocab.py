from collections import Counter
from collections.abc import Iterable

from common import *


class VocabDict(object):
    def __init__(self, name):
        self._name = name
        self._counter = Counter()
        self._str2id = {}
        self._id2str = []
        self._unknown_index = -1
        self._is_locked = False

    def __len__(self):
        return len(self._str2id)

    @property
    def name(self):
        return self._name

    #  ------ _counter ------
    def add_key_into_counter(self, k):
        assert not self.is_locked()
        self._counter[k] += 1

    def save(self, file_name):
        assert not self.is_locked()
        # self.set_lock(True)  # bzhang
        assert len(self._counter) > 0
        total_num = len(self._counter)
        with open(file_name, mode='w', encoding='utf-8') as f:
            f.write("total-num=%d\n" % len(self._counter))
            for s, cnt in self._counter.most_common():
                f.write("%s\t%d\n" % (s, cnt))
        print("\tSaved %d vocab into %s\n" % (total_num, file_name))
        self._counter.clear()
    #  ------

    def load(self, file_name, cutoff_freq=0, default_keys_ids=()):
        assert len(self._counter) == 0
        assert len(self._id2str) == 0
        assert not self.is_locked()
        for (k, v) in default_keys_ids:
            v2 = self.add_key_into_dict(k)
            assert v == v2

        with open(file_name, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) > 0
            tokens = lines[0].strip().split('=')
            assert 2 == len(tokens)
            total_num = int(tokens[1])
            assert total_num == len(lines) - 1
            for line in lines[1:]:
                tokens = line.strip().split('\t')
                assert 2 == len(tokens)
                if int(tokens[1]) <= cutoff_freq:  # descending order
                    break
                self.add_key_into_dict(tokens[0])
        self._id2str = [''] * self._size()
        for (k, v) in self._str2id.items():
            assert (v >= 0) and (v < self._size())
            self._id2str[v] = k
        self._unknown_index = self._get_id(unknown_str)
        self.set_lock(True)
        print('Loading dict %s done: %d keys; unknown-id=%d' % (self.name, self.size(), self._unknown_index),
              flush=True)

    def _size(self):
        return len(self._str2id)

    def size(self):
        assert self.is_locked() is True
        return self._size()

    def is_locked(self):
        return self._is_locked

    def _set_lock(self, value):
        self._is_locked = value

    def set_lock(self, value=True):
        assert value != self.is_locked()
        self._set_lock(value)

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

    def add_key_into_dict(self, k):
        assert not self.is_locked()
        assert -1 == self._get_id(k)
        self._str2id[k] = self._size()
        return self._get_id(k)
