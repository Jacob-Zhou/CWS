import math
import random
from collections import Counter

import numpy as np
import torch

from instance import Instance
from simple_bucketing import Bucketing
from vocab import VocabDict


class Dataset(object):

    def __init__(self, file_name,
                 max_bucket_num=80,
                 char_num_one_batch=5000,
                 sent_num_one_batch=200,
                 inst_num_max=-1,
                 min_len=1,
                 max_len=100,
                 shuffle=False):
        self._file_name = file_name
        self._file_name_short = file_name[-30:].replace('/', '_')
        self._instances = []

        self.char_num_total = 0
        with open(self._file_name, mode='r', encoding='utf-8') as f:
            lines = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    length = len(lines)
                    if length >= min_len and (max_len < 0 or length <= max_len):
                        inst = Instance(len(self._instances), lines)
                        self._instances.append(inst)
                        self.char_num_total += length
                        if (inst_num_max > 0) and (self.size() == inst_num_max):
                            break
                    lines = []
                else:
                    lines.append(line)
        assert self.size() > 0
        print('Reading %s done: %d instances %d chars' % (self._file_name_short,
                                                          self.size(),
                                                          self.char_num_total))

        self._sent_index = 0
        self._char_num_one_batch = char_num_one_batch
        self._sent_num_one_batch = sent_num_one_batch
        assert self._char_num_one_batch > 0 or self._sent_num_one_batch > 0

        self._bucket_num = -1
        self._use_bucket = (max_bucket_num > 1)
        self._buckets = None  # [(max_len, inst_num_one_batch, bucket)]
        self._bucket_sent_index = 0
        self._shuffle = shuffle

        if self._use_bucket:
            assert (self._char_num_one_batch > 0)
            len_counter = Counter()
            for inst in self.all_inst:
                len_counter[len(inst)] += 1

            # Automatically decide the bucket num according to the data
            self._bucket_num = int(min(max_bucket_num, math.ceil(len(len_counter)/1.5),
                                       np.ceil(self.char_num_total/(2*self._char_num_one_batch))))
            assert self._bucket_num > 0

            # k_classes = KMeans(self._bucket_num, len_counter)
            k_classes = Bucketing(self._bucket_num, len_counter)
            max_len_buckets = k_classes.max_len_in_buckets
            len2bucket_idx = k_classes.len2bucket_idx
            self._bucket_num = len(max_len_buckets)
            buckets = [None] * self._bucket_num
            # Can NOT use [[]] * self._bucket_num, shallow copy issue!
            for inst in self.all_inst:
                b_idx = len2bucket_idx[len(inst)]
                if buckets[b_idx] is None:
                    buckets[b_idx] = [inst]
                else:
                    buckets[b_idx].append(inst)
            batch_num_total = 0
            inst_num_one_batch_buckets = []
            for (i, max_len) in enumerate(max_len_buckets):
                inst_num = len(buckets[i])
                batch_num_to_provide = max(
                    1, round(float(inst_num) * max_len / self._char_num_one_batch))
                print("i, inst_num, max_len, batch_num_to_provide, batch_num_total = ",
                      i, inst_num, max_len, batch_num_to_provide, batch_num_total)
                batch_num_total += batch_num_to_provide
                inst_num_one_batch_this_bucket = math.ceil(
                    inst_num / batch_num_to_provide)
                # The goal is to avoid the last batch of one bucket contains too few instances
                inst_num_one_batch_buckets.append(
                    inst_num_one_batch_this_bucket)
                # assert inst_num_one_batch_this_bucket * (batch_num_to_provide-0.5) < inst_num
            print('%s can provide %d batches in total with %d buckets' %
                  (self._file_name_short, batch_num_total, self._bucket_num))
            self._buckets = [(ml, nb, b) for ml, nb, b in zip(
                max_len_buckets, inst_num_one_batch_buckets, buckets)]

    def __len__(self):
        return len(self._instances)

    def __iter__(self):
        return self

    def __next__(self):
        if self._use_bucket:
            return self.get_one_batch_bucket()
        else:
            return self.get_one_batch()

    @property
    def file_name_short(self):
        return self._file_name_short

    def size(self):
        return len(self._instances)

    @property
    def all_inst(self):
        return self._instances

    @property
    def all_buckets(self):
        return self._buckets

    def get_one_batch_bucket(self):
        if self._bucket_sent_index >= self._bucket_num:
            self._bucket_sent_index = 0
            if self._shuffle:
                for (max_len, inst_num, bucket) in self._buckets:
                    random.shuffle(bucket)
                random.shuffle(self._buckets)
            raise StopIteration

        max_len, inst_num_one_batch, this_bucket = self._buckets[self._bucket_sent_index]
        assert self._sent_index < len(this_bucket)
        idx_next_batch = self._sent_index + inst_num_one_batch
        one_batch = this_bucket[self._sent_index:idx_next_batch]
        assert len(one_batch) > 0
        if idx_next_batch >= len(this_bucket):
            self._bucket_sent_index += 1
            self._sent_index = 0
        else:
            self._sent_index = idx_next_batch

        return one_batch

    def get_one_batch(self):
        if self._sent_index >= len(self):
            self._sent_index = 0
            if self._shuffle:
                random.shuffle(self._instances)
            raise StopIteration

        char_num_accum, one_batch = 0, []
        for inst in self._instances[self._sent_index:]:
            if char_num_accum + len(inst) > self._char_num_one_batch + 25:
                return one_batch  # not include this instance
            one_batch.append(inst)
            char_num_accum += len(inst)
            self._sent_index += 1

        return one_batch
