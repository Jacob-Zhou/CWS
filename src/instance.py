import numpy as np
from common import *
import re

class Instance(object):
    def __init__(self, id, lines):
        self.id = id
        n = len(lines)
        self.chars_s = [''] * n
        self.bichars_s = [''] * n
        self.labels_s = [''] * n
        self.labels_s_predict = [''] * n
        self.chars_i = np.array([-1] * n, dtype=data_type_int)
        self.bichars_i = np.array([-1] * n, dtype=data_type_int)
        self.labels_i = np.array([-1] * n, dtype=data_type_int)
        self.labels_i_predict = np.array([-1] * n, dtype=data_type_int)
        self.decompose_sent(lines)

    def size(self):
        return len(self.chars_s)

    def pad_size(self):
        return len(self.labels_i) - self.size()

    @staticmethod
    def compose_sent(chars_s, bichars_s, labels_s):
        n = len(chars_s)
        assert n == len(bichars_s) == len(labels_s)
        lines = []
        for i in np.arange(n):
            lines.append('%s %s %s\n' % (chars_s[i], bichars_s[i], labels_s[i]))
        return lines

    def write(self, out_file):
        lines = Instance.compose_sent(self.chars_s, self.bichars_s, self.labels_s_predict)
        for line in lines:
            out_file.write(line)
        out_file.write('\n')

    def decompose_sent(self, lines):
        for (i, line) in enumerate(lines):
            tokens = re.split('\s+', line.strip())
            assert(len(tokens) == 8)
            self.chars_s[i], self.bichars_s[i], self.labels_s[i] = \
                tokens[0], tokens[1], tokens[2]

    def eval_label(self):
        return np.sum(np.equal(self.labels_i_predict, self.labels_i, )) - self.pad_size()

    def eval_word(self):
        return 0, 0, 0 # gold sys correct in word num


