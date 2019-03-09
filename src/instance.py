import re

import numpy as np

from common import *


class Decoder(object):

    def __init__(self):
        inf = float('-inf')
        T = [[0., inf, inf, 0.],
             [inf, 0., 0., inf],
             [inf, inf, 0., inf],
             [0., inf, inf, 0.],
             [0., inf, inf, 0.]]
        self.gram = torch.tensor(T)

    def is_continue_label(self, label, startlabel, distance):
        if distance == 0:
            return True
        if startlabel[0] == 'S' or self.is_start_label(label) or label[2:] != startlabel[2:]:
            return False
        return True

    def is_start_label(self, label):
        return (label[0] == 'B' or label[0] == 'S')

    def evaluate(self, predict, target):
        preds, golds = [], []
        length = len(predict)

        for idx in range(length):
            if self.is_start_label(predict[idx]):
                s = ''
                for idy in range(idx, length):
                    if not self.is_continue_label(predict[idy], predict[idx], idy - idx):
                        endpos = idy - 1
                        s += target[idy][0]
                        break
                    endpos = idy
                    s += predict[idy][0]
                ss = '[' + str(idx) + ',' + str(endpos) + ']'
                preds.append(s + predict[idx][1:] + ss)

        for idx in range(length):
            if self.is_start_label(target[idx]):
                s = ''
                for idy in range(idx, length):
                    if not self.is_continue_label(target[idy], target[idx], idy - idx):
                        endpos = idy - 1
                        s += target[idy][0]
                        break
                    endpos = idy
                    s += target[idy][0]
                ss = '[' + str(idx) + ',' + str(endpos) + ']'
                golds.append(s + target[idx][1:] + ss)
        gold_num = len(golds)
        pred_num = len(preds)
        correct_num = 0
        for pred in preds:
            if pred in golds:
                correct_num += 1

        return gold_num, pred_num, correct_num


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
            lines.append('%s %s %s\n' %
                         (chars_s[i], bichars_s[i], labels_s[i]))
        return lines

    def write(self, out_file):
        lines = Instance.compose_sent(self.chars_s, self.bichars_s,
                                      self.labels_s_predict)
        for line in lines:
            out_file.write(line)
        out_file.write('\n')

    def decompose_sent(self, lines):
        for (i, line) in enumerate(lines):
            tokens = line.split()
            assert(len(tokens) == 3)
            self.chars_s[i], self.bichars_s[i], self.labels_s[i] = \
                tokens[0], tokens[1], tokens[2]

    def evaluate(self):
        return Decoder().evaluate(self.labels_s_predict, self.labels_s)


