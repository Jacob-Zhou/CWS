# -*- coding: utf-8 -*-

import torch


class Instance(object):

    def __init__(self, inst_id, lines):
        self.inst_id = inst_id
        n = len(lines)
        self.chars_s = [''] * n
        self.bichars_s = [''] * n
        self.labels_s = [''] * n
        self.labels_s_pred = [''] * n
        self.chars_i = torch.zeros(n).long()
        self.bichars_i = torch.zeros(n).long()
        self.labels_i = torch.zeros(n).long()
        self.labels_i_predict = torch.zeros(n).long()

        self.decompose_sent(lines)

    def __len__(self):
        return len(self.chars_s)

    @staticmethod
    def compose_sent(chars_s, bichars_s, labels_s):
        n = len(chars_s)
        assert n == len(bichars_s) == len(labels_s)
        lines = []
        for i in range(n):
            lines.append('%s %s %s\n' %
                         (chars_s[i], bichars_s[i], labels_s[i]))
        return lines

    def write(self, out_file):
        lines = Instance.compose_sent(self.chars_s,
                                      self.bichars_s,
                                      self.labels_s_pred)
        for line in lines:
            out_file.write(line)
        out_file.write('\n')

    def decompose_sent(self, lines):
        for (i, line) in enumerate(lines):
            tokens = line.split()
            assert(len(tokens) == 3)
            self.chars_s[i], self.bichars_s[i], self.labels_s[i] = tokens

    def evaluate(self):
        preds = self.get_spans(self.labels_s_pred)
        golds = self.get_spans(self.labels_s)

        gold_num = len(golds)
        pred_num = len(preds)
        correct_num = len(golds & preds)
        total_labels = len(self.labels_s)
        correct_labels = sum(i == j
                             for i, j in zip(self.labels_i.tolist(),
                                             self.labels_i_pred.tolist()))

        return gold_num, pred_num, correct_num, total_labels, correct_labels

    @classmethod
    def get_spans(cls, labels):
        # record span of each words and their split points
        spans, splits = set(), set()

        for i, label in enumerate(labels):
            if label.startswith('b'):
                splits.add(i)
            elif label.startswith('e'):
                splits.add(i + 1)
            elif label.startswith('s'):
                splits.update({i, i + 1})
        splits = sorted(splits | {0, len(labels)})
        for i, j in zip(splits[:-1], splits[1:]):
            spans.add((i, j))

        return spans

    @classmethod
    def recover(cls, label, length):
        # the labels must be sorted, that is [B, E, M, S]
        if length < 2:
            return [label]
        # [B] -> [B, M, ..., M]
        elif label == 0:
            return [0] + [2 for _ in range(length - 1)]
        # [E] -> [M, M, ..., E]
        elif label == 1:
            return [2 for _ in range(length - 1)] + [1]
        # [M] -> [M, M, ..., M]
        elif label == 2:
            return [2 for _ in range(length)]
        # [S] -> [B, M, ..., E]
        elif label == 3:
            return [0] + [2 for _ in range(length - 2)] + [1]
