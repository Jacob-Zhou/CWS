# -*- coding: utf-8 -*-

import time


class Metric(object):

    def __init__(self,  eps=1e-6):
        self.clear()
        self.eps = eps

    def clear(self):
        self.sent_num = 0
        self.gold_num = 0
        self.pred_num = 0
        self.correct_num = 0
        self.total_labels = 0
        self.correct_labels = 0
        self.loss_accumulated = 0.
        self.start_time = time.time()
        self.time_gap = 0.
        self.forward_time = 0.
        self.loss_time = 0.
        self.backward_time = 0.
        self.decode_time = 0.

    @property
    def precision(self):
        return 100. * self.correct_num / (self.pred_num + self.eps)

    @property
    def recall(self):
        return 100. * self.correct_num / (self.gold_num + self.eps)

    @property
    def fscore(self):
        return 100.*self.correct_num*2 / (self.pred_num + self.gold_num + self.eps)

    @property
    def accuracy(self):
        return 100. * self.correct_labels / (self.total_labels + self.eps)

    def compute_and_output(self, dataset, eval_cnt):
        self.time_gap = float(time.time() - self.start_time)
        print("\n%30s(%5d): loss=%.3f " %
              (dataset.filename_short, eval_cnt, self.loss_accumulated), end='')
        if self.gold_num > 0:
            print("precision=%.3f, recall=%.3f, fscore=%.3f, " %
                  (self.precision, self.recall, self.fscore), end='')
        print("accuracy=%.3f, " % self.accuracy, end='')
        print("%d sentences, time=%.3f (%.1f %.1f %.1f %.1f) [%s]" %
              (self.sent_num, self.time_gap, self.forward_time, self.loss_time,
               self.backward_time, self.decode_time,
               time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time()))))
