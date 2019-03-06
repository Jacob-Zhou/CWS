# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


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
        if startlabel[0] == "S" or self.is_start_label(label) or label[2:] != startlabel[2:]:
            return False
        return True

    def is_start_label(self, label):
        return (label[0] == "B" or label[0] == "S")

    def evaluate(self, predict, target):
        preds, golds = [], []
        length = len(predict)

        for idx in range(length):
            if self.is_start_label(predict[idx]):
                s = ""
                for idy in range(idx, length):
                    if not self.is_continue_label(predict[idy], predict[idx], idy - idx):
                        endpos = idy - 1
                        s += target[idy][0]
                        break
                    endpos = idy
                    s += predict[idy][0]
                ss = "[" + str(idx) + "," + str(endpos) + "]"
                preds.append(s + predict[idx][1:] + ss)

        for idx in range(length):
            if self.is_start_label(target[idx]):
                s = ""
                for idy in range(idx, length):
                    if not self.is_continue_label(target[idy], target[idx], idy - idx):
                        endpos = idy - 1
                        s += target[idy][0]
                        break
                    endpos = idy
                    s += target[idy][0]
                ss = "[" + str(idx) + "," + str(endpos) + "]"
                golds.append(s + target[idx][1:] + ss)
        overall_count = len(golds)
        predict_count = len(preds)
        correct_count = 0
        for pred in preds:
            if pred in golds:
                correct_count += 1

        return overall_count, predict_count, correct_count

    def viterbi(self, score, labelSize):
        length = len(score)
        viterbi_score = torch.zeros_like(score)
        for idx in range(length):
            if idx == 0:
                viterbi_score[idx] = score[idx] + self.gram[0]
            else:
                viterbi_score[idx] = score[idx] + \
                    torch.max(self.gram[1:] + viterbi_score[idx-1], 1)[0]


class Evaluator(object):

    def __init__(self, vocab, use_crf=True):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0
        self.vocab = vocab
        self.use_crf = use_crf

    def clear_num(self):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0

    @torch.no_grad()
    def evaluate(self, network, data_loader):
        network.eval()
        total_loss = 0.0

        for batch in data_loader:
            batch_size = batch[0].size(0)
            # mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            # mask = word_idxs.gt(0)
            mask, out, targets = network.forward_batch(batch)
            sen_lens = mask.sum(1)

           # print("mask:", mask)
           # print("sen_lens:", sen_lens)

            batch_loss = network.get_loss(out, targets)
            total_loss += batch_loss * batch_size

            # predicts = Decoder.viterbi_batch(network.crf, out, mask)
            # predicts = [torch.max(F.softmax(out_sen, dim = 1), 1)[1] for out_sen in out]
            predicts = []
            for i, out_sen in enumerate(out):
                pre = torch.max(F.softmax(out_sen[:sen_lens[i]], dim=1), 1)[1]
                predicts.append(pre)

            #print("predicts:", predicts)
#            targets = torch.split(targets[mask], sen_lens.tolist())
            # print(targets[0].shape)
            #self.pred_num += torch.sum(sen_lens)
            for predict, target in zip(predicts, targets):
                predict = self.vocab.id2label(predict.tolist())
                target = self.vocab.id2label(target.tolist())
                overall_count, predict_count, correct_count = Decoder().evaluate(predict, target)
                self.correct_num += correct_count
                self.pred_num += predict_count
                self.gold_num += overall_count

        total_loss /= len(data_loader)
        precision = self.correct_num/self.pred_num
        recall = self.correct_num/self.gold_num
        fmeasure = self.correct_num*2/(self.pred_num+self.gold_num)

        self.clear_num()
        return total_loss, precision, recall, fmeasure
