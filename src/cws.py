# -*- coding: utf-8 -*-

import os
import re
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
from src.common import pad, unk
from src.metric import Metric
from src.model import CWSModel
from src.utils import Dataset, Embedding, VocabDict
from torch.nn.utils.rnn import pad_sequence


class CWS(object):

    def __init__(self, conf):
        self._conf = conf
        self._device = torch.device(self._conf.device)
        # self._cpu_device = torch.device('cpu')
        self._use_cuda, self._cuda_device = ('cuda' == self._device.type,
                                             self._device.index)
        if self._use_cuda:
            # please note that the index is the relative index
            # in CUDA_VISIBLE_DEVICES=6,7 (0, 1)
            assert 0 <= self._cuda_device < 8
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._cuda_device)
            # an alternative way: CUDA_VISIBLE_DEVICE=6 python ../main.py ...
            self._cuda_device = 0
        self.training = False

        self._optimizer = None
        self._use_bucket = (self._conf.max_bucket_num > 1)
        self._train_datasets = []
        self._dev_datasets = []
        self._test_datasets = []
        self._subword_pretrained = None
        self._char_dict = VocabDict('chars')
        self._bichar_dict = VocabDict('bichars')
        self._subword_dict = VocabDict('subwords')
        # there may be more than one label dictionaries
        self._label_dict = VocabDict('labels')

        # transition scores of the labels
        # make sure the label dict have been sorted
        # [B, E, M, S]
        self._strans = torch.tensor([1., 0., 0., 1.]).log()
        self._etrans = torch.tensor([0., 1., 0., 1.]).log()
        self._trans = torch.tensor([
            [0., 1., 1., 0.],  # B
            [1., 0., 0., 1.],  # E
            [0., 1., 1., 0.],  # M
            [1., 0., 0., 1.]   # S
        ]).log()  # (FROM->TO)

        self._metric = Metric()
        self._model = CWSModel('ws', conf, self._use_cuda)

    def run(self):
        self._subword_pretrained = Embedding.load(self._conf.emb_subword_file)
        if self._conf.is_train:
            self.load_datasets(self._conf.train_files,
                               self._train_datasets,
                               inst_num_max=self._conf.inst_num_max,
                               shuffle=True)
            if not self._conf.is_dictionary_exist:
                print("create dict...")
                for dataset in self._train_datasets:
                    self.create_dictionaries(dataset)

                self.save_dictionaries(self._conf.dict_dir)
                self.load_dictionaries(self._conf.dict_dir)

                return
        self.load_dictionaries(self._conf.dict_dir)
        if self._conf.is_train:
            self.load_datasets(self._conf.dev_files,
                               self._dev_datasets,
                               inst_num_max=self._conf.inst_num_max)

        self.load_datasets(self._conf.test_files,
                           self._test_datasets,
                           inst_num_max=self._conf.inst_num_max)

        self._subword_dict.read_embeddings(embed=self._subword_pretrained,
                                           init=nn.init.zeros_,
                                           smooth=True)
        print('numericalizing all instances in all datasets')
        for dataset in self._train_datasets + self._dev_datasets + \
                self._test_datasets:
            self.numericalize_all_instances(dataset)

        self._model.init_models(self._char_dict,
                                self._bichar_dict,
                                self._subword_dict,
                                self._label_dict)
        if not self._conf.is_train:
            self._model.load_model(self._conf.model_dir,
                                   self._conf.model_eval_num)
        print(self._model)

        if self._use_cuda:
            # self._model.cuda()
            self._model.to(self._cuda_device)
            self._strans = self._strans.to(self._cuda_device)
            self._etrans = self._etrans.to(self._cuda_device)
            self._trans = self._trans.to(self._cuda_device)

        if self._conf.is_train:
            assert self._optimizer is None
            self._optimizer = optim.Adam(lr=self._conf.lr,
                                         params=self._model.parameters())
            self.train()
            return

        assert self._conf.is_test
        for dataset in self._test_datasets:
            self.evaluate(dataset=dataset,
                          output_filename=dataset.filename_short + '.out')
            self._metric.compute_and_output(self._test_datasets[0],
                                            self._conf.model_eval_num)
            self._metric.clear()

    def train(self):
        best_eval_cnt, best_accuracy = 0, 0.
        self._metric.clear()
        for eval_cnt in range(1, self._conf.train_max_eval_num + 1):
            self.set_training_mode(training=True)
            for batch in self._train_datasets[0]:
                self._optimizer.zero_grad()
                self.train_or_eval_one_batch(batch)
            self._metric.compute_and_output(self._train_datasets[0],
                                            eval_cnt)
            self._metric.clear()

            self.evaluate(self._dev_datasets[0])
            self._metric.compute_and_output(self._dev_datasets[0],
                                            eval_cnt)
            current_fmeasure = self._metric.fscore
            self._metric.clear()

            if best_accuracy < current_fmeasure - 1e-3:
                if eval_cnt > self._conf.save_model_after_eval_num:
                    self._model.save_model(self._conf.model_dir,
                                           eval_cnt)
                    self.evaluate(dataset=self._test_datasets[0],
                                  output_filename=None)
                    self._metric.compute_and_output(self._test_datasets[0],
                                                    eval_cnt)
                    self._metric.clear()

                best_eval_cnt = eval_cnt
                best_accuracy = current_fmeasure

            if best_eval_cnt + self._conf.patience <= eval_cnt:
                break
        print("The training ended at epoch %d" % eval_cnt)
        print("The best fscore of dev is %.3f at epoch %d" %
              (best_accuracy, best_eval_cnt))

    def train_or_eval_one_batch(self, insts):
        print('.', end='')
        chars, bichars, subwords, labels = self.compose_batch(insts)
        mask = chars.ne(self._char_dict.pad_index)
        time1 = time.time()
        out = self._model(chars, bichars, subwords)
        time2 = time.time()

        loss = self._model.get_loss(out, labels, mask)
        self._metric.loss_accumulated += loss.item()
        time3 = time.time()

        if self.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(),
                                     max_norm=self._conf.clip)
            self._optimizer.step()
        time4 = time.time()

        self.decode(out, insts, mask)
        time5 = time.time()

        self._metric.sent_num += len(insts)
        self._metric.forward_time += time2 - time1
        self._metric.loss_time += time3 - time2
        self._metric.backward_time += time4 - time3
        self._metric.decode_time += time5 - time4

    @torch.no_grad()
    def evaluate(self, dataset, output_filename=None):
        self.set_training_mode(training=False)
        for batch in dataset:
            self.train_or_eval_one_batch(batch)

        if output_filename is not None:
            with open(output_filename, 'w', encoding='utf-8') as out_file:
                all_inst = dataset.all_inst
                for inst in all_inst:
                    inst.write(out_file)

    def decode(self, emit, insts, mask):
        lens = [len(i) for i in insts]
        if self.training:
            predicts = torch.split(emit.argmax(-1)[mask], lens)
        else:
            emit = emit.transpose(0, 1).log_softmax(dim=-1)
            T, B, N = emit.shape

            delta = emit.new_zeros(T, B, N)
            paths = emit.new_zeros(T, B, N, dtype=torch.long)

            delta[0] = self._strans + emit[0]  # [B, N]

            for i in range(1, T):
                scores = self._trans.unsqueeze(0) + delta[i - 1].unsqueeze(-1)
                scores, paths[i] = torch.max(scores, dim=1)
                delta[i] = scores + emit[i]

            predicts = []
            for i, length in enumerate(lens):
                # trace the best tag sequence from the end of the sentence
                # add end transition scores to the total scores before tracing
                prev = torch.argmax(delta[length - 1, i] + self._etrans)

                predict = [prev]
                for j in reversed(range(1, length)):
                    prev = paths[j, i, prev]
                    predict.append(prev)
                # flip the predicted sequence before appending it to the list
                predicts.append(paths.new_tensor(predict).flip(0))

        for (inst, pred) in zip(insts, predicts):
            CWS.set_predict_result(inst, pred, self._label_dict)
            CWS.compute_accuracy_one_inst(inst, self._metric, self.training)

    def create_dictionaries(self, dataset):
        for inst in dataset.all_inst:
            for i in range(len(inst)):
                self._char_dict.count(inst.chars_s[i])
                self._bichar_dict.count(inst.bichars_s[i])
                self._label_dict.count(inst.labels_s[i])
                for subword in inst.subwords_s[i]:
                    if subword in self._subword_pretrained:
                        self._subword_dict.count(re.sub(r'\d', '0', subword))

    def numericalize_all_instances(self, dataset):
        for inst in dataset.all_inst:
            inst.chars_i = torch.tensor([self._char_dict.get_id(i)
                                         for i in inst.chars_s])
            inst.bichars_i = torch.tensor([self._bichar_dict.get_id(i)
                                           for i in inst.bichars_s])
            inst.labels_i = torch.tensor([self._label_dict.get_id(i)
                                          for i in inst.labels_s])
            # each position has a list of subword indices
            # if the subword [i, j) does not exist in vocabularies,
            # then numericalize it with unk_index
            inst.subwords_i = torch.zeros(
                len(inst), self._conf.max_word_length, dtype=torch.long
            )
            for i in range(len(inst)):
                word_indices = torch.tensor([
                    self._subword_dict.get_id(re.sub(r'\d', '0', j))
                    for j in inst.subwords_s[i]
                ])
                inst.subwords_i[i, :len(word_indices)] = word_indices

    def load_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path)
        self._char_dict.load(path + self._char_dict.name,
                             cutoff_freq=self._conf.cutoff_freq,
                             default_keys=[pad, unk])
        self._bichar_dict.load(path + self._bichar_dict.name,
                               cutoff_freq=self._conf.cutoff_freq,
                               default_keys=[pad, unk])
        self._subword_dict.load(path + self._subword_dict.name,
                                cutoff_freq=self._conf.cutoff_freq,
                                default_keys=[pad, unk])
        self._label_dict.load(path + self._label_dict.name)
        print("load dict done")

    def save_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        if not os.path.exists(path):
            os.mkdir(path)
        self._char_dict.save(path + self._char_dict.name)
        self._bichar_dict.save(path + self._bichar_dict.name)
        self._subword_dict.save(path + self._subword_dict.name)
        self._label_dict.save(path + self._label_dict.name)
        print("save dict done")

    @staticmethod
    def del_model(path, eval_num):
        path = os.path.join(path, 'models-%d/' % eval_num)
        if os.path.exists(path):
            # os.rmdir(path)
            shutil.rmtree(path)
            print('Delete model %s done.' % path)
        else:
            print('Delete model %s error, not exist.' % path)

    def load_datasets(self, filenames, datasets, inst_num_max, shuffle=False):
        assert len(datasets) == 0
        names = filenames.strip().split(':')
        assert len(names) > 0
        for name in names:
            dataset = Dataset(filename=name,
                              max_bucket_num=self._conf.max_bucket_num,
                              max_word_length=self._conf.max_word_length,
                              char_batch_size=self._conf.char_batch_size,
                              sent_batch_size=self._conf.sent_batch_size,
                              inst_num_max=inst_num_max,
                              shuffle=shuffle)
            datasets.append(dataset)

    @staticmethod
    def set_predict_result(inst, pred, label_dict):
        inst.labels_i_pred = pred
        inst.labels_s_pred = [label_dict.get_str(i) for i in pred.tolist()]

    @staticmethod
    def compute_accuracy_one_inst(inst, eval_metrics, training):
        gold_num, pred_num, correct_num, total_labels, correct_labels = inst.evaluate()
        if not training:
            eval_metrics.gold_num += gold_num
            eval_metrics.pred_num += pred_num
            eval_metrics.correct_num += correct_num
        eval_metrics.total_labels += total_labels
        eval_metrics.correct_labels += correct_labels

    def set_training_mode(self, training=True):
        self.training = training
        self._model.train(training)

    def compose_batch(self, insts):
        chars = pad_sequence([inst.chars_i for inst in insts], True)
        bichars = pad_sequence([inst.bichars_i for inst in insts], True)
        subwords = pad_sequence([inst.subwords_i for inst in insts], True)
        labels = pad_sequence([inst.labels_i for inst in insts], True)

        # MUST assign for Tensor.cuda() unlike nn.Module
        if self._use_cuda:
            chars = chars.cuda()
            bichars = bichars.cuda()
            subwords = subwords.cuda()
            labels = labels.cuda()
        return chars, bichars, subwords, labels
