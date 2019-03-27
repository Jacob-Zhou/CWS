# -*- coding: utf-8 -*-

import os
import shutil
import time

import torch
import torch.nn as nn
from src.common import eos, pad, bos, unk
from src.cws_model import CWSModel
from src.metric import Metric
from src.optimizer import Optimizer
from src.utils import Dataset, VocabDict, Instance
from torch.nn.utils.rnn import pad_sequence


class CWS(object):

    def __init__(self, conf):
        self._conf = conf
        self._device = torch.device(self._conf.device)
        # self._cpu_device = torch.device('cpu')
        self._use_cuda, self._cuda_device = ('cuda' == self._device.type,
                                             self._device.index)
        if self._use_cuda:
            # please note that the index is the relative index in CUDA_VISIBLE_DEVICES=6,7 (0, 1)
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
        self._char_dict = VocabDict('chars')
        self._bichar_dict = VocabDict('bichars')
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

        self._eval_metrics = Metric()
        self._cws_model = CWSModel('ws', conf, self._use_cuda)

    def run(self):
        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.train_files,
                                        self._train_datasets,
                                        inst_num_max=self._conf.inst_num_max,
                                        shuffle=True)
            if not self._conf.is_dictionary_exist:
                print("create dict...")
                for dataset in self._train_datasets:
                    self.create_dictionaries(dataset)
                self.save_dictionaries(self._conf.dict_dir)
                self.load_dictionaries(self._conf.dict_dir)

                self._cws_model.init_models(len(self._char_dict),
                                            len(self._bichar_dict),
                                            len(self._label_dict))
                self._cws_model.reset_parameters()
                self._cws_model.save_model(self._conf.model_dir, 0)
                return
        self.load_dictionaries(self._conf.dict_dir)
        self._cws_model.init_models(len(self._char_dict),
                                    len(self._bichar_dict),
                                    len(self._label_dict))

        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.dev_files,
                                        self._dev_datasets,
                                        inst_num_max=self._conf.inst_num_max)

        self.open_and_load_datasets(self._conf.test_files,
                                    self._test_datasets,
                                    inst_num_max=self._conf.inst_num_max)

        print('numeralizing all instances in all datasets')
        for dataset in self._train_datasets + self._dev_datasets + self._test_datasets:
            self.numeralize_all_instances(dataset, self._label_dict)

        if self._conf.is_train:
            self._cws_model.load_model(self._conf.model_dir, 0)
        else:
            self._cws_model.load_model(self._conf.model_dir,
                                       self._conf.model_eval_num)

        if self._use_cuda:
            # self._cws_model.cuda()
            self._cws_model.to(self._cuda_device)
            self._strans = self._strans.to(self._cuda_device)
            self._etrans = self._etrans.to(self._cuda_device)
            self._trans = self._trans.to(self._cuda_device)
        print(self._cws_model)

        if self._conf.is_train:
            assert self._optimizer is None
            self._optimizer = Optimizer(self._cws_model.parameters(),
                                        self._conf)
            self.train()
            return

        assert self._conf.is_test
        for dataset in self._test_datasets:
            self.evaluate(dataset=dataset,
                          output_filename=dataset.filename_short + '.out')
            self._eval_metrics.compute_and_output(self._test_datasets[0],
                                                  self._conf.model_eval_num)
            self._eval_metrics.clear()

    def train(self):
        best_eval_cnt, best_accuracy = 0, 0.
        self._eval_metrics.clear()
        for eval_cnt in range(1, self._conf.train_max_eval_num + 1):
            self.set_training_mode(training=True)
            for batch in self._train_datasets[0]:
                self.train_or_eval_one_batch(batch)
            self._eval_metrics.compute_and_output(self._train_datasets[0],
                                                  eval_cnt)
            self._eval_metrics.clear()

            self.evaluate(self._dev_datasets[0])
            self._eval_metrics.compute_and_output(self._dev_datasets[0],
                                                  eval_cnt)
            current_fmeasure = self._eval_metrics.fscore
            self._eval_metrics.clear()

            if best_accuracy < current_fmeasure - 1e-3:
                if eval_cnt > self._conf.save_model_after_eval_num:
                    if best_eval_cnt > self._conf.save_model_after_eval_num:
                        self.del_model(self._conf.model_dir, best_eval_cnt)
                    self._cws_model.save_model(self._conf.model_dir,
                                               eval_cnt)
                    self.evaluate(dataset=self._test_datasets[0],
                                  output_filename=None)
                    self._eval_metrics.compute_and_output(self._test_datasets[0],
                                                          eval_cnt)
                    self._eval_metrics.clear()

                best_eval_cnt = eval_cnt
                best_accuracy = current_fmeasure

            if best_eval_cnt + self._conf.patience <= eval_cnt:
                break

    def train_or_eval_one_batch(self, one_batch):
        print('.', end='')
        # shape of the following tensors:
        # chars: [batch_size, seq_len + 2]
        # bichars: [batch_size, seq_len + 2]
        # labels: [batch_size, seq_len]
        # the bos and eos tokens are added to inputs of the model
        chars, bichars, labels = self.compose_batch_data(one_batch)
        # ignore all pad, bos, and eos tokens
        mask = chars.ne(self._char_dict.pad_index)
        mask &= chars.ne(self._char_dict.bos_index)
        mask &= chars.ne(self._char_dict.eos_index)
        # cut off the first and last ones
        mask = mask[:, 1:-1]

        time1 = time.time()
        out = self._cws_model(chars, bichars)
        subword_mask = mask.new_ones(out.shape[:-1])
        subword_mask &= torch.ones_like(subword_mask[0]).tril()
        subword_mask &= torch.ones_like(subword_mask[0]).triu()
        subword_mask &= mask.unsqueeze(-1)
        out = out.masked_fill(~subword_mask.unsqueeze(-1), float('-inf'))
        time2 = time.time()

        label_loss = self._cws_model.get_loss(out[subword_mask], labels[mask])
        self._eval_metrics.loss_accumulated += label_loss.item()
        time3 = time.time()

        if self.training:
            label_loss.backward()
            nn.utils.clip_grad_norm_(self._cws_model.parameters(),
                                     max_norm=self._conf.clip)
            self._optimizer.step()
        time4 = time.time()

        self.decode(out, one_batch, subword_mask)
        time5 = time.time()

        self._eval_metrics.sent_num += len(one_batch)
        self._eval_metrics.forward_time += time2 - time1
        self._eval_metrics.loss_time += time3 - time2
        self._eval_metrics.backward_time += time4 - time3
        self._eval_metrics.decode_time += time5 - time4

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

    ''' 2018.11.3 by Zhenghua
    I found that using multi-thread for non-viterbi (local) decoding is actually
    much slower than single-thread (ptb labeled-crf-loss train 1-iter: 150s vs. 5s)
    NOTICE:
        multi-process: CAN NOT CWS.set_predict_result(inst, head_pred, label_pred, label_dict),
        this will not change inst of the invoker
    '''

    def decode(self, emit, insts, mask):
        if self.training:
            lengths = [len(i) for i in insts]
            predicts = torch.split(emit.argmax(-1)[mask], lengths)
        else:
            emit = emit.permute(1, 2, 0, 3)
            seq_len, batch_size, n_labels = emit.shape[1:]
            lens = mask.sum(dim=(1, 2)).tolist()

            # [seq_len, batch_size, n_labels]
            delta = emit.new_zeros(seq_len, batch_size, n_labels)
            labels = emit.new_zeros(seq_len, batch_size, n_labels).long()
            splits = emit.new_zeros(seq_len, batch_size, n_labels).long()

            # only records of the upper triangular part are valid shortcuts
            # [seq_len, seq_len, batch_size, n_labels]
            shortcuts = torch.zeros_like(emit)

            # [seq_len, batch_size, n_labels]
            shortcuts[0] = emit[0] + self._strans
            delta[0] = shortcuts[0, 0]

            for i in range(1, seq_len):
                scores = self._trans + delta[i - 1].unsqueeze(-1)
                scores, labels[i] = torch.max(scores, dim=1)
                shortcuts[i] = scores + emit[i]
                delta[i], splits[i] = torch.max(shortcuts[:i+1, i], dim=0)

            predicts = []
            for i, length in enumerate(lens):
                # trace the best tag sequence from the end of the sentence
                # add end transition scores to the total scores before tracing
                prev = torch.argmax(delta[length - 1, i] + self._etrans)
                split = splits[length - 1, i, prev]

                predict, word_lens = [prev], [length - split]
                while split > 0:
                    prev = labels[split - 1, i, prev]
                    predict.append(prev)
                    word_lens.append(split - splits[split - 1, i, prev])
                    split = splits[split - 1, i, prev]
                predict = [Instance.recover(label, word_length)
                           for label, word_length in zip(predict, word_lens)]
                predict = [label for pred in reversed(predict)
                           for label in pred]
                predicts.append(labels.new_tensor(predict))
                assert len(predict) == length

        for (inst, pred) in zip(insts, predicts):
            CWS.set_predict_result(inst, pred, self._label_dict)
            CWS.compute_accuracy_one_inst(inst, self._eval_metrics,
                                          self.training)

    def create_dictionaries(self, dataset):
        for inst in dataset.all_inst:
            for i in range(len(inst)):
                self._char_dict.add_key_into_counter(inst.chars_s[i])
                self._bichar_dict.add_key_into_counter(inst.bichars_s[i])
                self._label_dict.add_key_into_counter(inst.labels_s[i])

    def numeralize_all_instances(self, dataset, label_dict):
        # the bos and eos tokens are added to the sequences of each inst here,
        # acting as representations of the beginning and end of the sentence
        for inst in dataset.all_inst:
            inst.chars_i = torch.tensor([self._char_dict.get_id(i)
                                         for i in [bos] + inst.chars_s + [eos]])
            inst.bichars_i = torch.tensor([self._bichar_dict.get_id(i)
                                           for i in [bos] + inst.bichars_s + [eos]])
            inst.labels_i = torch.tensor([self._label_dict.get_id(i)
                                          for i in inst.labels_s])

    def load_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path)
        self._char_dict.load(path + self._char_dict.name,
                             default_keys=[pad, unk, bos, eos])
        self._bichar_dict.load(path + self._bichar_dict.name,
                               default_keys=[pad, unk, bos, eos])
        self._label_dict.load(path + self._label_dict.name)
        print("load dict done")

    def save_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        if not os.path.exists(path):
            os.mkdir(path)
        self._char_dict.save(path + self._char_dict.name)
        self._bichar_dict.save(path + self._bichar_dict.name)
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

    def open_and_load_datasets(self, filenames, datasets, inst_num_max, shuffle=False):
        assert len(datasets) == 0
        names = filenames.strip().split(':')
        assert len(names) > 0
        for name in names:
            datasets.append(Dataset(filename=name,
                                    max_bucket_num=self._conf.max_bucket_num,
                                    char_num_one_batch=self._conf.char_num_one_batch,
                                    sent_num_one_batch=self._conf.sent_num_one_batch,
                                    inst_num_max=inst_num_max,
                                    shuffle=shuffle))

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
        self._cws_model.train(training)

    def compose_batch_data(self, one_batch):
        chars = pad_sequence([inst.chars_i for inst in one_batch], True)
        bichars = pad_sequence([inst.bichars_i for inst in one_batch], True)
        labels = pad_sequence([inst.labels_i for inst in one_batch], True)
        # MUST assign for Tensor.cuda() unlike nn.Module
        if torch.cuda.is_available():
            chars = chars.cuda()
            bichars = bichars.cuda()
            labels = labels.cuda()
        return chars, bichars, labels
