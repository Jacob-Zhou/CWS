import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

from common import (data_type, get_time_str, padding_id, padding_str,
                    unknown_id, unknown_str)
from dataset import Dataset
from optimizer import Optimizer
from parser_model import ParserModel
from vocab import VocabDict


class Parser(object):
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
            # an alternative way: CUDA_VISIBLE_DEVICE=6 python ../src/main.py ...
            self._cuda_device = 0
        self._optimizer = None
        self._use_bucket = (self._conf.max_bucket_num > 1)
        self._train_datasets = []
        self._dev_datasets = []
        self._test_datasets = []
        self._word_dict = VocabDict('words')
        self._char_dict = VocabDict('chars')
        self._bichar_dict = VocabDict('bichars')
        # there may be more than one label dictionaries
        self._label_dict = VocabDict('labels')

        self._eval_metrics = Metric()
        self._parser_model = ParserModel('ws', conf, self._use_cuda)

    def run(self):
        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.train_files, self._train_datasets,
                                        inst_num_max=self._conf.inst_num_max)
            if not self._conf.is_dictionary_exist:
                print("create dict...")
                for dataset in self._train_datasets:
                    self.create_dictionaries(dataset, self._label_dict)
                self.save_dictionaries(self._conf.dict_dir)
                self.load_dictionaries(self._conf.dict_dir)
                self._parser_model.init_models(self._char_dict.size(
                ), self._bichar_dict.size(), self._label_dict.size())
                self._parser_model.reset_parameters()
                self._parser_model.save_model(self._conf.model_dir, 0)
                return
        self.load_dictionaries(self._conf.dict_dir)
        self._parser_model.init_models(self._char_dict.size(
        ), self._bichar_dict.size(), self._label_dict.size())

        if self._conf.is_train:
            self.open_and_load_datasets(self._conf.dev_files, self._dev_datasets,
                                        inst_num_max=self._conf.inst_num_max)

        self.open_and_load_datasets(self._conf.test_files, self._test_datasets,
                                    inst_num_max=self._conf.inst_num_max)

        print(
            'numeralizing [and pad if use-bucket] all instances in all datasets')
        for dataset in self._train_datasets + self._dev_datasets + self._test_datasets:
            self.numeralize_all_instances(dataset, self._label_dict)
            if self._use_bucket:
                self.pad_all_inst(dataset)

        if self._conf.is_train:
            self._parser_model.load_model(self._conf.model_dir, 0)
        else:
            self._parser_model.load_model(
                self._conf.model_dir, self._conf.model_eval_num)

        if self._use_cuda:
            # self._parser_model.cuda()
            self._parser_model.to(self._cuda_device)

        if self._conf.is_train:
            assert self._optimizer is None
            self._optimizer = Optimizer(
                [param for param in self._parser_model.parameters() if param.requires_grad], self._conf)
            self.train()
            return

        assert self._conf.is_test
        for dataset in self._test_datasets:
            self.evaluate(
                dataset, output_file_name=dataset.file_name_short + '.out')
            self._eval_metrics.compute_and_output(
                self._test_datasets[0], self._conf.model_eval_num)
            self._eval_metrics.clear()

    def train(self):
        update_step_cnt, eval_cnt, best_eval_cnt, best_accuracy = 0, 0, 0, 0.
        self._eval_metrics.clear()
        self.set_training_mode(is_training=True)
        while True:
            inst_num = self.train_or_eval_one_batch(
                self._train_datasets[0], is_training=True)
            assert inst_num > 0
            update_step_cnt += 1
            print('.', end='')

            if 0 == update_step_cnt % self._conf.eval_every_update_step_num:
                eval_cnt += 1
                self._eval_metrics.compute_and_output(
                    self._train_datasets[0], eval_cnt)
                self._eval_metrics.clear()

                self.evaluate(self._dev_datasets[0])
                self._eval_metrics.compute_and_output(
                    self._dev_datasets[0], eval_cnt)
                current_fmeasure = self._eval_metrics.fscore
                self._eval_metrics.clear()

                if best_accuracy < current_fmeasure - 1e-3:
                    if eval_cnt > self._conf.save_model_after_eval_num:
                        if best_eval_cnt > self._conf.save_model_after_eval_num:
                            self.del_model(self._conf.model_dir, best_eval_cnt)
                        self._parser_model.save_model(
                            self._conf.model_dir, eval_cnt)
                        self.evaluate(
                            self._test_datasets[0], output_file_name=None)
                        self._eval_metrics.compute_and_output(
                            self._test_datasets[0], eval_cnt)
                        self._eval_metrics.clear()

                    best_eval_cnt = eval_cnt
                    best_accuracy = current_fmeasure

                self.set_training_mode(is_training=True)

            if (best_eval_cnt + self._conf.train_stop_after_eval_num_no_improve < eval_cnt) or \
                    (eval_cnt > self._conf.train_max_eval_num):
                break

    def train_or_eval_one_batch(self, dataset, is_training):
        one_batch, total_word_num, max_len = dataset.get_one_batch(
            rewind=is_training)
        # NOTICE: total_word_num does not include w_0
        inst_num = len(one_batch)
        if 0 == inst_num:
            return 0

        chars, bichars, gold_labels, lstm_masks = self.compose_batch_data(
            one_batch, max_len)

        time1 = time.time()
        mlp_out = self._parser_model(chars, bichars)
        time2 = time.time()

        label_loss = self._parser_model.get_loss(
            mlp_out, gold_labels) / total_word_num
        self._eval_metrics.loss_accumulated += label_loss.item()
        time3 = time.time()

        if is_training:
            label_loss.backward()
            nn.utils.clip_grad_norm_(self._parser_model.parameters(),
                                     max_norm=self._conf.clip)
            self._optimizer.step()
            self.zero_grad()
        time4 = time.time()

        self.decode(mlp_out, one_batch, self._label_dict)
        time5 = time.time()

        self._eval_metrics.forward_time += time2 - time1
        self._eval_metrics.loss_time += time3 - time2
        self._eval_metrics.backward_time += time4 - time3
        self._eval_metrics.decode_time += time5 - time4
        return inst_num

    def evaluate(self, dataset, output_file_name=None):
        with torch.no_grad():
            self.set_training_mode(is_training=False)
            while True:
                inst_num = self.train_or_eval_one_batch(
                    dataset, is_training=False)
                print('.', end='')
                if 0 == inst_num:
                    break

            if output_file_name is not None:
                with open(output_file_name, 'w', encoding='utf-8') as out_file:
                    all_inst = dataset.all_inst
                    for inst in all_inst:
                        inst.write(out_file)

    ''' 2018.11.3 by Zhenghua
    I found that using multi-thread for non-viterbi (local) decoding is actually much slower than single-thread (ptb labeled-crf-loss train 1-iter: 150s vs. 5s)
    NOTICE:
        multi-process: CAN NOT Parser.set_predict_result(inst, head_pred, label_pred, label_dict), this will not change inst of the invoker
    '''

    def decode(self, scores, one_batch, label_dict):
        inst_num = scores.size(0)
        assert inst_num == len(one_batch)
        ret = [score[:inst.size()]
               for (score, inst) in zip(scores.argmax(-1), one_batch)]

        self._eval_metrics.sent_num += len(one_batch)
        for (inst, pred) in zip(one_batch, ret):
            Parser.set_predict_result(inst, pred, label_dict)
            Parser.compute_accuracy_one_inst(inst, self._eval_metrics)

    def create_dictionaries(self, dataset, label_dict):
        all_inst = dataset.all_inst
        for inst in all_inst:
            for i in range(1, inst.size()):
                self._char_dict.add_key_into_counter(inst.chars_s[i])
                self._bichar_dict.add_key_into_counter(inst.bichars_s[i])
                self._label_dict.add_key_into_counter(inst.labels_s[i])

    @staticmethod
    def get_candidate_heads(length, gold_arcs):
        candidate_heads = np.array(
            [0] * length * length, dtype=data_type_int32).reshape(length, length)
        for m in range(1, length):
            h = gold_arcs[m]
            if h < 0:
                for i in range(length):
                    candidate_heads[m][i] = 1
            else:
                candidate_heads[m][h] = 1
        return candidate_heads

    def numeralize_all_instances(self, dataset, label_dict):
        all_inst = dataset.all_inst
        for inst in all_inst:
            for i in range(0, inst.size()):
                inst.chars_i[i] = self._char_dict.get_id(inst.chars_s[i])
                inst.bichars_i[i] = self._bichar_dict.get_id(inst.bichars_s[i])
                inst.labels_i[i] = self._label_dict.get_id(inst.labels_s[i])

    def load_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path)
        self._char_dict.load(path + self._char_dict.name,
                             cutoff_freq=self._conf.word_freq_cutoff,
                             default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._bichar_dict.load(path + self._bichar_dict.name,
                               default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._label_dict.load(
            path + self._label_dict.name, default_keys_ids=())
        print("load  dict done")

    def save_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert not os.path.exists(path)
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

    def open_and_load_datasets(self, file_names, datasets, inst_num_max):
        assert len(datasets) == 0
        names = file_names.strip().split(':')
        assert len(names) > 0
        for name in names:
            datasets.append(Dataset(name, max_bucket_num=self._conf.max_bucket_num,
                                    word_num_one_batch=self._conf.word_num_one_batch,
                                    sent_num_one_batch=self._conf.sent_num_one_batch,
                                    inst_num_max=inst_num_max,
                                    max_len=self._conf.sent_max_len))

    @staticmethod
    def set_predict_result(inst, pred, label_dict):
        # assert arc_pred.size(0) == inst.size()
        for i in range(inst.size()):
            inst.labels_i_predict[i] = pred[i]
            inst.labels_s_predict[i] = label_dict.get_str(pred[i])

    @staticmethod
    def compute_accuracy_one_inst(inst, eval_metrics):
        gold_num, pred_num, correct_num = inst.evaluate()
        eval_metrics.gold_num += gold_num
        eval_metrics.pred_num += pred_num
        eval_metrics.correct_num += correct_num

    def set_training_mode(self, is_training=True):
        self._parser_model.train(mode=is_training)

    def zero_grad(self):
        self._parser_model.zero_grad()

    def pad_all_inst(self, dataset):
        for (max_len, inst_num_one_batch, this_bucket) in dataset.all_buckets:
            for inst in this_bucket:
                inst.chars_i, inst.bichars_i, inst.labels_i, inst.lstm_mask = \
                    self.pad_one_inst(inst, max_len)

    def pad_one_inst(self, inst, max_sz):
        sz = inst.size()
        assert len(inst.chars_i) == sz
        assert max_sz >= sz
        pad_sz = (0, max_sz - sz)
        '''
        return torch.from_numpy(np.pad(inst.chars_i, pad_sz, 'constant', constant_values=padding_id)), \
            torch.from_numpy(np.pad(inst.bichars_i, pad_sz, 'constant', constant_values=padding_id)), \
            torch.from_numpy(np.pad(inst.labels_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label)), \
            torch.from_numpy(np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=padding_id))
        '''
        return np.pad(inst.chars_i, pad_sz, 'constant', constant_values=0), \
            np.pad(inst.bichars_i, pad_sz, 'constant', constant_values=0), \
            np.pad(inst.labels_i, pad_sz, 'constant', constant_values=0), \
            np.pad(np.ones(sz, dtype=data_type), pad_sz,
                   'constant', constant_values=0)

    def compose_batch_data(self, one_batch, max_len):
        chars, bichars, labels, lstm_masks = [], [], [], []
        for inst in one_batch:
            if self._use_bucket:
                chars.append(inst.chars_i)
                bichars.append(inst.bichars_i)
                labels.append(inst.labels_i)
                lstm_masks.append(inst.lstm_mask)
            else:
                ret = self.pad_one_inst(inst, max_len)
                chars.append(ret[0])
                bichars.append(ret[1])
                labels.append(ret[2])
                lstm_masks.append(ret[3])
        # dim: batch max-len
        chars, bichars, labels, lstm_masks = \
            torch.from_numpy(np.stack(chars, axis=0)),\
            torch.from_numpy(np.stack(bichars, axis=0)), \
            torch.from_numpy(np.stack(labels, axis=0)),\
            torch.from_numpy(np.stack(lstm_masks, axis=0))

        # MUST assign for Tensor.cuda() unlike nn.Module
        if self._use_cuda:
            chars, bichars, labels, lstm_masks = \
                chars.cuda(self._cuda_device),\
                bichars.cuda(self._cuda_device), \
                labels.cuda(self._cuda_device),\
                lstm_masks.cuda(self._cuda_device)
        return chars, bichars, labels, lstm_masks


class Metric(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.sent_num = 0
        self.gold_num = 0
        self.pred_num = 0
        self.correct_num = 0
        self.precision = 0.
        self.recall = 0.
        self.fscore = 0.
        self.loss_accumulated = 0.
        self.start_time = time.time()
        self.time_gap = 0.
        self.forward_time = 0.
        self.loss_time = 0.
        self.backward_time = 0.
        self.decode_time = 0.

    def compute_and_output(self, dataset, eval_cnt):
        assert self.gold_num > 0
        self.precision = 100. * self.correct_num/self.pred_num
        self.recall = 100. * self.correct_num/self.gold_num
        self.fscore = 100. * self.correct_num*2/(self.pred_num+self.gold_num)
        self.time_gap = float(time.time() - self.start_time)
        print(
            "\n%30s(%5d): loss=%.3f precision=%.3f, recall=%.3f, fscore=%.3f, %d sentences, time=%.3f (%.1f %.1f %.1f %.1f) [%s]" %
            (dataset.file_name_short, eval_cnt, self.loss_accumulated, self.precision, self.recall, self.fscore,
             self.sent_num, self.time_gap, self.forward_time, self.loss_time,
             self.backward_time, self.decode_time, get_time_str())
        )
