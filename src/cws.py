# -*- coding: utf-8 -*-

import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from src.common import pad, unk
from src.metric import Metric
from src.model import CWSModel
from src.utils import Dataset, VocabDict
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import data_parallel


class CWS(object):

    def __init__(self, conf):
        self._conf = conf
        # self._cpu_device = torch.device('cpu')
        self.training = False

        self._optimizer = None
        self._use_bucket = (self._conf.max_bucket_num > 1)
        self._train_datasets = []
        self._dev_datasets = []
        self._test_datasets = []

        self._tokenizer = BertTokenizer.from_pretrained(self._conf.bert_vocab)
        self._char_dict = VocabDict('chars')
        self._bichar_dict = VocabDict('bichars')
        # there may be more than one label dictionaries
        self._label_dict = VocabDict('labels')
        self._extra_dicts = {}
        self._extra_dicts["train"] = self.load_extra_dicts(self._conf.train_extra_dictionarys)
        self._extra_dicts["dev"] = self.load_extra_dicts(self._conf.dev_extra_dictionarys)
        self._extra_dicts["test"] = self.load_extra_dicts(self._conf.test_extra_dictionarys)

        # transition scores of the labels
        # NOTE: make sure the label dict MUST have been sorted
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
        self._model = CWSModel('ws', conf)

    def run(self):
        if self._conf.is_train:
            self._train_datasets = self.load_datasets(self._conf,
                                                      self._conf.train_files,
                                                      shuffle=True)
            if not self._conf.is_dictionary_exist:
                print("create dict...")
                self.create_dictionaries(self._train_datasets)
                self.save_dictionaries(self._conf.path)
                self.load_dictionaries(self._conf.path)

                return
        self.load_dictionaries(self._conf.path)
        if self._conf.is_train:
            self._dev_datasets = self.load_datasets(self._conf,
                                                    self._conf.dev_files)

        self._test_datasets = self.load_datasets(self._conf,
                                                 self._conf.test_files)

        print('numericalizing all instances in all datasets')
        self.numericalize_all_instances(self._train_datasets, type="train")
        self.numericalize_all_instances(self._dev_datasets, type="dev")
        self.numericalize_all_instances(self._test_datasets, type="test")

        self._model.init_models(self._char_dict,
                                self._bichar_dict,
                                self._label_dict,
                                self._extra_dicts["train"])
        if not self._conf.is_train:
            self._model.load_model(self._conf.path,
                                   self._conf.model_eval_num)
        print(self._model)

        if torch.cuda.is_available():
            self._model.to(self._conf.device)
            self._strans = self._strans.to(self._conf.device)
            self._etrans = self._etrans.to(self._conf.device)
            self._trans = self._trans.to(self._conf.device)

        # non-sense hack (very bad)
        for test in self._test_datasets:
            test.index = 0

        if self._conf.is_train:
            assert self._optimizer is None
            self._optimizer = optim.Adam(lr=self._conf.lr,
                                         params=self._model.parameters())
            self.train()
            return

        assert self._conf.is_test
        dataset = self._test_datasets[0]
        for i in range(len(self._conf.train_files)):
            self._test_datasets[0].index = i
            self.evaluate(dataset,
                          dataset.filename_short + '.out.' + str(i))
            self._metric.compute_and_output(dataset,
                                            self._conf.model_eval_num)
            self._metric.clear()

    def train(self):
        best_eval_cnt, best_accuracy = 0, 0.
        self._metric.clear()
        for eval_cnt in range(1, self._conf.train_max_eval_num + 1):
            self.set_training_mode(training=True)
            iters = [None] + [iter(aux) for aux in self._train_datasets[1:]]
            for batch in self._train_datasets[0]:
                for i in range(1, len(iters)):
                    self._optimizer.zero_grad()
                    try:
                        self.train_or_eval_one_batch(next(iters[i]), i)
                    except StopIteration:
                        iters[i] = iter(self._train_datasets[i])
                        self.train_or_eval_one_batch(next(iters[i]), i)
                self._optimizer.zero_grad()
                self.train_or_eval_one_batch(batch, 0)
            self._metric.compute_and_output(self._train_datasets[0], eval_cnt)
            self._metric.clear()

            for dev in self._dev_datasets:
                self.evaluate(dev)
                self._metric.compute_and_output(dev, eval_cnt)
                if dev.index == 0:
                    current_fmeasure = self._metric.fscore
                self._metric.clear()

            if best_accuracy < current_fmeasure - 1e-3:
                if eval_cnt > self._conf.patience:
                    self._model.save_model(self._conf.path,
                                           eval_cnt)
                    for test in self._test_datasets:
                        self.evaluate(test)
                        self._metric.compute_and_output(test, eval_cnt)
                        self._metric.clear()

                best_eval_cnt = eval_cnt
                best_accuracy = current_fmeasure

            if best_eval_cnt + self._conf.patience <= eval_cnt:
                break
        print("The training ended at epoch %d" % eval_cnt)
        print("The best fscore of dev is %.3f at epoch %d" %
              (best_accuracy, best_eval_cnt))

    def train_or_eval_one_batch(self, insts, index=0):
        print('.', end='')
        subwords, chars, bichars, labels, dict_feats = self.compose_batch(insts)
        mask = chars.ne(self._char_dict.pad_index)
        time1 = time.time()
        out = self._model(subwords, chars, bichars, dict_feats, index)
        time2 = time.time()

        loss = self._model.get_loss(out, labels, mask)
        self._metric.total_loss += loss.item()
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
            self.train_or_eval_one_batch(batch, dataset.index)

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

    def numericalize_all_instances(self, datasets, type="train"):
        for dataset in datasets:
            for inst in dataset.all_inst:
                # for chinese, no extra operations is required
                inst.subwords_i = torch.tensor(
                    self._tokenizer.convert_tokens_to_ids(
                        ['[CLS]']+[i if i in self._tokenizer.vocab else '[UNK]'
                                   for i in inst.chars_s]+['[SEP]'])
                )
                inst.dict_feats_i = [None] * len(self._extra_dicts[type])
                for dict_k in range(len(self._extra_dicts[type])):
                    inst.dict_feats_i[dict_k] = torch.tensor(self.extract_dict_features(inst.chars_s, dict_k, type=type))
                inst.chars_i = torch.tensor([self._char_dict.get_id(i)
                                             for i in inst.chars_s])
                inst.bichars_i = torch.tensor([self._bichar_dict.get_id(i)
                                               for i in inst.bichars_s])
                inst.labels_i = torch.tensor([self._label_dict.get_id(i)
                                              for i in inst.labels_s])

    def extract_dict_features(self, chars, dict_k, type="train"):
        max_len = self._conf.max_word_len[dict_k]
        min_len = self._conf.min_word_len[dict_k]
        result = np.zeros((len(chars), max_len - min_len + 1), dtype=int)
        for i in range(len(chars)):
            # fw
            for l in range(min_len, max_len + 1):
                if (i + l - 1) >= len(chars):
                    continue
                word = ''.join(chars[i: i + l])
                if word in self._extra_dicts[type][dict_k]:
                    result[i][l - min_len] |= 1
                    result[i + l - 1][l - min_len] |= 4
                    for mid in range(i + 1, i + l - 1):
                        result[mid][l - min_len] |= 2
        return result

        # AAAI-18 的词典特征
        # result = []
        # for i in range(len(chars)):
        #     # fw
        #     word_tag = []
        #     for l in range(max_len - 1, min_len - 2, -1):
        #         if (i - l) < 0:
        #             word_tag.append(0)
        #             continue
        #         word = ''.join(chars[i - l:i + 1])
        #         if word in self._extra_dicts:
        #             word_tag.append(self._extra_dicts[word])
        #         else:
        #             word_tag.append(0)
        #     # bw
        #     for l in range(min_len - 1, max_len):
        #         if (i + l) >= len(chars):
        #             word_tag.append(0)
        #             continue
        #         word = ''.join(chars[i:i + l + 1])
        #         if word in self._extra_dicts:
        #             word_tag.append(self._extra_dicts[word])
        #         else:
        #             word_tag.append(0)
        #     result.append(word_tag)
        # return result
    
    def load_extra_dicts(self, dict_files):

        def load_extra_dict(sub_dict_files):
            files = sub_dict_files.split(",")
            dictionary = dict()
            for file in files:
                file = file.strip()
                if not str.isspace(file):
                    with open(file, mode='r', encoding='utf-8') as reader:
                        for line in reader:
                            token = line.strip().split(" ")
                            if len(token) == 2:
                                dictionary[token[0]] = token[1]
                            else:
                                dictionary[token[0]] = 1
            return dictionary

        dicts = []
        for dictionary_file in dict_files:
            dicts.append(load_extra_dict(dictionary_file))
        return dicts

    def create_dictionaries(self, datasets):
        for dataset in datasets:
            for inst in dataset.all_inst:
                for i in range(len(inst)):
                    self._char_dict.add_key_into_counter(inst.chars_s[i])
                    self._bichar_dict.add_key_into_counter(inst.bichars_s[i])
                    self._label_dict.add_key_into_counter(inst.labels_s[i])

    def load_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path)
        self._char_dict.load(path + self._char_dict.name,
                             cutoff_freq=self._conf.cutoff_freq,
                             default_keys=[pad, unk])
        self._bichar_dict.load(path + self._bichar_dict.name,
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

    def load_datasets(self, conf, filenames, shuffle=False):
        datasets = [
            Dataset(filename=name,
                    max_bucket_num=conf.max_bucket_num,
                    max_sent_length=conf.max_sent_length,
                    char_batch_size=conf.char_batch_sizes[i],
                    sent_batch_size=conf.sent_batch_size,
                    inst_num_max=conf.inst_num_max,
                    shuffle=shuffle)
            for i, name in enumerate(filenames)
        ]
        for i, dataset in enumerate(datasets):
            dataset.index = i

        return datasets

    @staticmethod
    def set_predict_result(inst, pred, label_dict):
        inst.labels_i_pred = pred
        inst.labels_s_pred = [label_dict.get_str(i) for i in pred.tolist()]

    @staticmethod
    def compute_accuracy_one_inst(inst, eval_metrics, training):
        gold_num, pred_num, correct_num,\
            total_labels, correct_labels = inst.evaluate()
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
        subwords = pad_sequence([inst.subwords_i for inst in insts], True)
        dict_feats = [None] * len(self._extra_dicts['train'])
        for dict_k in range(len(dict_feats)):
            dict_feats[dict_k] = pad_sequence([inst.dict_feats_i[dict_k] for inst in insts], True)
        chars = pad_sequence([inst.chars_i for inst in insts], True)
        bichars = pad_sequence([inst.bichars_i for inst in insts], True)
        labels = pad_sequence([inst.labels_i for inst in insts], True)
        # MUST assign for Tensor.cuda() unlike nn.Module
        if torch.cuda.is_available():
            subwords = subwords.cuda()
            for dict_k in range(len(dict_feats)):
                dict_feats[dict_k]= dict_feats[dict_k].cuda()
            chars = chars.cuda()
            bichars = bichars.cuda()
            labels = labels.cuda()
        return subwords, chars, bichars, labels, dict_feats
