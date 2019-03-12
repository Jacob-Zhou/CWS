# -*- coding: utf-8 -*-

import os
import sys
from configparser import ConfigParser


class Configurable(object):
    def __init__(self, config_file, extra_args):
        self.config_file = config_file
        print("read config from " + config_file)
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(
                extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._conf = config
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        assert self.model_dir.endswith('/')
        config.write(open(self.model_dir + self.config_file + '.bak', 'w'))
        print('Loaded config file successfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    @property
    def cpu_thread_num(self):
        return self._conf.getint('Run', 'cpu_thread_num')

    @property
    def multi_thread_decode(self):
        return self._conf.getint('Run', 'multi_thread_decode') != 0

    @property
    def viterbi_decode(self):
        return self._conf.getint('Run', 'viterbi_decode') != 0

    @property
    def use_labeled_crf_loss(self):
        return self._conf.getint('Run', 'use_labeled_crf_loss') != 0

    # only when use labeled-crf-loss
    # default: arc prob = sum prob of all labels
    # if setting this to 0, arc prob = max prob among all labels
    @property
    def max_label_prob_as_arc_prob_when_decode(self):
        ret = self._conf.getint(
            'Run', 'max_label_prob_as_arc_prob_when_decode') != 0
        if ret:
            assert self.use_labeled_crf_loss
        return ret

    @property
    def use_first_child_score(self):
        ret = self._conf.getint('Run', 'use_first_child_score') != 0
        if ret:
            assert self.use_sib_score
        return ret

    @property
    def use_sib_score(self):
        ret = self._conf.getint('Run', 'use_sib_score') != 0
        if ret:
            assert self.use_labeled_crf_loss or self.use_unlabeled_crf_loss
        return ret

    @property
    def use_unlabeled_crf_loss(self):
        return not self.use_labeled_crf_loss and self._conf.getint('Run', 'use_unlabeled_crf_loss') != 0

    @property
    def sent_num_one_batch(self):
        return self._conf.getint('Run', 'sent_num_one_batch')

    @property
    def char_num_one_batch(self):
        return self._conf.getint('Run', 'char_num_one_batch')

    @property
    def max_bucket_num(self):
        # negative means not using bucket
        return self._conf.getint('Run', 'max_bucket_num')

    @property
    def is_train(self):
        return self._conf.getint('Run', 'is_train') > 0

    @property
    def is_test(self):
        return self._conf.getint('Run', 'is_test') > 0

    @property
    def device(self):
        return self._conf.get('Run', 'device')

    @property
    def dict_dir(self):
        return self._conf.get('Run', 'dict_dir')

    @property
    def model_dir(self):
        return self._conf.get('Run', 'model_dir')

    @property
    def inst_num_max(self):
        return self._conf.getint('Run', 'inst_num_max')

    @property
    def model_eval_num(self):
        return self._conf.getint('Test', 'model_eval_num')

    @property
    def test_files(self):
        return self._conf.get('Train', 'test_files')

    @property
    def train_files(self):
        # use ; to split multiple training datasets
        return self._conf.get('Train', 'train_files')

    @property
    def dev_files(self):
        return self._conf.get('Train', 'dev_files')

    @property
    def is_dictionary_exist(self):
        return self._conf.getint('Train', 'is_dictionary_exist') > 0

    @property
    def train_max_eval_num(self):
        return self._conf.getint('Train', 'train_max_eval_num')

    @property
    def save_model_after_eval_num(self):
        return self._conf.getint('Train', 'save_model_after_eval_num')

    @property
    def patience(self):
        return self._conf.getint('Train', 'patience')

    @property
    def save_model_after_eval_num(self):
        return self._conf.getint('Train', 'save_model_after_eval_num')

    @property
    def lstm_layer_num(self):
        return self._conf.getint('Network', 'lstm_layer_num')

    @property
    def char_emb_dim(self):
        return self._conf.getint('Network', 'char_emb_dim')

    @property
    def emb_dropout(self):
        return self._conf.getfloat('Network', 'emb_dropout')

    @property
    def lstm_hidden_dim(self):
        return self._conf.getint('Network', 'lstm_hidden_dim')

    @property
    def lstm_dropout(self):
        return self._conf.getfloat('Network', 'lstm_dropout')

    @property
    def learning_rate(self):
        return self._conf.getfloat('Optimizer', 'learning_rate')

    @property
    def decay(self):
        return self._conf.getfloat('Optimizer', 'decay')

    @property
    def decay_steps(self):
        return self._conf.getint('Optimizer', 'decay_steps')

    @property
    def beta_1(self):
        return self._conf.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._conf.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._conf.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._conf.getfloat('Optimizer', 'clip')
