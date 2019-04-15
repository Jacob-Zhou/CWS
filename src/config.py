# -*- coding: utf-8 -*-

import os
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
    def sent_batch_size(self):
        return self._conf.getint('Run', 'sent_batch_size')

    @property
    def char_batch_size(self):
        return self._conf.getint('Run', 'char_batch_size')

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
    def train_files(self):
        # use ; to split multiple training datasets
        return self._conf.get('Train', 'train_files')

    @property
    def dev_files(self):
        return self._conf.get('Train', 'dev_files')

    @property
    def test_files(self):
        return self._conf.get('Train', 'test_files')

    @property
    def emb_char_file(self):
        file = self._conf.get('Train', 'emb_char_file')
        return file if os.path.exists(file) else None

    @property
    def emb_bichar_file(self):
        file = self._conf.get('Train', 'emb_bichar_file')
        return file if os.path.exists(file) else None

    @property
    def emb_subword_file(self):
        file = self._conf.get('Train', 'emb_subword_file')
        return file if os.path.exists(file) else None

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
    def max_word_length(self):
        return self._conf.getint('Train', 'max_word_length')

    @property
    def cutoff_freq(self):
        return self._conf.getint('Train', 'cutoff_freq')

    @property
    def patience(self):
        return self._conf.getint('Train', 'patience')

    @property
    def n_char_emb(self):
        return self._conf.getint('Network', 'n_char_emb')

    @property
    def n_subword_emb(self):
        return self._conf.getint('Network', 'n_subword_emb')

    @property
    def emb_dropout(self):
        return self._conf.getfloat('Network', 'emb_dropout')

    @property
    def n_lstm_hidden(self):
        return self._conf.getint('Network', 'n_lstm_hidden')

    @property
    def n_lstm_layers(self):
        return self._conf.getint('Network', 'n_lstm_layers')

    @property
    def lstm_dropout(self):
        return self._conf.getfloat('Network', 'lstm_dropout')

    @property
    def lr(self):
        return self._conf.getfloat('Optimizer', 'lr')

    @property
    def clip(self):
        return self._conf.getfloat('Optimizer', 'clip')
