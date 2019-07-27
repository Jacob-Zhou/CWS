# -*- coding: utf-8 -*-

import sys
from ast import literal_eval
from configparser import ConfigParser


class Config(object):

    def __init__(self, path):
        print("read config from " + path)
        config = ConfigParser()
        config.read(path)
        self._conf = config
        self.kwargs = dict((option, literal_eval(value))
                           for section in self._conf.sections()
                           for option, value in self._conf.items(section))
        print('Loaded config file successfully.')

    def __repr__(self):
        s = "─" * 20 + "─┬─" + "─" * 25 + "\n"
        s += f"{'Param':20} │ {'Value':25}\n"
        s += "─" * 20 + "─┼─" + "─" * 25 + "\n"
        for i, (option, value) in enumerate(self.kwargs.items()):
            s += f"{option:20} │ {value}\n"
        s += "─" * 20 + "─┴─" + "─" * 25 + "\n"
        sys.stdout.flush()
        return s

    def __getattr__(self, attr):
        return self.kwargs.get(attr, None)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, kwargs):
        self.kwargs.update(kwargs)
