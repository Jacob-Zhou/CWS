# -*- coding: utf-8 -*-

from torch import optim


class Optimizer:

    def __init__(self, params_to_optimize, conf):
        self.optim = optim.Adam(params=params_to_optimize,
                                lr=conf.learning_rate)

    def step(self):
        self.optim.step()
        self.optim.zero_grad()
