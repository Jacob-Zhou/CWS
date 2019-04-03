# -*- coding: utf-8 -*-

from torch import optim


class Optimizer:

    def __init__(self, params_to_optimize, conf):
        def annealing(x): return conf.lr * ((1 - conf.decay)**x)
        self.optimizer = optim.SGD(params=params_to_optimize,
                                   lr=conf.lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                     lr_lambda=annealing)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
