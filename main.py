# -*- coding: utf-8 -*-

import argparse
import random

import torch
from src import CWS, Config

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp-des',
                           default='description of this experiment')
    argparser.add_argument('--config',
                           default='config.ini')
    argparser.add_argument('--seed', type=int,
                           default=1)
    argparser.add_argument('--thread', default=4, type=int)

    args = argparser.parse_args()
    conf = Config(args.config)
    conf.update(vars(args))
    print(conf)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # run with CPU, then use multi-thread? What does this mean?
    torch.set_num_threads(conf.cpu_thread_num)

    cws = CWS(conf)
    cws.run()
