# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from parser import Parser

import numpy as np
import torch

from config import Configurable

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp_des',
                           default='description of this experiment')
    argparser.add_argument('--config_file',
                           default='config.txt')
    argparser.add_argument('--seed', type=int,
                           default=1)
    # argparser.add_argument('--thread', default=4, type=int, help='thread num')

    args, extra_args = argparser.parse_known_args()
    conf = Configurable(args.config_file, extra_args)
    # cudaNo = conf.cudaNo
    # os.environ["CUDA_VISIBLE_DEVICES"] = cudaNo

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print('random_seeds = ', args.seed)

    # run with CPU, then use multi-thread? What does this mean?
    torch.set_num_threads(conf.cpu_thread_num)

    parser = Parser(conf)
    parser.run()
