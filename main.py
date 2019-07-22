# -*- coding: utf-8 -*-

import argparse
import os
import random

import torch
import numpy as np
import ast
from src import CWS, Config

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp-des',
                           default='description of this experiment')
    argparser.add_argument('--config', default='config.ini',
                           help='config file')
    argparser.add_argument('--path', '-p', default='exp',
                           help='path to saved files')
    argparser.add_argument('--device', '-d', default='-1',
                           help='ID of GPU to use')
    argparser.add_argument('--seed', '-s', default=1, type=int,
                           help='seed for generating random numbers')
    argparser.add_argument('--threads', default=4, type=int)
    argparser.add_argument('--is-dictionary-exist', action='store_true')
    argparser.add_argument('--is-train', action='store_true')
    argparser.add_argument('--is-test', action='store_true')
    argparser.add_argument('--dict-feature-type', default='ours',
                           help="type of dictionary 'ours' or 'aaai'")
    argparser.add_argument('--dict-concat-type', default=None,
                           help="type of dictionary 'pre' or 'post'")
    argparser.add_argument('--with-dict-emb', default=True, type=ast.literal_eval)
    argparser.add_argument('--with_extra_dictionarys', default=True, type=ast.literal_eval)

    args = argparser.parse_args()
    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    conf = Config(args.config)
    conf.update(vars(args))
    print(conf)
    if not os.path.exists(conf.path):
        os.mkdir(conf.path)
    torch.save(conf, os.path.join(conf.path, 'config'))

    cws = CWS(conf)
    cws.run()
