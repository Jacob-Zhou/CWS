# -*- coding: utf-8 -*-

import argparse
import datetime

import torch
from torch.utils.data import DataLoader

from config import config
from model import CHAR_LSTM_CRF
from train import process_data

if __name__ == '__main__':
    # init config
    model_name = 'char_lstm_crf'
    config = config[model_name]

    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--device', '-d', default='-1',
                        help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=4, type=int,
                        help='max num of threads')
    args = parser.parse_args()
    print('setting:')
    print(args)

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # loading vocab
    vocab = torch.load(config.vocab_file)
    # loading network
    print("loading model...")
    network = torch.load(config.net_file)
    # if use GPU , move all needed tensors to CUDA
    if torch.cuda.is_available():
        network.cuda()
    print('loading three datasets...')
    test = Corpus(config.test_file)
    # process test data , change string to index
    print('processing datasets...')
    test_data = process_data(vocab, test, max_len=30)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn
    )

    # init evaluator
    evaluator = Evaluator(vocab)
    print('evaluating test data...')

    time_start = datetime.datetime.now()
    test_loss, test_p = evaluator.eval(network, test_loader)
    print('test  : loss = %.4f  precision = %.4f' % (test_loss, test_p))
    time_end = datetime.datetime.now()
    print('iter executing time is ' + str(time_end - time_start))
