# -*- coding: utf-8 -*-

import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import config
from models import CHAR_LSTM_CRF
from utils import Corpus, Evaluator, TextDataset, Trainer, Vocab, collate_fn


def process_data(vocab, dataset, max_len=30):
    char_idxs, bichar_idxs, label_idxs = [], [], []

    for charseq, bicharseq, labelseq in zip(dataset.char_seqs, dataset.bichar_seqs, dataset.label_seqs):
        _char_idxs = vocab.char2id(charseq)
        _label_idxs = vocab.label2id(labelseq)
        _bichar_idxs = vocab.bichar2id(bicharseq)

        char_idxs.append(torch.tensor(_char_idxs))
        bichar_idxs.append(torch.tensor(_bichar_idxs))
        label_idxs.append(torch.tensor(_label_idxs))

    return char_idxs, bichar_idxs, label_idxs


if __name__ == '__main__':
    # init config
    model_name = 'char_lstm_crf'
    config = config[model_name]
    for name, value in vars(config).items():
        print('%s = %s' % (name, str(value)))

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--device', '-d', default='-1',
                        help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=4, type=int,
                        help='max num of threads')
    parser.add_argument('--pre_emb', action='store_true',
                        help='choose if use pretrain embedding')
    args = parser.parse_args()
    print('setting:')
    print(args)
    print()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # read training , dev and test file
    print('loading three datasets...')
    train = Corpus(config.train_file)
    dev = Corpus(config.dev_file)
    test = Corpus(config.test_file)

    # collect all words, characters and labels in trainning data
    vocab = Vocab(train, min_freq=1)

    # choose if use pretrained word embedding
    if args.pre_emb and config.embedding_file != None:
        print('loading pretrained embedding...')
        pre_embedding = vocab.read_embedding(config.embedding_file)
    print('Chars : %d，BiChars : %d，labels : %d' %
          (vocab.num_chars, vocab.num_bichars, vocab.num_labels))
    torch.save(vocab, config.vocab_file)

    # process training data , change string to index
    print('processing datasets...')
    train_data = TextDataset(process_data(vocab, train, max_len=20))
    dev_data = TextDataset(process_data(vocab, dev, max_len=20))
    test_data = TextDataset(process_data(vocab, test, max_len=20))

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn
    )

    # create neural network
    model = CHAR_LSTM_CRF(vocab.num_chars,
                          config.char_dim,
                          config.char_hidden,
                          vocab.num_bichars,
                          config.word_dim,
                          config.layers,
                          config.word_hidden,
                          vocab.num_labels,
                          config.dropout)
    if args.pre_emb:
        model.load_pretrained_embedding(pre_embedding)
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    # init evaluator
    evaluator = Evaluator(vocab)
    # init trainer
    trainer = Trainer(model, config)
    # start to train
    trainer.train((train_loader, dev_loader, test_loader), evaluator)
