import argparse
from collections import Counter
import re
import sys

'''
Zhenghua 2019.1.18
input:

dict:
    one word a line: length word frequency \t bpes
'''


def load_dict(dict_file_name, max_word_len, max_entry_num):
    dicts = [dict() for i in range(max_word_len + 1)]
    with open(dict_file_name, mode='r', encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            if i > max_entry_num > 0:
                break
            tokens = re.split('\t', line.strip())
            assert len(tokens) == 2, 'format error: ' % line
            w_info = re.split('\s+', tokens[0].strip())
            w_len = int(w_info[0])
            if w_len == 1:
                continue
            if w_len > max_word_len:
                break
            word = w_info[1]
            dicts[w_len][word] = [0]
            bpes = re.split('@@\s+', tokens[1].strip())
            posi = 0
            for bpe in bpes:
                posi += len(bpe)
                dicts[w_len][word].append(posi)
            #print(word, bpes, dicts[w_len][word])
            if i % 10000 == 0:
                print(i//10000, end=' ', file=sys.stderr)

    for i in range(len(dicts)-1, 0, -1):
        if len(dicts[i]) == 0:
            del dicts[i]
        else:
            break
    return dicts


def segment_word_seq(word_seq_file, dicts):
    with open(word_seq_file, mode='r', encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            tokens = re.split('\s+', line.strip())
            for token in tokens:
                w_len = len(token)
                bpes = dicts[w_len].get(token) if w_len < len(dicts) else None
                if bpes is None:
                    for ch in token:
                        print(ch, end=' ')
                else:
                    for (j, posi) in enumerate(bpes[1:]):
                        print(token[bpes[j]:posi], end=' ')
            print()
            if i % 10000 == 0:
                print(i//10000, end=' ', file=sys.stderr)


def one_word_to_bpes(token, dicts):
    w_len = len(token)
    bpes_idx = []
    bpes = dicts[w_len].get(token) if w_len < len(dicts) else None
    if bpes is None:
        for i in range(w_len):
            bpes_idx.append((i, i+1))
    else:
        for i in range(len(bpes)-1):
            bpes_idx.append((bpes[i], bpes[i+1]))
    return bpes_idx


def segment_word_seq(word_seq_file, dicts):
    with open(word_seq_file, mode='r', encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            tokens = re.split('\s+', line.strip())
            for token in tokens:
                bpes_idx = one_word_to_bpes(token, dicts)
                for (j, k) in bpes_idx:
                    print(token[j:k], end=' ')
            print()
            if i % 10000 == 0:
                print(i//10000, end=' ', file=sys.stderr)


def segment_chars_seq(chars_seq_file, dicts):
    max_word_len = len(dicts) - 1
    with open(chars_seq_file, mode='r', encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            line = line.strip()
            l_len = len(line)
            cntr = Counter()
            for b_posi in range(0, l_len):
                for w_len in range(1, min(max_word_len+1, l_len-b_posi)):
                    w = line[b_posi:b_posi+w_len]
                    bpes_idx = one_word_to_bpes(w, dicts)
                    for (j, k) in bpes_idx:
                        if k-j > 1:
                            cntr[(b_posi+j, w[j:k])] += 1
            for k in sorted(cntr.keys(), key=lambda x: x[0]):
                print('%d_%s_%d' % (k[0], k[1], cntr[k]), end=' ')
            print()
            if i % 10000 == 0:
                print(i//10000, end=' ', file=sys.stderr)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--dict_file', default='Giga-coarse-fine.dict-with-bpe.txt')
    argparser.add_argument('--max_entry_num', default=-10000, type=int)
    argparser.add_argument('--max_word_len', default=100, type=int)
    argparser.add_argument(
        '--word_seq_file', default='Giga.CTB5.len-less-than-150.hwc.coarse-and-fine-100')
    argparser.add_argument(
        '--chars_seq_file', default='Giga.CTB5.len-less-than-150.hwc.coarse-and-fine-100-chars')  # not segmented

    args, extra_args = argparser.parse_known_args()
    dicts = load_dict(args.dict_file, args.max_word_len, args.max_entry_num)

    if args.word_seq_file:
        segment_word_seq(args.word_seq_file, dicts)

    if args.chars_seq_file:
        segment_chars_seq(args.chars_seq_file, dicts)
