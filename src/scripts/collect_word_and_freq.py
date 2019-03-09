import argparse
from collections import Counter
import re
import sys

'''
Zhenghua 2019.1.17
input:
    one sentence each line, words are separated by whitespaces
output:
    one word a line: length word frequency
'''


def collect(inf, counter):
    with open(inf, mode='r', encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            tokens = re.split('\s+', line.strip())
            for token in tokens:
                counter[token] += 1
            if i % 10000 == 0:
                print(i, ' ', end='')


def output_dict(outf, counter):
    words = counter.keys()
    # first according to word length
    # then according to freq
    words_sorted = sorted(words, key=lambda w: '%03d%015d%s' % (100-len(w), counter[w], w), reverse=True)
    with open(outf, mode='w', encoding='utf-8') as f:
        for k in words_sorted:
            f.write('%d %s %d\n' % (len(k), k, counter[k]))
            print(k, file=sys.stderr)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--inf', default='in.txt')
    argparser.add_argument('--out_dict_file', default='dict.txt')

    args, extra_args = argparser.parse_known_args()
    counter = Counter()
    collect(args.inf, counter)
    output_dict(args.out_dict_file, counter)




