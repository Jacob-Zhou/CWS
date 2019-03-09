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

# 戴 戴#START# b-beg (bmes)
def conll_2_line(conll_file):
    with open(conll_file, mode='r', encoding='utf-8') as f:
        words = []
        for (i, line) in enumerate(f):
            line = line.strip()
            if line == '':
                if len(words) > 0:
                    print(' '.join(words))
                    words = []
                continue
            tokens = re.split('\s+', line)
            ws_tag = tokens[2][0].lower()
            if ws_tag == 'b' or ws_tag == 's':
                words.append(tokens[0])
            else:
                words[-1] = words[-1] + tokens[0]
            if i % 10000 == 0:
                print(i // 10000, end=' ', file=sys.stderr)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--conll-file', default='ctb5-dev.conll')
    args, extra_args = argparser.parse_known_args()
    conll_2_line(args.conll_file)





