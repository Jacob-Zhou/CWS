# -*- coding: utf-8 -*-


class Corpus(object):

    def __init__(self, filename=None):
        self.filename = filename
        self.sentence_num = 0
        self.char_num = 0
        self.bichar_num = 0
        self.char_seqs = []
        self.bichar_seqs = []
        self.label_seqs = []
        chars = []
        bichars = []
        sequence = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    self.char_seqs.append(chars)
                    self.bichar_seqs.append(bichars)
                    self.label_seqs.append(sequence)
                    self.sentence_num += 1
                    chars = []
                    bichars = []
                    sequence = []
                else:
                    conll = line.split()
                    chars.append(conll[0])
                    bichars.append(conll[1])
                    sequence.append(conll[2])
                    self.char_num += 1
        print('%s : sentences:%d，chars:%d，bichars:%d' %
              (filename, self.sentence_num, self.char_num, self.bichar_num))
