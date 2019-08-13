#!/usr/bin/env python
# -*- coding: utf-8 -*-

from krnnt.structure import Paragraph, Sentence, Token, Form
from krnnt.serial_pickle import SerialPickler, SerialUnpickler
import sys
from optparse import OptionParser

usage = """%prog CORPUS_GOLD CORPUS_ANALYZED SAVE_PATH

Combines analyzed corpus with gold. Analyzed corpus must be with gold segmentation.

E.g. %prog train-gold.spickle train-analyzed.spickle train-merged.spickle
"""





if __name__=='__main__':
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 3:
        print('Provide paths to corpora and to save path.')
        sys.exit(1)

    file_path1 = args[0]
    file_path2 = args[1]
    output_path = args[2]

    file1 = open(file_path1, 'rb')
    su_gold = SerialUnpickler(file1)

    file2 = open(file_path2,'rb')
    su_analyzed = SerialUnpickler(file2)

    file3=open(output_path,'wb')
    sp = SerialPickler(file3)

    for paragraph_gold in su_gold:
        for sentence_gold in paragraph_gold:
            paragraph_analyzed = next(su_analyzed.__iter__())
            assert len(paragraph_analyzed.sentences), 1
            sentence_analyzed = paragraph_analyzed.sentences[0]
            assert len(sentence_analyzed.tokens), len(sentence_gold.tokens)
            for token_gold, token_analyzed in zip(sentence_gold, sentence_analyzed):
                token_gold.interpretations = token_analyzed.interpretations
        sp.add(paragraph_gold)


    file3.close()