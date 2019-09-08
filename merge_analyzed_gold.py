#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from krnnt.serial_pickle import SerialPickler, SerialUnpickler

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Combines analyzed corpus with gold. Analyzed corpus must be with gold segmentation.')
    parser.add_argument('gold_path', help='')
    parser.add_argument('analyzed_path', help='')
    parser.add_argument('output_path', help='')
    args = parser.parse_args()

    file_path1 = args.gold_path
    file_path2 = args.analyzed_path
    output_path = args.output_path

    file1 = open(file_path1, 'rb')
    su_gold = SerialUnpickler(file1)

    file2 = open(file_path2, 'rb')
    su_analyzed = SerialUnpickler(file2)

    file3 = open(output_path, 'wb')
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
