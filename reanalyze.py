#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from tqdm import tqdm

from krnnt.aligner import align_paragraphs
from krnnt.analyzers import MacaAnalyzer
from krnnt.structure import Paragraph
from krnnt.serial_pickle import SerialPickler, SerialUnpickler

usage = """prog CORPUS_GOLD CORPUS_SAVE

Reanalyze corpus with Maca.

E.g. prog train-gold.spickle train-reanalyzed.spickle
"""

if __name__ == '__main__':
    parser = ArgumentParser(usage=usage)
    parser.add_argument('file_path', type=str, help='paths to corpus')
    parser.add_argument('output_path', type=str, help='save path')
    parser.add_argument('--maca_config', default='morfeusz2-nkjp', help='Maca config')
    parser.add_argument('--toki_config_path', default='', help='Toki config path (directory)')
    args = parser.parse_args()

    file1 = open(args.file_path, 'rb')
    su_gold = SerialUnpickler(file1)

    file2 = open(args.output_path, 'wb')
    sp = SerialPickler(file2)

    maca_analyzer = MacaAnalyzer(args.maca_config)

    paragraph_gold: Paragraph
    for j, paragraph_gold in tqdm(enumerate(su_gold), total=18484, desc='Morphological analysis'):
        paragraph_raw = paragraph_gold.text()

        paragraph_reanalyzed = maca_analyzer.analyze(paragraph_raw)

        print('Number of sentences by Maca vs gold', len(paragraph_reanalyzed.sentences), len(paragraph_gold.sentences))

        paragraph_reanalyzed = align_paragraphs(paragraph_reanalyzed, paragraph_gold)

        sp.add(paragraph_reanalyzed)

    file2.close()

    # TODO: count mismatched sentences
