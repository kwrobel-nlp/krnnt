#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from argparse import ArgumentParser

from tqdm import tqdm

from krnnt.classes import SerialUnpickler, SerialPickler, Paragraph

usage = """%prog CORPUS SAVE_PATH

Shuffle training data.

E.g. %prog train-merged.spickle train-merged.shuf.spickle
"""

if __name__ == '__main__':
    parser = ArgumentParser(usage=usage)
    parser.add_argument('file_path', type=str, help='paths to corpus')
    parser.add_argument('output_path', type=str, help='save path')
    parser.add_argument('--seed', '-s', type=int, default=1337, help='seed')
    args = parser.parse_args()

    file_path1 = args.file_path
    file_path2 = args.output_path

    file = open(file_path1, 'rb')
    su = SerialUnpickler(file)

    paragraphs = []
    paragraph: Paragraph
    for paragraph in tqdm(su, desc='Loading', total=18484):
        paragraphs.append(paragraph)
    file.close()

    random.seed(args.seed)
    random.shuffle(paragraphs)

    file2 = open(file_path2, 'wb')
    sp = SerialPickler(file2)

    for paragraph in tqdm(paragraphs, desc='Saving'):
        sp.add(paragraph)

    file2.close()
