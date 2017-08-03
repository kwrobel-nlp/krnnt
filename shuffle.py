#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from optparse import OptionParser

usage = """%prog CORPUS SAVE_PATH

Shuffle training data.

E.g. %prog train-merged.spickle train-merged.shuf.spickle
"""


import random


from progress.bar import Bar


from krnnt.classes import SerialUnpickler, SerialPickler


if __name__=='__main__':
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 2:
        print('Provide paths to corpus and to save path.')
        sys.exit(1)

    file_path1 = args[0]
    file_path2 = args[1]




    file = open(file_path1,'rb')
    su = SerialUnpickler(file)



    paragraphs = []
    bar = Bar('Loading', suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=18484)
    for paragraph in su:
        bar.next()
        paragraphs.append(paragraph)
    file.close()

    random.seed(1337)
    random.shuffle(paragraphs)


    file2 = open(file_path2, 'wb')
    sp = SerialPickler(file2)

    bar = Bar('Saving', suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=18484)
    for paragraph in paragraphs:
        sp.add(paragraph)
        bar.next()
    file2.close()