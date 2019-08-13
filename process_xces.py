#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob

from krnnt.serial_pickle import SerialPickler
from argparse import ArgumentParser

from krnnt.readers import read_xces

usage = """%prog CORPUS SAVE_PATH

Converts XCES corpus to internal KRNNT representation and saves it to file.

E.g. %prog train-analyzed.xml train-analyzed.spickle
"""

if __name__ == '__main__':
    parser = ArgumentParser(usage=usage)
    parser.add_argument('file_path', type=str, help='path to XCES corpus (or path with wildcard)')
    parser.add_argument('output_path', type=str, help='save path')
    args = parser.parse_args()

    with open(args.output_path, 'wb') as file:
        sp = SerialPickler(file)

        for path in glob.iglob(args.file_path):
            print(path)
            for paragraph in read_xces(path):
                sp.add(paragraph)
