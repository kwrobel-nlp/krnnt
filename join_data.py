#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from tqdm import tqdm

from krnnt.serial_pickle import SerialPickler, SerialUnpickler

if __name__ == '__main__':
    parser = ArgumentParser(description='Join data')
    parser.add_argument('output_path', help='output path to data')
    parser.add_argument('input_paths', nargs='+', help='input paths to data')

    args = parser.parse_args()

    sp = SerialPickler(open(args.output_path, 'wb'))
    for input_path in args.input_paths:
        su = SerialUnpickler(open(input_path, 'rb'))
        for paragraph in tqdm(su):
            sp.add(paragraph)
    sp.close()
