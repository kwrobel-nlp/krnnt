#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from argparse import ArgumentParser

from krnnt.serial_pickle import SerialPickler, SerialUnpickler, count_samples

if __name__ == '__main__':
    parser = ArgumentParser(description='Split data')
    parser.add_argument('input_path', help='input path to data')
    parser.add_argument('output_path1', help='output path to data')
    parser.add_argument('output_path2', help='output path to data')
    parser.add_argument('ratio', type=float, help='ratio of data to write to the first output')

    args = parser.parse_args()

    num_data = count_samples(args.input_path)
    first_part = math.ceil(num_data * args.ratio)

    sp1 = SerialPickler(open(args.output_path1, 'wb'))
    sp2 = SerialPickler(open(args.output_path2, 'wb'))

    su = SerialUnpickler(open(args.input_path, 'rb'))
    for i, paragraph in enumerate(su):
        if i < first_part:
            sp1.add(paragraph)
        else:
            sp2.add(paragraph)
    sp1.close()
    sp2.close()
