#!/usr/bin/env python
# -*- coding: utf-8 -*-

from krnnt.classes import SerialPickler
import sys
from optparse import OptionParser

from krnnt.new import read_xces

usage = """%prog CORPUS SAVE_PATH

Converts XCES corpus to internal KRNNT representation and saves it to file.

E.g. %prog train-analyzed.xml train-analyzed.spickle
"""

if __name__=='__main__':
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    if len(args) != 2:
        print('Provide path to XCES corpus and to save path.')
        sys.exit(1)

    file_path = args[0]
    output_path = args[1]

    file = open(output_path, 'wb')
    sp = SerialPickler(file)
    for paragraph in read_xces(file_path):
        sp.add(paragraph)
    file.close()