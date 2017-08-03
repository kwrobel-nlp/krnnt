#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import sys
from optparse import OptionParser

from krnnt.keras_models import BEST
from krnnt.new import results_to_plain, results_to_xces, read_xces
from krnnt.pipeline import KRNNTSingle

usage = """%prog MODEL LEMMATISATION_MODEL DICTIONARY < CORPUS_PATH



E.g. %prog
"""

if __name__ == '__main__':
    parser = OptionParser(usage=usage)
    parser.add_option('-p', '--preanalyzed', action='store_false',
                      default=True, dest='reanalyzed',
                      help='training data have not been reanalyzed')
    # parser.add_option('-i', '--input-format', action='store',
    #                   default='xces', dest='input_format',
    #                   help='input format of preanalyzed data: xces')
    parser.add_option('-o', '--output-format', action='store',
                      default='xces', dest='output_format',
                      help='output format: xces, plain')
    parser.add_option('-g', '--debug', action='store_true', dest='debug_mode')  # TODO
    (options, args) = parser.parse_args()

    pref = {'keras_batch_size': 32, 'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'keras_model_class': BEST}

    if len(args) != 3:
        print('Provide paths to corpus and to save path.')
        sys.exit(1)

    pref['reanalyze'] = options.reanalyzed
    # pref['input_format'] = options.input_format
    pref['output_format'] = options.output_format

    pref['weight_path'] = args[0]
    pref['lemmatisation_path'] = args[1]
    pref['UniqueFeaturesValues'] = args[2]

    krnnt = KRNNTSingle(pref)

    if options.reanalyzed:
        results = krnnt.tag_sentences(sys.stdin.read().split('\n\n')) # ['Ala ma kota.', 'Ale nie ma psa.']
    else:
        #f = io.StringIO(sys.stdin.read())
        corpus = read_xces(sys.stdin)
        results = krnnt.tag_sentences(corpus, preana=True)

    #print(results)

    if options.output_format == 'xces':
        conversion = results_to_xces
    elif options.output_format == 'plain':
        conversion = results_to_plain
    else:
        print('Wrong output format.')
        sys.exit(1)

    conversion(results)
