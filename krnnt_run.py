#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys

from argparse import ArgumentParser

from krnnt.aglt import remove_aglt_from_results_rule1_3
from krnnt.keras_models import BEST
from krnnt.new import Lemmatisation, Lemmatisation2, get_morfeusz, analyze_tokenized
from krnnt.pipeline import KRNNTSingle, chunk
from krnnt.readers import read_xces, read_jsonl
from krnnt.writers import results_to_jsonl_str, results_to_conll_str, results_to_conllu_str, \
    results_to_xces_str, results_to_plain_str

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run tagger')
    parser.add_argument('weight_path', help='path to weights, lemmatisation data and dictionary')
    parser.add_argument('lemmatisation_data', help='path to lemmatisation data')
    parser.add_argument('dictionary', help='path to dictionary')
    parser.add_argument('-p', '--preanalyzed', action='store_false',
                      default=True, dest='reanalyzed',
                      help='training data have not been reanalyzed')
    parser.add_argument('-i', '--input-format', default='xces', dest='input_format',
                      help='input format of preanalyzed data: xces, jsonl')
    parser.add_argument('-o', '--output-format',
                      default='xces', dest='output_format',
                      help='output format: xces, plain, conll, conllu, jsonl')
    parser.add_argument('--maca_config',
                      default='morfeusz2-nkjp',
                      help='Maca config')
    parser.add_argument('--toki_config_path',
                      default='',
                      help='Toki config path (directory)')
    parser.add_argument('--lemmatisation',
                      default='sgjp',
                      help='lemmatization mode (sgjp, simple)')
    parser.add_argument('-g', '--debug', action='store_true', dest='debug_mode')  # TODO
    parser.add_argument('--tokenized', action='store_true',
                      help='input data are tokenized, but not analyzed')
    parser.add_argument('--reproducible', action='store_true', default=False, help='set seeds')
    parser.add_argument('--chunk_size',
                        default=100000, type=int,
                        help='chunk size')
    parser.add_argument('--remove_aglt', action='store_true')
    args = parser.parse_args()

    if args.reproducible:
        from numpy.random import seed
        seed(1337)
        import random as rn
        rn.seed(1337)
        import tensorflow as tf
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        from keras import backend as K
        tf.set_random_seed(1337)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    pref = {'keras_batch_size': 32, 'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'keras_model_class': BEST, 'maca_config':args.maca_config, 'toki_config_path':args.toki_config_path}

    if args.lemmatisation== 'simple':
        pref['lemmatisation_class'] = Lemmatisation2
    else:
        pref['lemmatisation_class'] = Lemmatisation

    pref['reanalyze'] = args.reanalyzed
    # pref['input_format'] = options.input_format
    pref['output_format'] = args.output_format

    pref['weight_path'] = args.weight_path
    pref['lemmatisation_path'] = args.lemmatisation_data
    pref['UniqueFeaturesValues'] = args.dictionary

    krnnt = KRNNTSingle(pref)
#time python3 -m cProfile -o gpu_run_train2.profil  krnnt_run.py ../krnnt/data/weights.hdf5 ../krnnt/data/lemmatisation.pkl ../krnnt/data/dictionary.pkl -o xces  > /tmp/out.xces < ../krnnt-refactor/tests/data/full/train-raw.txt

    if args.tokenized:
        if args.input_format == 'jsonl':
            corpus = read_jsonl(sys.stdin)
        else:
            print('Wrong input format.')
            sys.exit(1)

        morf=get_morfeusz()
        corpus = analyze_tokenized(morf, corpus)
        results = krnnt.tag_paragraphs(corpus, preana=True)
    elif args.reanalyzed:
        data=sys.stdin.read().split('\n\n')
        results=[]
        for batch in chunk(data, args.chunk_size):
            results += krnnt.tag_paragraphs(batch) # ['Ala ma kota.', 'Ale nie ma psa.']
            #TODO: print here
    else:
        #f = io.StringIO(sys.stdin.read())
        if args.input_format== 'xces':
            corpus = read_xces(sys.stdin)
        elif args.input_format== 'jsonl':
            corpus = read_jsonl(sys.stdin)
        else:
            print('Wrong input format.')
            sys.exit(1)

        results = krnnt.tag_paragraphs(corpus, preana=True)

    # print(results)

    if args.output_format == 'xces':
        conversion = results_to_xces_str
    elif args.output_format == 'plain':
        conversion = results_to_plain_str
    elif args.output_format == 'conll':
        conversion = results_to_conll_str
    elif args.output_format == 'conllu':
        conversion = results_to_conllu_str
    elif args.output_format == 'jsonl':
        conversion = results_to_jsonl_str
    else:
        print('Wrong output format.')
        sys.exit(1)


    if args.remove_aglt:
        remove_aglt_from_results_rule1_3(results)

    print(conversion(results), end='')
