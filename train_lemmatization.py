#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from krnnt.keras_models import ExperimentParameters
from krnnt.tagger_exps import KerasData2, RunLemma

if __name__ == '__main__':
    parser = ArgumentParser(description='Train lemmatization')
    parser.add_argument('data_path', help='path to preprocessed data')


    parser.add_argument('-t', '--train_ratio',
                        default=1.0, dest='train_ratio', type=float,
                        help='percentage of data for training')
    parser.add_argument('-d', '--dev_ratio',
                        default=0.0, dest='dev_ratio', type=float,
                        help='percentage of training data for development')
    parser.add_argument('--dev_data', default='0.1', help='dev data ratio or path to dev data')
    parser.add_argument('--test_data', default='0.1', help='test data ratio or path to test data')
    parser.add_argument('-g', '--debug', action='store_true', dest='debug_mode')  # TODO
    parser.add_argument('--hash', action='store', default=None, dest='hash')
    parser.add_argument('--reproducible', action='store_true', default=False, help='set seeds')

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

    pref = {
        'train_data_ratio': float(args.train_ratio),
        'dev_data_ratio': float(args.dev_ratio),
        'dev_data': args.dev_data,
        'test_data': args.test_data
    }

    if args.hash is not None:
        pref['h'] = args.hash


    parameters = ExperimentParameters(pref)

    kd = KerasData2(args.data_path, None, parameters)
    re = RunLemma(kd)
    re.learn_lemma()

    print('Lemmatisation model is saved under: %s' % parameters.pref['lemmatisation_path'])

    #TODO CV, usunac zaleznosc od TF, KerasData2 bez słownika