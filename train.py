#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from argparse import ArgumentParser

from keras.models import load_model

from krnnt.keras_models import BEST, ExperimentParameters
from krnnt.new import UnalignedSimpleEvaluator
from krnnt.tagger_exps import RunFolds2, KerasData, RunExperiment, KerasData2, RunExperiment2

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_path', help='path to preprocessed data')
    parser.add_argument('features_dict', help='path to features dict')

    parser.add_argument('-p', '--preanalyzed', action='store_false',
                        default=True, dest='reanalyzed',
                        help='training data have not been reanalyzed')
    parser.add_argument('-c', '--cv', action='store_true',
                        default=False, dest='cv',
                        help='run 10-fold cross-validation')
    parser.add_argument('-t', '--train_ratio',
                        default=1.0, dest='train_ratio', type=float,
                        help='percentage of data for training')
    parser.add_argument('-d', '--dev_ratio',
                        default=0.0, dest='dev_ratio', type=float,
                        help='percentage of training data for development')
    parser.add_argument('--dev_data', default='0.0', help='dev data ratio or path to dev data')
    parser.add_argument('--test_data', default='0.0', help='test data ratio or path to test data')
    parser.add_argument('--load_model', default=None, help='path to pretrained model')
    parser.add_argument('-e', '--epochs',
                        default=100, dest='epochs', type=int,
                        help='number of epochs')
    parser.add_argument('--patience',
                        default=10, dest='patience', type=int,
                        help='patience')
    parser.add_argument('--maca_config',
                        default='morfeusz2-nkjp',
                        help='Maca config')
    parser.add_argument('--tensor_board',
                        action='store_true',
                        help='save data for TensorBoard')
    parser.add_argument('-g', '--debug', action='store_true', dest='debug_mode')  # TODO
    parser.add_argument('--hash', action='store', default=None, dest='hash')
    parser.add_argument('--reproducible', action='store_true', default=False, help='set seeds')
    parser.add_argument('-f', '--fold', default=None, dest='fold')
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

    pref = {'nb_epoch': 100, 'batch_size': 256,
            'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'evaluator': UnalignedSimpleEvaluator, 'patience': 10,
            'weight_path': 'weights.hdf5', 'samples_per_epoch': 10000, 'keras_model_class': BEST,
            'corpus_path': 'data/train-reanalyzed.spickle', 'reanalyze': True, 'train_data_ratio': 0.9,
            'dev_data_ratio': 0.1}

    pref['reanalyze'] = args.reanalyzed
    pref['train_data_ratio'] = float(args.train_ratio)
    pref['dev_data_ratio'] = float(args.dev_ratio)

    pref['tensor_board']= args.tensor_board
    pref['nb_epoch'] = args.epochs

    pref['dev_data'] = args.dev_data
    if pref['dev_data']=='0.0':
        pref['patience'] = pref['nb_epoch']
    pref['test_data'] = args.test_data
    pref['load_model'] = args.load_model


    # pref['corpus_path'] = args.corpus_path
    pref['patience'] = args.patience
    pref['maca_config'] = args.maca_config
    if args.hash is not None:
        pref['h'] = args.hash
    if args.fold is not None:
        pref['fold'] = int(args.fold)

    keras_model_class = pref['keras_model_class']

    if args.cv:
        logging.error('CV is not supported')
        # rf = RunFolds2(keras_model_class, pref)
        # rf.run()
    else:
        parameters = ExperimentParameters(pref)

        km = keras_model_class(parameters)




        print('Model will be saved under: %s.final' % parameters.pref['weight_path'])

        kd = KerasData2(args.data_path, args.features_dict, parameters)
        re = RunExperiment2(kd, km)
        re.run()

        print('Model is saved under: %s' % parameters.pref['weight_path'])

