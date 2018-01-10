#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from optparse import OptionParser

from krnnt.keras_models import BEST, ExperimentParameters
from krnnt.new import UnalignedSimpleEvaluator
from krnnt.tagger_exps import RunFolds2, KerasData, RunExperiment

usage = """%prog CORPUS_PATH



E.g. %prog
"""

if __name__ == '__main__':
    parser = OptionParser(usage=usage)
    parser.add_option('-p', '--preanalyzed', action='store_false',
                      default=True, dest='reanalyzed',
                      help='training data have not been reanalyzed')
    parser.add_option('-c', '--cv', action='store_true',
                      default=False, dest='cv',
                      help='run 10-fold cross-validation')
    parser.add_option('-t', '--train_ratio', action='store',
                      default=1.0, dest='train_ratio', type="float",
                      help='percentage of data for training')
    parser.add_option('-d', '--dev_ratio', action='store',
                      default=0.0, dest='dev_ratio', type="float",
                      help='percentage of training data for development')
    parser.add_option('-e', '--epochs', action='store',
                      default=100, dest='epochs', type="int",
                      help='number of epochs')
    parser.add_option('--patience', action='store',
                      default=10, dest='patience', type="int",
                      help='patience')
    parser.add_option('-g', '--debug', action='store_true', dest='debug_mode') # TODO
    parser.add_option('--hash', action='store', default=None, dest='hash')
    parser.add_option('-f', '--fold', action='store', default=None, dest='fold')
    (options, args) = parser.parse_args()

    if len(args) != 1:
        print('Provide paths to corpus and to save path.')
        sys.exit(1)

    pref = {'nb_epoch': 100, 'batch_size': 256,
            'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'evaluator': UnalignedSimpleEvaluator, 'patience': 10,
            'weight_path': 'weights.hdf5', 'samples_per_epoch': 10000, 'keras_model_class': BEST,
            'corpus_path': 'data/train-reanalyzed.spickle', 'reanalyze': True, 'train_data_ratio':0.9, 'dev_data_ratio':0.1}

    pref['reanalyze'] = options.reanalyzed
    pref['train_data_ratio']=float(options.train_ratio)
    pref['dev_data_ratio']=float(options.dev_ratio)
    pref['nb_epoch']=int(options.epochs)
    pref['corpus_path'] = args[0]
    pref['patience'] = options.patience
    if options.hash is not None:
        pref['h'] = options.hash
    if options.fold is not None:
        pref['fold'] = int(options.fold)

    keras_model_class = pref['keras_model_class']

    if options.cv:
        rf = RunFolds2(keras_model_class, pref)
        rf.run()
    else:
        parameters = ExperimentParameters(pref)
        km = keras_model_class(parameters)

        print('Model will be saved under: %s.final' % parameters.pref['weight_path'])
        print('Lemmatisation model will be saved under: %s' % parameters.pref['lemmatisation_path'])

        kd = KerasData(pref['corpus_path'], pref['reanalyze'])
        re = RunExperiment(kd, km)
        re.run()


        print('Model is saved under: %s' % parameters.pref['weight_path'])
        print('Lemmatisation model is saved under: %s' % parameters.pref['lemmatisation_path'])
        if pref['reanalyze']:
            print('Dictionary is saved under: %s' % parameters.pref['corpus_path']+'_FormatData2_PreprocessData_UniqueFeaturesValues')
        else:
            print('Dictionary is saved under: %s' % parameters.pref['corpus_path']+'_FormatDataPreAnalyzed_PreprocessData_UniqueFeaturesValues')

