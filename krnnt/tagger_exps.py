#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys

import collections

from .new import FormatData, FormatData2, FormatDataPreAnalyzed, FeaturePreprocessor, TagsPreprocessor, \
    JoinProcessingRule, PreprocessData, UniqueFeaturesValues, batch_generator, generate_arrays_from_file, Xy_generator, \
    pad_generator, to_plain, LossHistory, DataList, DataGenerator, Lemmatisation
from .classes import count_samples

sys.setrecursionlimit(10000)
import logging
from krnnt import keras_models
import uuid


import keras


class KerasData():
    def __init__(self, corpus_path, reanalyze=True):
        self.unique_features_dict = None
        self.corpus_path=corpus_path
        self.reanalyze=reanalyze

    def load_data(self):
        if self.unique_features_dict is not None:
            return

        if self.reanalyze=='old':
            data_path = FormatData(self.corpus_path).load()
        elif self.reanalyze:
            data_path = FormatData2(self.corpus_path).load()
        else:
            data_path = FormatDataPreAnalyzed(self.corpus_path).load()

        from krnnt.new import ProcessingRule
        operations = [ProcessingRule(FeaturePreprocessor.nic, 'token', 'nic'),
                    ProcessingRule(FeaturePreprocessor.cases, 'token', 'cases'),
                      ProcessingRule(FeaturePreprocessor.interps, 'token', 'interps'),
                      ProcessingRule(FeaturePreprocessor.qubliki, 'token', 'qubs'),
                      ProcessingRule(FeaturePreprocessor.shape, 'token', 'shape'),
                      ProcessingRule(FeaturePreprocessor.prefix1, 'token', 'prefix1'),
                      ProcessingRule(FeaturePreprocessor.prefix2, 'token', 'prefix2'),
                      ProcessingRule(FeaturePreprocessor.prefix3, 'token', 'prefix3'),
                      ProcessingRule(FeaturePreprocessor.suffix1, 'token', 'suffix1'),
                      ProcessingRule(FeaturePreprocessor.suffix2, 'token', 'suffix2'),
                      ProcessingRule(FeaturePreprocessor.suffix3, 'token', 'suffix3'),
                      # ProcessingRule(TagsPreprocessor().create_tags2, 'tags', 'tags2'),
                      ProcessingRule(TagsPreprocessor.create_tags4_without_guesser, 'tags', 'tags4'),
                      ProcessingRule(TagsPreprocessor.create_tags5_without_guesser, 'tags', 'tags5'),
                      # ProcessingRule(TagsPreprocessor().create_tag4, 'label', 'y4'),
                      # ProcessingRule(TagsPreprocessor().create_tag2, 'label', 'y'),
                      # HistoryRule('tags2', 'tags2h+1', 1, '-1-'),
                      # HistoryRule('tags2', 'tags2h-1', -1, '+1-'),
                      # JoinProcessingRule(['tags2', 'tags2h-1', 'tags2h+1'], 'tags2x'),
                      # HistoryRule('tags4', 'tags4h+1', 1, '-1-'),
                      # HistoryRule('tags4', 'tags4h-1', -1, '+1-'),
                      # JoinProcessingRule(['tags4', 'tags4h-1', 'tags4h+1'], 'tags4x'),
                      #JoinProcessingRule(
                      #    ['prefix1', 'prefix2','prefix3', 'suffix1', 'suffix2', 'suffix3'], 'presuffix'),
                      # JoinProcessingRule(['tags4', 'tags5'], 'tags4tags5'),
                      # JoinProcessingRule(['tags4', 'shape'], 'tags4shape'),
                      # JoinProcessingRule(['tags4', 'cases'], 'tags4cases'),
                      # JoinProcessingRule(['tags4', 'interps'], 'tags4interps'),
                      # JoinProcessingRule(['tags4', 'qubs'], 'tags4qubs'),
                      #JoinProcessingRule(['tags4', 'presuffix'], 'tags4presuffix'),
                      #JoinProcessingRule(['tags4', 'presuffix','tags'], 'tags4presuffixtags'),
                      #JoinProcessingRule(['tags4', 'presuffix','tags5'], 'tags4presuffixtags5'),
                      #JoinProcessingRule(['tags4', 'presuffix','shape'], 'tags4presuffixshape'),
                      #JoinProcessingRule(['tags4', 'presuffix','cases'], 'tags4presuffixcases'),
                      # JoinProcessingRule(['tags4', 'presuffix','interps'], 'tags4presuffixinterps'),
                      # JoinProcessingRule(['tags4', 'presuffix','qubs'], 'tags4presuffixqubs'),
                      # JoinProcessingRule(['tags4', 'tags'], 'tags4tags'),
                      JoinProcessingRule(
                          ['tags4', 'tags5', 'cases', 'space_before', 'interps', 'qubs', 'shape', 'prefix1', 'prefix2',
                           'prefix3', 'suffix1', 'suffix2', 'suffix3'], 'tags4e3')]
        self.operations=operations


        data_path = PreprocessData(data_path, operations).load()



        unique_features_dict = UniqueFeaturesValues(data_path).get()

        self.data_path = data_path
        self.unique_features_dict = unique_features_dict


        logging.info('KerasData loaded')



class Train():
    def __init__(self, keras_data, keras_model):
        """

        @type keras_data: KerasData
        @type keras_model: keras_models.KerasModel
        """
        self.keras_data=keras_data
        self.keras_model=keras_model






    def train(self):
        pref = self.keras_model.parameters.pref




        logging.info('Training...')


        # self.train_data = pad_generator(
        #     batch_generator(
        #         generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
        #                                   keep_infinity=False, keep_unaligned=True),
        #         batch_size=pref['batch_size']))



        try:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
                batch_generator(
                    generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name']),
                    batch_size=pref['batch_size']))),
                steps_per_epoch=int(pref['samples_per_epoch']/pref['batch_size']), nb_epoch=pref['nb_epoch'])
        except TypeError:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
                batch_generator(
                    generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name']),
                    batch_size=pref['batch_size']))),
                samples_per_epoch=int(pref['samples_per_epoch']), nb_epoch=pref['nb_epoch'])

    def propagate_data_info(self):
        pref = self.keras_model.parameters.pref
        pref['features_length'] = len(self.keras_data.unique_features_dict[pref['feature_name']])
        pref['output_length'] = len(self.keras_data.unique_features_dict[pref['label_name']])

        pref['operations']=str(self.keras_data.operations)

    def print_parameters(self):
        print(self.keras_model.parameters.pref)
        print(self.keras_model.yaml_model())

    def run(self):
        self.keras_data.load_data()
        self.propagate_data_info()

        self.keras_model.create_model()
        self.keras_model.compile()

        self.print_parameters()
        self.train()
        self.keras_model.model.save_weights('train_all.weights')



class RunExperiment():
    def __init__(self, keras_data, keras_model):
        """

        @type keras_data: KerasData
        @type keras_model: keras_models.KerasModel
        """
        self.keras_data=keras_data
        self.keras_model=keras_model

    def learn_lemma(self):
        l = Lemmatisation()
        print('Training lemmatisation...')
        l.learn(self.keras_data.data_path)
        l.save(self.keras_model.parameters.pref['lemmatisation_path'])
        print('Training lemmatisation - done')

    def save_plain_data(self):
        logging.info('Saving training, dev, test data')
        #TODO predictions too in plain
        #txt, plain preds and ref
        #train
        pref = self.keras_model.parameters.pref

        self.train_data = pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          stop=pref['train_data_size'], keep_infinity=False, keep_unaligned=True),
                batch_size=pref['batch_size']))

        unique_tags_dict=self.keras_data.unique_features_dict[self.keras_model.parameters.pref['label_name']]
        for data,name in [(self.train_data,'train'),(self.test_data,'test'),(self.dev_data,'dev')]:
            logging.info(name)
            file_test=open(pref['h']+'_'+name+'.ref.plain','wt')
            file_pred=open(pref['h']+'_'+name+'.pred.plain','wt')
            #TODO txt

            for batch in data:
                X_train2, y_train2, sentences2,sentences_orig2 = batch
                logging.info(name+' Predicting %s' % len(X_train2))
                preds =  self.keras_model.model.predict_on_batch(X_train2)
                logging.info(name+' Predicted')
                to_plain(batch,preds, file_test, file_pred,unique_tags_dict)
                logging.info(name+' DONE: to_plain')
                #test
                #dev
        logging.info('Saved training, dev, test data')


    def create_data(self):
        pref = self.keras_model.parameters.pref



        test_data = []
        test_data2 = []
        if pref['test_data_size']>0:
            print('test_data2')
            for x in pad_generator(batch_generator(
                    generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                              start=pref['train_data_size']+pref['dev_data_size'],
                                              stop=pref['train_data_size'] + pref['dev_data_size']+pref['test_data_size'], keep_infinity=False),
                    batch_size=pref['test_data_size'], return_all=True)):
                test_data.extend(x)
                break

            test_data2 = [x for x in Xy_generator([test_data])][0]

            print('test_data')
            test_data = []
            for x in pad_generator(batch_generator(
                    generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                              start=pref['train_data_size']+pref['dev_data_size'],
                                              stop=pref['train_data_size'] + pref['dev_data_size']+pref['test_data_size'], keep_infinity=False, keep_unaligned=True),
                    batch_size=pref['batch_size'])):
                # print(len(x))
                test_data.append(x)

        # print(len(test_data), len(test_data[0]), len(test_data[0][0]))

        # print(len(test_data2), len(test_data2[0]), len(test_data2[0][0]))

        dev_data = []
        if pref['dev_data_size']>0:
            print('dev_data')
            for x in pad_generator(batch_generator(
                    generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                              start=pref['train_data_size'],
                                              stop=pref['train_data_size'] + pref['dev_data_size'], keep_infinity=False, keep_unaligned=True),
                    batch_size=pref['batch_size'])):
                dev_data.append(x)

        #self.test_data=test_data
        #self.test_data2=test_data2
        #self.dev_data=dev_data

        print('train_data')
        train_data = DataGenerator(self.keras_data.data_path, self.keras_data.unique_features_dict, pref, range(pref['train_data_size']))
        self.train_data=train_data

        self.test_data=DataList(test_data)
        self.test_data2=test_data2
        self.dev_data=DataList(dev_data)

        logging.info('Data created')

    def train(self):
        pref = self.keras_model.parameters.pref

        loss_history = LossHistory(pref['evaluator'](self.test_data, self.keras_data.unique_features_dict[pref['label_name']]),
                                   'log_' + str(pref['h']) + '.log')
        loss_history.write_str(pref)
        loss_history.write_str(self.keras_model.yaml_model)

        dev_loss_history = LossHistory(pref['evaluator'](self.dev_data, self.keras_data.unique_features_dict[pref['label_name']]),
                                   'devlog_' + str(pref['h']) + '.log', name='dev_')

        callbacks = [dev_loss_history, loss_history,
                     keras.callbacks.ModelCheckpoint(pref['weight_path'], save_best_only=True, monitor='dev_val_score'),
                     keras.callbacks.EarlyStopping(monitor='dev_val_score', patience=pref['patience'], mode='max'),
                     keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)]  # , save_weights_only=True
        logging.info('Training...')

        # self.train_data = pad_generator(
        #     batch_generator(
        #         generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
        #                                   stop=pref['train_data_size'], keep_infinity=False, keep_unaligned=True),
        #         batch_size=pref['batch_size']))

        try:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          stop=pref['train_data_size']),
                batch_size=pref['batch_size']))),
            steps_per_epoch=int(pref['samples_per_epoch']/pref['batch_size']), nb_epoch=pref['nb_epoch'],
            validation_data=self.test_data2, callbacks=callbacks)
        except TypeError:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          stop=pref['train_data_size']),
                batch_size=pref['batch_size']))),
            samples_per_epoch=int(pref['samples_per_epoch']), nb_epoch=pref['nb_epoch'],
            validation_data=self.test_data2, callbacks=callbacks)

    def propagate_data_info(self):
        pref = self.keras_model.parameters.pref
        pref['features_length'] = len(self.keras_data.unique_features_dict[pref['feature_name']])
        pref['output_length'] = len(self.keras_data.unique_features_dict[pref['label_name']])

        pref['operations']=str(self.keras_data.operations)

        #count data
        pref['data_size']=count_samples(self.keras_data.data_path)
        ts=int(pref['data_size']*pref['train_data_ratio'])
        pref['train_data_size']=int(ts*(1-pref['dev_data_ratio']))
        pref['dev_data_size']=int(ts*pref['dev_data_ratio'])-5
        pref['test_data_size']=max(0, pref['data_size']-pref['train_data_size']-pref['dev_data_size'])

    def print_parameters(self):
        print(self.keras_model.parameters.pref)
        print(self.keras_model.yaml_model())

    def run(self):
        print('keras_data.load_data')
        self.keras_data.load_data()
        print('propagate_data_info')
        self.propagate_data_info()
        print('learn_lemma')
        self.learn_lemma()
        print('keras_model.create_model')
        self.keras_model.create_model()
        print('keras_model.compile')
        self.keras_model.compile()
        print('create_data')
        self.create_data()
        print('print_parameters')
        self.print_parameters()
        print('train')
        self.train()

        self.keras_model.model.save_weights(self.keras_model.parameters.pref['weight_path']+'.final')
        #self.save_plain_data()

    def run_test(self):
        self.keras_data.load_data()
        self.propagate_data_info()

        self.keras_model.create_model()
        self.keras_model.load_weights(self.keras_model.parameters.pref['weight_path'])
        self.keras_model.compile()

        self.create_data()

        self.print_parameters()
        pref=self.keras_model.parameters.pref
        evaluator = pref['evaluator'](self.test_data, self.keras_data.unique_features_dict[pref['label_name']])
        results = evaluator.evaluate(self.keras_model.model)
        print(results)

class RunExperimentFold(RunExperiment):
    def __init__(self, keras_data, keras_model, train_ids, test_ids):
        """

        @type keras_data: KerasData
        @type keras_model: keras_models.KerasModel
        """
        self.test_ids=test_ids
        self.train_ids=train_ids

        super().__init__(keras_data, keras_model)

    def create_data(self):
        logging.info('Data creating')
        pref = self.keras_model.parameters.pref


        test_data = []
        for x in pad_generator(batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.test_ids, keep_infinity=False),
                batch_size=pref['test_data_size'], return_all=True)):
            test_data.extend(x)
            break


        test_data2 = [x for x in Xy_generator([test_data])][0]
        logging.info('Test data 2 created: %s' % len(test_data2[0]))

        test_data = []
        for x in pad_generator(batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.test_ids, keep_infinity=False, keep_unaligned=True),
                batch_size=pref['batch_size'])):
            test_data.append(x)
        logging.info('Test data created: %s' % len(test_data))

        dev_data = []
        for x in pad_generator(batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.train_ids[int(len(self.train_ids)*0.9):], keep_infinity=False, keep_unaligned=True),
                batch_size=pref['batch_size'])):
            dev_data.append(x)

        self.test_data=test_data
        self.test_data2=test_data2
        self.dev_data=dev_data

        logging.info('Data created. Test: %s Dev: %s' % (len(test_data), len(dev_data)))

    def train(self):
        pref = self.keras_model.parameters.pref

        loss_history = LossHistory(pref['evaluator'](self.test_data, self.keras_data.unique_features_dict[pref['label_name']]),
                                   'log_' + str(pref['h']) + '.log')
        self.loss_history = loss_history
        loss_history.write_str(pref)
        loss_history.write_str(self.keras_model.yaml_model)

        dev_loss_history = LossHistory(pref['evaluator'](self.dev_data, self.keras_data.unique_features_dict[pref['label_name']]),
                                   'devlog_' + str(pref['h']) + '.log', name='dev_')

        callbacks = [dev_loss_history, loss_history,
                     keras.callbacks.ModelCheckpoint(pref['weight_path'], save_best_only=True, monitor='dev_val_score'),
                     keras.callbacks.EarlyStopping(monitor='dev_val_score', patience=pref['patience'], mode='max')]  # , save_weights_only=True
        logging.info('Training...')

        self.train_data = pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.train_ids[:int(len(self.train_ids)*0.9)]),
                batch_size=pref['batch_size']))

        try:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.train_ids[:int(len(self.train_ids)*0.9)]),
                batch_size=pref['batch_size']))),
            steps_per_epoch=int(pref['samples_per_epoch']/pref['batch_size']), nb_epoch=pref['nb_epoch'],
            validation_data=self.test_data2, callbacks=callbacks)
        except TypeError:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.train_ids[:int(len(self.train_ids)*0.9)]),
                batch_size=pref['batch_size']))),
            samples_per_epoch=int(pref['samples_per_epoch']), nb_epoch=pref['nb_epoch'],
            validation_data=self.test_data2, callbacks=callbacks)

class RunExperimentFold2(RunExperiment):
    def __init__(self, keras_data, keras_model, train_ids, test_ids):
        """

        @type keras_data: KerasData
        @type keras_model: keras_models.KerasModel
        """
        self.test_ids=test_ids
        self.train_ids=train_ids

        super().__init__(keras_data, keras_model)

    def create_data(self):
        logging.info('Data creating')
        pref = self.keras_model.parameters.pref

        print('test_data2')
        test_data = []
        for x in pad_generator(batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.test_ids, keep_infinity=False),
                batch_size=len(self.test_ids), return_all=True)):
            test_data.extend(x)
            break


        test_data2 = [x for x in Xy_generator([test_data])][0]
        logging.info('Test data 2 created: %s' % len(test_data2[0]))

        print('test_data')
        test_data = []
        for x in pad_generator(batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.test_ids, keep_infinity=False, keep_unaligned=True),
                batch_size=pref['batch_size'])):
            test_data.append(x)
        logging.info('Test data created: %s' % len(test_data))

        dev_data = []
        print('dev_data')
        print(pref)
        if pref['dev_data_ratio']>0.0:
            print('dev_data go')
            for x in pad_generator(batch_generator(
                    generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                              ids=self.train_ids[int(len(self.train_ids)*(1-pref['dev_data_ratio'])):], keep_infinity=False, keep_unaligned=True),
                    batch_size=len(self.train_ids[int(len(self.train_ids)*(1-pref['dev_data_ratio'])):]))):
                dev_data.append(x)

        print('train_data')
        train_data = DataGenerator(self.keras_data.data_path, self.keras_data.unique_features_dict, pref, self.train_ids[:int(len(self.train_ids)*(1-pref['dev_data_ratio']))])
        self.train_data=train_data

        self.test_data=DataList(test_data)
        self.test_data2=test_data2
        self.dev_data=DataList(dev_data)

        logging.info('Data created. Test: %s Dev: %s' % (len(test_data), len(dev_data)))

    def train(self):
        pref = self.keras_model.parameters.pref

        loss_history = LossHistory(pref['evaluator'](self.test_data, self.keras_data.unique_features_dict[pref['label_name']]),
                                   'log_' + str(pref['h']) + '.log')
        self.loss_history = loss_history
        loss_history.write_str(pref)
        loss_history.write_str(self.keras_model.yaml_model)

        dev_loss_history = LossHistory(pref['evaluator'](self.dev_data, self.keras_data.unique_features_dict[pref['label_name']]),
                                   'devlog_' + str(pref['h']) + '.log', name='dev_')

        #train_loss_history = LossHistory(pref['evaluator'](self.train_data, self.keras_data.unique_features_dict[pref['label_name']], tagset_utils.tags),
        #                           'trainlog_' + str(pref['h']) + '.log', name='train_')

        callbacks = [ dev_loss_history, loss_history, #train_loss_history,
                     keras.callbacks.ModelCheckpoint(pref['weight_path'], save_best_only=True, monitor='dev_val_score'),
                     keras.callbacks.EarlyStopping(monitor='dev_val_score', patience=pref['patience'], mode='max')]  # , save_weights_only=True
        logging.info('Training...')

        #self.train_data = pad_generator(
        #    batch_generator(
        #        generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
        #                                  ids=self.train_ids[:int(len(self.train_ids)*0.9)]),
        #        batch_size=pref['batch_size']))

        try:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.train_ids[:int(len(self.train_ids)*0.9)]),
                batch_size=pref['batch_size']))),
            steps_per_epoch=int(pref['samples_per_epoch']/pref['batch_size']), nb_epoch=pref['nb_epoch'],
            validation_data=self.test_data2, callbacks=callbacks)
        except TypeError:
            self.keras_model.model.fit_generator(Xy_generator(pad_generator(
            batch_generator(
                generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
                                          ids=self.train_ids[:int(len(self.train_ids)*0.9)]),
                batch_size=pref['batch_size']))),
            samples_per_epoch=int(pref['samples_per_epoch']), nb_epoch=pref['nb_epoch'],
            validation_data=self.test_data2, callbacks=callbacks)

    def save_plain_data(self):
        logging.info('Saving training, dev, test data')
        #TODO predictions too in plain
        #txt, plain preds and ref
        #train
        pref = self.keras_model.parameters.pref

        # self.train_data = pad_generator(
        #     batch_generator(
        #         generate_arrays_from_file(self.keras_data.data_path, self.keras_data.unique_features_dict, pref['feature_name'], pref['label_name'],
        #                                   stop=pref['train_data_size'], keep_infinity=False, keep_unaligned=True),
        #         batch_size=pref['batch_size']))

        unique_tags_dict=self.keras_data.unique_features_dict[self.keras_model.parameters.pref['label_name']]
        for data,name in [(self.train_data,'train'),(self.test_data,'test'),(self.dev_data,'dev')]:
            logging.info(name)
            file_test=open(pref['h']+'_'+name+'.ref.plain','wt')
            file_pred=open(pref['h']+'_'+name+'.pred.plain','wt')
            #TODO txt

            for batch in data.get():
                X_train2, y_train2, sentences2,sentences_orig2 = batch
                logging.info(name+' Predicting %s' % len(X_train2))
                preds =  self.keras_model.model.predict_on_batch(X_train2)
                logging.info(name+' Predicted')
                to_plain(batch,preds, file_test, file_pred,unique_tags_dict)
                logging.info(name+' DONE: to_plain')
                #test
                #dev
        logging.info('Saved training, dev, test data')

class RunFolds():
    def __init__(self, keras_model_class, preferences, fold=None):
        self.keras_model_class=keras_model_class
        self.preferences=preferences
        self.fold=fold



    def run(self):
        import numpy as np
        from sklearn.cross_validation import KFold


        kd = KerasData()

        folds_scores = []
        folds_scores2 = collections.defaultdict(list)

        kf = KFold(self.preferences['data_size'], n_folds=10)

        for i, (train, test) in enumerate(kf):
            if self.fold is not None and self.fold!=i: continue
            logging.info('Fold %s' % i)
            parameters = keras_models.ExperimentParameters(self.preferences)
            parameters.pref['fold']=i
            km = self.keras_model_class(parameters)
            re = RunExperimentFold(kd, km, train, test)

            re.run()

            folds_scores.append(re.loss_history.history[-1]['val_score'])

            for name in re.loss_history.history[-1].keys():
                folds_scores2[name].append(re.loss_history.history[-1][name])

        print(folds_scores)
        print(np.mean(folds_scores))

        for name,sc in sorted(folds_scores2.items()):
            print(name, np.mean(sc))


class RunFolds2():
    def __init__(self, keras_model_class, preferences):
        self.keras_model_class=keras_model_class
        self.preferences=preferences



    def run(self):
        import numpy as np
        from sklearn.cross_validation import KFold

        pref=self.preferences
        kd = KerasData(pref['corpus_path'], pref['reanalyze'])

        folds_scores = []
        folds_scores2 = collections.defaultdict(list)

        pref['data_size']=count_samples(pref['corpus_path'])

        kf = KFold(self.preferences['data_size'], n_folds=10)

        fmain=open('mainlog'+str(uuid.uuid1()),'wt')

        for i, (train, test) in enumerate(kf):
            print(i)
            if 'fold' in pref and i!=pref['fold']: continue
            logging.info('Fold %s' % i)
            parameters = keras_models.ExperimentParameters(self.preferences)
            fmain.write(parameters.h)
            fmain.write('\n')
            fmain.flush()
            parameters.pref['fold']=i
            km = self.keras_model_class(parameters)

            #if self.preferences['train_data_size'] is not None: train = train[:self.preferences['train_data_size']]

            re = RunExperimentFold2(kd, km, train, test)

            re.run()

            folds_scores.append(re.loss_history.history[-1]['val_score'])

            for name in re.loss_history.history[-1].keys():
                folds_scores2[name].append(re.loss_history.history[-1][name])

        print(folds_scores)
        print(np.mean(folds_scores))

        for name,sc in sorted(folds_scores2.items()):
            print(name, np.mean(sc))
