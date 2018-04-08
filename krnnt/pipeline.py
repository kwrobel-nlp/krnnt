# -*- coding: utf-8 -*-
import pickle
import sys

from .keras_models import ExperimentParameters
from .classes import uniq
from .new import FeaturePreprocessor, TagsPreprocessor, k_hot, UniqueFeaturesValues, Lemmatisation

sys.setrecursionlimit(10000)
import threading
import logging
import multiprocessing
mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.DEBUG)
from timeit import default_timer as timer
import csv
from subprocess import Popen, PIPE
import setproctitle
from keras.preprocessing import sequence
import numpy as np


from keras import backend as K
#K.set_learning_phase(False)

#from krnnt import keras_models


class KRNNTSingle:
    def __init__(self, pref):
        self.pref=pref
        self.unique_features_dict = pickle.load(open(pref['UniqueFeaturesValues'],'rb'))
        self.km = KerasThread.create_model(pref, testing=True)
        self.lemmatisation = Lemmatisation()
        self.lemmatisation.load(pref['lemmatisation_path'])

    def tag_sentence(self, sentence, preana=False):
        return self.__tag([sentence], preana)

    def tag_sentences(self, sentences, preana=False):
        return self.__tag(sentences, preana)


    def __tag(self, sentences, preana):
        if preana:
            sequences = Preprocess.process_batch_preana([(i,s) for i,s in enumerate(sentences)])
        else:
            sequences = Preprocess.process_batch([(i,s) for i,s in enumerate(sentences)])
        result = []
        for batch in chunk(sequences, self.pref['keras_batch_size']):
            pad_batch=Preprocess.pad(batch, self.unique_features_dict, 'tags4e3')
            preds = self.km.model.predict_on_batch(pad_batch)
            for plain in KerasThread.return_results(batch, preds, self.km.classes, self.lemmatisation):
                #print(plain)
                result.append(plain)
        return result


class LogTime(multiprocessing.Process):
    def __init__(self, queue_log):
        super(LogTime, self).__init__()
        self.queue_log = queue_log
        self.f=open('log2cores2workersWsort.csv','w', newline='')
        self.csv_writer = csv.writer(self.f)

    def run(self):
        while True:
            item = self.queue_log.get()
            if item is None:
                self.queue_log.task_done()
                break
            self.csv_writer.writerow(item)
            self.f.flush()
            self.queue_log.task_done()

class StdInThread(threading.Thread):
    def __init__(self, queue, queue_log=None):
        super(StdInThread, self).__init__()
        self.queue=queue
        self.queue_log = queue_log

    def log(self, desc):
        if self.queue_log:
            # print(self.name, timer(), desc, file=sys.stderr)
            self.queue_log.put([self.name, timer(), desc])

    def run(self):
        self.log('START')
        ss = []
        for i,line in enumerate(sys.stdin):
            # self.queue.put(line.strip())
            ss.append((i,line.strip()))


        #ss = sorted(ss, key=lambda sentence: sentence[1].count(' '))
        for s in ss:
            self.queue.put(s)

        self.queue.put(1)
        self.log('STOP')

class BatcherThread(threading.Thread):
    def __init__(self, input_queue, output_queue, batch_size, number_of_consumers, queue_log=None):
        super(BatcherThread, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        self.number_of_consumers=number_of_consumers
        self.queue_log = queue_log

    def log(self, desc):
        if self.queue_log:
            # print(self.name, timer(), desc)
            self.queue_log.put([self.name, timer(), desc])

    def run(self):
        # print(self.name, 'RUN', self.input_queue.qsize())
        self.log('START')
        batch = []
        while True:
            self.log('WORKING')
            item = self.input_queue.get()
            self.log('WAIT')
            if isinstance( item, int ):
                if batch:
                    pass
                    self.output_queue.put(batch)
                if item>1:
                    self.input_queue.put(item-1)
                else:
                    pass
                    self.output_queue.put(self.number_of_consumers)

                self.input_queue.task_done()
                break

            batch.append(item)

            if len(batch) == self.batch_size:
                self.log('PUT0')
                self.output_queue.put(batch)
                self.log('PUT1')
                batch = []
            self.input_queue.task_done()


        self.log('STOP')
        # print('batcher stop')

class Sample:
    def __init__(self):
        self.features = {}

class Preprocess:
    @staticmethod
    def maca(batch):
        p = Popen(['maca-analyse', '-c', 'morfeusz-nkjp-official', '-l'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        stdout = p.communicate(input='\n'.join(batch).encode('utf-8'))[0]
        return  [i for i in stdout.decode('utf-8').split('\n\n') if len(i) > 0]

    @staticmethod
    def parse(output):
        data = []
        lemma_lines = []
        token_line = None
        for line in output.split("\n"):
            if line.startswith("\t"):
                lemma_lines.append(line)
            else:
                if token_line is not None:
                    data.append((token_line, lemma_lines))
                    lemma_lines = []
                token_line = line
        data.append((token_line, lemma_lines))

        tokens = []
        for index, (token_line, lemma_lines) in enumerate(data):
            token = Preprocess.construct(token_line, lemma_lines) #80%
            if token is None: continue
            tokens.append(token)

        return tokens

    @staticmethod
    def create_features(sequence):
        #TODO długo trwa
        for sample in sequence:
            f = []

            f.extend(FeaturePreprocessor.cases(sample.features['token']))
            f.extend(FeaturePreprocessor.interps(sample.features['token'],sample.features))
            f.extend(FeaturePreprocessor.qubliki(sample.features['token']))
            f.extend(FeaturePreprocessor.shape(sample.features['token'])) #90%
            f.extend(FeaturePreprocessor.prefix1(sample.features['token']))
            f.extend(FeaturePreprocessor.prefix2(sample.features['token']))
            f.extend(FeaturePreprocessor.prefix3(sample.features['token']))
            f.extend(FeaturePreprocessor.suffix1(sample.features['token']))
            f.extend(FeaturePreprocessor.suffix2(sample.features['token']))
            f.extend(FeaturePreprocessor.suffix3(sample.features['token']))
            f.extend(TagsPreprocessor.create_tags4_without_guesser(sample.features['tags'])) #3% moze cache dla wszystkich tagów
            f.extend(TagsPreprocessor.create_tags5_without_guesser(sample.features['tags'])) #3%
            f.extend(sample.features['space_before'])

            sample.features['tags4e3'] = uniq(f)

            # print()
            # print(sample.features['token'])
            # print(sample.features['tags'])
            # print(FeaturePreprocessor.cases(sample.features['token']))
            # print(FeaturePreprocessor.interps(sample.features['token'],sample.features))
            # print(FeaturePreprocessor.qubliki(sample.features['token']))
            # print(FeaturePreprocessor.shape(sample.features['token']))
            # print(FeaturePreprocessor.prefix1(sample.features['token']))
            # print(FeaturePreprocessor.prefix2(sample.features['token']))
            # print(FeaturePreprocessor.prefix3(sample.features['token']))
            # print(FeaturePreprocessor.suffix1(sample.features['token']))
            # print(FeaturePreprocessor.suffix2(sample.features['token']))
            # print(FeaturePreprocessor.suffix3(sample.features['token']))
            # print(TagsPreprocessor.create_tags4_without_guesser(sample.features['tags']))
            # print(TagsPreprocessor.create_tags5_without_guesser(sample.features['tags']))
            # print(sample.features['space_before'])

    @staticmethod
    def construct(token_line, lemma_lines):
        try:
            if token_line == '': return None
            form, separator_before = token_line.split("\t")
        except ValueError:
            raise Exception('Probably concraft-pl not working.') #TODO what?

        form = form
        space_before = separator_before
        interpretations = []

        for lemma_line in lemma_lines:
            try:
                lemma, tags, _ = lemma_line.strip().split("\t") #30%
                disamb = True
            except ValueError:
                lemma, tags = lemma_line.strip().split("\t") #16%
                disamb = False
            lemma = (lemma, tags)
            # lemma.disamb=disamb
            interpretations.append(lemma)

        return (form, space_before, interpretations)

    @staticmethod
    def process_batch(batch):
        #indexes=[]
        batchC = []
        for index, line in batch:
            #indexes.append(i)
            batchC.append(line)



        results = Preprocess.maca(batchC)
        #self.log('MACA')
        # print('MACA', len(results))
        sequences=[]
        for res in  results:
            result = Preprocess.parse(res)

            # TODO cechy
            sequence = []
            for form, space_before, interpretations in result:
                sample = Sample()
                sequence.append(sample)
                sample.features['token'] = form
                # print(interpretations)
                sample.features['tags'] = uniq([t for l,t in  interpretations])
                sample.features['maca_lemmas'] = interpretations
                sample.features['space_before'] = ['space_before'] if space_before=='space' else ['no_space_before']

            Preprocess.create_features(sequence)

            if sequence:
                sequences.append(sequence)
        return sequences

    @staticmethod
    def process_batch_preana(batch):
        sequences=[]
        for index, paragraph in batch:
            for sentence in paragraph:
                sequence=[]
                for token in sentence:
                    sample = Sample()
                    sequence.append(sample)
                    sample.features['token'] = token.form
                    # print(interpretations)
                    sample.features['tags'] = uniq([form.tags for form in  token.interpretations])
                    sample.features['maca_lemmas'] = uniq([(form.lemma, form.tags) for form in  token.interpretations])
                    sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']

                Preprocess.create_features(sequence)

                if sequence:
                    sequences.append(sequence)
        return sequences

    @staticmethod
    def pad(batch, unique_features_dict, feature_name):
        if not batch:
            return []

        #feature_name='tags4e3'
        result_batchX = []
        # print('batch len',len(batch))
        for sentence in batch:
            X_sentence = []
            # y_sentence = []
            for sample in sentence:
                #print(feature_name, sample.features[feature_name])
                X_sentence.append(np.array(k_hot(sample.features[feature_name], unique_features_dict[feature_name])))

            result_batchX.append(X_sentence)


        # max_sentence_length = max([len(x) for x in result_batchX])
        #self.log('KHOT')
        return sequence.pad_sequences(result_batchX) #, sequence.pad_sequences(result_batchY, maxlen=max_sentence_length)


def chunk(l, batch_size):
    if not l:
        return

    n=max(len(l)//batch_size,1)
    # print('n', n)
    k, m = divmod(len(l), n)
    for i in range(n):
        batch = l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        yield batch


class WorkerThread(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, maca_batch_size, keras_batch_size, number_of_consumers, queue_log=None, pref=None):
        # super(WorkerThread, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.number_of_consumers=number_of_consumers
        self.queue_log = queue_log
        self.maca_batch_size=maca_batch_size
        self.keras_batch_size=keras_batch_size
        self.pref=pref
        super(WorkerThread, self).__init__()

    def log(self, desc):
        if self.queue_log:
            # print(self.name, timer(), desc)
            self.queue_log.put([self.name, timer(), desc])





    def run(self):
        setproctitle.setproctitle(self.name)
        # print(self.name, 'WORKER0')

        if 'UniqueFeaturesValues' in self.pref:
            self.unique_features_dict = pickle.load(open(self.pref['UniqueFeaturesValues'],'rb'))
        else:
            #data_path = 'nkjp_paragraphs_shuffled_concraft.spickle_FormatData_PreprocessData'
            data_path = self.pref['data_path']
            self.unique_features_dict = UniqueFeaturesValues(data_path).get()
        #data_path = 'train-merged.spickle_FormatData2_PreprocessData'

        # print(self.name, 'WORKER1') #TODO problem


        # print(self.name, 'WORKER')
        self.log('START')

        while True:
            self.log('WORKING')
            batch = self.input_queue.get()
            self.log('WAIT')

            if isinstance( batch, int ):
                if batch>1:
                    self.input_queue.put(batch-1)
                else:
                    self.output_queue.put(self.number_of_consumers)

                self.input_queue.task_done()
                break

            # print('MACA0')

            if self.pref['reanalyze']==True:
                sequences=Preprocess.process_batch(batch)
            else:
                sequences=Preprocess.process_batch_preana(batch)
            self.log('PARSER_F')


            # print('jgfjhg', len(indexes), len(sequences))

            #TODO dziel na x czesci

            for batch in chunk(sequences, self.keras_batch_size):
            #create batches
            # while sequences:
            #     batch = sequences[:self.keras_batch_size]
                self.log('CHUNK')
                pad_batch=Preprocess.pad(batch, self.unique_features_dict, 'tags4e3')
                self.log('PAD')
                # print(self.name, 'put batch', len(pad_batch))
                self.output_queue.put((pad_batch, batch))
                self.log('PUT')
                # sequences=sequences[self.keras_batch_size:]

            self.input_queue.task_done()


        self.log('STOP')




class KerasThread(threading.Thread):
    def __init__(self, input_queue, output_queue, number_of_consumers,queue_log=None, pref=None):
        """

        :param input_queue:
        :param output_queue:
        :return:
        """
        super(KerasThread, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.number_of_consumers=number_of_consumers
        self.queue_log = queue_log
        self.name='Keras'+self.name
        self.pref=pref

    @staticmethod
    def create_model(pref, testing=False):
        keras_model_class = pref['keras_model_class']



        parameters = ExperimentParameters(pref, testing)


        km = keras_model_class(parameters)

        if 'UniqueFeaturesValues' in pref:
            km.unique_features_dict = pickle.load(open(pref['UniqueFeaturesValues'],'rb'))
        else:
            #data_path = 'nkjp_paragraphs_shuffled_concraft.spickle_FormatData_PreprocessData'
            data_path = pref['data_path']
            km.unique_features_dict = UniqueFeaturesValues(data_path).get()

        unique_tags_dict = km.unique_features_dict[pref['label_name']]
        km.classes = list(map(lambda k: k[0], sorted(unique_tags_dict.items(), key=lambda k: k[1])))
        pref = km.parameters.pref
        pref['features_length'] = len(km.unique_features_dict[pref['feature_name']])
        pref['output_length'] = len(km.unique_features_dict[pref['label_name']])


        km.create_model()
        #self.km.load_weights('weight_7471898792961270266.hdf5')
        #km.load_weights('weight_7471898792961270266.hdf5')
        #km.load_weights('../artykul/compare/train_on_all.weights')
        km.load_weights(pref['weight_path'])
        km.compile()

        return km

    @staticmethod
    def return_results(sentences, preds, classes, lemmatisation):
        for sentence,preds2 in zip(sentences, preds): #TODO sentences
            # print(preds2.shape)
            # print(preds2)

            response=[]

            preds3 = preds2.argmax(axis=-1)
            preds3max = preds2.max(axis=-1)
            # print(len(sentence), len(preds3))
            first=True
            for sample,max_index, prob in zip(sentence,list(preds3)[-len(sentence):], list(preds3max)[-len(sentence):]):
                # print(sample, max_index)
                # max_index, max_value = max(enumerate(d), key=lambda x: x[1])

                token_response = {}
                response.append(token_response)
                predicted_tag=classes[max_index]

                if sample.features['space_before']==['space_before']:
                    if first:
                        sep='newline'
                    else:
                        sep = 'space'
                else:
                    sep='none'


                #print(sample.features['token']+'\t'+sep)
                #response.append(sample.features['token']+'\t'+sep)
                token_response['token']=sample.features['token']
                token_response['sep']=sep
                token_response['prob']=prob

                lemmas = [ x for x in sample.features['maca_lemmas'] if x[1]==predicted_tag]
                token_response['tag']=predicted_tag
                token_response['lemmas']=[]


                if not lemmas:
                    lemmas.append((sample.features['token'], predicted_tag))
                lemma = lemmatisation.disambiguate(token_response['token'], lemmas)

                token_response['lemmas'].append(lemma)

                # if lemmas:
                #     for l, t in lemmas:
                #         #print('\t'+l+'\t'+t+'\tdisamb')
                #         #response.append('\t'+l+'\t'+t+'\tdisamb')
                #         token_response['lemmas'].append(l)
                # else:
                #     #print('\t'+sample.features['token']+'\t'+predicted_tag+'\tdisamb')
                #     #response.append('\t'+sample.features['token']+'\t'+predicted_tag+'\tdisamb')
                #     token_response['lemmas'].append(sample.features['token'])


                first=False
            #print()
            #response.append('')

            yield response

    @staticmethod
    def return_plain(sentences, preds, classes, lemmatisation):
        for sentence,preds2 in zip(sentences, preds): #TODO sentences
            # print(preds2.shape)
            # print(preds2)

            response=[]

            preds3 = preds2.argmax(axis=-1)
            # print(len(sentence), len(preds3))
            first=True
            for sample,max_index in zip(sentence,list(preds3)[-len(sentence):]):
                # print(sample, max_index)
                # max_index, max_value = max(enumerate(d), key=lambda x: x[1])
                predicted_tag=classes[max_index]

                if sample.features['space_before']==['space_before']:
                    if first:
                        sep='newline'
                    else:
                        sep = 'space'
                else:
                    sep='none'


                #print(sample.features['token']+'\t'+sep)
                response.append(sample.features['token']+'\t'+sep)

                lemmas = [ x for x in sample.features['maca_lemmas'] if x[1]==predicted_tag]

                if not lemmas:
                    lemmas.append((sample.features['token'], predicted_tag))
                lemma = lemmatisation.disambiguate(sample.features['token'], lemmas)

                response.append('\t'+lemma+'\t'+predicted_tag+'\tdisamb')

                # if lemmas:
                #     for l, t in lemmas:
                #         #print('\t'+l+'\t'+t+'\tdisamb')
                #         response.append('\t'+l+'\t'+t+'\tdisamb')
                # else:
                #     #print('\t'+sample.features['token']+'\t'+predicted_tag+'\tdisamb')
                #     response.append('\t'+sample.features['token']+'\t'+predicted_tag+'\tdisamb')


                first=False
            #print()
            response.append('')

            yield '\n'.join(response)

    def run(self):
        if self.queue_log is not None: self.queue_log.put([self.name, timer(), 'START'])
        # if self.name=='Thread-2':
        #     import theano.sandbox.cuda
        #     theano.sandbox.cuda.use('gpu')
        # self.km = KerasModel()


        #pref = {'data_size':int(10), 'train_data_size': int(16635*0.9), 'dev_data_size':int(16635*0.1), 'test_data_size': int(1849), 'nb_epoch': 1, 'batch_size': 32,
        #'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
        #'evaluator': UnalignedSimpleEvaluator,
        #'weight_path': 'weight_836cd228-603a-11e7-aaf4-a0000220fe80.hdf5', 'samples_per_epoch': 128, 'keras_model_class': keras_models.BEST,'UniqueFeaturesValues':'train-merged.spickle_FormatData2_PreprocessData_UniqueFeaturesValues'}

        # pref['features_length']=123
        # pref['output_length']=132


        self.km = KerasThread.create_model(self.pref, testing=True)
        lemmatisation=Lemmatisation()
        lemmatisation.load(self.pref['lemmatisation_path'])

        if self.queue_log is not None: self.queue_log.put([self.name, timer(), 'START2'])
        # setproctitle.setproctitle('KERAS '+self.name)
        start_timer = timer()
        # print('start compiling model')

        # print('compiled')

        while True:
            # print('keras', self.name,  self.input_queue.qsize())
            if self.queue_log is not None: self.queue_log.put([self.name, timer(), 'WORKING'])
            item = self.input_queue.get()
            if self.queue_log is not None: self.queue_log.put([self.name, timer(), 'WAIT'])
            if isinstance( item, int ):
                # print('INT', item)
                if item>1:
                    self.input_queue.put(item-1)
                elif item==1:
                    pass
                    #self.output_queue.put(self.number_of_consumers)

                # print('parser koneic')
                self.input_queue.task_done()
                break
            item,sentences=item
            # print(len(item))
            #batch = self.km.prepare(item)
            batch=item
            # print('keras', self.name, 'prepared')
            if self.queue_log is not None: self.queue_log.put([self.name, timer(), 'GPU0'])
            preds = self.km.model.predict_on_batch(batch)
            if self.queue_log is not None: self.queue_log.put([self.name, timer(), 'GPU1'])
            # print('keras', self.name, 'predicted')
            # self.input_queue.task_done()
            #print(preds)

            for plain in KerasThread.return_plain(sentences, preds, self.km.classes, lemmatisation):
                print(plain)


            # self.output_queue.put((item,preds))
            # self.queue_log.put([self.name, timer(), 'PUT1'])
            self.input_queue.task_done()

        end_timer = timer()
        print('KerasThread', (end_timer - start_timer), file=sys.stderr)
        if self.queue_log is not None: self.queue_log.put([self.name, timer(), 'STOP'])


