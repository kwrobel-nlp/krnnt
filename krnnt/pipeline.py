# -*- coding: utf-8 -*-
import math
import pickle
import re
import sys
from typing import List, Iterable, Generator, Union

from krnnt.analyzers import MacaAnalyzer
from krnnt.structure import Paragraph

from .keras_models import ExperimentParameters, KerasModel
from krnnt.utils import uniq
from .new import k_hot, UniqueFeaturesValues, Lemmatisation, Lemmatisation2
from krnnt.features import create_token_features

sys.setrecursionlimit(10000)

from keras.preprocessing import sequence
import numpy as np
import krnnt_utils


class KRNNTSingle:
    def __init__(self, pref):
        self.pref = pref
        self.unique_features_dict = pickle.load(open(pref['UniqueFeaturesValues'], 'rb'))
        self.km = KerasThread.create_model(pref, testing=True)
        self.lemmatisation = pref['lemmatisation_class']()
        self.lemmatisation.load(pref['lemmatisation_path'])

        self.configure()

    def tag_sentence(self, sentence: str, preana=False):
        return self.__tag([sentence], preana)

    def tag_sentences(self, sentences: List[str], preana=False):
        return self.__tag(sentences, preana)

    def tag_sentences_preana(self, sentences: List[Paragraph]):
        return self.__tag(sentences, preana=True)

    def tag_paragraphs(self, paragraphs: Iterable[str], preana=False):
        return self.__tag_paragraphs(paragraphs, preana)

    def __tag_paragraphs(self, paragraphs: Iterable[str], preana):


        if preana:
            sequences = Preprocess.process_batch_preana(enumerate(paragraphs))
        else:
            sequences = Preprocess.process_batch(paragraphs, self.pref['maca_config'], self.pref['toki_config_path'])

        # batch_size=math.ceil(len_sequences/max(math.floor(len_sequences/self.pref['keras_batch_size']), 1)) # dynamic batch

        result = []
        for batch in chunk(sequences, self.pref['keras_batch_size']):
            pad_batch = self.pad(batch, self.unique_features_dict, 'tags4e3')
            preds = self.km.model.predict_on_batch(pad_batch)
            for plain in KerasThread.return_results(batch, preds, self.km.classes, self.lemmatisation):
                result.append(plain)


        # podziel na paragrafy
        result2=[]
        result_paragraph=[]
        for sentence in result:
            if not result_paragraph or sentence[0]['document_id']==result_paragraph[-1][0]['document_id']:
                result_paragraph+= (sentence, )
            else:
                result2+=(result_paragraph,)
                result_paragraph=[sentence]

        if result_paragraph:
            result2 += (result_paragraph,)

        return result2

    def configure(self):
        if 'krnnt_utils' in sys.modules:
            self.pad = krnnt_utils.pad
        else:
            self.pad = Preprocess.pad

    def __tag(self, sentences: List[str], preana: bool):
        if preana:
            sequences = Preprocess.process_batch_preana(enumerate(sentences))
        else:
            sequences = Preprocess.process_batch(sentences, self.pref['maca_config'], self.pref['toki_config_path'])

        # batch_size=math.ceil(len_sequences/max(math.floor(len_sequences/self.pref['keras_batch_size']), 1)) # dynamic batch

        result = []
        for batch in chunk(sequences, self.pref['keras_batch_size']):
            pad_batch = self.pad(batch, self.unique_features_dict, 'tags4e3')
            preds = self.km.model.predict_on_batch(pad_batch)
            for plain in KerasThread.return_results(batch, preds, self.km.classes, self.lemmatisation):
                result.append(plain)

        return result


class Sample:
    def __init__(self):
        self.features = {}


class Preprocess:
    @staticmethod
    def create_features(sequence: List[Sample]):
        for sample in sequence:
            sample.features['tags4e3'] = create_token_features(sample.features['token'], sample.features['tags'],
                                                               sample.features['space_before'])

    @staticmethod
    def process_batch(documents: Iterable[str], maca_config: str, toki_config_path: str) -> Generator[
        List[Sample], None, None]:
        maca_analyzer = MacaAnalyzer(maca_config, toki_config_path)

        for document_id, document in enumerate(documents):
            results = maca_analyzer._maca(document)

            for res in results:
                result = maca_analyzer._parse(res)

                sequence = []
                for form, space_before, interpretations, start, end in result:
                    sample = Sample()
                    sequence.append(sample)
                    sample.features['token'] = form
                    sample.features['tags'] = uniq([t for l, t in interpretations])
                    interpretations = [(re.sub(r':[abcdijnopqsv]\d?$', '', l), t) for l, t in
                                       interpretations]
                    sample.features['maca_lemmas'] = [(l.replace('_', ' '), t) for l, t in uniq(interpretations)]

                    # TODO: cleanup space before
                    sample.features['space_before'] = ['space_before'] if space_before !='none' else [
                        'no_space_before']
                    sample.features['space_before'].append(space_before)
                    sample.features['start'] = start
                    sample.features['end'] = end
                    sample.features['document_id'] = document_id
                Preprocess.create_features(sequence)

                if sequence:
                    yield sequence

    @staticmethod
    def process_batch_preana(batch: Iterable[Paragraph]) -> Generator[List[Sample], None, None]:
        for document_id, paragraph in batch:
            for sentence in paragraph:
                sequence = []
                for token in sentence:
                    sample = Sample()
                    sequence.append(sample)
                    sample.features['token'] = token.form
                    sample.features['tags'] = uniq([form.tags for form in token.interpretations])
                    sample.features['maca_lemmas'] = uniq([(form.lemma, form.tags) for form in token.interpretations])
                    sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']
                    sample.features['space_before'].append(token.space_before)
                    sample.features['document_id'] = document_id
                Preprocess.create_features(sequence)

                if sequence:
                    yield sequence

    @staticmethod
    def pad(batch: List[List[Sample]], unique_features_dict, feature_name: str):
        if not batch:
            return []

        result_batchX = []
        for sentence in batch:
            X_sentence = []
            for sample in sentence:
                X_sentence.append(np.array(k_hot(sample.features[feature_name], unique_features_dict[feature_name])))

            result_batchX.append(X_sentence)

        return sequence.pad_sequences(result_batchX)


def chunk(l: Iterable, batch_size: int) -> List:
    batch = []
    for element in l:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class KerasThread():

    @staticmethod
    def create_model(pref, testing=False) -> KerasModel:
        keras_model_class = pref['keras_model_class']

        parameters = ExperimentParameters(pref, testing)

        km = keras_model_class(parameters)

        if 'UniqueFeaturesValues' in pref:
            km.unique_features_dict = pickle.load(open(pref['UniqueFeaturesValues'], 'rb'))
        else:
            # data_path = 'nkjp_paragraphs_shuffled_concraft.spickle_FormatData_PreprocessData'
            data_path = pref['data_path']
            km.unique_features_dict = UniqueFeaturesValues(data_path).get()

        unique_tags_dict = km.unique_features_dict[pref['label_name']]
        km.classes = list(map(lambda k: k[0], sorted(unique_tags_dict.items(), key=lambda k: k[1])))
        pref = km.parameters.pref
        pref['features_length'] = len(km.unique_features_dict[pref['feature_name']])
        pref['output_length'] = len(km.unique_features_dict[pref['label_name']])

        km.create_model()
        # self.km.load_weights('weight_7471898792961270266.hdf5')
        # km.load_weights('weight_7471898792961270266.hdf5')
        # km.load_weights('../artykul/compare/train_on_all.weights')
        km.load_weights(pref['weight_path'])
        km.compile()

        return km

    @staticmethod
    def return_results(sentences: List[List[Sample]], preds, classes: List[str],
                       lemmatisation: Union[Lemmatisation, Lemmatisation2]):
        for sentence, preds2 in zip(sentences, preds):  # TODO sentences
            # print(preds2.shape)
            # print(preds2)

            response = []

            preds3 = preds2.argmax(axis=-1)
            preds3max = preds2.max(axis=-1)
            # print(len(sentence), len(preds3))
            first = True
            for sample, max_index, prob in zip(sentence, list(preds3)[-len(sentence):],
                                               list(preds3max)[-len(sentence):]):
                # print(sample.features, max_index)
                # max_index, max_value = max(enumerate(d), key=lambda x: x[1])

                token_response = {}
                response.append(token_response)
                predicted_tag = classes[max_index]

                # TODO
                if sample.features['space_before'] == ['space_before']:
                    sep = 'space'
                else:
                    sep = 'none'

                if 'newline' in sample.features['space_before'] or 'newlines' in sample.features['space_before']:
                    sep = 'newline'
                elif 'space' in sample.features['space_before'] or 'spaces' in sample.features['space_before']:
                    sep = 'space'
                elif 'none' in sample.features['space_before']:
                    sep = 'none'

                # print(sample.features['token']+'\t'+sep)
                # response.append(sample.features['token']+'\t'+sep)
                token_response['token'] = sample.features['token']
                token_response['sep'] = sep
                token_response['prob'] = float(prob)
                token_response['document_id'] = sample.features['document_id']

                lemmas = [x for x in sample.features['maca_lemmas']]
                token_response['tag'] = predicted_tag
                token_response['lemmas'] = []
                try:
                    token_response['start'] = sample.features['start']
                    token_response['end'] = sample.features['end']
                except KeyError:
                    token_response['start'] = None
                    token_response['end'] = None

                # if not lemmas:
                #    lemmas.append((sample.features['token'], predicted_tag))
                lemma = lemmatisation.disambiguate(token_response['token'], lemmas, predicted_tag)

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

                first = False
            # print()
            # response.append('')

            yield response
