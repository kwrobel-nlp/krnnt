#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numbers
import os.path
import pickle
import random
import time
import traceback

import collections
from typing import List

import numpy as np
import regex
from keras.callbacks import Callback
from keras.preprocessing import sequence
from krnnt.features import create_token_features
from progress.bar import Bar
from tqdm import tqdm

from krnnt.utils import uniq
from krnnt.serial_pickle import SerialPickler, SerialUnpickler
from krnnt.structure import Paragraph, Form


class Sample:
    def __init__(self):
        self.features = {}  # also labels


class Module(object):
    def __init__(self, input_path: str):
        self.input_path = input_path

    def load(self) -> str:
        output_path = self.output_path()
        if not os.path.isfile(output_path) or os.path.getmtime(output_path) < os.path.getmtime(self.input_path):
            try:
                self._create()
            except BaseException as exception:
                # print exception
                traceback.print_exc()
                os.remove(output_path)

        return output_path

    def _create(self):
        raise NotImplementedError()

    def output_path(self) -> str:
        return self.input_path + '_' + str(self.__class__.__name__)


class FormatDataPreAnalyzed(Module):

    def _create(self):
        file = open(self.input_path, 'rb')
        su = SerialUnpickler(file)

        file2 = open(self.output_path(), 'wb')
        sp = SerialPickler(file2)

        for paragraph in tqdm(su, total=18484, desc='Processing'):
            paragraph_sequence = []
            for sentence in paragraph:
                sequence = []
                for token in sentence.tokens:
                    sample = Sample()
                    sequence.append(sample)
                    sample.features['token'] = token.form
                    sample.features['tags'] = uniq(map(lambda form: form.tags, token.interpretations))
                    sample.features['label'] = token.gold_form.tags
                    sample.features['lemma'] = token.gold_form.lemma
                    sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']

                paragraph_sequence.append((sequence, sequence))
            sp.add(paragraph_sequence)

        file.close()
        file2.close()


class FormatData2(Module):

    def _create(self):
        file = open(self.input_path, 'rb')
        su = SerialUnpickler(file)

        file2 = open(self.output_path(), 'wb')
        sp = SerialPickler(file2)

        paragraph: Paragraph
        for paragraph in tqdm(su, total=18484, desc='Processing'):
            paragraph_sequence = []
            for sentence, sentence_gold in zip(paragraph, paragraph.concraft):
                sequence = []
                if len(sentence_gold.tokens) == len(sentence.tokens) and len(
                        [token.gold_form for token in sentence.tokens if token.gold_form is None]) == 0:
                    for token in sentence.tokens:
                        sample = Sample()
                        sequence.append(sample)
                        sample.features['token'] = token.form
                        sample.features['tags'] = uniq(map(lambda form: form.tags, token.interpretations))
                        sample.features['label'] = token.gold_form.tags
                        sample.features['lemma'] = token.gold_form.lemma
                        sample.features['space_before'] = ['space_before'] if token.space_before else [
                            'no_space_before']
                else:
                    for token in sentence.tokens:
                        sample = Sample()
                        sequence.append(sample)
                        sample.features['token'] = token.form
                        sample.features['tags'] = uniq(map(lambda form: form.tags, token.interpretations))
                        sample.features['space_before'] = ['space_before'] if token.space_before else [
                            'no_space_before']

                sequence2 = []
                for token_gold in sentence_gold.tokens:
                    sample = Sample()
                    sequence2.append(sample)
                    sample.features['token'] = token_gold.form
                    if token_gold.gold_form is None:
                        sample.features['label'] = 'ign'
                    else:
                        sample.features['label'] = token_gold.gold_form.tags
                        sample.features['lemma'] = token_gold.gold_form.lemma
                    sample.features['space_before'] = ['space_before'] if token_gold.space_before else [
                        'no_space_before']

                paragraph_sequence.append((sequence, sequence2))
            sp.add(paragraph_sequence)

        file.close()
        file2.close()


class PreprocessData(Module):
    def __init__(self, input_path: str, operations: List = None):
        super(PreprocessData, self).__init__(input_path)
        if operations is None:
            operations = []
        self.operations = operations

    def _create(self):
        file = open(self.input_path, 'rb')
        su = SerialUnpickler(file)

        file2 = open(self.output_path(), 'wb')
        sp = SerialPickler(file2)

        for paragraph in tqdm(su, total=18484, desc='Processing %s' % str(self.__class__.__name__)):
            paragraph_sequence = []
            for sentence, sentence_orig in paragraph:
                sequence = list(sentence)
                for sample in sentence:
                    sample.features['tags4e3'] = create_token_features(sample.features['token'],
                                                                       sample.features['tags'],
                                                                       sample.features['space_before'])
                    # print(sample.features)

                    # for operation in self.operations:
                #     for sample in sentence:
                #         operation.apply(sample, sentence)

                paragraph_sequence.append((sequence, sentence_orig))
            sp.add(paragraph_sequence)

        file.close()
        file2.close()


class UniqueFeaturesValues(Module):
    def _create(self):
        file = open(self.input_path, 'rb')
        su = SerialUnpickler(file)

        unique_dict = collections.defaultdict(dict)
        index = collections.defaultdict(int)
        for paragraph in tqdm(su, total=85663, desc='Processing %s' % str(self.__class__.__name__)):
            for sentence, sentence_orig in paragraph:
                for sample in sentence:
                    for name, values in sample.features.items():
                        if isinstance(values, str) or isinstance(values, numbers.Number):
                            values = [values]

                        for value in values:
                            if value not in unique_dict[name]:
                                unique_dict[name][value] = index[name]
                                index[name] += 1

                for sample in sentence_orig:
                    for name, values in sample.features.items():
                        if isinstance(values, str) or isinstance(values, numbers.Number):
                            values = [values]

                        for value in values:
                            if value not in unique_dict[name]:
                                unique_dict[name][value] = index[name]
                                index[name] += 1

        file.close()

        file = open(self.output_path(), 'wb')
        pickle.dump(unique_dict, file)
        file.close()

    def get(self):
        self.load()
        return pickle.load(open(self.output_path(), 'rb'))


class Lemmatisation2():
    def __init__(self):
        pass

    def learn(self, path, stop=-1, start=0, ids=None):
        return

    def save(self, path):
        return

    def load(self, path):
        self.gpp = pickle.load(open(path, 'rb'))
        return

    def base_tag(self, tag):
        if isinstance(tag, str):
            tag = tag.split(':')
        transformations = {
            'ger': [(['pl'], 'sg'),
                    (['gen', 'dat', 'acc', 'inst', 'loc', 'voc'], 'nom')],
            'pact': [(['pl'], 'sg'),
                     (['gen', 'dat', 'acc', 'inst', 'loc', 'voc'], 'nom'),
                     (['m2', 'm3', 'f', 'n'], 'm1')],
            'ppas': [(['pl'], 'sg'),
                     (['gen', 'dat', 'acc', 'inst', 'loc', 'voc'], 'nom'),
                     (['m2', 'm3', 'f', 'n'], 'm1')],
        }

        tag = list(tag)

        if tag[0] not in transformations: return None

        transforms = transformations[tag[0]]
        for sources, target in transforms:
            for source in sources:
                try:
                    index = tag.index(source)
                    tag[index] = target
                    break
                except ValueError:
                    pass
        return tag

    def same_lemma_tag(self, tag1, tag2):
        if isinstance(tag1, str):
            tag1 = tag1.split(':')
            tag2 = tag2.split(':')
        groups = [['subst', 'ger'], ['adj', 'ppas', 'pact']]
        for group in groups:
            if tag1[0] in group and tag2[0] in group:
                if tag1[1:4] == tag2[1:4]:
                    return True
                return False
        return False

    def disambiguate(self, form, lemmas_tags, predicted_tag):
        # print(form)
        # print(lemmas_tags)
        # print(predicted_tag)
        # print()

        lemmas_tags = [(regex.sub(r':[abcdijnopqsv][0-9]?$', '', x), y) for x, y in lemmas_tags]
        disamb_lemmas_tags = [x for x in lemmas_tags if x[1] == predicted_tag]

        if disamb_lemmas_tags:
            base_tag = self.base_tag(predicted_tag)
            if base_tag is None:
                return disamb_lemmas_tags[0][0]  # oryginalny lemat
            else:
                try:
                    return self.gpp[(disamb_lemmas_tags[0][0], tuple(base_tag))]  # szukamy "mianownika"
                except KeyError:
                    pass
        else:
            for lemma, tag in lemmas_tags:
                if self.same_lemma_tag(predicted_tag, tag):
                    base_tag = self.base_tag(tag)
                    if base_tag is None:
                        return lemma  # orygianlny lemat
                    else:
                        try:
                            return self.gpp[(lemma, tuple(base_tag))]  # szukamy "mianownika"
                        except KeyError:
                            break

        return form


class Lemmatisation():
    def __init__(self):
        self.lemmas = {}

    def learn(self, path, stop=-1, start=0, ids=None):
        lemma_count = collections.defaultdict(lambda: collections.defaultdict(int))
        if ids is None:
            ids = []
        su = SerialUnpickler(open(path, 'rb'), stop=stop, start=start, ids=ids)
        for paragraph in su:
            for sentence, sentence_orig in paragraph:
                for sample in sentence_orig:
                    # print(sample.features)
                    if 'lemma' in sample.features:  # some samples doesnt have lemma, because not on gold segmentation
                        lemma_count[(sample.features['token'], sample.features['label'])][sample.features['lemma']] += 1

        for k, v in lemma_count.items():
            best = sorted(v.items(), key=lambda x: (x[1], x[0]), reverse=True)[0]  # TODO kilka z taka sama statystyka
            self.lemmas[k] = best[0]

    def disambiguate(self, form, ilosc_disamb, predicted_tag):
        if form is None: return
        tag = predicted_tag

        ilosc_disamb = [x for x in ilosc_disamb if x[1] == predicted_tag]

        if not ilosc_disamb:
            ilosc_disamb = [(form, predicted_tag)]

        if (form, tag) in self.lemmas:
            pred_lemma = self.lemmas[(form, tag)]
        else:
            interp_lemmas = [lemma for (lemma, tag) in ilosc_disamb]
            pred_lemma = random.choice(interp_lemmas)
        return pred_lemma
        # print('\t%s\t%s\tdisamb' % (pred_lemma, tag))

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.lemmas, f)
        f.close()

    def load(self, path):
        self.lemmas = pickle.load(open(path, 'rb'))


def generate_arrays_from_file(path, unique_features_dict, feature_name, label_name, stop=-1, start=0, ids=None,
                              keep_unaligned=False, keep_infinity=True):
    if ids is None:
        ids = []
    while 1:
        su = SerialUnpickler(open(path, 'rb'), stop=stop, start=start, ids=ids)
        for paragraph in su:
            for sentence, sentence_orig in paragraph:
                X_sentence = []
                y_sentence = []
                if not sentence: continue  # TODO

                same_segmentation = len(sentence) == len(sentence_orig) and len(
                    [sample for sample in sentence if 'label' in sample.features])
                if (not same_segmentation) and not keep_unaligned:
                    continue

                if keep_unaligned and same_segmentation:
                    for sample in sentence:
                        X_sentence.append(
                            np.array(k_hot(sample.features[feature_name], unique_features_dict[feature_name])))
                        if label_name == 'label':
                            y_sentence.append(
                                np.array(k_hot([sample.features[label_name]], unique_features_dict[label_name])))
                        else:
                            y_sentence.append(
                                np.array(k_hot(sample.features[label_name], unique_features_dict[label_name])))
                else:
                    for sample in sentence:
                        X_sentence.append(
                            np.array(k_hot(sample.features[feature_name], unique_features_dict[feature_name])))
                    for sample in sentence_orig:
                        if label_name == 'label':
                            y_sentence.append(
                                np.array(k_hot([sample.features[label_name]], unique_features_dict[label_name])))
                        else:
                            y_sentence.append(
                                np.array(k_hot(sample.features[label_name], unique_features_dict[label_name])))

                # print len(X_sentence), len(y_sentence)
                yield (X_sentence, y_sentence, sentence, sentence_orig)
        if not keep_infinity: break


def batch_generator(generator, batch_size=32, return_all=False, sort=False):
    batch_X = []
    batch_y = []
    sentences = []
    sentences_orig = []

    if sort:
        generator = sorted(generator, key=lambda x: len(x[0]))

    for X, y, sentence, sentence_orig in generator:
        batch_X.append(X)
        batch_y.append(y)
        sentences.append(sentence)
        sentences_orig.append(sentence_orig)
        if (not return_all) and len(batch_X) == batch_size:
            yield (batch_X, batch_y, sentences, sentences_orig)
            batch_X = []
            batch_y = []
            sentences = []
            sentences_orig = []

    if batch_X or return_all:
        yield (batch_X, batch_y, sentences, sentences_orig)


def pad_generator(generator, sequence_length=20):
    for batch_X, batch_y, sentences, sentences_orig in generator:
        if not batch_X or not batch_y:
            continue

        # TODO pad multi inputs
        max_sentence_length = max([len(x) for x in batch_X])
        # print('max_sentence_length',max_sentence_length)
        yield (sequence.pad_sequences(batch_X, maxlen=max_sentence_length),
               sequence.pad_sequences(batch_y, maxlen=max_sentence_length), sentences, sentences_orig)


def Xy_generator(generator):
    for batch_X, batch_y, sequences, sentences_orig in generator:
        # print len(x), len(x[0])
        yield (batch_X, batch_y)


def k_hot(tags, unique_tags_dict, zero=0):
    # print
    result = [zero] * len(unique_tags_dict)
    for tag in tags:
        # print len(unique_tags_dict), unique_tags_dict[tag]
        try:
            result[unique_tags_dict[tag]] = 1
        except KeyError:
            # print('KeyError:', tag, file=sys.stderr)
            pass
    return result


def predictions_to_classes(unique_tags_dict, predictions):
    out = []
    classes = map(lambda k, v: k, sorted(unique_tags_dict.items(), key=lambda k, v: v))
    for i, a in enumerate(predictions):
        if a: out.append(classes[i])
    return out


class LossHistory(Callback):
    def __init__(self, evaluator, log_path, name=''):
        super().__init__()
        self.evaluator = evaluator
        self.file = open(log_path, 'wt')
        self.history = []
        self.name = name

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # X,y=self.test_data
        # predictions = self.model.predict(X, verbose=1)

        if logs is None:
            logs = {}
        end = time.time()
        logs[self.name + 'time'] = end - self.start

        out = self.evaluator.evaluate(self.model)
        print(out)
        logs[self.name + 'val_score'] = out[2]
        try:
            logs[self.name + 'val_score_pos_em'] = out[5]
            logs[self.name + 'val_score_k'] = out[8]
            logs[self.name + 'val_score_u'] = out[11]
            logs[self.name + 'val_score_upper'] = out[14]
            logs[self.name + 'test_time'] = out[-1]
        except:
            pass
        self.write_str(sorted(logs.items(), key=lambda x: x[0]))
        self.history.append(logs)

    def write_str(self, logs):
        self.file.write(str(logs))
        self.file.write('\n')
        self.file.flush()


class Evaluator:
    def __init__(self, test_data, unique_tags_dict):
        self.test_data = test_data
        self.unique_tags_dict = unique_tags_dict


class Accuracy(object):
    def __init__(self):
        self.tp = 0
        self.tn = 0

    def accuracy(self):
        l = self.tp + self.tn
        if l == 0:
            return 0.0
        return float(self.tp) / l

    def __repr__(self):
        return str([self.tp, self.tn])


def to_plain(batch, preds, file_test, file_pred, unique_tags_dict):
    X_train2, y_train2, sentences2, sentences_orig2 = batch
    for x, y, s1, s2, pr in zip(X_train2, y_train2, sentences2, sentences_orig2, preds):
        for t, p in zip(s1, list(pr)[-len(s1):]):
            # print(t)
            form = t.features['token']
            form = form.replace('\t', '').replace('­', '')
            # print(t.features)
            # if 'space_before' in t.features['space_before']:
            #     form=' '+form
            file_pred.write('%s\t%s\n' % (form, 'space' if 'space_before' in t.features['space_before'] else 'none'))
            max_index, max_value = max(enumerate(p), key=lambda x: x[1])
            predicted_tag = list(unique_tags_dict.keys())[list(unique_tags_dict.values()).index(max_index)]
            file_pred.write('\t%s\t%s\tdisamb\n' % (form, predicted_tag))
        file_pred.write('\n')

        for t in s2:
            form = t.features['token']
            form = form.replace('\t', '').replace('­', '')
            file_test.write('%s\t%s\n' % (form, 'space' if 'space_before' in t.features['space_before'] else 'none'))
            file_test.write('\t%s\t%s\tdisamb\n' % (form, t.features['label']))
        file_test.write('\n')

    file_pred.flush()
    file_test.flush()


class DataGenerator():
    def __init__(self, data_path, unique_features_dict, pref, ids):
        self.data_path = data_path
        self.unique_features_dict = unique_features_dict
        self.pref = pref
        self.ids = ids

    def _create_generator(self):
        return pad_generator(batch_generator(
            generate_arrays_from_file(self.data_path, self.unique_features_dict, self.pref['feature_name'],
                                      self.pref['label_name'],
                                      ids=self.ids, keep_infinity=False, keep_unaligned=True),
            batch_size=self.pref['batch_size']))

    def get(self):
        return self._create_generator()


class DataList():
    def __init__(self, data):
        self.data = data

    def get(self):
        return self.data


class UnalignedSimpleEvaluator(Evaluator):
    def evaluate(self, model, verbose=False, plain=False):
        # print len(self.test_data), len(self.test_data[0])
        # X_train, y_train, sentences,sentences_orig = self.test_data

        X_train = []
        y_train = []
        sentences = []
        sentences_orig = []

        # predictions = model.predict(X_train, verbose=1, batch_size=64)
        predictions = []

        file_test = open('test.plain', 'wt')
        file_pred = open('pred.plain', 'wt')

        start = time.time()

        for batch in self.test_data.get():
            X_train2, y_train2, sentences2, sentences_orig2 = batch

            X_train.extend(X_train2)
            y_train.extend(y_train2)
            sentences.extend(sentences2)
            sentences_orig.extend(sentences_orig2)
            preds = model.predict_on_batch(X_train2)
            predictions.extend(preds)

            if plain: to_plain(batch, preds, file_test, file_pred, self.unique_tags_dict)

        # TODO if generator then calculate acc on batches

        self.predictions = predictions

        end = time.time()
        test_time = len(X_train) / float(end - start)

        acc_lower = Accuracy()
        acc_upper = Accuracy()
        acc_k_lower = Accuracy()
        acc_u_lower = Accuracy()
        acc_pos_em = Accuracy()
        acc_verb_em = Accuracy()

        i = 0
        zero = 0
        errors = []
        bar = Bar('Processing %s' % str(self.__class__.__name__),
                  suffix='%(index)d/%(max)d %(percent).1f%% - %(eta_td)s - %(elapsed_td)s', max=len(predictions))
        for a, b, e, sentence, sentence_orig in zip(y_train, predictions, X_train, sentences, sentences_orig):
            bar.next()
            # print tested_sentences[i]
            length = min(len(a), len(sentence))
            # print length
            # if length==0: continue

            if verbose:
                print(length)

            sentence_valid = True

            # predictions
            pred = []
            ref = []
            for predd, sample in zip(list(b)[-len(sentence):], sentence):
                max_index, max_value = max(enumerate(predd), key=lambda x: x[1])  # mozna wczesniej obliczyc jako argmax
                pred.append((sample.features['token'], max_index, sample.features['space_before'][0] == 'space_before',
                             'ign' in sample.features['tags']))

            # print(pred)

            # reference
            for predd, sample in zip(list(a)[-len(sentence_orig):], sentence_orig):
                max_index, max_value = max(enumerate(predd), key=lambda x: x[1])
                ref.append((sample.features['token'], max_index, sample.features['space_before'][0] == 'space_before'))

            # if len(pred)!=len(ref): print(pred,ref)
            # align(pred,ref)

            # print(pred,ref)
            # print('dupa',acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em)
            accuracy(pred, ref, acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em)
            # print('dupa2',acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em)

            i += 1

        bar.finish()

        return acc_lower.tp, acc_lower.tn, acc_lower.accuracy(), acc_pos_em.tp, acc_pos_em.tn, acc_pos_em.accuracy(), acc_k_lower.tp, acc_k_lower.tn, acc_k_lower.accuracy(), acc_u_lower.tp, acc_u_lower.tn, acc_u_lower.accuracy(), acc_upper.tp, acc_upper.tn, acc_upper.accuracy(), test_time


def text(buffer):
    return ''.join(
        [' ' + x[0].replace('\xa0', '') if x[2] is True or x[2] == 'space' else x[0].replace('\xa0', '') for x in
         buffer])


def text_verbose(buffer):
    return ''.join([' ' + x[0].features['token'].replace('\xa0', '') if x[2] is True or x[2] == 'space' else
                    x[0].features['token'].replace('\xa0', '') for x in buffer])


def text_sample(buffer):
    return ''.join([' ' + x.features['token'].replace('\xa0', '') if x.features['space_before'] is True or x.features[
        'space_before'] == 'space' else x.features['token'].replace('\xa0', '') for x in buffer])


def accuracy(pred, ref, acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em):
    sentence_valid = True
    for p, r in align(pred, ref):
        # print(p,r)
        if len(p) == len(r) and len(p) == 1:
            if p[0][1] == r[0][1]:
                acc_lower.tp += 1
                acc_upper.tp += 1
                if p[0][3]:  # unknown
                    acc_u_lower.tp += 1
                else:
                    acc_k_lower.tp += 1
            else:
                sentence_valid = False
                acc_lower.tn += 1
                acc_upper.tn += 1
                if p[0][3]:  # unknown
                    acc_u_lower.tn += 1
                else:
                    acc_k_lower.tn += 1
        else:
            sentence_valid = False
            acc_lower.tn += len(r)
            acc_upper.tp += len(r)
            acc_k_lower.tn += len(r)

    if sentence_valid:
        acc_pos_em.tp += 1
    else:
        acc_pos_em.tn += 1


def accuracy_verbose(pred, ref, acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em, acc_pred_not_in_tags,
                     acc_ref_not_in_tags, classes):
    sentence_valid = True
    raw_s = text_verbose(pred)

    for p, r in align_verbose(pred, ref):
        pred_sample = p[0][0]
        ref_sample = r[0][0]
        pmax_index, pmax_value = max(enumerate(p[0][1]), key=lambda x: x[1])
        rmax_index, rmax_value = max(enumerate(r[0][1]), key=lambda x: x[1])
        pred_not_in_tags = classes[pmax_index] not in pred_sample.features['tags']
        ign = 'ign' in pred_sample.features['tags']
        if len(p) == len(r) and len(p) == 1:

            if not ign:
                if classes[rmax_index] not in pred_sample.features['tags']:
                    acc_ref_not_in_tags.tp += 1
                else:
                    acc_ref_not_in_tags.tn += 1

            if pmax_index == rmax_index:
                acc_lower.tp += 1
                acc_upper.tp += 1
                if not ign and pred_not_in_tags:
                    acc_pred_not_in_tags.tp += 1

                if p[0][3]:  # unknown
                    acc_u_lower.tp += 1
                else:
                    acc_k_lower.tp += 1
            else:
                if not ign and pred_not_in_tags:
                    acc_pred_not_in_tags.tn += 1

                sss = sorted(enumerate(p[0][1]), key=lambda x: x[1], reverse=True)

                print(raw_s)
                print(pred_sample.features['token'])
                print('True: \t\t%s' % classes[rmax_index])
                print('Predicted: \t%s' % classes[pmax_index])
                print(pred_sample.features['tags'])
                print(pred_sample.features['tags4e3'])
                for index, value in sss[:10]:
                    desc = ''
                    if index == rmax_index: desc = 'TRUE '
                    if classes[index] not in pred_sample.features['tags']:
                        desc += 'NOT_IN_TAGS'
                    print(' - %.4f %s \t%s' % (value, classes[index], desc))
                print()

                sentence_valid = False
                acc_lower.tn += 1
                acc_upper.tn += 1
                if p[0][3]:  # unknown
                    acc_u_lower.tn += 1
                else:
                    acc_k_lower.tn += 1
        else:
            sentence_valid = False
            acc_lower.tn += len(r)
            acc_upper.tp += len(r)
            acc_k_lower.tn += len(r)

    if sentence_valid:
        acc_pos_em.tp += 1
    else:
        acc_pos_em.tn += 1


def align(pred, ref):
    pred_buffer = [pred.pop(0)]
    ref_buffer = [ref.pop(0)]

    while pred_buffer or ref_buffer:
        pred_text = text(pred_buffer)
        ref_text = text(ref_buffer)
        # print(pred_text, ref_text)
        if len(pred_text) == len(ref_text):  # aligned
            if pred_text != ref_text:
                print('alignment ERROR', pred_text, ref_text, ref, pred)

            yield (pred_buffer, ref_buffer)

            pred_buffer = []
            ref_buffer = []

            if not pred or not ref:
                break

            pred_buffer = [pred.pop(0)]
            ref_buffer = [ref.pop(0)]
        elif len(pred_text) < len(ref_text):
            if pred:
                pred_buffer.append(pred.pop(0))
            else:
                break
        else:
            if ref:
                ref_buffer.append(ref.pop(0))
            else:
                break

    rest = ref_buffer + ref
    if rest:
        yield (pred_buffer + pred, rest)
    # print('rest', pred, ref)


def align_verbose(pred, ref):
    pred_buffer = [pred.pop(0)]
    ref_buffer = [ref.pop(0)]

    while pred_buffer or ref_buffer:
        pred_text = text_verbose(pred_buffer)
        ref_text = text_verbose(ref_buffer)
        # print(pred_text, ref_text)
        if len(pred_text) == len(ref_text):  # aligned
            if pred_text != ref_text:
                print('alignment ERROR', pred_text, ref_text, ref, pred)

            yield (pred_buffer, ref_buffer)

            pred_buffer = []
            ref_buffer = []

            if not pred or not ref:
                break

            pred_buffer = [pred.pop(0)]
            ref_buffer = [ref.pop(0)]
        elif len(pred_text) < len(ref_text):
            if pred:
                pred_buffer.append(pred.pop(0))
            else:
                break
        else:
            if ref:
                ref_buffer.append(ref.pop(0))
            else:
                break

    rest = ref_buffer + ref
    if rest:
        yield (pred_buffer + pred, rest)


def count_sentences(path, ids=None):
    if ids is None:
        ids = []
    count = 0
    su = SerialUnpickler(open(path, 'rb'), ids=ids)
    for paragraph in su:
        for sentence in paragraph:
            count += 1
    return count


def get_morfeusz():
    import morfeusz2
    morf = morfeusz2.Morfeusz(
        analyse=True,  # load analyze dictionary
        generate=False,  # dont load generator dictionary
        expand_tags=True,  # expand tags (return tags without dots)
        aggl='isolated',  # 'isolated' - token 'm' has aglt interpretation, token 'np' has brev interpretation
        praet='composite',  # aglt and 'by' are not divided
        #    whitespace=morfeusz2.KEEP_WHITESPACES
    )
    return morf


def analyze_token(morf, token):
    segment_interpretations = morf.analyse(token)
    # if token is tokenized then take all interpretation of segments starting from beginning of the token
    interpretations = []
    for start, end, interpretation in segment_interpretations:
        if start == 0:
            form, lemma, tag, domain, qualifier = interpretation
            interpretations.append((lemma, tag))

    return interpretations


def analyze_tokenized(morf, paragraphs):
    for p in paragraphs:
        for s in p:
            for token in s:
                interpretations = analyze_token(morf, token.form)
                print(interpretations)
                token.interpretations.extend([Form(base, ctag) for (base, ctag) in interpretations])
        yield p
