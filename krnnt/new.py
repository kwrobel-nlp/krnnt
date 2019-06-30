#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import numbers
import os.path
import pickle
import random
import time
import traceback

import collections
import numpy as np
import regex
from keras.callbacks import Callback
from keras.preprocessing import sequence
from progress.bar import Bar

from .classes import geo_mean, SerialUnpickler, SerialPickler, uniq, flatten, mean, Form


class Sample:
    def __init__(self):
        self.features = {} # also labels


class Module(object):
    def __init__(self,input_path):
        self.input_path=input_path

    def load(self):
        output_path=self.output_path()
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

    def output_path(self):
        return self.input_path+'_'+str(self.__class__.__name__)


class FormatDataPreAnalyzed(Module):

    def _create(self):
        file = open(self.input_path,'rb')
        su = SerialUnpickler(file)

        file2=open(self.output_path(),'wb')
        sp = SerialPickler(file2)

        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=18484)
        for paragraph in su:
            bar.next()
            paragraph_sequence = []
            for sentence in paragraph:
                sequence =[]
                for token in sentence.tokens:
                    sample = Sample()
                    sequence.append(sample)
                    sample.features['token'] = token.form
                    sample.features['tags'] = uniq(map(lambda form: form.tags, token.interpretations))
                    sample.features['label'] = token.gold_form.tags
                    sample.features['lemma'] = token.gold_form.lemma
                    sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']

                paragraph_sequence.append((sequence,sequence))
            sp.add(paragraph_sequence)
        bar.finish()

        file.close()
        file2.close()

class FormatData2(Module):

    def _create(self):
        file = open(self.input_path,'rb')
        su = SerialUnpickler(file)

        file2=open(self.output_path(),'wb')
        sp = SerialPickler(file2)

        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=18484)
        for paragraph in su:
            bar.next()
            paragraph_sequence = []
            for sentence, sentence_gold in zip(paragraph, paragraph.concraft):
                sequence =[]
                if len(sentence_gold.tokens) == len(sentence.tokens) and len([token.gold_form for token in sentence.tokens if token.gold_form is None])==0:
                    for token in sentence.tokens:
                        sample = Sample()
                        sequence.append(sample)
                        sample.features['token'] = token.form
                        sample.features['tags'] = uniq(map(lambda form: form.tags, token.interpretations))
                        sample.features['label'] = token.gold_form.tags
                        sample.features['lemma'] = token.gold_form.lemma
                        sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']
                else:
                    for token in sentence.tokens:
                        sample = Sample()
                        sequence.append(sample)
                        sample.features['token'] = token.form
                        sample.features['tags'] = uniq(map(lambda form: form.tags, token.interpretations))
                        sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']

                sequence2 =[]
                for token_gold in sentence_gold.tokens:
                    sample = Sample()
                    sequence2.append(sample)
                    sample.features['token'] = token_gold.form
                    if token_gold.gold_form is None:
                        sample.features['label'] = 'ign'
                    else:
                        sample.features['label'] = token_gold.gold_form.tags
                        sample.features['lemma'] = token_gold.gold_form.lemma
                    sample.features['space_before'] = ['space_before'] if token_gold.space_before else ['no_space_before']

                paragraph_sequence.append((sequence,sequence2))
            sp.add(paragraph_sequence)
        bar.finish()

        file.close()
        file2.close()



class FormatData(Module):

    def _create(self):
        file = open(self.input_path,'rb')
        su = SerialUnpickler(file)

        file2=open(self.output_path(),'wb')
        sp = SerialPickler(file2)

        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=18484)
        for paragraph in su:
            bar.next()
            paragraph_sequence = []
            for sentence in paragraph:
                sequence =[]
                if len(sentence.concraft) == len(sentence.tokens):
                    for token_concraft, token in zip(sentence.concraft, sentence.tokens):
                        sample = Sample()
                        sequence.append(sample)
                        sample.features['token'] = token.form
                        sample.features['tags'] = uniq(map(lambda form: form.tags, token_concraft.interpretations))
                        sample.features['label'] = token.gold_form.tags
                        sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']
                        # if len(sample.features['space_before'])>1:
                        #     print(sample.features['space_before'])
                else:
                    for token_concraft in sentence.concraft:
                        sample = Sample()
                        sequence.append(sample)
                        sample.features['token'] = token_concraft.form
                        sample.features['tags'] = uniq(map(lambda form: form.tags, token_concraft.interpretations))
                        sample.features['space_before'] = ['space_before'] if token_concraft.space_before == 'space' else ['no_space_before']

                sequence2 =[]
                for token in sentence.tokens:
                    sample = Sample()
                    sequence2.append(sample)
                    sample.features['token'] = token.form
                    sample.features['label'] = token.gold_form.tags
                    sample.features['space_before'] = ['space_before'] if token.space_before else ['no_space_before']

                paragraph_sequence.append((sequence,sequence2))
            sp.add(paragraph_sequence)
        bar.finish()

        file.close()
        file2.close()

class FeaturePreprocessor:
    qubs = set(['a','abo','aby','akurat','albo','ale','amen','ani','aż','aza','bądź','blisko','bo','bogać','by','byle','byleby','choć','choćby','chociaż','chociażby','chyba','ci','co','coś','czy','czyli','czyż','dalibóg','dobra','dokładnie','doprawdy','dość','dosyć','dziwna','dziwniejsza','gdyby','gdzie','gdzieś','hale','i','ino','istotnie','jakby','jakoby','jednak','jedno','jeno','koło','kontra','lada','ledwie','ledwo','li','maksimum','minimum','może','najdziwniejsza','najmniej','najwidoczniej','najwyżej','naturalnie','nawzajem','ni','niby','nie','niechaj','niejako','niejakoś','no','nuż','oczywiście','oczywista','okay','okej','około','oto','pewnie','pewno','podobno','ponad','ponoś','prawda','prawie','przecie','przeszło','raczej','skąd','skądinąd','skądże','szlus','ta','taj','tak','tam','też','to','toż','tuż','tylko','tylo','widocznie','właśnie','wprost','wręcz','wszakże','wszelako','za','zaledwie','zaledwo','żali','zaliż','zaraz','że','żeby','zwłaszcza'])
    safe_chars = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', u'?', u'-', u'a', u'ą', u'c', u'ć', u'b', u'e',
                  u'ę', u'd', u'g', u'f', u'i', u'h', u'k', u'j', u'm', u'l', u'ł', u'o', u'ó', u'n', u'ń', u'q', u'p',
                  u's', u'ś', u'r', u'u', u't', u'w', u'y', u'x', u'z', u'ź', u'ż'}

    @staticmethod
    def nic(form, features = None):
        return ['NIC']

    @staticmethod
    def cases(form, features = None):
        if form.islower():
            return ['islower']
        elif form.isupper():
            return ['isupper']
        elif form.istitle():
            return ['istitle']
        elif form.isdigit():
            return ['isdigit']
        else:
            return ['ismixed']

    @staticmethod
    def interps(form, features):
        if 'interp' in features['tags'] and len(features['token'])==1:
            return [features['token']]
        else:
            return []

    @staticmethod
    def qubliki(form, features = None):
        if form.lower() in FeaturePreprocessor.qubs:
            return [form]
        else:
            return []

    @staticmethod
    def shape(form, features = None):
        # print(form, shape(form))
        return [shape(form)]

    @staticmethod
    def prefix(n, form, features= None):
        try:
            char = form[n].lower()
            if char not in FeaturePreprocessor.safe_chars:
                char = '??'
        except IndexError:
            char = 'xx'

        return ['P'+str(n)+char]

    @staticmethod
    def prefix1(form, features = None):
        return FeaturePreprocessor.prefix(0, form, features)

    @staticmethod
    def prefix2(form, features = None):
        return FeaturePreprocessor.prefix(1, form, features)

    @staticmethod
    def prefix3(form, features = None):
        return FeaturePreprocessor.prefix(2, form, features)

    @staticmethod
    def suffix(n, form, features= None):
        try:
            char = form[-n].lower()
            if char not in FeaturePreprocessor.safe_chars:
                char = '??'
        except IndexError:
            char = 'xx'

        return ['S'+str(n)+char]

    @staticmethod
    def suffix1(form, features = None):
        return FeaturePreprocessor.suffix(1, form, features)

    @staticmethod
    def suffix2(form, features = None):
        return FeaturePreprocessor.suffix(2, form, features)

    @staticmethod
    def suffix3(form, features = None):
        return FeaturePreprocessor.suffix(3, form, features)

class TagsPreprocessor:
    cas=['nom', 'gen', 'dat', 'acc', 'inst', 'loc', 'voc']

    per=['pri', 'sec', 'ter']
    nmb=['sg', 'pl']
    gnd=['m1', 'm2', 'm3', 'f', 'n']

    def create_tags2(self, tags, features = None):
        return uniq(flatten(map(lambda tag: self.create_tag2(tag), tags)))

    def create_tag2(self, tag, features = None):
        tags = flatten(map(lambda x: x.split('.'), tag.split(':')))
        pos = tags[0]
        tags = tags[1:]
        tags2 = []

        if not tags:
            tags2.append(pos)
        for tag in tags:
            tags2.append(pos+'-'+tag)

        return uniq(tags2)

    def create_tag3(self, tag, features = None):
        tags = flatten(map(lambda x: x.split('.'), tag.split(':')))
        pos = tags[0]
        tags = tags[1:]
        tags2 = []

        if not tags:
            tags2.append(pos)
        for tag in tags:
            tags2.append(pos+'-'+tag)
            tags2.append(tag)
            tags2.append(pos)

        return uniq(tags2)

    def create_tags3(self, tags, features = None):
        return uniq(flatten(map(lambda tag: self.create_tag3(tag), tags)))

    @staticmethod
    def create_tags4(tags, features = None, keep_guesser=True): #concraft
        if not keep_guesser and 'ign' in tags:
            return ['ign']
            # return ['1ign','2ign','1subst:nom','2subst:sg:f','1adj:nom','1subst:gen','2subst:sg:n','2subst:sg:m1','2adj:sg:m3:pos','2subst:sg:m3','1num:acc','2num:pl:m3:rec','1brev','2adj:sg:n:pos','2num:pl:m3:congr','1num:nom','1adj:gen','1adj:loc']
        return uniq(flatten(map(lambda tag: TagsPreprocessor.create_tag4(tag), tags)))

    @staticmethod
    def create_tags4_without_guesser(tags, features = None):
        return TagsPreprocessor.create_tags4(tags, features = features, keep_guesser=False)

    @staticmethod
    def create_tag4(otag, features = None):
        tags = flatten(map(lambda x: x.split('.'), otag.split(':')))
        pos = tags[0]
        tags = tags[1:]
        tags2 = []

        first=None
        for tag in tags:
            if tag in TagsPreprocessor.cas or tag in TagsPreprocessor.per:
                first=tag
                break

        if first:
            tags.remove(first)
            tags2.append('1'+pos+':'+first)
        else:
            tags2.append('1'+pos) #TODO sprawdzic

        tags2.append('2'+(':'.join([pos]+tags)))

        # print otag, tags2
        return uniq(tags2)

    @staticmethod
    def create_tags5(tags, features = None, keep_guesser=True): #concraft
        if not keep_guesser and 'ign' in tags:
            return ['ign']
            # return ['ign','sg:loc:m3','sg:nom:n','pl:nom:m3','pl:acc:m3','loc','sg:gen:m3','pl:gen:m3','sg:nom:m1','sg:nom:m3','gen','nom','acc','sg:nom:f']

        return uniq(flatten(map(lambda tag: TagsPreprocessor.create_tag5(tag), tags)))

    @staticmethod
    def create_tags5_without_guesser(tags, features = None):
        return TagsPreprocessor.create_tags5(tags, features = features, keep_guesser=False)
    @staticmethod
    def create_tag5(otag, features = None):

        tags = flatten(map(lambda x: x.split('.'), otag.split(':')))

        tags_out = []
        tags2 = []
        tags3= []
        for tag in tags:
            if tag in TagsPreprocessor.nmb:
                tags2.append(tag)
            elif tag in TagsPreprocessor.cas:
                tags2.append(tag)
                tags3.append(tag)
            elif tag in TagsPreprocessor.gnd:
                tags2.append(tag)

        for tagsX in [tags2, tags3]:
            if tagsX:
                tags_out.append(':'.join(tagsX))

        return uniq(tags_out)





class ProcessingRule:
    def __init__(self, method, inpupt_name, output_name):
        self.method = method
        self.input_name = inpupt_name
        self.output_name = output_name

    def apply(self, sample, sentence):
        sample.features[self.output_name] = self.method(sample.features[self.input_name], features = sample.features)

    def __repr__(self):
        return str([self.method.__name__, self.input_name, self.output_name])

class HistoryRule:
    def __init__(self, inpupt_name, output_name, history_index, prefix):
        self.history_index = history_index
        self.input_name = inpupt_name
        self.output_name = output_name
        self.prefix = prefix

    def apply(self, sample, sentence):
        sequence = list(sentence)
        sample_index = sequence.index(sample)+self.history_index
        if sample_index <0 or sample_index>=len(sequence):
            sample.features[self.output_name] = []
            return

        sample.features[self.output_name] = map(lambda f: self.prefix+f, sequence[sample_index].features[self.input_name])

class RemoveRule:
    def __init__(self, inpupt_name):
        self.input_name = inpupt_name

    def apply(self, sample, sentence):
        del sample.features[self.input_name]

class JoinProcessingRule:
    def __init__(self, input_names, output_name):
        self.input_names = input_names
        self.output_name = output_name

    def apply(self, sample, sentence):
        # print(self.input_names, self.output_name)
        # print(list(map(lambda input_name: sample.features[input_name], self.input_names)))
        #
        # for input_name in self.input_names:
        #     print(input_name, sample.features[input_name])
        # print()

        sample.features[self.output_name] = uniq(flatten(map(lambda input_name: sample.features[input_name], self.input_names)))

    def __repr__(self):
        return str(self.input_names)

class PreprocessData(Module):
    def __init__(self, input_path, operations=None):
        super(PreprocessData,self).__init__(input_path)
        if operations is None:
            operations = []
        self.operations = operations


    def _create(self):
        file = open(self.input_path,'rb')
        su = SerialUnpickler(file)

        file2=open(self.output_path(),'wb')
        sp = SerialPickler(file2)

        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=85663)

        for paragraph in su:
            paragraph_sequence = []
            for sentence,sentence_orig in paragraph:
                bar.next()
                sequence =list(sentence)
                for operation in self.operations:
                    for sample in sentence:
                        operation.apply(sample, sentence)
                    # print sample.features
                    # sequence.append(sample)

                paragraph_sequence.append((sequence,sentence_orig))
            sp.add(paragraph_sequence)
        bar.finish()

        file.close()
        file2.close()

class UniqueFeatures(Module):
    def __init__(self, input_path, name):
        super(UniqueFeatures,self).__init__(input_path)
        self.name = name

    def output_path(self):
        return self.input_path+'_'+str(self.__class__.__name__)+'_'+self.name

    def _create(self):
        file = open(self.input_path,'rb')
        su = SerialUnpickler(file)

        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=85663)
        unique_dict = {}
        i=0
        for sentence in su:
            bar.next()
            for sample in sentence:
                values = sample.features[self.name]
                if isinstance(values, str) or isinstance(values, numbers.Number):
                    values=[values]

                for value in values:
                    if value not in unique_dict:
                        unique_dict[value]=i
                        i+=1

        bar.finish()

        file.close()

        file = open(self.output_path(),'wb')
        pickle.dump(unique_dict, file)
        file.close()

    def get(self):
        self.load()
        return pickle.load(open(self.output_path(),'rb'))

class UniqueFeaturesValues(Module):
    def _create(self):
        file = open(self.input_path,'rb')
        su = SerialUnpickler(file)

        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=85663)
        unique_dict = collections.defaultdict(dict)
        index=collections.defaultdict(int)
        for paragraph in su:
            for sentence,sentence_orig in paragraph:
                bar.next()
                for sample in sentence:
                    for name,values in sample.features.items():
                        if isinstance(values, str) or isinstance(values, numbers.Number):
                            values=[values]

                        for value in values:
                            if value not in unique_dict[name]:
                                unique_dict[name][value]=index[name]
                                index[name]+=1

                for sample in sentence_orig:
                    for name,values in sample.features.items():
                        if isinstance(values, str) or isinstance(values, numbers.Number):
                            values=[values]

                        for value in values:
                            if value not in unique_dict[name]:
                                unique_dict[name][value]=index[name]
                                index[name]+=1

        bar.finish()

        file.close()

        file = open(self.output_path(),'wb')
        pickle.dump(unique_dict, file)
        file.close()

    def get(self):
        self.load()
        return pickle.load(open(self.output_path(),'rb'))

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
            tag=tag.split(':')
        transformations = {
            'ger':  [(['pl'],'sg'), 
                     (['gen','dat','acc','inst','loc','voc'],'nom')], 
            'pact': [(['pl'],'sg'), 
                     (['gen','dat','acc','inst','loc','voc'],'nom'), 
                     (['m2','m3','f','n'], 'm1')],
            'ppas': [(['pl'],'sg'), 
                     (['gen','dat','acc','inst','loc','voc'],'nom'), 
                     (['m2','m3','f','n'], 'm1')],
        }
        
        tag=list(tag)
        
        if tag[0] not in transformations: return None
        
        transforms = transformations[tag[0]]
        for sources, target in transforms:
            for source in sources:
                try:
                    index = tag.index(source)
                    tag[index]=target
                    break
                except ValueError:
                    pass
        return tag
      
    def same_lemma_tag(self, tag1, tag2):
        if isinstance(tag1, str):
            tag1=tag1.split(':')
            tag2=tag2.split(':')
        groups=[['subst','ger'],['adj','ppas','pact']]
        for group in groups:
            if tag1[0] in group and tag2[0] in group:
                if tag1[1:4]==tag2[1:4]:
                    return True
                return False
        return False
      
    def disambiguate(self, form, lemmas_tags, predicted_tag):
        #print(form)
        #print(lemmas_tags)
        #print(predicted_tag)
        #print()
      
        lemmas_tags = [(regex.sub(r':[abcdijnopqsv][0-9]?$','', x),y) for x,y in lemmas_tags]
        disamb_lemmas_tags = [x for x in lemmas_tags if x[1]==predicted_tag]
        

        if disamb_lemmas_tags:
            base_tag = self.base_tag(predicted_tag)
            if base_tag is None:
                return disamb_lemmas_tags[0][0] # oryginalny lemat
            else:
                try:
                    return self.gpp[(disamb_lemmas_tags[0][0],tuple(base_tag))] # szukamy "mianownika"
                except KeyError:
                    pass
        else:
            for lemma,tag in lemmas_tags:
                if self.same_lemma_tag(predicted_tag, tag):
                    base_tag = self.base_tag(tag)
                    if base_tag is None:
                        return lemma # orygianlny lemat
                    else:
                        try:
                            return self.gpp[(lemma, tuple(base_tag))] # szukamy "mianownika"
                        except KeyError:
                            break
        
        return form
    
class Lemmatisation():
    def __init__(self):
        self.lemmas={}

    def learn(self, path, stop=-1, start=0, ids=None):
        lemma_count=collections.defaultdict(lambda: collections.defaultdict(int))
        if ids is None:
            ids = []
        su = SerialUnpickler(open(path,'rb'), stop=stop,start=start, ids=ids)
        for paragraph in su:
            for sentence,sentence_orig in paragraph:
                for sample in sentence_orig:
                    # print(sample.features)
                    if 'lemma' in sample.features: #some samples doesnt have lemma, because not on gold segmentation
                        lemma_count[(sample.features['token'], sample.features['label'])][sample.features['lemma']]+=1

        for k, v in lemma_count.items():
            best = sorted(v.items(), key=lambda x: (x[1], x[0]), reverse=True)[0] #TODO kilka z taka sama statystyka
            self.lemmas[k]=best[0]

    def disambiguate(self, form, ilosc_disamb, predicted_tag):
        if form is None: return
        tag=predicted_tag
        
        ilosc_disamb=[x for x in ilosc_disamb if x[1]==predicted_tag]
        
        if not ilosc_disamb:
            ilosc_disamb=[(form, predicted_tag)]
        
        interp_lemmas = [lemma for (lemma,tag) in ilosc_disamb]
        if (form,tag) in self.lemmas:
            pred_lemma = self.lemmas[(form,tag)]
        else:
            #print('XXX', form, tag, interp_lemmas)
            pred_lemma = random.choice(interp_lemmas)
        return pred_lemma
        #print('\t%s\t%s\tdisamb' % (pred_lemma, tag))

    def save(self, path):
        f=open(path, 'wb')
        pickle.dump(self.lemmas, f)
        f.close()

    def load(self, path):
        self.lemmas = pickle.load(open(path, 'rb'))

def generate_arrays_from_file(path, unique_features_dict, feature_name, label_name, stop=-1, start=0, ids=None, keep_unaligned=False, keep_infinity=True):
    if ids is None:
        ids = []
    while 1:
        su = SerialUnpickler(open(path,'rb'), stop=stop,start=start, ids=ids)
        for paragraph in su:
            for sentence,sentence_orig in paragraph:
                X_sentence = []
                y_sentence = []
                if not sentence: continue # TODO

                same_segmentation = len(sentence)==len(sentence_orig) and len([sample for sample in sentence if 'label' in sample.features])
                if (not same_segmentation) and not keep_unaligned:
                    continue

                if keep_unaligned and same_segmentation:
                    for sample in sentence:
                        X_sentence.append(np.array(k_hot(sample.features[feature_name], unique_features_dict[feature_name])))
                        if label_name=='label':
                            y_sentence.append(np.array(k_hot([sample.features[label_name]], unique_features_dict[label_name])))
                        else:
                            y_sentence.append(np.array(k_hot(sample.features[label_name], unique_features_dict[label_name])))
                else:
                    for sample in sentence:
                        X_sentence.append(np.array(k_hot(sample.features[feature_name], unique_features_dict[feature_name])))
                    for sample in sentence_orig:
                        if label_name=='label':
                            y_sentence.append(np.array(k_hot([sample.features[label_name]], unique_features_dict[label_name])))
                        else:
                            y_sentence.append(np.array(k_hot(sample.features[label_name], unique_features_dict[label_name])))

                # print len(X_sentence), len(y_sentence)
                yield (X_sentence, y_sentence, sentence,sentence_orig)
        if not keep_infinity: break



def batch_generator(generator, batch_size=32, return_all=False, sort=False):
    batch_X = []
    batch_y = []
    sentences = []
    sentences_orig = []

    if sort:
        generator = sorted(generator, key=lambda x: len(x[0]))

    for X,y,sentence,sentence_orig in generator:
        batch_X.append(X)
        batch_y.append(y)
        sentences.append(sentence)
        sentences_orig.append(sentence_orig)
        if (not return_all) and len(batch_X)==batch_size:
            yield (batch_X, batch_y,sentences,sentences_orig)
            batch_X = []
            batch_y=[]
            sentences=[]
            sentences_orig=[]


    if batch_X or return_all:
        yield (batch_X, batch_y,sentences,sentences_orig)


def pad_generator(generator,sequence_length=20):
    for batch_X,batch_y, sentences,sentences_orig in generator:
        if not batch_X or not batch_y:
            continue

        #TODO pad multi inputs
        max_sentence_length = max([len(x) for x in batch_X])
        # print('max_sentence_length',max_sentence_length)
        yield (sequence.pad_sequences(batch_X, maxlen=max_sentence_length), sequence.pad_sequences(batch_y, maxlen=max_sentence_length), sentences,sentences_orig)

def pad_generatorE(generator,sequence_length=20):
    for batch_X,batch_y, sentences,sentences_orig,batch_X_e  in generator:
        if not batch_X or not batch_y or not batch_X_e:
            continue

        #TODO pad multi inputs
        max_sentence_length = max([len(x) for x in batch_X])
        # print('max_sentence_length',max_sentence_length)
        yield (sequence.pad_sequences(batch_X, maxlen=max_sentence_length), sequence.pad_sequences(batch_y, maxlen=max_sentence_length), sentences,sentences_orig, sequence.pad_sequences(batch_X_e, maxlen=max_sentence_length))


def Xy_generator(generator):
    for batch_X,batch_y,sequences,sentences_orig in generator:
        # print len(x), len(x[0])
        yield (batch_X,batch_y)

def k_hot(tags, unique_tags_dict, zero=0):
    # print
    result = [zero]*len(unique_tags_dict)
    for tag in tags:
        # print len(unique_tags_dict), unique_tags_dict[tag]
        try:
            result[unique_tags_dict[tag]]=1
        except KeyError:
            # print('KeyError:', tag, file=sys.stderr)
            pass
    return result

def predictions_to_classes(unique_tags_dict, predictions):
    out=[]
    classes = map(lambda k,v: k,sorted(unique_tags_dict.items(), key=lambda k,v: v))
    for i,a in enumerate(predictions):
        if a: out.append(classes[i])
    return out

class LossHistory(Callback):
    def __init__(self, evaluator, log_path, name=''):
        self.evaluator=evaluator
        self.file = open(log_path, 'wt')
        self.history = []
        self.name =name

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
        logs[self.name+'time']=end-self.start

        out = self.evaluator.evaluate(self.model)
        print(out)
        logs[self.name+'val_score'] = out[2]
        try:
            logs[self.name+'val_score_pos_em'] = out[5]
            logs[self.name+'val_score_k'] = out[8]
            logs[self.name+'val_score_u'] = out[11]
            logs[self.name+'val_score_upper'] = out[14]
            logs[self.name+'test_time'] = out[-1]
        except:
            pass
        self.write_str(sorted(logs.items(),key=lambda x: x[0]))
        self.history.append(logs)


    def write_str(self,logs):
        self.file.write(str(logs))
        self.file.write('\n')
        self.file.flush()

from termcolor import colored





class Evaluator:
    def __init__(self, test_data, unique_tags_dict):
        self.test_data = test_data
        self.unique_tags_dict = unique_tags_dict
        #self.tags=tags
        #self.init_evaluator()
        # print self.tags[0:5]
        # print self.divided_tags[0:5]



    def init_evaluator(self):
        self.divided_tags = map(lambda tag: TagsPreprocessor().create_tag2(':'.join(tag)), self.tags)

    def choose_best_tag(self, pred):
        out=[]
        for set_of_tags, tags in zip(self.divided_tags, self.tags):
            # print tag
            # set_of_tags = create_tags2(tag)
            probs=[]
            fuck=False
            for tag2 in set_of_tags:
                if tag2 not in self.unique_tags_dict:
                    probs.append(0.0)
                    fuck=True #???
                    continue # poprawic
                i = self.unique_tags_dict[tag2]
                probs.append(pred[i])

            if fuck: continue

            if probs:
                out.append((set_of_tags, probs, ':'.join(tags)))

        sorted_tags = sorted(out, key=lambda set_of_tags,probs,tags: geo_mean(probs), reverse=True)
        best_tag = sorted_tags[0]
        out2 = [0] * len(pred)
        # print 'BEST', best_tag
        for tag in best_tag[0]:
            out2[self.unique_tags_dict[tag]]=1
        return out2, sorted_tags

    def evaluate(self, model, verbose=False):
        # print len(self.test_data), len(self.test_data[0])
        X_train, y_train, sentences = self.test_data
        predictions = model.predict(X_train, verbose=1)
        tp2=0
        rc2=0
        pr2=0
        tp=fp=fn=tn=0
        i=0
        zero=0
        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=len(predictions))
        for a,b,e,sentence in zip(y_train,predictions,X_train, sentences):
            bar.next()
            # print tested_sentences[i]
            length = min(len(a), len(sentence))
            # print length
            # if length==0: continue

            if verbose:
                print(length)

            for j,(c,dd) in enumerate(list((zip(a,b)))[-length:]):
                # if j==length: break
                # print 'train', tested_sentences[i][0][-20:][j]
                # print X_train[i][j]

                # print 'train', sentences[i][2][-20:][j]
                # print 'train', sentences[i][0][-20:][j]


                d, sorted_tags=self.choose_best_tag(dd)


                sample = sentence[-20:][j].features
                predicted_full_tag = sorted_tags[0][2]
                true_full_tag = sample['label']

                if verbose:


                    # ap=predictions_to_classes(self.unique_tags_dict, c)
                    bp=predictions_to_classes(self.unique_tags_dict, d)
                    # print c,d
                    # if not ap: zero+=1

                    correct=False
                    # if set(bp)==set(sample['y']):
                    if predicted_full_tag == true_full_tag:
                        correct=True

                    print(colored('token: '+sample['token'], 'green' if correct else 'red'), '' if correct else '(INCORRECT)')
                    print('tags:', ', '.join(sorted(sample['tags'])))
                    print('tags2:', ', '.join(sorted(sample['tags2'])))
                    print('y:', ', '.join(map(lambda x: x if x in sample['tags2'] else colored(x, 'yellow'), sorted(sample['y']))), sample['label'])
                    for yy in sorted(sample['y']): #+['sg','pl','f','n','m1','m2','m3','nom','gen','dat','acc','loc','inst','subst','ger','ppas','adj','pact','num']:
                        print(yy, dd[self.unique_tags_dict[yy]])
                    print('predicted:', ', '.join(map(lambda x: x if x in sample['tags2'] else colored(x, 'yellow'), sorted(bp))))
                    for x,y,z in sorted_tags[0:100]:
                        # print type(y), type(sample['label'])
                        if z == sample['label']:
                            print(colored(z, 'green'), y, mean(y), geo_mean(y))
                            break
                        else:
                            if z in sample['tags']:
                                print(z, y, mean(y), geo_mean(y))
                            else:
                                print(colored(z, 'yellow'), y, mean(y), geo_mean(y))
                    print()

                # tap= tagset_utils.find_correct_tags2(ap)
                # tbp= tagset_utils.find_correct_tags2(bp)
                # print tap, ap
                # print tbp
                #
                # print
                # if len(tap)==0: print 'Error', tap
                # if len(tbp)==0: zero+=1
                # pr2+=len(tbp)
                # rc2+=len(tap)
                # tp2+=len(set(tbp)&set(tap))


                # print c, d
                # if list(c)==d:

                # print 'XXXX', predicted_full_tag, true_full_tag
                if predicted_full_tag == true_full_tag:
                    tp+=1
                else:
                    tn+=1
            i+=1

            if verbose: print()

        bar.finish()

        return tp, tn, float(tp)/(tp+tn)


class Accuracy(object):
    def __init__(self):
        self.tp=0
        self.tn=0

    def accuracy(self):
        l=self.tp+self.tn
        if l==0:
            return 0.0
        return float(self.tp)/l

    def __repr__(self):
        return str([self.tp,self.tn])

class SimpleEvaluator(Evaluator):
    def evaluate(self, model, verbose=False):
        # print len(self.test_data), len(self.test_data[0])
        X_train, y_train, sentences, sentences_orig = self.test_data
        predictions = model.predict(X_train, verbose=1)

        tp2=0
        rc2=0
        pr2=0
        tp=fp=fn=tn=0
        acc = Accuracy()
        i=0
        zero=0
        errors = []
        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s - %(elapsed_td)s', max=len(predictions))
        for a,b,e,sentence in zip(y_train,predictions,X_train, sentences):
            bar.next()
            # print tested_sentences[i]
            length = min(len(a), len(sentence))
            # print length
            # if length==0: continue

            if verbose:
                print(length)

            for j,(c,dd) in enumerate(list((zip(a,b)))[-length:]):
                # if j==length: break
                # print 'train', tested_sentences[i][0][-20:][j]
                # print X_train[i][j]

                # print 'train', sentences[i][2][-20:][j]
                # print 'train', sentences[i][0][-20:][j]
                # print c
                # print dd
                # print

                cmax_index, cmax_value = max(enumerate(c), key=lambda x: x[1])
                max_index, max_value = max(enumerate(dd), key=lambda x: x[1])


                if verbose:

                    correct = (max_index == cmax_index)
                    sample = sentence[-20:][j].features

                    predicted_tag= self.unique_tags_dict.keys()[self.unique_tags_dict.values().index(max_index)]
                    true_tag  = self.unique_tags_dict.keys()[self.unique_tags_dict.values().index(cmax_index)]

                    print(colored('token: '+sample['token'], 'green' if correct else 'red'), '' if correct else '(INCORRECT)')
                    print('tags:', ', '.join(sorted(sample['tags'])))
                    print('tags4:', ', '.join(sorted(sample['tags4e'])))
                    # print 'y:', ', '.join(map(lambda x: x if x in sample['tags4'] else colored(x, 'yellow'), sorted(sample['label']))), sample['label']
                    # for yy in sorted(sample['label']): #+['sg','pl','f','n','m1','m2','m3','nom','gen','dat','acc','loc','inst','subst','ger','ppas','adj','pact','num']:
                    #     print yy, dd[self.unique_tags_dict[yy]]
                    # print 'predicted:', ', '.join(map(lambda (x,y): str((x,y)) if x in sample['tags4'] else colored(str((x,y)), 'yellow'), sorted(sorted_tags, key=lambda (x,y): y, reverse=True)))
                    # print 'predicted:', predicted_full_tag
                    print('true\t\t', colored(true_tag,'yellow') if true_tag not in sample['tags'] else true_tag)
                    print('predicted\t', colored(predicted_tag,'yellow') if predicted_tag not in sample['tags'] else predicted_tag)

                    sorted_indexes = sorted(enumerate(dd), key=lambda x: x[1], reverse=True)
                    stop=0
                    for index,value in sorted_indexes[:30]:
                        pt = self.unique_tags_dict.keys()[self.unique_tags_dict.values().index(index)]
                        if pt == true_tag:
                            print(colored(pt,'green'), value)
                            stop = 1
                        else:
                            print(colored(pt,'yellow') if pt not in sample['tags'] else pt, value)
                        if stop>0: stop+=1
                        if stop==5: break
                    print()
                    if not correct:
                        errors.append((sample['token'], true_tag, predicted_tag))

                if max_index == cmax_index:
                    acc.tp+=1
                else:
                    acc.tn+=1
            i+=1

            if verbose: print()

        bar.finish()

        if verbose:
            x= collections.defaultdict(int)
            for form, true, pred in errors:
                x[form]+=1

            for a,b in sorted(x.items(), key=lambda a,b: b):
                print(a, b)


        return acc.tp, acc.tn, acc.accuracy()

def to_plain(batch, preds,file_test,file_pred,unique_tags_dict):
    X_train2, y_train2, sentences2,sentences_orig2 = batch
    for x,y,s1,s2,pr in zip(X_train2, y_train2, sentences2,sentences_orig2, preds):
        for t,p in zip(s1, list(pr)[-len(s1):]):
            # print(t)
            form = t.features['token']
            form = form.replace('\t','').replace('­','')
            # print(t.features)
            # if 'space_before' in t.features['space_before']:
            #     form=' '+form
            file_pred.write('%s\t%s\n' % (form, 'space' if 'space_before' in t.features['space_before'] else 'none'))
            max_index, max_value = max(enumerate(p), key=lambda x: x[1])
            predicted_tag= list(unique_tags_dict.keys())[list(unique_tags_dict.values()).index(max_index)]
            file_pred.write( '\t%s\t%s\tdisamb\n' % (form, predicted_tag))
        file_pred.write('\n')

        for t in s2:
            form = t.features['token']
            form = form.replace('\t','').replace('­','')
            file_test.write('%s\t%s\n' % (form, 'space' if 'space_before' in t.features['space_before'] else 'none'))
            file_test.write( '\t%s\t%s\tdisamb\n' % (form, t.features['label']))
        file_test.write('\n')

    file_pred.flush()
    file_test.flush()



class DataGenerator():
    def __init__(self, data_path, unique_features_dict, pref, ids):
        self.data_path=data_path
        self.unique_features_dict=unique_features_dict
        self.pref=pref
        self.ids=ids

    def _create_generator(self):
        return pad_generator(batch_generator(
                    generate_arrays_from_file(self.data_path, self.unique_features_dict, self.pref['feature_name'], self.pref['label_name'],
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

        file_test = open('test.plain','wt')
        file_pred = open('pred.plain','wt')

        start = time.time()


        for batch in self.test_data.get():
            X_train2, y_train2, sentences2,sentences_orig2 = batch

            X_train.extend(X_train2)
            y_train.extend(y_train2)
            sentences.extend(sentences2)
            sentences_orig.extend(sentences_orig2)
            preds = model.predict_on_batch(X_train2)
            predictions.extend(preds)

            if plain: to_plain(batch, preds, file_test,file_pred, self.unique_tags_dict)

        #TODO if generator then calculate acc on batches

        self.predictions = predictions

        end = time.time()
        test_time=len(X_train)/float(end-start)


        acc_lower=Accuracy()
        acc_upper=Accuracy()
        acc_k_lower=Accuracy()
        acc_u_lower=Accuracy()
        acc_pos_em=Accuracy()
        acc_verb_em=Accuracy()

        i=0
        zero=0
        errors = []
        bar = Bar('Processing %s' % str(self.__class__.__name__), suffix = '%(index)d/%(max)d %(percent).1f%% - %(eta_td)s - %(elapsed_td)s', max=len(predictions))
        for a,b,e,sentence,sentence_orig in zip(y_train,predictions,X_train, sentences, sentences_orig):
            bar.next()
            # print tested_sentences[i]
            length = min(len(a), len(sentence))
            # print length
            # if length==0: continue

            if verbose:
                print(length)

            sentence_valid = True


            #predictions
            pred = []
            ref = []
            for predd, sample in zip(list(b)[-len(sentence):], sentence):
                max_index, max_value = max(enumerate(predd), key=lambda x: x[1])# mozna wczesniej obliczyc jako argmax
                pred.append((sample.features['token'], max_index, sample.features['space_before'][0] == 'space_before', 'ign' in sample.features['tags']))

            # print(pred)

            #reference
            for predd, sample in zip(list(a)[-len(sentence_orig):], sentence_orig):
                max_index, max_value = max(enumerate(predd), key=lambda x: x[1])
                ref.append((sample.features['token'], max_index, sample.features['space_before'][0] == 'space_before'))

            # if len(pred)!=len(ref): print(pred,ref)
            # align(pred,ref)


            # print(pred,ref)
            # print('dupa',acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em)
            accuracy(pred, ref, acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em)
            # print('dupa2',acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em)

            i+=1


        bar.finish()




        return acc_lower.tp, acc_lower.tn, acc_lower.accuracy(), acc_pos_em.tp, acc_pos_em.tn, acc_pos_em.accuracy(), acc_k_lower.tp, acc_k_lower.tn, acc_k_lower.accuracy(), acc_u_lower.tp, acc_u_lower.tn, acc_u_lower.accuracy(), acc_upper.tp, acc_upper.tn, acc_upper.accuracy(), test_time


def text(buffer):
    return ''.join([' ' + x[0].replace('\xa0', '') if x[2] is True or x[2] == 'space' else x[0].replace('\xa0', '') for x in buffer])

def text_verbose(buffer):
    return ''.join([' ' + x[0].features['token'].replace('\xa0', '') if x[2] is True or x[2] == 'space' else x[0].features['token'].replace('\xa0', '') for x in buffer])

def text_sample(buffer):
    return ''.join([' ' + x.features['token'].replace('\xa0', '') if x.features['space_before'] is True or x.features['space_before'] == 'space' else x.features['token'].replace('\xa0', '') for x in buffer])


def accuracy(pred, ref, acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em):
    sentence_valid=True
    for p,r in align(pred,ref):
        # print(p,r)
        if len(p)==len(r) and len(p)==1:
            if p[0][1]==r[0][1]:
                acc_lower.tp+=1
                acc_upper.tp+=1
                if p[0][3]: #unknown
                    acc_u_lower.tp+=1
                else:
                    acc_k_lower.tp+=1
            else:
                sentence_valid=False
                acc_lower.tn+=1
                acc_upper.tn+=1
                if p[0][3]: #unknown
                    acc_u_lower.tn+=1
                else:
                    acc_k_lower.tn+=1
        else:
            sentence_valid=False
            acc_lower.tn+=len(r)
            acc_upper.tp+=len(r)
            acc_k_lower.tn+=len(r)

    if sentence_valid:
        acc_pos_em.tp+=1
    else:
        acc_pos_em.tn+=1

def accuracy_verbose(pred, ref, acc_lower, acc_upper, acc_k_lower, acc_u_lower, acc_pos_em, acc_pred_not_in_tags, acc_ref_not_in_tags, classes):
    sentence_valid=True
    raw_s=text_verbose(pred)

    for p,r in align_verbose(pred,ref):
        pred_sample = p[0][0]
        ref_sample = r[0][0]
        pmax_index, pmax_value = max(enumerate(p[0][1]), key=lambda x: x[1])
        rmax_index, rmax_value = max(enumerate(r[0][1]), key=lambda x: x[1])
        pred_not_in_tags = classes[pmax_index] not in pred_sample.features['tags']
        ign = 'ign' in pred_sample.features['tags']
        if len(p)==len(r) and len(p)==1:

            if not ign:
                if classes[rmax_index] not in pred_sample.features['tags']:
                    acc_ref_not_in_tags.tp+=1
                else:
                    acc_ref_not_in_tags.tn+=1


            if pmax_index==rmax_index:
                acc_lower.tp+=1
                acc_upper.tp+=1
                if not ign and pred_not_in_tags:
                    acc_pred_not_in_tags.tp+=1

                if p[0][3]: #unknown
                    acc_u_lower.tp+=1
                else:
                    acc_k_lower.tp+=1
            else:
                if not ign and pred_not_in_tags:
                    acc_pred_not_in_tags.tn+=1

                sss = sorted(enumerate(p[0][1]), key=lambda x: x[1], reverse=True)

                print(raw_s)
                print(pred_sample.features['token'])
                print('True: \t\t%s' % classes[rmax_index])
                print('Predicted: \t%s' % classes[pmax_index])
                print(pred_sample.features['tags'])
                print(pred_sample.features['tags4e3'])
                for index,value in sss[:10]:
                    desc=''
                    if index==rmax_index: desc = 'TRUE '
                    if classes[index] not in pred_sample.features['tags']:
                        desc+='NOT_IN_TAGS'
                    print(' - %.4f %s \t%s' % (value, classes[index], desc))
                print()



                sentence_valid=False
                acc_lower.tn+=1
                acc_upper.tn+=1
                if p[0][3]: #unknown
                    acc_u_lower.tn+=1
                else:
                    acc_k_lower.tn+=1
        else:
            sentence_valid=False
            acc_lower.tn+=len(r)
            acc_upper.tp+=len(r)
            acc_k_lower.tn+=len(r)

    if sentence_valid:
        acc_pos_em.tp+=1
    else:
        acc_pos_em.tn+=1

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

            pred_buffer=[]
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
        yield (pred_buffer+pred, rest)
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

            pred_buffer=[]
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
        yield (pred_buffer+pred, rest)

def reverse_func(x):
    import keras.backend as K
    assert K.ndim(x) == 3, "Should be a 3D tensor."
    rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
    return K.permute_dimensions(rev, (1, 0, 2))

def count_sentences(path, ids=None):
    if ids is None:
        ids = []
    count=0
    su = SerialUnpickler(open(path,'rb'), ids=ids)
    for paragraph in su:
        for sentence in paragraph:
            count+=1
    return count


def assign_ids(unique_values):
    classes = {}
    for i,sign in enumerate(unique_values):
        classes[sign]=i
    return classes

def unix_uniq(l):
    packed = []

    for el in l:
        if not packed or packed[-1]!=el:
            packed.append(el)
    return ''.join(packed)


def shape(word): #TODO zredukowac czas
    word = regex.sub(u'(?V1)\p{Lowercase}', 'l', word, flags=regex.U) #80%
    word = regex.sub(u'(?V1)\p{Uppercase}','u', word, flags=regex.U)
    word = regex.sub(u'\p{gc=Decimal_Number}','d', word, flags=regex.U)
    word = regex.sub(u'[^A-Za-z0-9]','x', word, flags=regex.LOCALE)
    return unix_uniq(list(word))

def results_to_jsonl(results):
    print(results_to_jsonl_str(results))

def results_to_xces(results):
    print(results_to_xces_str(results))

def results_to_plain(results):
    print(results_to_plain_str(results))

def results_to_conll(results):
    print(results_to_conll_str(results))

def results_to_conllu(results):
    print(results_to_conllu_str(results))

def results_to_conll_str(results):
    result_str = ""
    for sentence in results:
        for token in sentence:
            try:
                start=token['start']
            except KeyError:
                start=''

            try:
                end=token['end']
            except KeyError:
                end=''

            result_str += ('%s\t%s\t%s\t%s\t%s\t%s\n' % (token['token'], token['lemmas'][0], 1 if token['sep']=='space' else 0, token['tag'], start, end))
        result_str += "\n"
    return result_str

def results_to_jsonl_str(results):
    fp = io.StringIO()
    with jsonlines.Writer(fp) as writer:
        for sentence in results:
            ss=[(token['token'], token['lemmas'][0], token['tag']) for token in sentence]
            writer.write(ss)
    return fp.getvalue()


def results_to_conllu_str(results):
    result_str = ""
    for sentence in results:
        for i,token in enumerate(sentence):
            result_str += ('%s\t%s\t%s\t_\t%s\t_\t_\t_\t_\t_\n' % (i+1, token['token'], token['lemmas'][0], token['tag']))
        result_str += "\n"
    return result_str

def results_to_plain_str(results):
    result_str = ""
    for sentence in results:
        for token in sentence:
            result_str += ('%s\t%s' % (token['token'], token['sep'])) + "\n"
            for lemma in token['lemmas']:
                result_str += ('\t%s\t%s\tdisamb' % (lemma, token['tag'])) + "\n"
        result_str += "\n"
    return result_str

def results_to_xces_str(results):
    result_str = ""
    result_str += ('<?xml version="1.0" encoding="UTF-8"?>\n'
          '<!DOCTYPE cesAna SYSTEM "xcesAnaIPI.dtd">\n'
          '<cesAna xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0" type="lex disamb">\n'
          '<chunkList>') + "\n"

    for sentence in results:
        result_str += (' <chunk type="p">\n'
              '  <chunk type="s">') + "\n"
        for token in sentence:
            if token['sep']=='none':
                result_str += ('   <ns/>') + "\n"
            result_str += ('   <tok>') + "\n"
            result_str += ('    <orth>%s</orth>' % escape_xml(token['token'])) + "\n"
            for lemma in token['lemmas']:
                result_str += ('    <lex disamb="1"><base>%s</base><ctag>%s</ctag></lex>' % (escape_xml(lemma),
                    token['tag'])) + "\n"
            result_str += ('   </tok>') + "\n"
        result_str += ('  </chunk>\n'
              ' </chunk>') + "\n"
    result_str += ('</chunkList>\n'
        '</cesAna>') + "\n"
    return result_str

def escape_xml(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;').replace('\'','&apos;')


import jsonlines


def get_morfeusz():
    import morfeusz2
    morf = morfeusz2.Morfeusz(
        analyse=True, #load analyze dictionary
        generate=False, #dont load generator dictionary
        expand_tags=True, #expand tags (return tags without dots)
        aggl='isolated', # 'isolated' - token 'm' has aglt interpretation, token 'np' has brev interpretation
        praet='composite', # aglt and 'by' are not divided
    #    whitespace=morfeusz2.KEEP_WHITESPACES
    )
    return morf

def analyze_token(morf, token):
    segment_interpretations = morf.analyse(token)
    #if token is tokenized then take all interpretation of segments starting from beginning of the token
    interpretations=[]
    for start, end, interpretation in segment_interpretations:
        if start==0:
            form, lemma, tag, domain, qualifier = interpretation
            interpretations.append((lemma, tag))

    return interpretations

def analyze_tokenized(morf, paragraphs):
    for p in paragraphs:
        for s in p:
            for token in s:
                interpretations=analyze_token(morf, token.form)
                print(interpretations)
                token.interpretations.extend([Form(base, ctag) for (base, ctag) in interpretations])
        yield p
