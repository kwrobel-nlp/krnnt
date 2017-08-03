#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce

poses = {u'pant', u'adjp', u'adja', u'ppron3', u'adv', u'siebie', u'adj', u'ger', u'pcon', u'interp', u'num', u'adjc',
         u'comp', u'qub', u'burk', u'praet', u'numcol', u'fin', u'winien', u'pred', u'subst', u'ppas', u'prep', u'xxx',
         u'pact', u'imps', u'impt', u'depr', u'interj', u'inf', u'ppron12', u'conj', u'aglt', u'brev', u'bedzie',
         u'ign'}

try:
    import cPickle as pickle
except ImportError:
    import pickle

class SerialPickler:
    def __init__(self, file, mode=3):
        self.file = file
        self.p = pickle.Pickler(file, mode)

    def add(self, obj):
        self.p.dump(obj)
        self.p.memo.clear()

    def extend(self, objs):
        for obj in objs:
            self.p.dump(obj)
            self.p.memo.clear()

    def close(self):
        self.file.close()

class SerialUnpickler:
    def __init__(self, file, stop=-1, start=0, ids=None):
        if ids is None:
            ids=[]
        self.file = file
        self.p = pickle.Unpickler(file)
        self.c=0
        self.stop=stop
        self.start=start
        self.ids=set(ids)

    def __iter__(self):
        if self.ids:
            return self.__iter2()
        else:
            return self.__iter1()

    def __iter1(self):
        while True:
            try:
                if self.c==self.stop:
                    break
                self.c+=1
                x = self.p.load()
                if self.c-1<self.start:
                    continue

                # print self.c
                yield x
            except EOFError:
                break

    def __iter2(self):
        while True:
            try:
                x = self.p.load()
                if self.c in self.ids:
                    yield x
                self.c+=1
            except EOFError:
                break

def count_samples(path):
    file = open(path,'rb')
    su = SerialUnpickler(file)

    count=0
    for paragraph in su:
        count+=1

    return count

def uniq(seq):
    seen = set()
    return [ x for x in seq if not (x in seen or seen.add(x))]

# def shape(word):
#     word = regex.sub(u'(?V1)\p{Lowercase}','l', word, regex.U)
#     word = regex.sub(u'(?V1)\p{Uppercase}','u', word, regex.U)
#     word = regex.sub(u'\p{gc=Decimal_Number}','d', word, regex.U)
#     word = regex.sub(u'[^A-Za-z0-9]','x', word, regex.LOCALE)
#     return unix_uniq(list(word))
#
# def unix_uniq(l):
#     packed = []
#
#     for el in l:
#         if not packed or packed[-1]!=el:
#             packed.append(el)
#     return ''.join(packed)

def flatten(l):
    return [item for sublist in l for item in sublist]

class Paragraph:
    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def __iter__(self):
        return self.sentences.__iter__()

class Sentence:
    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def text(self):
        return ''.join(map(lambda token: ' '+token.form if token.space_before else token.form, self.tokens))

    def __iter__(self):
        return self.tokens.__iter__()

class Token:
    def __init__(self):
        self.form = None
        self.space_before = None
        self.interpretations = []
        self.gold_form = None

    def add_interpretation(self, interpretation):
        self.interpretations.append(interpretation)

    def __str__(self):
        return 'Token(%s, %s)' % (self.form, self.interpretations)

def geomean(nums):
    return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))


class Form:
    def __init__(self, base_form, tags):
        self.lemma = base_form
        self.tags = tags

    def __str__(self):
        return 'Form(%s, %s)' % (self.lemma, self.tags)

    def __eq__(x, y):
        return x.lemma == y.lemma and x.tags==y.tags

    def __hash__(self):
        return hash((self.lemma, self.tags))

def create_tags2(tags):
    pos = tags[0]
    tags = tags[1:]
    tags2 = []

    if not tags:
        tags2.append(pos)
    for tag in tags:
        tags2.append(pos+'-'+tag)

    return uniq(tags2)

def mean(nums):
    return (reduce(lambda x, y: x+y, nums))*(1.0/len(nums))

def choose_best_tag(unique_tags_dict, all_tags, pred):
    out=[]
    for tag in all_tags:
        # print tag
        set_of_tags = create_tags2(tag)
        probs=[]
        fuck=False
        for tag2 in set_of_tags:
            if tag2 not in unique_tags_dict:
                probs.append(0.0)
                fuck=True #???
                continue # poprawic
            i = unique_tags_dict[tag2]
            probs.append(pred[i])

        if fuck: continue

        if probs:
            out.append((set_of_tags, probs))

    sorted_tags = sorted(out, key=lambda set_of_tags,probs: mean(probs), reverse=True)
    best_tag = max(out, key=lambda set_of_tags,probs: mean(probs))
    out2 = [0] * len(pred)
    # print 'BEST', best_tag
    for tag in best_tag[0]:
        out2[unique_tags_dict[tag]]=1
    return out2, sorted_tags