#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce

import pickle
from typing import BinaryIO, Iterable, List, TypeVar


class SerialPickler:
    def __init__(self, file: BinaryIO, mode=3): #don't work with protocol 4
        self.file = file
        self.p = pickle.Pickler(file, mode)

    def add(self, obj):
        self.p.dump(obj)
        self.p.memo.clear()

    def extend(self, objs: Iterable):
        for obj in objs:
            self.p.dump(obj)
            self.p.memo.clear()

    def close(self):
        self.file.close()


class SerialUnpickler:
    def __init__(self, file: BinaryIO, stop=-1, start=0, ids=None):
        if ids is None:
            ids = []
        self.file = file
        self.p = pickle.Unpickler(file)
        self.c = 0
        self.stop = stop
        self.start = start
        self.ids = set(ids)

    def __iter__(self):
        if self.ids:
            return self.__iter2()
        else:
            return self.__iter1()

    def __iter1(self):
        while True:
            try:
                if self.c == self.stop:
                    break
                self.c += 1
                x = self.p.load()
                if self.c - 1 < self.start:
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
                self.c += 1
            except EOFError:
                break


def count_samples(path: str) -> int:
    file = open(path, 'rb')
    su = SerialUnpickler(file)

    count = 0
    for paragraph in su:
        count += 1

    return count


def uniq(seq: Iterable) -> List:
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def flatten(l: Iterable) -> List:
    return [item for sublist in l for item in sublist]


class Paragraph:
    sentences: List['Sentences']

    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence: 'Sentence'):
        self.sentences.append(sentence)

    def __iter__(self):
        return self.sentences.__iter__()

    def text(self) -> str:
        raw = ''.join([sentence.text() for sentence in self.sentences])
        return raw[1:]  # omit first separator


class Sentence:
    tokens: List['Token']

    def __init__(self):
        self.tokens = []

    def add_token(self, token: 'Token'):
        self.tokens.append(token)

    def text(self) -> str:
        return ''.join(map(lambda token: ' ' + token.form if token.space_before else token.form, self.tokens))

    def __iter__(self):
        return self.tokens.__iter__()


class Token:
    form: str
    interpretations: List['Form']
    gold_form: 'Form'

    def __init__(self):
        self.form = None
        self.space_before = None
        self.interpretations = []
        self.gold_form = None

    def add_interpretation(self, interpretation: 'Form'):
        self.interpretations.append(interpretation)

    def __str__(self):
        return 'Token(%s, %s)' % (self.form, self.interpretations)


class Form:
    def __init__(self, lemma: str, tags: str):
        self.lemma = lemma
        self.tags = tags

    def __str__(self):
        return 'Form(%s, %s)' % (self.lemma, self.tags)

    def __eq__(x, y):
        return x.lemma == y.lemma and x.tags == y.tags

    def __hash__(self):
        return hash((self.lemma, self.tags))


TNum = TypeVar('TNum', int, float)


def geo_mean(numbers: List[TNum]) -> float:
    return (reduce(lambda x, y: x * y, numbers)) ** (1.0 / len(numbers))


def mean(numbers: List[TNum]) -> float:
    return (reduce(lambda x, y: x + y, numbers)) * (1.0 / len(numbers))
