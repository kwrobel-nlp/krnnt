#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List


class Paragraph:
    sentences: List['Sentences']

    __slots__ = ['sentences', 'concraft']

    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence: 'Sentence'):
        self.sentences.append(sentence)

    def __iter__(self):
        return self.sentences.__iter__()

    def text(self) -> str:
        raw = ''.join([sentence.text() for sentence in self.sentences])
        try:
            if self.sentences[0].tokens[0].space_before:
                return raw[1:]
            else:
                return raw
        except:
            return raw


class Sentence:
    tokens: List['Token']

    __slots__ = ['tokens']

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

    __slots__ = ['form', 'space_before', 'interpretations', 'gold_form', 'start', 'end']

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

    def __eq__(self, y):
        return self.lemma == y.lemma and self.tags == y.tags

    def __hash__(self):
        return hash((self.lemma, self.tags))
