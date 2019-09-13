import copy

from krnnt.aglt import rewrite_praet, remove_aglt, rule1, rule3

paragraph = [[{'token': 'Zrobił', 'sep': 'newline', 'prob': 0.99893767, 'document_id': 0, 'tag': 'praet:sg:m1:perf',
               'lemmas': ['zrobić'], 'start': 0, 'end': 6},
              {'token': 'by', 'sep': 'none', 'prob': 0.99995995, 'document_id': 0, 'tag': 'qub', 'lemmas': ['by'],
               'start': 6, 'end': 8},
              {'token': 'm', 'sep': 'none', 'prob': 0.99999964, 'document_id': 0, 'tag': 'aglt:sg:pri:imperf:nwok',
               'lemmas': ['być'], 'start': 8, 'end': 9},
              {'token': 'to', 'sep': 'space', 'prob': 0.99836236, 'document_id': 0, 'tag': 'subst:sg:acc:n',
               'lemmas': ['to'], 'start': 10, 'end': 12},
              {'token': '.', 'sep': 'none', 'prob': 1.0, 'document_id': 0, 'tag': 'interp', 'lemmas': ['.'],
               'start': 12, 'end': 13}], [
                 {'token': 'Czy', 'sep': 'space', 'prob': 0.9954, 'document_id': 0, 'tag': 'qub', 'lemmas': ['czy'],
                  'start': 14, 'end': 17},
                 {'token': 'by', 'sep': 'space', 'prob': 0.99997103, 'document_id': 0, 'tag': 'qub', 'lemmas': ['by'],
                  'start': 18, 'end': 20},
                 {'token': 'm', 'sep': 'none', 'prob': 0.9999995, 'document_id': 0, 'tag': 'aglt:sg:pri:imperf:nwok',
                  'lemmas': ['być'], 'start': 20, 'end': 21},
                 {'token': 'to', 'sep': 'space', 'prob': 0.9997142, 'document_id': 0, 'tag': 'subst:sg:acc:n',
                  'lemmas': ['to'], 'start': 22, 'end': 24},
                 {'token': 'zrobił', 'sep': 'space', 'prob': 0.9960322, 'document_id': 0, 'tag': 'praet:sg:m1:perf',
                  'lemmas': ['zrobić'], 'start': 25, 'end': 31},
                 {'token': '?', 'sep': 'none', 'prob': 0.99999976, 'document_id': 0, 'tag': 'interp', 'lemmas': ['?'],
                  'start': 31, 'end': 32}], [
                 {'token': 'Zrobił', 'sep': 'space', 'prob': 0.99980086, 'document_id': 0, 'tag': 'praet:sg:m1:perf',
                  'lemmas': ['zrobić'], 'start': 33, 'end': 39},
                 {'token': 'em', 'sep': 'none', 'prob': 0.99999976, 'document_id': 0, 'tag': 'aglt:sg:pri:imperf:wok',
                  'lemmas': ['być'], 'start': 39, 'end': 41},
                 {'token': 'to', 'sep': 'space', 'prob': 0.98835105, 'document_id': 0, 'tag': 'subst:sg:acc:n',
                  'lemmas': ['to'], 'start': 42, 'end': 44},
                 {'token': '.', 'sep': 'none', 'prob': 1.0, 'document_id': 0, 'tag': 'interp', 'lemmas': ['.'],
                  'start': 44, 'end': 45}], [
                 {'token': 'Aby', 'sep': 'space', 'prob': 0.999451, 'document_id': 0, 'tag': 'comp', 'lemmas': ['aby'],
                  'start': 46, 'end': 49},
                 {'token': 'm', 'sep': 'none', 'prob': 0.99999976, 'document_id': 0, 'tag': 'aglt:sg:pri:imperf:nwok',
                  'lemmas': ['być'], 'start': 49, 'end': 50},
                 {'token': 'to', 'sep': 'space', 'prob': 0.9997446, 'document_id': 0, 'tag': 'subst:sg:acc:n',
                  'lemmas': ['to'], 'start': 51, 'end': 53},
                 {'token': 'zrobił', 'sep': 'space', 'prob': 0.994592, 'document_id': 0, 'tag': 'praet:sg:m1:perf',
                  'lemmas': ['zrobić'], 'start': 54, 'end': 60},
                 {'token': '?', 'sep': 'none', 'prob': 0.9999999, 'document_id': 0, 'tag': 'interp', 'lemmas': ['?'],
                  'start': 60, 'end': 61}]]


def test_rewrite_praet():
    sentence1 = copy.deepcopy(paragraph[0])

    rewrite_praet(sentence1[2], sentence1[0])
    assert sentence1[0]['tag'] == 'praet:sg:m1:pri:perf'


def test_rewrite_cond():
    sentence1 = copy.deepcopy(paragraph[0])
    rewrite_praet(sentence1[2], sentence1[0], sentence1[1])
    assert sentence1[0]['tag'] == 'cond:sg:m1:pri:perf'


def test_rule1_cond():
    sentence1 = copy.deepcopy(paragraph[0])

    remove_aglt(sentence1, [rule1])
    assert sentence1[0]['tag'] == 'cond:sg:m1:pri:perf'
    assert sentence1[1]['token'] != 'by'
    assert sentence1[2]['token'] != 'm'


def test_rule1_praet():
    sentence1 = copy.deepcopy(paragraph[2])

    remove_aglt(sentence1, [rule1])
    assert sentence1[0]['tag'] == 'praet:sg:m1:pri:perf'
    assert sentence1[1]['token'] != 'm'


def test_rule3_1():
    sentence1 = copy.deepcopy(paragraph[1])

    remove_aglt(sentence1, [rule1, rule3])

    assert sentence1[3]['tag'] == 'praet:sg:m1:pri:perf'


def test_rule3_2():
    sentence1 = copy.deepcopy(paragraph[3])

    remove_aglt(sentence1, [rule1, rule3])

    assert sentence1[2]['tag'] == 'praet:sg:m1:pri:perf'
