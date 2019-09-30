import copy

from krnnt.aglt import rewrite_praet, remove_aglt, rule1, rule3, rule1b
from krnnt.blanks import remove_blanks

sentence = [

        {'token': '200', 'sep': 'newline', 'tag': 'num:pl:nom:m2:rec',
         'lemmas': ['200'], 'start': 0, 'end': 3},
        {'token': '.', 'sep': 'none', 'tag': 'blank', 'lemmas': ['.'],
         'start': 3, 'end': 4},
        {'token': '000', 'sep': 'none', 'tag': 'blank',
         'lemmas': ['000'], 'start': 4, 'end': 7},
        {'token': 'zł', 'sep': 'space', 'tag': 'brev:npun',
         'lemmas': ['złoty'], 'start': 8, 'end': 10}
]


def test_remove_blanks():
    sentence1 = copy.deepcopy(sentence)
    remove_blanks(sentence1)
    print(sentence1)

    assert len(sentence1)==2


    assert sentence1[0]['tag'] == 'num:pl:nom:m2:rec'
    assert sentence1[0]['token'] == '200.000'
    assert sentence1[0]['start'] == 0
    assert sentence1[0]['end'] == 7

    assert sentence1[1] == sentence[-1]