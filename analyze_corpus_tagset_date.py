#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import glob
import sys

from argparse import ArgumentParser

from krnnt.readers import read_xces

usage = """%prog CORPUS

Analyze corpus for changes in dictionary.
"""

if __name__ == '__main__':
    parser = ArgumentParser(usage=usage)
    parser.add_argument('corpus_path', type=str, help='path to XCES corpus (or path with wildcard)')
    args = parser.parse_args()

    # read corpus
    stats_forms = collections.defaultdict(int)
    stats_tags = collections.defaultdict(int)

    for path in glob.iglob(args.corpus_path):
        print(path, file=sys.stderr)
        for paragraph in read_xces(path):
            for sentence in paragraph:
                for token in sentence:
                    form = token.form
                    try:
                        tag = token.gold_form.tags
                        stats_forms[(form, tag)] += 1
                        stats_tags[tag] += 1
                    except:  # no disamb
                        print("Missing disamb", path, form, file=sys.stderr)
                        pass

    # stats
    print('Tokens: %s' % sum(stats_forms.values()))
    print('Unique tokens: %s' % len(set([x[0] for x in stats_forms])))
    print('Unique token+tag: %s' % len(stats_forms))
    print('Unique tags: %s' % len(stats_tags))
    print()

    # analyse
    TAGS = 'tags'
    FORMS = 'forms'
    POSITIVE = '+'
    NEGATIVE = '-'
    checks = {}

    checks['20141013 brev -> brev:n?pun'] = {
        TAGS: {
            POSITIVE: [lambda tag: tag in ('brev:pun', 'brev:npun')],
            NEGATIVE: [lambda tag: tag == 'brev']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }
    checks['20150127 siebie, ale w NKJP jest'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag.startswith('siebie:')]
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['20150617 _'] = {
        TAGS: {
            POSITIVE: [lambda tag: '_' not in tag],
            NEGATIVE: [lambda tag: '_' in tag]
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['20160126 pacta'] = {
        TAGS: {
            POSITIVE: [lambda tag: tag == 'pacta'],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['20170301 bardzo:adv:pos, bardziej:adv:com'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [lambda form, tag: form == 'bardziej' and tag == 'adv:com',
                       lambda form, tag: form == 'bardzo' and tag == 'adv:pos'],
            NEGATIVE: [lambda form, tag: form == 'bardziej' and tag == 'adv',
                       lambda form, tag: form == 'bardzo' and tag == 'adv']
        }
    }
    checks['20170409 n1,n2,p1,p2,p3 -> n'] = {
        TAGS: {
            POSITIVE: [lambda tag: {'n'} & set(tag.split(':'))],
            NEGATIVE: [lambda tag: {'n1', 'n2', 'n3', 'p1', 'p2', 'p3'} & set(tag.split(':'))]
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }
    checks['no col,ncol,pt'] = {
        TAGS: {
            POSITIVE: [lambda tag: not {'col', 'ncol', 'pt'} & set(tag.split(':'))],
            NEGATIVE: [lambda tag: {'col', 'ncol', 'pt'} & set(tag.split(':'))]
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }
    checks['20170430 num:comp -> numcomp'] = {
        TAGS: {
            POSITIVE: [lambda tag: tag == 'numcomp'],
            NEGATIVE: [lambda tag: tag == 'num:comp']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['20170625 jak nie adv'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: [lambda form, tag: form == 'jak' and tag == 'adv']
        }
    }

    checks['20170702 jak:comp'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [lambda form, tag: form == 'jak' and tag == 'comp'],
            NEGATIVE: []
        }
    }

    checks['20170914 adv na qub'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [lambda form, tag: form == 'niedaleko' and tag == 'prep:gen',
                       lambda form, tag: form == 'doprawdy' and tag == 'qub'],
            NEGATIVE: [lambda form, tag: form == 'doprawdy' and tag == 'adv']
        }
    }

    conj_to_comp = ['czym', 'ergo', 'jakokolwiek', 'jakoż', 'przeto', 'tedy', 'to', 'toteż', 'więc', 'zatem']
    checks['20170917 conj na comp'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }
    for token in conj_to_comp:
        checks['20170917 conj na comp'][FORMS][POSITIVE].append(lambda form, tag: form == token and tag == 'comp')
        checks['20170917 conj na comp'][FORMS][NEGATIVE].append(lambda form, tag: form == token and tag == 'conj')

    checks['20171224 num:..:congr'] = {
        TAGS: {
            POSITIVE: [lambda tag: tag.startswith('num:') and tag.endswith(':congr')],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['20180722 adjp -> adjp:dat, adjp:gen; burk -> frag; qub -> part'] = {
        TAGS: {
            POSITIVE: [lambda tag: tag == 'adjp:dat',
                       lambda tag: tag == 'adjp:gen',
                       lambda tag: tag == 'adjp:frag',
                       lambda tag: tag == 'adjp:part'],
            NEGATIVE: [lambda tag: tag == 'adjp',
                       lambda tag: tag == 'burk',
                       lambda tag: tag == 'qub']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['NKJP tagset'] = {
        TAGS: {
            POSITIVE: [lambda tag: tag == 'interj',
                       lambda tag: tag == 'adjc',
                       lambda tag: tag == 'burk',
                       lambda tag: tag == 'numcol'],
            NEGATIVE: []
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['dig'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag == 'dig']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['romandig'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag == 'romandig']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['blank'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag == 'blank']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['emoticon'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag == 'emoticon']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['emo'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag == 'emo']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['emo'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag == 'ign']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    checks['morfeusz2 tags not in NKJP'] = {
        TAGS: {
            POSITIVE: [],
            NEGATIVE: [lambda tag: tag == 'prefa',
                       lambda tag: tag == 'prefppas',
                       lambda tag: tag == 'prefs',
                       lambda tag: tag == 'prefv',
                       lambda tag: tag == 'nie',
                       lambda tag: tag == 'naj',
                       lambda tag: tag == 'cond',
                       lambda tag: tag == 'substa']
        },
        FORMS: {
            POSITIVE: [],
            NEGATIVE: []
        }
    }

    for date, functions in checks.items():
        print('Checking: %s' % date)
        for i, function in enumerate(functions[TAGS][POSITIVE]):
            if any([function(tag) for tag in stats_tags]):
                print('%s. +' % (i,))
            else:
                print('%s. ?' % (i,))
        for i, function in enumerate(functions[TAGS][NEGATIVE]):
            if any([function(tag) for tag in stats_tags]):
                print('%s. -' % (i,))
            else:
                print('%s. ?' % (i,))
        for i, function in enumerate(functions[FORMS][POSITIVE]):
            if any([function(form, tag) for form, tag in stats_forms]):
                print('%s. +' % (i,))
            else:
                print('%s. ?' % (i,))

        for i, function in enumerate(functions[FORMS][NEGATIVE]):
            if any([function(form, tag) for form, tag in stats_forms]):
                print('%s. -' % (i,))
            else:
                print('%s. ?' % (i,))
        print()
