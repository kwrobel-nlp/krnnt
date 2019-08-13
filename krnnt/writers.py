import io

import jsonlines


def results_to_txt_str(results):
    result_str = []
    for sentence in results:
        for i, token in enumerate(sentence):
            # print(token['sep'])
            if i > 0 and token['sep'] != 'none':
                result_str += (' ',)
            result_str += (token['token'],)
        result_str += ("\n",)
    return ''.join(result_str)


def results_to_conll_str(results):
    result_str = []
    for sentence in results:
        for token in sentence:
            try:
                start = token['start']
            except KeyError:
                start = ''

            try:
                end = token['end']
            except KeyError:
                end = ''

            result_str += ('%s\t%s\t%s\t%s\t%s\t%s' % (
                token['token'], token['lemmas'][0], 0 if token['sep'] == 'none' else 1, token['tag'], start, end),)
        result_str += ("",)
    return '\n'.join(result_str)


def results_to_jsonl_str(results):
    fp = io.StringIO()
    with jsonlines.Writer(fp) as writer:
        for sentence in results:
            ss = [(token['token'], token['lemmas'][0], token['tag']) for token in sentence]
            writer.write(ss)
    return fp.getvalue()


def results_to_conllu_str(results):
    result_str = []
    for sentence in results:
        for i, token in enumerate(sentence):
            result_str += ('%s\t%s\t%s\t_\t%s\t_\t_\t_\t_\t_' % (
                i + 1, token['token'], token['lemmas'][0], token['tag']),)
        result_str += ("",)
    return '\n'.join(result_str)


def results_to_plain_str(results):
    result_str = []
    for sentence in results:
        for token in sentence:
            result_str += ('%s\t%s' % (token['token'], token['sep']),)
            for lemma in token['lemmas']:
                result_str += ('\t%s\t%s\tdisamb' % (lemma, token['tag']),)
        result_str += ("",)
    return '\n'.join(result_str)


def results_to_xces_str(results):
    result_str = []
    result_str += ('<?xml version="1.0" encoding="UTF-8"?>',
                   '<!DOCTYPE cesAna SYSTEM "xcesAnaIPI.dtd">',
                   '<cesAna xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0" type="lex disamb">',
                   '<chunkList>')

    for sentence in results:
        result_str += (' <chunk type="p">',
                       '  <chunk type="s">')
        for token in sentence:
            if token['sep'] == 'none':
                result_str += ('   <ns/>',)
            result_str += ('   <tok>',)
            result_str += ('    <orth>%s</orth>' % escape_xml(token['token']),)
            for lemma in token['lemmas']:
                result_str += ('    <lex disamb="1"><base>%s</base><ctag>%s</ctag></lex>' % (escape_xml(lemma),
                                                                                             token['tag']),)
            result_str += ('   </tok>',)
        result_str += ('  </chunk>',
                       ' </chunk>')
    result_str += ('</chunkList>',
                   '</cesAna>')
    return '\n'.join(result_str)


def escape_xml(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace('\'',
                                                                                                            '&apos;')
