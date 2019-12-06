import io
import json
import logging
import sys
from typing import Callable

import jsonlines


def results_to_txt_str(result_paragraphs):
    result_str = []
    for paragraph in result_paragraphs:
        for sentence in paragraph:
            for i, token in enumerate(sentence):
                # print(token['sep'])
                if i > 0 and token['sep'] != 'none':
                    result_str += (' ',)
                result_str += (token['token'],)
            result_str += ("\n",)
        result_str += ("\n",)
    return ''.join(result_str)


def results_to_conll_str(result_paragraphs):
    result_str = []
    for paragraph in result_paragraphs:
        for sentence in paragraph:
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
        result_str += ("",)
    return '\n'.join(result_str)


def results_to_jsonl_str(result_paragraphs):
    fp = io.StringIO()
    with jsonlines.Writer(fp) as writer:
        for paragraph in result_paragraphs:
            output_paragraph=[]
            for sentence in paragraph:
                ss = [(token['token'], token['lemmas'][0], token['tag']) for token in sentence]
                output_paragraph+=(ss,)
            writer.write(output_paragraph)
    return fp.getvalue()

def results_to_json_str(result_paragraphs):
    return json.dumps(result_paragraphs)


def results_to_conllu_str(result_paragraphs):
    result_str = []
    for paragraph in result_paragraphs:
        for sentence in paragraph:
            for i, token in enumerate(sentence):
                result_str += ('%s\t%s\t%s\t_\t%s\t_\t_\t_\t_\t_' % (
                    i + 1, token['token'], token['lemmas'][0], token['tag']),)
            result_str += ("",)
        result_str += ("",)
    return '\n'.join(result_str)


def results_to_plain_str(result_paragraphs):
    result_str = []
    for paragraph in result_paragraphs:
        for sentence in paragraph:
            for token in sentence:
                result_str += ('%s\t%s' % (token['token'], token['sep']),)
                for lemma in token['lemmas']:
                    result_str += ('\t%s\t%s\tdisamb' % (lemma, token['tag']),)
            result_str += ("",)
        result_str += ("",)
    return '\n'.join(result_str)


def results_to_xces_str(result_paragraphs):
    result_str = []
    result_str += ('<?xml version="1.0" encoding="UTF-8"?>',
                   '<!DOCTYPE cesAna SYSTEM "xcesAnaIPI.dtd">',
                   '<cesAna xmlns:xlink="http://www.w3.org/1999/xlink" version="1.0" type="lex disamb">',
                   '<chunkList>')
    for paragraph in result_paragraphs:
        result_str += (' <chunk type="p">', )
        for sentence in paragraph:
            result_str += ('  <chunk type="s">',)
            for token in sentence:
                if token['sep'] == 'none':
                    result_str += ('   <ns/>',)
                result_str += ('   <tok>',)
                result_str += ('    <orth>%s</orth>' % escape_xml(token['token']),)
                for lemma in token['lemmas']:
                    result_str += ('    <lex disamb="1"><base>%s</base><ctag>%s</ctag></lex>' % (escape_xml(lemma),
                                                                                                 token['tag']),)
                result_str += ('   </tok>',)
            result_str += ('  </chunk>',)
        result_str += (' </chunk>',)

    result_str += ('</chunkList>',
                   '</cesAna>')
    return '\n'.join(result_str)


def escape_xml(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace('\'',
                                                                                                            '&apos;')


def get_output_converter(output_format: str) -> Callable:
    output_format=output_format.lower()
    if output_format == 'xces':
        conversion = results_to_xces_str
    elif output_format == 'plain':
        conversion = results_to_plain_str
    elif output_format in ('conll','tsv'):
        conversion = results_to_conll_str
    elif output_format == 'conllu':
        conversion = results_to_conllu_str
    elif output_format == 'jsonl':
        conversion = results_to_jsonl_str
    elif output_format == 'json':
        conversion = results_to_json_str
    elif output_format in ('txt','text'):
        conversion = results_to_txt_str
    else:
        logging.error('Wrong output format.')
        sys.exit(1)

    return conversion