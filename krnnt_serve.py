#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
import sys
from argparse import ArgumentParser
from optparse import OptionParser
import time

import flask
import tensorflow
from flask import Flask
from flask import request
from flask import g, current_app
from keras import backend as K
from krnnt.analyzers import MacaAnalyzer
from krnnt.structure import Paragraph, Sentence, Token
from krnnt.keras_models import BEST
from krnnt.new import Lemmatisation, Lemmatisation2, get_morfeusz, analyze_tokenized
from krnnt.writers import results_to_conll_str, results_to_jsonl_str, results_to_conllu_str, results_to_plain_str, \
    results_to_xces_str
from krnnt.readers import read_xces
from krnnt.pipeline import KRNNTSingle

import threading

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
application = app

global krnntx, conversion, maca_analyzer, morfeusz


def render(text='', str_results=''):
    return """
<html>
<head>
<meta charset="utf-8">
<title>KRNNT</title>
</head>
<body>
<h1>KRNNT: Polish Recurrent Neural Network Tagger</h1>
<form action="/" method="post">
<textarea name="text" rows=10 cols=100>%s</textarea><br>
<input type="submit">
</form>
<pre>%s</pre>
<p>The tagset is described here: <a href="http://nkjp.pl/poliqarp/help/ense2.html">http://nkjp.pl/poliqarp/help/ense2.html</a></p>
<p>Wróbel Krzysztof, <a href="http://ltc.amu.edu.pl/book/papers/PolEval1-6.pdf">KRNNT: Polish Recurrent Neural Network Tagger</a></p>
</body>
</html>""" % (text, str_results)


@app.route('/', methods=['GET'])
def gui():
    return render()


def json_to_objects(data):
    paragraphs = []
    for input_paragraph in data['documents']:
        paragraph = Paragraph()
        paragraphs.append(paragraph)
        for input_sentence in input_paragraph['sentences']:
            sentence = Sentence()
            paragraph.add_sentence(sentence)
            for input_token in input_sentence['tokens']:
                token = Token()
                token.form = input_token['form']
                if len(input_token)>=2:
                    separator=input_token['separator']
                    if separator is not None:
                        token.space_before=separator
                    elif len(input_token)>=4:
                        token.start=input_token['start']
                        token.end = input_token['end']
                        #infer separator before from positions
                        if len(sentence.tokens)==0:
                            token.space_before='space'
                        else:
                            if sentence.tokens[-1].end==token.start:
                                token.space_before = 'none'
                            else:
                                token.space_before = 'space'
                else:
                    token.space_before = 'space'  # TODO ?
                sentence.add_token(token)
    return paragraphs

def json_compact_to_objects(data):
    paragraphs = []
    for input_paragraph in data:
        paragraph = Paragraph()
        paragraphs.append(paragraph)
        for input_sentence in input_paragraph:
            sentence = Sentence()
            paragraph.add_sentence(sentence)
            for input_token in input_sentence:
                token = Token()
                token.form = input_token[0]
                if len(input_token) >= 2:
                    separator = input_token[1]
                    if separator is not None:
                        token.space_before = separator
                    elif len(input_token) >= 4:
                        token.start = input_token[2]
                        token.end = input_token[3]
                        # infer separator before from positions
                        if len(sentence.tokens) == 0:
                            token.space_before = 'space'
                        else:
                            if sentence.tokens[-1].end == token.start:
                                token.space_before = 'none'
                            else:
                                token.space_before = 'space'
                else:
                    token.space_before = 'space'  # TODO ?
                sentence.add_token(token)
    return paragraphs


@app.route('/', methods=['POST'])
def tag_raw():
    request.get_data()

    output_format = request.args.get('output_format', default='plain', type=str)
    # filter = request.args.get('filter', default='*', type=str)
    conversion = get_output_converter(output_format)

    if request.is_json:
        data = request.get_json()

        if 'documents' in data:
            paragraphs=json_to_objects(data)
        else:
            paragraphs = json_compact_to_objects(data)


        corpus = analyze_tokenized(morfeusz, paragraphs)
        results = krnntx.tag_sentences(corpus, preana=True)
        return conversion(results)

    elif 'text' in request.form:
        text = request.form['text']
        results = krnntx.tag_sentences(text.split('\n\n'))  # ['Ala ma kota.', 'Ale nie ma psa.']
        return render(text, conversion(results))
    else:
        text = request.get_data()
        # print(text.decode('utf-8').split('\n\n'))
        results = krnntx.tag_sentences(text.decode('utf-8').split('\n\n'))
        return conversion(results)


@app.route('/tag/', methods=['POST'])
def tag():
    text = request.form['text']
    results = krnntx.tag_sentences(text.split('\n\n'))  # ['Ala ma kota.', 'Ale nie ma psa.']
    return render(text, conversion(results))

@app.route('/maca/', methods=['POST'])
def maca():
    text = request.get_data()
    # print(text.decode('utf-8').split('\n\n'))

    results = maca_analyzer._maca(text.decode('utf-8').split('\n\n'))
    results = list(results)
    return str(results)

def get_output_converter(output_format):
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
    else:
        logging.error('Wrong output format.')
        sys.exit(1)

    return conversion

def main(argv=sys.argv[1:]):
    print(argv)
    global conversion,krnntx,maca_analyzer, morfeusz

    parser = ArgumentParser(usage='HTTP Tagger server')
    parser.add_argument('model_path', help='path to directory woth weights, lemmatisation data and dictionary')
    parser.add_argument('-p', '--port',
                        default=9200,
                        help='server port (defaults to 9200)')
    parser.add_argument('-t', '--host',
                        default='0.0.0.0',
                        help='server host (defaults to localhost)')
    parser.add_argument('--maca_config',
                        default='morfeusz-nkjp-official',
                        help='Maca config')
    parser.add_argument('--toki_config_path',
                        default='',
                        help='Toki config path (directory)')
    parser.add_argument('--lemmatisation',
                        default='sgjp',
                        help='lemmatization mode (sgjp, simple)')
    parser.add_argument('-o', '--output-format',
                        default='plain', dest='output_format',
                        help='output format: xces, plain, conll, conllu, jsonl')
    parser.add_argument('-b', '--batch_size',
                        default=32, type=int,
                        help='batch size')
    args = parser.parse_args(argv)

    pref = {'keras_batch_size': args.batch_size, 'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'keras_model_class': BEST, 'maca_config': args.maca_config, 'toki_config_path': args.toki_config_path}

    if args.lemmatisation == 'simple':
        pref['lemmatisation_class'] = Lemmatisation2
    else:
        pref['lemmatisation_class'] = Lemmatisation

    pref['reanalyze'] = True

    pref['weight_path'] = args.model_path + "/weights.hdf5"
    pref['lemmatisation_path'] = args.model_path + "/lemmatisation.pkl"
    pref['UniqueFeaturesValues'] = args.model_path + "/dictionary.pkl"

    morfeusz = get_morfeusz()
    maca_analyzer = MacaAnalyzer(args.maca_config)
    krnntx = KRNNTSingle(pref)

    krnntx.tag_sentences(['Ala'])

    conversion=get_output_converter(args.output_format)

    return app, args.host, args.port



if __name__ == '__main__':
    app,host,port = main()
    # from werkzeug.middleware.profiler import ProfilerMiddleware
    # app.config['PROFILE'] = True
    # app = ProfilerMiddleware(app)
    # app.wsgi_app = ProfilerMiddleware(
    #     app.wsgi_app, profile_dir="."
    # )
    app.run(host=host, port=port, debug=False) # threaded=False on GPU

def start(*args, **kwargs):
    app, host, port = main(args)
    return app

#gunicorn -b 127.0.0.1:9200 -w 4 -k gevent -t 3600 --threads 4 'krnnt_serve:start("model_data","--maca_config","morfeusz2-nkjp","--toki_config_path","/home/krnnt/")'