#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import sys
from argparse import ArgumentParser
from optparse import OptionParser
import time

import flask
from flask import Flask
from flask import request
from flask import g, current_app

from krnnt.keras_models import BEST
from krnnt.new import Lemmatisation, Lemmatisation2
from krnnt.writers import results_to_conll_str, results_to_jsonl_str, results_to_conllu_str, results_to_plain_str, \
    results_to_xces_str
from krnnt.readers import read_xces
from krnnt.pipeline import KRNNTSingle

import threading

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
application = app

global krnntx, conversion


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


@app.route('/', methods=['POST'])
def tag_raw():
    request.get_data()
    if 'text' in request.form:
        text = request.form['text']
        results = krnntx.tag_sentences(text.split('\n\n'))  # ['Ala ma kota.', 'Ale nie ma psa.']
        return render(text, conversion(results))
    else:
        text = request.get_data()
        # print(text)
        print(text.decode('utf-8').split('\n\n'))
        print(threading.active_count())
        results = krnntx.tag_sentences(text.decode('utf-8').split('\n\n'))
        return conversion(results)


@app.route('/tag/', methods=['POST'])
def tag():
    text = request.form['text']
    results = krnntx.tag_sentences(text.split('\n\n'))  # ['Ala ma kota.', 'Ale nie ma psa.']
    return render(text, conversion(results))


def main(argv=sys.argv[1:]):
    print(argv)
    global conversion,krnntx

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
    args = parser.parse_args(argv)

    # TODO args = parser.parse_args(argv)

    pref = {'keras_batch_size': 32, 'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'keras_model_class': BEST, 'maca_config': args.maca_config, 'toki_config_path': args.toki_config_path}

    if args.lemmatisation == 'simple':
        pref['lemmatisation_class'] = Lemmatisation2
    else:
        pref['lemmatisation_class'] = Lemmatisation

    pref['reanalyze'] = True

    pref['weight_path'] = args.model_path + "/weights.hdf5"
    pref['lemmatisation_path'] = args.model_path + "/lemmatisation.pkl"
    pref['UniqueFeaturesValues'] = args.model_path + "/dictionary.pkl"

    krnntx = KRNNTSingle(pref)

    krnntx.tag_sentences(['Ala'])

    if args.output_format == 'xces':
        conversion = results_to_xces_str
    elif args.output_format == 'plain':
        conversion = results_to_plain_str
    elif args.output_format == 'conll':
        conversion = results_to_conll_str
    elif args.output_format == 'conllu':
        conversion = results_to_conllu_str
    elif args.output_format == 'jsonl':
        conversion = results_to_jsonl_str
    else:
        print('Wrong output format.')
        sys.exit(1)

    return app, args.host, args.port



if __name__ == '__main__':
    app,host,port = main()
    app.run(host=host, port=port, debug=False)

def start(*args, **kwargs):
    app, host, port = main(args)
    return app

#gunicorn -b 127.0.0.1:9200 -w 4 -k gevent -t 3600 --threads 4 'krnnt_serve:start("model_data","--maca_config","morfeusz2-nkjp","--toki_config_path","/home/krnnt/")'