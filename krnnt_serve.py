#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from argparse import ArgumentParser

from flask import Flask
from flask import request
from krnnt.additional_format import additional_format
from krnnt.aglt import remove_aglt_from_results_rule1_3
from krnnt.analyzers import MacaAnalyzer
from krnnt.blanks import remove_blanks_from_results
from krnnt.keras_models import BEST
from krnnt.new import Lemmatisation, Lemmatisation2, get_morfeusz, analyze_tokenized
from krnnt.writers import get_output_converter
from krnnt.readers import json_to_objects, json_compact_to_objects
from krnnt.pipeline import KRNNTSingle

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
<p>Wróbel Krzysztof, <a href="https://www.researchgate.net/publication/333566748_KRNNT_Polish_Recurrent_Neural_Network_Tagger">KRNNT: Polish Recurrent Neural Network Tagger</a></p>
<p>Source code: <a href="https://github.com/kwrobel-nlp/krnnt">https://github.com/kwrobel-nlp/krnnt</a></p>
</body>
</html>""" % (text, str_results)


@app.route('/', methods=['GET'])
def gui():
    return render()


@app.route('/', methods=['POST'])
def tag_raw():
    request.get_data()

    input_format = request.args.get('input_format', default=None, type=str)
    output_format = request.args.get('output_format', default='plain', type=str)
    remove_aglt = request.args.get('remove_aglt', default='0', type=str)
    remove_blank = request.args.get('remove_blank', default='1', type=str)

    conversion2 = get_output_converter(output_format)

    if remove_aglt!='0':
        conversionx=conversion2
        conversion2=lambda x: conversionx(remove_aglt_from_results_rule1_3(x))

    if remove_blank!='0':
        conversionx2=conversion2
        conversion2=lambda x: conversionx2(remove_blanks_from_results(x))

    if request.is_json:
        data = request.get_json()

        if 'docs' in data:
            return additional_format(data, krnntx, morfeusz)
        else:
            if 'documents' in data:
                paragraphs = json_to_objects(data)
            else:
                paragraphs = json_compact_to_objects(data)

            corpus = analyze_tokenized(morfeusz, paragraphs)
            results = krnntx.tag_paragraphs(corpus, preana=True)
            print(results)
            return conversion2(results)
    elif 'text' in request.form:
        text = request.form['text']



        results = krnntx.tag_paragraphs([text])  # ['Ala ma kota.', 'Ale nie ma psa.']
        return render(text, conversion(results))
    else:
        text = request.get_data()

        if input_format == 'lines':
            data = text.decode('utf-8').split('\n\n') #TODO
        else:
            data = [text.decode('utf-8')]

        results = krnntx.tag_paragraphs(data)
        print(results)
        return conversion2(results)


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


def main(argv=sys.argv[1:]):
    print(argv)
    global conversion,krnntx,maca_analyzer, morfeusz

    parser = ArgumentParser(usage='HTTP Tagger server')
    parser.add_argument('model_path', help='path to directory woth weights, lemmatisation data and dictionary')
    parser.add_argument('-p', '--port',
                        default=9003,
                        help='server port (defaults to 9003)')
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
    parser.add_argument('--remove_aglt', action='store_true')
    parser.add_argument('--dont_remove_blank', action='store_false')
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

    conversion= get_output_converter(args.output_format)

    if args.remove_aglt:
        conversionx = conversion
        conversion=lambda x: conversionx(remove_aglt_from_results_rule1_3(x))

    if args.dont_remove_blank:
        conversionx2 = conversion
        conversion=lambda x: conversionx2(remove_blanks_from_results(x))


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

#gunicorn -b 127.0.0.1:9003 -w 4 -k gevent -t 3600 --threads 4 'krnnt_serve:start("model_data","--maca_config","morfeusz2-nkjp","--toki_config_path","/home/krnnt/")'