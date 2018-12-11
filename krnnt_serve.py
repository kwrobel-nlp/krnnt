#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import sys
from optparse import OptionParser
import time
from flask import Flask
from flask import request
from flask import g, current_app

from krnnt.keras_models import BEST
from krnnt.new import results_to_plain_str, results_to_xces_str, read_xces, Lemmatisation, Lemmatisation2
from krnnt.pipeline import KRNNTSingle

app = Flask(__name__)


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
    global krnnt
    request.get_data()
    if 'text' in request.form:
        text=request.form['text']
        results = krnnt.tag_sentences(text.split('\n\n')) # ['Ala ma kota.', 'Ale nie ma psa.']
        return render(text, results_to_plain_str(results))
    else:
        print('wtf')
        text = request.get_data()
        print(text)
        results = krnnt.tag_sentences(text.decode('utf-8').split('\n\n'))
        return results_to_plain_str(results)

@app.route('/tag/', methods=['POST'])
def tag():
    global krnnt
    text=request.form['text']
    results = krnnt.tag_sentences(text.split('\n\n')) # ['Ala ma kota.', 'Ale nie ma psa.']
    return render(text, results_to_plain_str(results))

if __name__ == '__main__':
    parser = OptionParser(usage='HTTP Tagger server')
    parser.add_option('-p', '--port', action='store',
                      default=9200, dest='port',
                      help='server port (defaults to 9200)')
    parser.add_option('-t', '--host', action='store',
                      default='0.0.0.0', dest='host',
                      help='server host (defaults to localhost)')
    parser.add_option('--maca_config', action='store',
                      default='morfeusz-nkjp-official', dest='maca_config',
                      help='Maca config')
    parser.add_option('--toki_config_path', action='store',
                      default='', dest='toki_config_path',
                      help='Toki config path (directory)')
    parser.add_option('--lemmatisation', action='store',
                      default='sgjp', dest='lemmatisation',
                      help='lemmatization mode (sgjp, simple)')
    (options, args) = parser.parse_args()

    pref = {}
    pref = {'keras_batch_size': 32, 'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'keras_model_class': BEST, 'maca_config':options.maca_config, 'toki_config_path':options.toki_config_path}

    if len(args) != 1:
        print('Provide path to directory with weights, lemmatisation and dictionary.')
        sys.exit(1)

    if options.lemmatisation=='simple':
        pref['lemmatisation_class'] = Lemmatisation2
    else:
        pref['lemmatisation_class'] = Lemmatisation

    pref['reanalyze'] = True

    pref['weight_path'] = args[0] + "/weights.hdf5"
    pref['lemmatisation_path'] = args[0] + "/lemmatisation.pkl"
    pref['UniqueFeaturesValues'] = args[0] + "/dictionary.pkl"


    krnnt = KRNNTSingle(pref)
    
    krnnt.tag_sentences( ['Ala'] )
    
    app.run(host=options.host, port=options.port, debug=False)
