#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import sys
from optparse import OptionParser
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from krnnt.keras_models import BEST
from krnnt.new import results_to_plain_str, results_to_xces_str, read_xces
from krnnt.pipeline import KRNNTSingle

HOST_NAME = 'localhost'
PORT_NUMBER = 9200


class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Respond to a GET request."""
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(bytes("<html><body><h1>Tagger KRNNT</h1><p>Send a POST request and you will receive a tagged response</p></body></html>", "utf8"))

    def do_POST(self):
        """Respond to a POST request."""
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        results = krnnt.tag_sentences(post_data.decode('utf-8').split('\n\n')) # ['Ala ma kota.', 'Ale nie ma psa.']
        self.wfile.write(bytes(results_to_plain_str(results), "utf-8"))


if __name__ == '__main__':
    parser = OptionParser(usage='HTTP Tagger server')
    parser.add_option('-p', '--port', action='store',
                      default=9200, dest='port',
                      help='server port (defaults to 9200)')
    parser.add_option('-t', '--host', action='store',
                      default='localhost', dest='host',
                      help='server host (defaults to localhost)')
    (options, args) = parser.parse_args()

    pref = {}
    pref = {'keras_batch_size': 32, 'internal_neurons': 256, 'feature_name': 'tags4e3', 'label_name': 'label',
            'keras_model_class': BEST}

    if len(args) != 1:
        print('Provide path to directory with weights, lemmatisation and dictionary.')
        sys.exit(1)

    pref['reanalyze'] = True

    pref['weight_path'] = args[0] + "/weights.hdf5"
    pref['lemmatisation_path'] = args[0] + "/lemmatisation.pkl"
    pref['UniqueFeaturesValues'] = args[0] + "/dictionary.pkl"


    krnnt = KRNNTSingle(pref)

    host_port = ('0.0.0.0', options.port)
    httpd = HTTPServer(host_port, MyHandler)
    print(time.asctime(), "Server Starts - %s:%s" % host_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), "Server Stops - %s:%s" % host_port)
