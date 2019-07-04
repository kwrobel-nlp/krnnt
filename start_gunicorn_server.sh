#!/usr/bin/env bash

PORT=${PORT:-9200}
WORKERS=${WORKERS:-1}
THREADS=${THREADS:-1}

gunicorn -b 127.0.0.1:$PORT -w $WORKERS -k gevent -t 3600 --threads $THREADS 'krnnt_serve:start("model_data","--maca_config","morfeusz2-nkjp")'