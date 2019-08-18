#!/usr/bin/env bash

PORT=${PORT:-9200}
WORKERS=${WORKERS:-1}

export CUDA_VISIBLE_DEVICES=""

gunicorn -b 0.0.0.0:$PORT -w $WORKERS -k sync -t 3600 --threads 1 'krnnt_serve:start("model_data","--maca_config","morfeusz2-nkjp")'