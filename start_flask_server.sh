#!/usr/bin/env bash

PORT=${PORT:-9200}

python3 krnnt_serve.py model_data --maca_config morfeusz2-nkjp -p $PORT