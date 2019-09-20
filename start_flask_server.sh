#!/usr/bin/env bash

PORT=${PORT:-9003}

export CUDA_VISIBLE_DEVICES=""

python3 krnnt_serve.py model_data --maca_config morfeusz2-nkjp -p $PORT