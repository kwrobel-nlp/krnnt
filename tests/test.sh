#!/bin/bash

#TODO pytest-shell

MACA_CONFIG=morfeusz2-nkjp

cd ..

python3 process_xces.py tests/data/small/nkjp1m-1.2-xces.xml /tmp/nkjp.spickle
diff /tmp/nkjp.spickle tests/data/reference/nkjp1m-1.2.spickle
echo $?

python3 reanalyze.py --maca_config $MACA_CONFIG /tmp/nkjp.spickle /tmp/nkjp-reanalyzed.spickle
diff /tmp/nkjp-reanalyzed.spickle tests/data/reference/nkjp1m-1.2-reanalyzed.spickle
echo $?

python3 shuffle.py /tmp/nkjp-reanalyzed.spickle /tmp/nkjp-reanalyzed.shuf.spickle
diff /tmp/nkjp-reanalyzed.shuf.spickle tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle
echo $?

CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 krnnt_train.py --maca_config $MACA_CONFIG /tmp/nkjp-reanalyzed.shuf.spickle -e 2
echo $?