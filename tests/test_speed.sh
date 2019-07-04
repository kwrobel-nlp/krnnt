#!/usr/bin/env bash

MACA_CONFIG=morfeusz2-nkjp


time cat tests/data/full/test-raw.txt | CUDA_VISIBLE_DEVICES="" python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces  > /tmp/out.xces
#12s

time cat tests/data/full/train-raw.txt | CUDA_VISIBLE_DEVICES="" python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces  > /tmp/out.xces
#7m16s

#one thread
time cat tests/data/full/test-raw.txt | CUDA_VISIBLE_DEVICES="" python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces --reproducible > /tmp/out.xces
#22s

#GPU 1050GTX
#train
#5m12s

#time maca-analyse -c morfeusz2-nkjp < tests/data/full/train-raw.txt > /dev/null
#35s

#time maca-analyse -c morfeusz2-nkjp < tests/data/full/test-raw.txt > /dev/null
#0.9s

#maca per line test-raw.txt
#45s

#i tak zrównolegla

# test-raw.txt API 1w1t GPU 44s
# test-raw.txt API 1w2t GPU 44s
# test-raw.txt API 2w1t GPU 44s

# test-raw.txt API 1w1t CPU 43s
# test-raw.txt API 1w2t CPU 43s
# test-raw.txt API 2w1t CPU 42s

# pool=2 test-raw.txt API 1w1t CPU 29s
# pool=2 test-raw.txt API 1w2t CPU 28s
# pool=2 test-raw.txt API 2w1t CPU 25s

# pool=2 test-raw.txt API 1w1t GPU 21s
# pool=2 test-raw.txt API 1w2t GPU 30s
# pool=2 test-raw.txt API 2w1t GPU 23s

# pool=10 test-raw.txt API 1w1t CPU 20s
# pool=10 test-raw.txt API 1w2t CPU 20s
# pool=10 test-raw.txt API 2w1t CPU 17s
# pool=10 test-raw.txt API 4w1t CPU 15s
# pool=10 test-raw.txt API 4w2t CPU 16s
# pool=10 test-raw.txt API 8w1t CPU 16s
# pool=100 test-raw.txt API 10w1t CPU 14s
# pool=100 test-raw.txt API 20w1t CPU 14s

# pool=10 test-raw.txt API 1w1t GPU 21s
# pool=10 test-raw.txt API 1w2t GPU 21s
# pool=10 test-raw.txt API 2w1t GPU 14s
# pool=10 test-raw.txt API 4w1t GPU OOM