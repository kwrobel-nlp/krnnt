#!/usr/bin/env bash

MACA_CONFIG=morfeusz2-nkjp

time cat tests/data/full/test-raw.txt | CUDA_VISIBLE_DEVICES="" python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces  > /tmp/out.xces
time cat tests/data/full/train-raw.txt | CUDA_VISIBLE_DEVICES="" python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces  > /tmp/out.xces

#one thread
time cat tests/data/full/test-raw.txt | CUDA_VISIBLE_DEVICES="" python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces --reproducible > /tmp/out.xces