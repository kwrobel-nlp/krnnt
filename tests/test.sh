#!/usr/bin/env bash

#TODO pytest-shell

#version of morfeusz dictionary may influence results

MACA_CONFIG=morfeusz2-nkjp

cd ..

python3 process_xces.py tests/data/small/nkjp1m-1.2-xces.xml /tmp/nkjp.spickle
echo $?
diff /tmp/nkjp.spickle tests/data/reference/nkjp1m-1.2.spickle

python3 reanalyze.py --maca_config $MACA_CONFIG /tmp/nkjp.spickle /tmp/nkjp-reanalyzed.spickle
echo $?
diff /tmp/nkjp-reanalyzed.spickle tests/data/reference/nkjp1m-1.2-reanalyzed.spickle

python3 shuffle.py /tmp/nkjp-reanalyzed.spickle /tmp/nkjp-reanalyzed.shuf.spickle
echo $?
diff /tmp/nkjp-reanalyzed.shuf.spickle tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle

rm /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2 /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues
CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python3 krnnt_train.py --maca_config $MACA_CONFIG /tmp/nkjp-reanalyzed.shuf.spickle -e 2 --reproducible --hash test
echo $?
h5diff weight_test.hdf5 tests/data/reference/weight_test.hdf5
h5diff weight_test.hdf5.final tests/data/reference/weight_test.hdf5.final
diff lemmatisation_test.pkl tests/data/reference/lemmatisation_test.pkl
diff /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2 tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2
diff /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData
diff /tmp/nkjp-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues

echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o xces > /tmp/out.xces
echo $?
diff /tmp/out.xces tests/data/reference/out.xces

echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o plain > /tmp/out.plain
echo $?
diff /tmp/out.plain tests/data/reference/out.plain

echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o conll > /tmp/out.conll
echo $?
diff /tmp/out.conll tests/data/reference/out.conll

echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o conllu > /tmp/out.conllu
echo $?
diff /tmp/out.conllu tests/data/reference/out.conllu

echo "Lubię placki." | python3 krnnt_run.py tests/data/reference/weight_test.hdf5.final tests/data/reference/lemmatisation_test.pkl tests/data/reference/nkjp1m-1.2-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues --maca_config $MACA_CONFIG -o jsonl > /tmp/out.jsonl
echo $?
diff /tmp/out.jsonl tests/data/reference/out.jsonl
