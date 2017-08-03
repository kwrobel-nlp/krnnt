## Training

1. Install KRNNT.

krnnt]$ pip3 install -e .


2. Prepare training data.

krnnt]$ time python3 process_xces.py train-gold.xml train-gold.spickle
real	0m37.769s


3. Reanalyze corpus with Maca.

krnnt]$ python3 reanalyze.py train-gold.spickle train-reanalyzed.spickle
real	26m35.013s


4. Shuffle data (optional).

krnnt]$ time python3 shuffle.py train-reanalyzed.spickle train-reanalyzed.shuf.spickle
real	1m26.350s


5. Train.

krnnt]$ time python3 krnnt_train.py train-reanalyzed.shuf.spickle
Model is saved under: weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5
Lemmatisation model is saved under: lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl
Dictionary is saved under: train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues
real    197m44,568s


6. Testing.

#krnnt]$ time python3 krnnt_single.py -w weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5 -d train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues -l lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl < test-raw.txt
krnnt]$ time python3 krnnt_run.py weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5.final lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues < train-raw.txt
real	7m22.892s


## Training on gold segmentation:

2. Prepare training data.

krnnt]$ time python3 process_xces.py train-analyzed.xml train-analyzed.spickle
real	1m51.836s

krnnt]$ time python3 process_xces.py train-gold.xml train-gold.spickle
real	0m37.769s

krnnt]$ time python3 merge_analyzed_gold.py train-gold.spickle train-analyzed.spickle train-merged.spickle
real	0m36.049s


3. Shuffle data (optional).

krnnt]$ time python3 shuffle.py train-merged.spickle train-merged.shuf.spickle
real	1m41.192s


4. Train.

krnnt]$ time python3 krnnt_train.py -p train-merged.shuf.spickle
Model is saved under: weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5
Lemmatisation model is saved under: lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl
Dictionary is saved under: train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues
real    190m44,568s

5. Testing.

#krnnt]$ time python3 krnnt_single.py -p -w weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5 -d train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues -l lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl test-analyzed.xml


krnnt]$ time python3 krnnt_run.py -p weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5.final lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues < train-analyzed.xces
real	7m38.660s


python krnnt_run.py -p models/preanalyze90/weight_b2c10680-6354-11e7-87c6-a0000220fe80.hdf5.final models/preanalyze90/lemmatisation_b2c10680-6354-11e7-87c6-a0000220fe80.pkl models/preanalyze90/train-merged.shuf.spickle_FormatDataPreAnalyzed_PreprocessData_UniqueFeaturesValues < poleval/test-analyzed.xml.txt.reana > models/preanalyze90/test-analyzed.xml.txt.reana.xml


