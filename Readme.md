# KRNNT

Accuracy tested with 10-fold cross validation on National Corpus of Polish.

Accuracy lower bound | Accuracy lower bound for unknown tokens
------------ | -------------
93.72% | 69.03%

## PolEval

### Training

1. Install KRNNT.

```
krnnt]$ pip3 install -e .
```


2. Prepare training data.

```
krnnt]$ time python3 process_xces.py train-gold.xml train-gold.spickle
real	0m37.769s
```


3. Reanalyze corpus with Maca.

```
krnnt]$ python3 reanalyze.py train-gold.spickle train-reanalyzed.spickle
0 MACA 9 10
1 MACA 7 8
...
real	26m35.013s
```
Ensure that last two numbers in each row are usually the same. Zeros indicates problems with Maca.

4. Shuffle data (optional).

```
krnnt]$ time python3 shuffle.py train-reanalyzed.spickle train-reanalyzed.shuf.spickle
real	1m26.350s
```


5. Train.

```
krnnt]$ time python3 krnnt_train.py train-reanalyzed.shuf.spickle
Model is saved under: weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5
Lemmatisation model is saved under: lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl
Dictionary is saved under: train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues
real    197m44,568s
```
Zeros on output indicates some problem `(0, 0, 0.0, 0, 0, 0.0, 0, 0, 0.0, 0, 0, 0.0, 0, 0, 0.0, 0.0)`.
Check `~/.keras/keras.json` for `"image_dim_ordering": "th"` (for old Keras) and `"image_data_format": "channels_first"` (for Keras 2).

6. Testing.

```
krnnt]$ time python3 krnnt_run.py weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5.final lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues < train-raw.txt
real	7m22.892s
```


## Training on gold segmentation

2. Prepare training data.

```
krnnt]$ time python3 process_xces.py train-analyzed.xml train-analyzed.spickle
real	1m51.836s

krnnt]$ time python3 process_xces.py train-gold.xml train-gold.spickle
real	0m37.769s

krnnt]$ time python3 merge_analyzed_gold.py train-gold.spickle train-analyzed.spickle train-merged.spickle
real	0m36.049s
```


3. Shuffle data (optional).

```
krnnt]$ time python3 shuffle.py train-merged.spickle train-merged.shuf.spickle
real	1m41.192s
```


4. Train.

```
krnnt]$ time python3 krnnt_train.py -p train-merged.shuf.spickle
Model is saved under: weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5
Lemmatisation model is saved under: lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl
Dictionary is saved under: train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues
real    190m44,568s
```

5. Testing.

```
krnnt]$ time python3 krnnt_run.py -p weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5.final lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues < train-analyzed.xces
real	7m38.660s
```
## Testing

Trained models are available with releases: https://github.com/kwrobel-nlp/krnnt/releases

```
krnnt]$ pip3 install -e .
```

`reana.zip` contains model trained with reanalyzed data:
```
krnnt]$ python3 krnnt_run.py reana/weights_reana.hdf5 reana/lemmatisation_reana.pkl reana/dictionary_reana.pkl < test-raw.txt > test-raw.krnnt.xml
```

`preana.zip` contains model trained with preanalyzed data:
```
krnnt]$ python3 krnnt_run.py -p preana/weights_preana.hdf5 preana/lemmatisation_preana.pkl preana/dictionary_preana.pkl < test-analyzed.xml > test-analyzed.krnnt.xml
```

### Voting

Training more models and performing simple voting increase accuracy. Voting over 10 models achieves about 94.30% accuracy lower bound.

`reana10.zip` and `preana10.zip` contain 10 models each.
```
for i in {0..9}
do
   krnnt]$ python3 krnnt_run.py reana/$i.hdf5 reana/lemmatisation.pkl  reana/dictionary.pkl < test-raw.txt > reana/$i.xml
done
krnnt]$ python3 voting.py reana/ > reana/test-raw.krnnt.voting.xml
```

KRNNT is licensed under GNU LGPL v3.0.
