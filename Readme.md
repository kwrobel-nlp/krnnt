# KRNNT

KRNNT is a morphological tagger for Polish based on recurrent naural networks. It was presented at 8th Language & Technology Conference. More details are available in the paper:
```
@inproceedings{wrobel2017,
  author       = "Wróbel, Krzysztof",
  editor       = "Vetulani, Zygmunt and Paroubek, Patrick",
  title        = "KRNNT: Polish Recurrent Neural Network Tagger",
  year         = "2017",
  booktitle    = "Proceedings of the 8th Language \& Technology Conference: Human Language Technologies as a Challenge for Computer Science and Linguistics",
  publisher    = "Fundacja Uniwersytetu im. Adama Mickiewicza w~Poznaniu",
  pages        = {386-391},
  pdf          = "http://ltc.amu.edu.pl/book/papers/PolEval1-6.pdf"
}
```

Online version is available at: http://ltc.amu.edu.pl/book/papers/PolEval1-6.pdf

## External tools

The tagger uses external tools: tokenizer Toki and morphological analyzer Morfeusz. Maca (http://nlp.pwr.wroc.pl/redmine/projects/libpltagger/wiki) integrates both tools.

The tagset is described here: http://nkjp.pl/poliqarp/help/ense2.html

## Getting started

You can run KRNNT using docker or by manual installation.

### Docker

Docker image was prepared by Aleksander Smywiński-Pohl and instrutions are available at: https://hub.docker.com/r/djstrong/krnnt/

1. Download and starte the server.
```bash
docker run -it -p 9200:9200 djstrong/krnnt:0.1 python3 /home/krnnt/krnnt/krnnt_serve.py /home/krnnt/krnnt/data
```
2. Tag a text usig POST request or open http://localhost:9200 in a browser.
```bash
$ curl -XPOST localhost:9200 -d "Ala ma kota."
Ala    none
    Ala    subst:sg:nom:f    disamb
ma    space
    mieć    fin:sg:ter:imperf    disamb
kota    space
    kot    subst:sg:acc:m2    disamb
.    none
    .    interp    disamb
```

### Manual installation

1. Install Maca: http://nlp.pwr.wroc.pl/redmine/projects/libpltagger/wiki

Make sure that command `maca-analyse` works:
```bash
echo "Ala ma kota." | maca-analyse -qc morfeusz-nkjp-official
Ala	newline
	Al	subst:sg:gen:m1
	Al	subst:sg:acc:m1
	Ala	subst:sg:nom:f
	Alo	subst:sg:gen:m1
	Alo	subst:sg:acc:m1
ma	space
	mieć	fin:sg:ter:imperf
	mój	adj:sg:nom:f:pos
	mój	adj:sg:voc:f:pos
kota	space
	Kot	subst:sg:gen:m1
	Kot	subst:sg:acc:m1
	kot	subst:sg:gen:m1
	kot	subst:sg:acc:m1
	kot	subst:sg:gen:m2
	kot	subst:sg:acc:m2
	kota	subst:sg:nom:f
.	none
	.	interp
```

2. Clone KRNNT repository:
```bash
git clone https://github.com/kwrobel-nlp/krnnt.git
```

3. Install dependencies.
```bash
pip3 install -e .
```

## Evaluation

Accuracy tested with 10-fold cross validation on National Corpus of Polish.

Accuracy lower bound | Accuracy lower bound for unknown tokens
------------ | -------------
93.72% | 69.03%

## PolEval

The tagger particaipated in PolEval 2017 competition: http://poleval.pl/

There is some problem with Keras version higher than 2.1.2.

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


5. Train for 100 epochs. Add `-d 0.1` for using 10% of training data as development data set. 

```
krnnt]$ time python3 krnnt_train.py train-reanalyzed.shuf.spickle --patience 100
Model is saved under: weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5.final
Lemmatisation model is saved under: lemmatisation_1810e860-6351-11e7-ae0b-a0000220fe80.pkl
Dictionary is saved under: train-reanalyzed.shuf.spickle_FormatData2_PreprocessData_UniqueFeaturesValues
real    197m44,568s
```
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


4. Train for 100 epochs.

```
krnnt]$ time python3 krnnt_train.py -p train-merged.shuf.spickle --patience 100
Model is saved under: weight_1810e860-6351-11e7-ae0b-a0000220fe80.hdf5.final
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


# Input formats

* raw text
  * text format - documents are separated by double new line (empty line) 
* pretokenized text
  * JSON
    * verbose
    * compact

## Pretokenized text

The format consists of documents. Each document have sentences, and each sentence have tokens.

Fields for each token:
* `form`
* `separator` (optional) - white characters before token: `newline`, `space` or `none`
* `start` (optional) - staring offset of the token
* `end` (optional) - ending offset of the token

`Separator` is a feature for the classifier. 
If `separator` is not provided and `start` and `end` positions are provided then `separator` is computed.
If `separator`, `start` and `end` fields are not provided then `separator` is set to True.

### Verbose JSON

Verbose JSON uses dictionaries.

```json
{
  "documents": [
    {
      "text": "Lubię placki. Ala ma kota.\nRaz dwa trzy.",
      "sentences": [
        {
          "tokens": [
            {
              "form": "Lubię",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": "placki",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": ".",
              "separator": false,
              "start": 0,
              "end": 0
            }
          ]
        },
        {
          "tokens": [
            {
              "form": "Ala",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": "ma",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": "kota",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": ".",
              "separator": false,
              "start": 0,
              "end": 0
            }
          ]
        }
      ]
    },
    {
      "text": "",
      "sentences": [
        {
          "tokens": [
            {
              "form": "Raz",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": "dwa",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": "trzy",
              "separator": true,
              "start": 0,
              "end": 0
            },
            {
              "form": ".",
              "separator": false,
              "start": 0,
              "end": 0
            }
          ]
        }
      ]
    }
  ]
}
```

### Compact JSON

Compact JSON uses lists and positional fields (for speed and memory efficiency).

```json
[
  [
    [["Lubię",true],["placki",true],[".",false]],
    [["Ala",true],["ma",true],["kota",true],[".",false]]
  ],
  [
    [["Raz",true],["dwa",true],["trzy",true],[".",false]]
  ]
]
```


# Output formats

* JSON
* JSONL - each document in separate line
* TSV (CONLL)
* XCES
* plain - sentences divided by one empty line, documents by two empty lines


# HTTP options

Default output format is `plain`. It can be changed by request parameter `output_format`, 
e.g.:

```bash
$ curl -X POST "localhost:9200/?output_format=conll" -d "Ala ma kota."
Ala	Ala	1	subst:sg:nom:f	0	3
ma	mieć	1	fin:sg:ter:imperf	4	6
kota	kot	1	subst:sg:acc:m2	7	11
.	.	0	interp	11	12
```

# Scripts

* `analyze_corpus_tagset_date.py` - analyze corpus tagset version
