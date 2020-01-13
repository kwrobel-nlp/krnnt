# KRNNT

KRNNT is a morphological tagger for Polish based on recurrent neural networks. 

[Try KRNNT online](https://krnnt-f3esrhez2q-ew.a.run.app/)

It was presented at 8th Language & Technology Conference. More details are available in the paper:
```
@inproceedings{wrobel2017,
  author       = "Wróbel, Krzysztof",
  editor       = "Vetulani, Zygmunt and Paroubek, Patrick",
  title        = "KRNNT: Polish Recurrent Neural Network Tagger",
  year         = "2017",
  booktitle    = "Proceedings of the 8th Language \& Technology Conference: Human Language Technologies as a Challenge for Computer Science and Linguistics",
  publisher    = "Fundacja Uniwersytetu im. Adama Mickiewicza w~Poznaniu",
  pages        = {386-391},
  pdf          = "http://ltc.amu.edu.pl/book2017/papers/PolEval1-6.pdf"
}
```

Online version is available at: http://ltc.amu.edu.pl/book2017/papers/PolEval1-6.pdf

Copy: https://www.researchgate.net/publication/333566748_KRNNT_Polish_Recurrent_Neural_Network_Tagger

## External tools

The tagger uses external tools: tokenizer Toki and morphological analyzer Morfeusz. Maca (http://nlp.pwr.wroc.pl/redmine/projects/libpltagger/wiki) integrates both tools.

The tagset is described here: http://nkjp.pl/poliqarp/help/ense2.html

## Getting started

You can run KRNNT using docker or by manual installation.

### Docker

Docker image was prepared by Aleksander Smywiński-Pohl and instrutions are available at: https://hub.docker.com/r/djstrong/krnnt/

1. Download and start the server.
```bash
docker run -p 9003:9003 -it djstrong/krnnt:1.0.0
```
2. Tag a text using POST request or open http://localhost:9003 in a browser.
```bash
$ curl -XPOST localhost:9003 -d "Ala ma kota."
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

Please refer to the docker file: https://github.com/kwrobel-nlp/dockerfiles/blob/morfeusz2/tagger/Dockerfile

1. Install Maca: http://nlp.pwr.wroc.pl/redmine/projects/libpltagger/wiki

Make sure that command `maca-analyse` works:
```bash
echo "Ala ma kota." | maca-analyse -qc morfeusz2-nkjp
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

The tagger participated in PolEval 2017 competition: http://2017.poleval.pl/index.php/results/

The original submission was created using tag `poleval`.

### Training

1. Install KRNNT.

```
krnnt]$ pip3 install -e .
```


2. Prepare training data.

```
krnnt]$ time process_xces.py train-gold.xml train-gold.spickle 
0:14.75
```


3. Reanalyze corpus with Maca.

```
krnnt]$ python3 reanalyze.py train-gold.spickle train-reanalyzed.spickle
Number of sentences by Maca vs gold 9 10
Number of sentences by Maca vs gold 7 8                                                                                                                                         | 2/18484 [00:00<15:41, 19.63it/s]Number of sentences by Maca vs gold 7 6
Number of sentences by Maca vs gold 8 10
Number of sentences by Maca vs gold 3 3
Number of sentences by Maca vs gold 3 3
Number of sentences by Maca vs gold 7 7
Number of sentences by Maca vs gold 4 4
...
1:30.30
```
Ensure that last two numbers in each row are usually the same. Zeros indicates problems with Maca.

4. Shuffle data (optional).

```
krnnt]$ time python3 shuffle.py train-reanalyzed.spickle train-reanalyzed.shuf.spickle
0:28.95
```

5. Preprocess data.

```
krnnt]$ time python3 preprocess_data.py train-reanalyzed.shuf.spickle train-reanalyzed.shuf.spickle.preprocessed
0:56.80
```

6. Create dictionary for all features.

```
krnnt]$ time python3 create_dict.py train-reanalyzed.shuf.spickle.preprocessed train-reanalyzed.shuf.spickle.dict
0:18.93
```

7. Train lemmatization module.

```
krnnt]$ time python3 train_lemmatization.py train-reanalyzed.shuf.spickle.preprocessed --hash model_nkjp
0:09.55
```


8. Train for 150 epochs. Add `-d 0.1` for using 10% of training data as development data set. 

```
krnnt]$ python3 train.py train-reanalyzed.shuf.spickle.preprocessed train-reanalyzed.shuf.spickle.dict -e 150 --patience 150 --hash model_nkjp --test_data poleval-reanalyzed.shuf.spickle.preprocessed

```
Check `~/.keras/keras.json` for `"image_data_format": "channels_first"`.

9. Testing.

```
krnnt]$ time python3 krnnt_run.py weight_model_nkjp.hdf5.final lemmatisation_model_nkjp.pkl train-reanalyzed.shuf.spickle.dict < test-raw.txt > test-raw.xml
0:09.02
```
10. Evaluate.

```
krnnt]$ python2 tagger-eval.py gold-task-c.xml test-raw.xml -t poleval -s
PolEval 2017 competition scores
-------------------------------
POS accuracy (Subtask A score): 	92.3308%
POS accuracy (known words): 	92.3308%
POS accuracy (unknown words): 	0.0000%
Lemmatization accuracy (Subtask B score): 	96.8816%
Lemmatization accuracy (known words): 	96.8816%
Lemmatization accuracy (unknown words): 	0.0000%
Overall accuracy (Subtask C score): 	94.6062%
```

## Training on gold segmentation

2. Prepare training data.

```
krnnt]$ time python3 process_xces.py train-analyzed.xml train-analyzed.spickle
real	0m37.211s

krnnt]$ time python3 process_xces.py train-gold.xml train-gold.spickle
real	0m14.750s

krnnt]$ time python3 merge_analyzed_gold.py train-gold.spickle train-analyzed.spickle train-merged.spickle
real	0m18.215s
```


3. Shuffle data (optional).

```
krnnt]$ time python3 shuffle.py train-merged.spickle train-merged.shuf.spickle
real	0m21,636s
```

4. Preprocess data.

```
time python3 preprocess_data.py -p train-merged.shuf.spickle train-merged.shuf.spickle.preprocessed
real	0m52,872s
```

6. Create dictionary for all features.

```
krnnt]$ time python3 create_dict.py train-merged.shuf.spickle.preprocessed train-merged.shuf.spickle.dict
real	0m19,756s
```

7. Train lemmatization module.

```
krnnt]$ time python3 train_lemmatization.py train-merged.shuf.spickle.preprocessed --hash model_nkjp_pre
real	0m7,184s
```

8. Train for 150 epochs.

```
krnnt]$ python3 train.py train-merged.shuf.spickle.preprocessed train-merged.shuf.spickle.dict -e 150 --patience 150 --hash model_nkjp_pre --test_data poleval-reanalyzed.shuf.spickle.preprocessed

```

9. Testing.

```
krnnt]$ time python3 krnnt_run.py -p weight_model_nkjp_pre.hdf5.final lemmatisation_model_nkjp_pre.pkl train-merged.shuf.spickle.dict < test-analyzed.xml > test-analyzed.xml.pred
real	0m8,426s
```

10. Evaluate.

```
krnnt]$ python2 tagger-eval.py gold-task-a-b.xml test-analyzed.xml.pred -t poleval -s
PolEval 2017 competition scores
-------------------------------
POS accuracy (Subtask A score): 	93.9106%
POS accuracy (known words): 	93.9106%
POS accuracy (unknown words): 	0.0000%
Lemmatization accuracy (Subtask B score): 	97.8654%
Lemmatization accuracy (known words): 	97.8654%
Lemmatization accuracy (unknown words): 	0.0000%
Overall accuracy (Subtask C score): 	95.8880%
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

* text
  * raw format (default) - one document (e.g. Wikipedia article, need of token offsets)
  * `lines` format - documents are separated by empty line 
* pretokenized text
  * JSON
    * verbose
    * compact

Input format is determined automatically, except `lines` format.

## Text

### Raw format

### Lines format

?input_format=lines

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
$ curl -X POST "localhost:9003/?output_format=conll" -d "Ala ma kota."
Ala	Ala	1	subst:sg:nom:f	0	3
ma	mieć	1	fin:sg:ter:imperf	4	6
kota	kot	1	subst:sg:acc:m2	7	11
.	.	0	interp	11	12
```

`remove_aglt` (default `0`) - indicates if aglt tags should be removed 
`remove_blank` (default `1`) - indicates if blank tags should be removed 

# Scripts

* `analyze_corpus_tagset_date.py` - analyze corpus tagset version
