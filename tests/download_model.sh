#!/usr/bin/env bash
cd ..
mkdir model_data -p
cd model_data

if [ ! -f "weights.hdf5" ]; then
    wget "https://github.com/kwrobel-nlp/krnnt/releases/download/poleval/reanalyze_150epochs_train1.0.zip"
    unzip reanalyze_150epochs_train1.0.zip
    mv lemmatisation_reana150_1.0.pkl lemmatisation.pkl
    mv weights_reana150_1.0.hdf5 weights.hdf5
fi