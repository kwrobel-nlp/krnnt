import logging
import uuid
from typing import Dict

import keras
from keras.layers import Dense, Dropout, Input, GRU, TimeDistributed, \
    Masking
from keras.layers.wrappers import Bidirectional
from keras.models import Model


class ExperimentParameters:
    def __init__(self, pref: Dict, testing=False):
        self.pref = pref.copy()
        if testing:
            pass  # TODO self.h
        else:
            if 'h' not in self.pref:
                self.pref['h'] = str(uuid.uuid1())
            self.h = self.pref['h']
            self.pref['weight_path'] = 'weight_' + self.h + '.hdf5'
            self.pref['lemmatisation_path'] = 'lemmatisation_' + self.h + '.pkl'

    def save_prefs(self):
        # TODO
        print(self.pref)


class KerasModel:
    model: Model

    def __init__(self, parameters: ExperimentParameters):
        self.parameters = parameters

    def compile(self):
        logging.info('Model compiling')
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        logging.info('Model compiled')

    def make_predict_func(self):
        self.model._make_predict_function()

    def load_weights(self, path):
        self.model.load_weights(path)
        logging.info('Weights loaded')

    def yaml_model(self):
        model_yaml = self.model.to_yaml()
        # TODO
        return model_yaml

    def create_model(self):
        raise NotImplementedError


class BEST(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))
        x = inputs
        x = Masking(mask_value=0., input_shape=(None, features_length))(x)
        x = Bidirectional(
            GRU(self.parameters.pref['internal_neurons'], return_sequences=True, dropout=0.0, recurrent_dropout=0.5,
                implementation=1), input_shape=(None, features_length))(x)
        x = Bidirectional(
            GRU(self.parameters.pref['internal_neurons'], return_sequences=True, dropout=0.0, recurrent_dropout=0.5,
                implementation=1), input_shape=(None, features_length))(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(inputs=inputs, outputs=x)

        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Nadam()
