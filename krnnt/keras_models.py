
import keras

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, GRU, TimeDistributed, \
    Masking, Lambda, BatchNormalization
# from keras.layers.wrappers import Bidirectional
import uuid
import logging

def reverse_func(x):
    import keras.backend as K
    assert K.ndim(x) == 3, "Should be a 3D tensor."
    rev = K.permute_dimensions(x, (1, 0, 2))[::-1]
    return K.permute_dimensions(rev, (1, 0, 2))

class ExperimentParameters():
    def __init__(self, pref, testing=False):
        """

        @type pref: dict
        """
        self.pref=pref.copy()
        if testing:
            pass #TODO self.h
        else:
            if 'h' not in self.pref:
                self.pref['h']=str(uuid.uuid1())
            self.h=self.pref['h']
            self.pref['weight_path'] = 'weight_' + self.h + '.hdf5'
            self.pref['lemmatisation_path'] = 'lemmatisation_' + self.h + '.pkl'


    def save_prefs(self):
        #TODO
        print(self.pref)

class KerasModel():
    """
    @type model: Model
    """
    def __init__(self, parameters):
        """

        @type parameters: ExperimentParameters
        """
        self.parameters=parameters

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
        #TODO
        return model_yaml

    def create_model(self):
        raise NotImplementedError

class GRUx(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0)(
            inputs)

        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(forwards)

        self.model = Model(input=inputs, output=x)

        keras.objectives.categorical_crossentropy()
        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class LSTMx(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = LSTM(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0)(
            inputs)

        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(forwards)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')


import keras.backend as K
def categorical_crossentropy2(y_true, y_pred):
    print(y_pred.shape)
    print(y_pred)
    return K.categorical_crossentropy(y_pred, y_true)

class BDGRU(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0)(
            inputs)

        backwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                        consume_less='gpu',
                        go_backwards=True, dropout_W=0.0, dropout_U=0.0)(inputs)

        reverse = Lambda(reverse_func)
        backwards = reverse(backwards)

        x = merge([forwards, backwards], mode='concat')

        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class BDGRU2Ex(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        sx=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=False,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0), input_shape=(None, features_length))(
            inputs)

        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0), input_shape=(None, features_length))(
            inputs)


        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class BDGRU2(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0), input_shape=(None, features_length))(
            inputs)


        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class BDGRU_GRU(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0)(
            inputs)

        backwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                        consume_less='gpu',
                        go_backwards=True, dropout_W=0.0, dropout_U=0.0)(inputs)

        reverse = Lambda(reverse_func)
        backwards = reverse(backwards)

        x = merge([forwards, backwards], mode='concat')

        x = GRU(self.parameters.pref['output_length'], activation='softmax', return_sequences=True,
                        consume_less='gpu')(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class BDGRU_Dense_Dropout(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0)(
            inputs)

        backwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                        consume_less='gpu',
                        go_backwards=True, dropout_W=0.0, dropout_U=0.0)(inputs)

        reverse = Lambda(reverse_func)
        backwards = reverse(backwards)

        x = merge([forwards, backwards], mode='concat')

        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class BDGRU_Dense_Dropout_UDropout(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5)(
            inputs)

        backwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                        consume_less='gpu',
                        go_backwards=True, dropout_W=0.0, dropout_U=0.5)(inputs)

        reverse = Lambda(reverse_func)
        backwards = reverse(backwards)

        x = merge([forwards, backwards], mode='concat')

        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')


class BDGRU_Dense_Dropout_UDropout_CRF(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5)(
            inputs)

        backwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                        consume_less='gpu',
                        go_backwards=True, dropout_W=0.0, dropout_U=0.5)(inputs)

        reverse = Lambda(reverse_func)
        backwards = reverse(backwards)

        x = merge([forwards, backwards], mode='concat')

        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        crf = keras.layers.ChainCRF()
        x = crf(x)

        self.model = Model(input=inputs, output=x)


        self.loss = crf.loss
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class BDGRU_Dense_Dropout_UDropoutNadam(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5)(
            inputs)

        backwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                        consume_less='gpu',
                        go_backwards=True, dropout_W=0.0, dropout_U=0.5)(inputs)

        reverse = Lambda(reverse_func)
        backwards = reverse(backwards)

        x = merge([forwards, backwards], mode='concat')

        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Nadam()

class BDGRU2_Dense_Dropout_UDropout_Masking(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))
        x = Masking()(inputs)
        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0), input_shape=(None, features_length))(
            x)


        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class BDGRU_BDGRU_Dense_Dropout_UDropout(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5), input_shape=(None, features_length))(
            inputs)
        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5), input_shape=(None, features_length))(
            x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')



class BEST(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))
        x = inputs

        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='mem', dropout_W=0.0, dropout_U=0.5), input_shape=(None, features_length))(x)
        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='mem', dropout_W=0.0, dropout_U=0.5), input_shape=(None, features_length))(x)
        x = Dropout(0.5)(x)
        #x = BatchNormalization()(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Nadam()

class BDGRU_BDGRU_BDGRU_Dense_Dropout_UDropout(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']

        inputs = Input(shape=(None, features_length))

        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5), input_shape=(None, features_length))(
            inputs)
        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5), input_shape=(None, features_length))(
            x)
        x=keras.layers.wrappers.Bidirectional(GRU(self.parameters.pref['internal_neurons'], return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.5), input_shape=(None, features_length))(
            x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(x)

        self.model = Model(input=inputs, output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')

class GRUE(KerasModel):
    def __init__(self, parameters):
        super().__init__(parameters)

    def create_model(self):
        features_length = self.parameters.pref['features_length']


        input_e = Input(shape=(None,), dtype='int32', name='input_e')
        e = Embedding(output_dim=16, input_dim=10000)(input_e)

        inputs = Input(shape=(None, features_length))

        x = merge([inputs, e], mode='concat')

        forwards = GRU(self.parameters.pref['internal_neurons'], input_shape=(None, features_length), return_sequences=True,
                       consume_less='gpu', dropout_W=0.0, dropout_U=0.0)(
            x)

        x = TimeDistributed(Dense(self.parameters.pref['output_length'], activation='softmax'))(forwards)

        self.model = Model(input=[inputs,input_e], output=x)


        self.loss = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        logging.info('Model created')
