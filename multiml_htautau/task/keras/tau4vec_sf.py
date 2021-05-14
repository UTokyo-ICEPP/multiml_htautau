from tensorflow.keras.layers import Layer
from . import Tau4vec_BaseTask


class SF_layer(Layer):
    def __init__(self,
                 sf_initializer='ones',
                 bias_initializer='zeros',
                 **kwargs):
        self.sf_initializer = sf_initializer
        self.bias_initializer = bias_initializer
        super(SF_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.sf = self.add_weight(shape=(input_dim, ),
                                  initializer=self.sf_initializer,
                                  name='sf',
                                  trainable=True)
        self.bias = self.add_weight(shape=(input_dim, ),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=True)
        super(SF_layer, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.sf + self.bias

    def get_config(self):
        return {
            'sf_initializer': self.sf_initializer,
            'bias_initializer': self.bias_initializer
        }


class Tau4vec_SFTask(Tau4vec_BaseTask):
    ''' Tau4vec SF task
    '''
    def build_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras import backend as K

        input_energy, input_jet = self.get_inputs()

        x = K.reshape(input_jet, (-1, self._n_features))
        x = x[:, 0:3]  # mass is not used

        x = SF_layer(sf_initializer='ones', bias_initializer='zeros')(x)

        x = self._set_phi_within_valid_range(x)

        x = K.reshape(x, (-1, len(self._output_var_names)))

        self._model = Model(inputs=[input_energy, input_jet], outputs=[x])

        self.compile_model()
