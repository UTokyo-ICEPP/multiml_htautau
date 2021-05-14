from . import Tau4vec_BaseTask


class Tau4vec_ZeroTask(Tau4vec_BaseTask):
    ''' Tau4vec Zero task
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._trainable_model = False

    def build_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras import backend as K

        from .modules import zero_layer

        input_energy, input_jet = self.get_inputs()
        x = K.reshape(input_jet, (-1, self._n_features))[:, 0:3]  # mass is not used
        x = K.reshape(x, (-1, self._njets * (self._n_features - 1)))

        x = Lambda(zero_layer)(x)
        x = K.reshape(x, (-1, len(self._output_var_names)))

        self._model = Model(inputs=[input_energy, input_jet], outputs=[x])

        self.compile_model()
