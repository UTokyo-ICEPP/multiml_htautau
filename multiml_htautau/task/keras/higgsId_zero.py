from . import HiggsID_BaseTask


class HiggsID_ZeroTask(HiggsID_BaseTask):
    ''' HiggsID Zero task
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._trainable_model = False

    def build_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras import backend as K

        from .modules import zero_layer

        input = self.get_inputs()[0]
        x = K.reshape(input, (-1, len(self._input_var_names)))
        x = K.sum(x, axis=1)
        x = Lambda(zero_layer, output_shape=(1, ))(x)
        x = K.reshape(x, [-1, 1])
        self._model = Model(inputs=input, outputs=x)

        self.compile_model()
