from . import HiggsID_BaseTask


class HiggsID_LSTMTask(HiggsID_BaseTask):
    ''' HiggsID LSTM task
    '''
    def __init__(self, nodes=None, batch_norm=False, **kwargs):
        """

        Args:
            nodes (list(int)): the number of hidden hodes in LSTM layer.
                               If more than single node is given, stacked LSTM layers are used.
            batch_norm (bool): use batch normalization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._nodes = nodes
        self._batch_norm = batch_norm

    def build_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras import backend as K

        input = self.get_inputs()[0]
        x = K.reshape(input, (-1, self._njets, self._n_features))

        for i, node in enumerate(self._nodes):
            if i == len(self._nodes) - 1:
                layer = LSTM(units=node, return_sequences=False)
                layer._could_use_gpu_kernel = False
                x = layer(x)
            else:
                layer = LSTM(units=node, return_sequences=True)
                layer._could_use_gpu_kernel = False
                x = layer(x)

        x = Dense(1, activation=None)(x)
        if self._batch_norm:
            x = BatchNormalization()(x)

        x = Activation(self._activation_last)(x)

        self._model = Model(inputs=input, outputs=x)

        self.compile_model()
