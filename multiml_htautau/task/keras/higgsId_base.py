from multiml import logger
from multiml.task.keras import KerasBaseTask


class HiggsID_BaseTask(KerasBaseTask):
    ''' Keras LSTM task
    '''
    def __init__(self, input_njets=0, activation_last='sigmoid', **kwargs):
        '''

        Args:
            input_njets (int): the number of jets in inputs
            **kwargs: Arbitrary keyword arguments
        '''
        super().__init__(**kwargs)

        if not isinstance(self._true_var_names[0], str):
            self._true_var_names = self._true_var_names[0]

        self._input_shapes = [len(self._input_var_names)]
        self._njets = input_njets
        self._n_features = len(self._input_var_names) // self._njets

        self._activation_last = activation_last

    def get_inputs(self):
        from tensorflow.keras.layers import Input
        return [Input(shape=self._input_shapes)]
