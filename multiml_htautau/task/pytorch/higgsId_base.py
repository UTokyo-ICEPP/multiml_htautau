from multiml import logger
from multiml.task.pytorch import PytorchBaseTask


class HiggsID_BaseTask(PytorchBaseTask):
    ''' HiggsID Base task
    '''
    def __init__(self, input_njets=2, activation_last='Sigmoid', **kwargs):
        '''

        Args:
            label (list(str)): variable names of labels for classification
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

        if isinstance(self._output_var_names, str):
            self._output_var_names = [self._output_var_names]

    def get_inputs(self):
        return [self._input_shapes]
