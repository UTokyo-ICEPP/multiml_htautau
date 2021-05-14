from torch.nn import Module
from torch import transpose
from multiml.task.pytorch.modules import LSTMBlock
from multiml.task.pytorch.modules import MLPBlock

from . import HiggsID_BaseTask


class HiggsID_LSTMTask(HiggsID_BaseTask):
    ''' HiggsID LSTM task
    '''
    def __init__(self, layers_lstm=None,
                 layers_mlp=None, activation_last=None,
                 batch_norm=False, n_jets=2,
                 **kwargs):
        """

        Args:
            layers_lstm (list(int)): the number of hidden hodes in LSTM layer.
                               If more than single node is given, stacked LSTM layers are used.
            layers_mlp :
            activation_last :
            batch_norm (bool): use batch normalization
            n_jets :
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._layers_lstm = layers_lstm
        self._layers_mlp = layers_mlp
        self._activation_last = activation_last
        self._batch_norm = batch_norm
        self._n_jets = n_jets

    def build_model(self):
        self._model = _HiggsID_LSTMTask(
            layers_lstm=self._layers_lstm,
            layers_mlp=self._layers_mlp,
            activation_last=self._activation_last,
            batch_norm=self._batch_norm,
            n_jets=self._n_jets
        )

        #self._model_compile()


class _HiggsID_LSTMTask(Module):
    def __init__(self,
                 layers_lstm=[4, 32, 32, 32, 1],
                 layers_mlp=[1, 1],
                 activation_last='Identity',
                 batch_norm=False,
                 n_jets=2,
                 **kwargs):
        super(_HiggsID_LSTMTask, self).__init__(**kwargs)
        self.layers_lstm = layers_lstm
        self.n_jets = n_jets
        self.lstm = LSTMBlock(layers=layers_lstm,
                              batch_norm=batch_norm)
        self.mlp = MLPBlock(layers=layers_mlp,
                            activation='Identity',
                            activation_last=activation_last,
                            batch_norm=batch_norm)

    def forward(self, x):
        x = transpose(
            x.reshape(-1, self.n_jets, self.layers_lstm[0]),
            1, 0)
        x = self.lstm(x)[-1]
        x = self.mlp(x)
        return x
