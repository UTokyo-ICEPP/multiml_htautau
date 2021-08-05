from torch.nn import Module
from torch import transpose
from multiml.task.pytorch.modules import LSTMBlock, LSTMBlock_HPS
from multiml.task.pytorch.modules import MLPBlock, MLPBlock_HPS
from multiml import Hyperparameters

from . import HiggsID_BaseTask


class HiggsID_LSTMTask(HiggsID_BaseTask):
    ''' HiggsID LSTM task
    '''
    def __init__(
            self,
            hps=None,
            #  layers_lstm=None,
            #  layers_mlp=None,
            #  activation_last=None,
            #  batch_norm=False,
            n_jets=2,
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
        self._hps = hps
        self._n_jets = n_jets

    def build_model(self):
        self._model = _HiggsID_LSTMTask(self._hps, n_jets=self._n_jets)
        #self._model_compile()


class _HiggsID_LSTMTask(Module):
    def __init__(self, hps, n_jets=2, **kwargs):

        super().__init__(**kwargs)
        self._hps = hps

        self.n_first_layer = self._hps['layers_lstm'][0][0]  # this should be same in all choice
        self.n_jets = n_jets

        self._hps_lstm = Hyperparameters()
        self._hps_lstm.set_alias(alias={
            'layers_lstm': 'layers',
            'activation_lstm': 'activation',
            'batch_norm_lstm': 'batch_norm'
        })
        self._hps_lstm.add_hp_from_dict(self._hps, is_alias=True)

        self._hps_mlp = Hyperparameters()
        self._hps_mlp.set_alias(
            alias={
                'layers_mlp': 'layers',
                'activation_mlp': 'activation',
                'batch_norm_mlp': 'batch_norm',
                'activation_last': 'activation_last'
            })
        self._hps_mlp.add_hp_from_dict(self._hps, is_alias=True)

        self._lstm = LSTMBlock_HPS(self._hps_lstm)
        self._mlp = MLPBlock_HPS(self._hps_mlp)

        self.hps = Hyperparameters()
        self.hps.add_hp_from_dict(self._hps)

    def get_hps_parameters(self):
        return self.hps.get_hps_parameters()

    def choice(self):
        return self._choice

    def choice(self, choice):
        self._choice = choice
        self._lstm.set_active_hps(self._choice)
        self._mlp.set_active_hps(self._choice)

    def forward(self, x):
        x = transpose(x.reshape(-1, self.n_jets, self.n_first_layer), 1, 0)
        x = self._lstm(x)[-1]
        x = self._mlp(x)
        return x
