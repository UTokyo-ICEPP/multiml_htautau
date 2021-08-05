from torch.nn import Module
from multiml.task.pytorch.modules import MLPBlock, MLPBlock_HPS
from multiml import Hyperparameters

from . import HiggsID_BaseTask


class HiggsID_MLPTask(HiggsID_BaseTask):
    ''' HiggsID MLP task
    '''
    def __init__(self, hps=None, **kwargs):
        """

        Args:
            layers :
            activation :
            activation_last :
            batch_norm (bool): use batch normalization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)
        self._hps = hps

    def build_model(self):
        self._model = _HiggsID_MLPTask(self._hps)
        #self._model_compile()


class _HiggsID_MLPTask(Module):
    def __init__(self, hps, **kwargs):
        super().__init__(**kwargs)
        self._hps = hps
        self._mlp_hps = Hyperparameters()
        self._mlp_hps.add_hp_from_dict(self._hps)
        self._mlp = MLPBlock_HPS(self._mlp_hps)

        self.hps = Hyperparameters()
        self.hps.add_hp_from_dict(self._hps)

    def get_hps_parameters(self):
        return self.hps.get_hps_parameters()

    def choice(self):
        return self._choice

    def choice(self, choice):
        self._choice = choice
        self._mlp.set_active_hps(self._choice)

    def forward(self, x):
        x = self._mlp(x)
        return x
