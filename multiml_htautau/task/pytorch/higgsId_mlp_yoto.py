from torch.nn import Module
from multiml.task.pytorch.modules.yoto import MLPBlock_Yoto
from multiml import Hyperparameters

from . import HiggsID_BaseTask


class HiggsID_MLP_Yoto_Task(HiggsID_BaseTask):
    ''' HiggsID MLP task
    '''
    def __init__(self, hps=None, **kwargs):
        """

        Args:
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)
        self._hps = hps

    def build_model(self):
        self._model = _HiggsID_MLP_Yoto_Task(self._hps)
        #self._model_compile()
        
    def get_n_layers_to_yoto(self):
        self._model.get_n_layers_to_yoto()


class _HiggsID_MLP_Yoto_Task(Module):
    def __init__(self, hps, **kwargs):
        super().__init__(**kwargs)
        self._hps = hps # hps should be dict, not Hyperparameters...
        
        self._mlp = MLPBlock_Yoto(self._hps)
        
    def set_yoto_layer(self, gamma, beta) : 
        self._mlp.set_yoto_layer(gamma, beta)

    def get_n_layers_to_yoto(self):
        return self._mlp.get_n_layers_to_yoto()

    def forward(self, x, x_gamma, x_beta):
        x = self._mlp(x, x_gamma, x_beta)
        return x
