from torch.nn import Module
from multiml.task.pytorch.modules import MLPBlock

from . import HiggsID_BaseTask


class HiggsID_MLPTask(HiggsID_BaseTask):
    ''' HiggsID MLP task
    '''
    def __init__(self,
                 layers=None,
                 activation=None,
                 activation_last=None,
                 batch_norm=False, **kwargs):
        """

        Args:
            layers :
            activation :
            activation_last :
            batch_norm (bool): use batch normalization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._layers = layers
        self._activation = activation
        self._activation_last = activation_last
        self._batch_norm = batch_norm

    def build_model(self):
        self._model = _HiggsID_MLPTask(
            layers=self._layers,
            activation=self._activation,
            activation_last=self._activation_last,
            batch_norm=self._batch_norm
        )

        #self._model_compile()


class _HiggsID_MLPTask(Module):
    def __init__(self,
                 layers=[8, 32, 32, 32, 1],
                 activation='ReLU',
                 activation_last='Identity',
                 batch_norm=False,
                 **kwargs):
        super(_HiggsID_MLPTask, self).__init__(**kwargs)
        self.mlp = MLPBlock(layers=layers,
                            activation=activation,
                            activation_last=activation_last,
                            batch_norm=batch_norm)

    def forward(self, x):
        x = self.mlp(x)
        return x
