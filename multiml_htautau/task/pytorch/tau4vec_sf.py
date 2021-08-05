from torch.nn import Module
from torch.nn import Parameter
from torch import Tensor
from numpy import ones, zeros

from . import Tau4vec_BaseTask


class SF_layer(Module):
    def __init__(self, input_dim):
        super(SF_layer, self).__init__()
        self.sf = Parameter(Tensor(ones(input_dim)))
        self.bias = Parameter(Tensor(zeros(input_dim)))

    def forward(self, x):
        return x * self.sf + self.bias


class _Tau4vec_SFTask(Module):
    def __init__(self, n_input_vars=8, n_output_vars=6, n_jets=2):
        super(_Tau4vec_SFTask, self).__init__()
        self.sf_layer = SF_layer(input_dim=(1, n_output_vars // 2))
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars
        self.n_jets = n_jets

    def forward(self, x):
        x = x[1].reshape(-1, self.n_input_vars // self.n_jets)
        if self.n_output_vars == 6:
            x = x[:, :3]  # mass is not used
        x = self.sf_layer(x)

        x = Tau4vec_BaseTask.set_phi_within_valid_range(x)
        x = x.reshape(-1, self.n_output_vars)
        return x


class Tau4vec_SFTask(Tau4vec_BaseTask):
    ''' Tau4vec SF task
    '''
    def __init__(self, hps=None, n_input_vars=8, n_output_vars=6, n_jets=2, **kwargs):
        super(Tau4vec_SFTask, self).__init__(**kwargs)
        self._hps = hps
        self._n_input_vars = n_input_vars
        self._n_output_vars = n_output_vars
        self._n_jets = n_jets

    def build_model(self):
        self._model = _Tau4vec_SFTask(n_input_vars=self._n_input_vars,
                                      n_output_vars=self._n_output_vars,
                                      n_jets=self._n_jets)

        #self._model_compile()
