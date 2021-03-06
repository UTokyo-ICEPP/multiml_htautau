from torch.nn import Module
from multiml.task.pytorch.modules import MLPBlock

from . import HiggsID_BaseTask


class HiggsID_MassTask(HiggsID_BaseTask):
    ''' HiggsID Mass task
    '''
    def __init__(self,
                 layers=None,
                 activation=None,
                 batch_norm=False,
                 scale_mass=1.,
                 n_jets=2,
                 n_input_vars=6,
                 **kwargs):
        """

        Args:
            layers (list(int)): the number of nodes in hidden layers in MLP that used for mass transformation.
            activation (str): activation function for MLP.
            batch_norm (bool): use batch normalization
            scale_mass (float): scaling output of mass layer
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._layers = layers
        self._activation = activation
        self._batch_norm = batch_norm
        self._scale_mass = scale_mass
        self._n_jets = n_jets
        self._n_input_vars = n_input_vars

    def build_model(self):
        self._model = _HiggsID_MassTask(
            layers=self._layers,
            activation=self._activation,
            activation_last=self._activation_last,
            batch_norm=self._batch_norm,
            scale_mass=self._scale_mass,
            n_jets=self._n_jets,
            n_input_vars=self._n_input_vars
        )

        #self._model_compile()


class _HiggsID_MassTask(Module):
    def __init__(self,
                 layers=[1, 64, 64, 1],
                 activation='ReLU',
                 activation_last='Identity',
                 batch_norm=False,
                 scale_mass=1./125.,
                 n_jets=2,
                 n_input_vars=6,
                 **kwargs):
        super(_HiggsID_MassTask, self).__init__(**kwargs)
        self.scale_mass = scale_mass
        self.n_input_vars = n_input_vars
        self.n_jets = n_jets

        self.mlp = MLPBlock(layers=layers,
                            activation=activation,
                            activation_last=activation_last,
                            batch_norm=batch_norm)

    def forward(self, x):
        x = self.mass_layer(x, self.n_jets, self.n_input_vars)
        x = x * self.scale_mass
        x = self.mlp(x)
        return x

    @staticmethod
    def mass_layer(tau_4vec, n_jets, n_input_vars):
        from torch import chunk, clamp, cos, exp, sin, sinh, sqrt, stack, sum
        tau_4vec = tau_4vec.reshape(-1, n_jets, n_input_vars // n_jets)
        pt = exp(
            clamp(tau_4vec[:, :, 0], min=-7., max=7.)
        ) - 0.1
        eta = tau_4vec[:, :, 1]
        phi = tau_4vec[:, :, 2]
        mass = 1.777

        px = pt * cos(phi)
        py = pt * sin(phi)
        pz = pt * sinh(clamp(eta, min=-5, max=5))
        epsilon = 0.1  # avoid nan when e=0. sqrt(x)^' = -1/2 * 1/sqrt(x)
        e = sqrt(
            epsilon + px**2 + py**2 + pz**2 + mass**2
        )

        tau_4vec = stack([px, py, pz, e], dim=2)
        tau_4vec = sum(tau_4vec, dim=1)
        px, py, pz, e = chunk(tau_4vec, chunks=4, dim=1)
        mass = sqrt(
            epsilon + e**2 - (px**2 + py**2 + pz**2)
        )
        return mass
