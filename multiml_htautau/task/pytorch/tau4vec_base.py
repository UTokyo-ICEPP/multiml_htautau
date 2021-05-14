from multiml import logger
#from multiml.task.pytorch import PytorchBaseTask_Test
from multiml.task.pytorch import PytorchBaseTask


class Tau4vec_BaseTask(PytorchBaseTask):
    ''' Pytorch MLP task
    '''
    def __init__(self, input_vars_energy, input_vars_jet,
                 input_njets, **kwargs):
        '''

        Args:
            input_vars_energy (list(str)): variables names of energy distributions for inputs
            input_vars_jet (list(str)): variables names of jets for inputs
            input_njets (int): the number of jets in inputs
            **kwargs: Arbitrary keyword arguments
        '''
        super().__init__(**kwargs)

        self._input_vars_energy = input_vars_energy
        self._input_vars_jet = input_vars_jet
        self._input_var_names = [self._input_vars_energy, self._input_vars_jet]

        self._input_shape = [16, 16, 3]
        self._njets = input_njets
        self._n_features = len(self._input_vars_jet) // self._njets

    def get_inputs(self):
        shape = [self._njets] + self._input_shape
        input_energy = shape
        input_jet = [len(self._input_vars_jet)]
        return [input_energy, input_jet]

    @staticmethod
    def wrap_phi_to_2pi_torch(x):
        """Shift input angle x to the range of [-pi, pi]
        """
        import math
        from torch import fmod
        pi = math.pi
        x = fmod(2 * pi + fmod(x + pi, 2 * pi), 2 * pi) - pi
        return x

    @staticmethod
    def wrap_phi_to_2pi_numpy(x):
        """Shift input angle x to the range of [-pi, pi]
        """
        import math
        from numpy import fmod
        pi = math.pi
        x = fmod(2 * pi + fmod(x + pi, 2 * pi), 2 * pi) - pi
        return x

    @staticmethod
    def set_phi_within_valid_range(x):
        from torch import Tensor, cat
        from numpy import concatenate, expand_dims, ndarray
        if isinstance(x, Tensor):
            x_phi = x[:, 2]
            x_phi = Tau4vec_BaseTask.wrap_phi_to_2pi_torch(x_phi)
            x_phi.unsqueeze_(1)
            x = cat([x[:, 0:2], x_phi, x[:, 3:4]], axis=1)
        elif isinstance(x, ndarray):
            x_phi = x[:, 2]
            x_phi = Tau4vec_BaseTask.wrap_phi_to_2pi_numpy(x_phi)
            x_phi = expand_dims(x_phi, 1)
            x = concatenate([x[:, 0:2], x_phi, x[:, 3:4]], axis=1)
        return x
