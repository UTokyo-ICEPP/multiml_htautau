from multiml import logger
from multiml.task.keras import KerasBaseTask


class Tau4vec_BaseTask(KerasBaseTask):
    ''' Keras MLP task
    '''
    def __init__(self, input_vars_energy, input_vars_jet, input_njets, **kwargs):
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
        from tensorflow.keras.layers import Input
        shape = [self._njets] + self._input_shape
        input_energy = Input(shape=shape)
        input_jet = Input(shape=len(self._input_vars_jet))
        return [input_energy, input_jet]

    @staticmethod
    def _wrap_phi_to_2pi(x):
        ''' Shift input angle x to the range of [-pi, +pi]
        '''
        from tensorflow.math import floormod
        import math
        pi = math.pi
        x = floormod(2 * pi + floormod(x + pi, 2 * pi), 2 * pi) - pi
        # x = tf.atan2(tf.sin(x), tf.cos(x))  # Alternative
        return x

    @staticmethod
    def _set_phi_within_valid_range(x):
        from tensorflow.keras import backend as K

        assert (K.int_shape(x) == (None, 3))

        x_phi = x[:, 2]
        x_phi = Tau4vec_BaseTask._wrap_phi_to_2pi(x_phi)
        x_phi = K.expand_dims(x_phi)
        x = K.concatenate([x[:, 0:2], x_phi], axis=1)
        return x
