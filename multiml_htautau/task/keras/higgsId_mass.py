from . import HiggsID_BaseTask


class HiggsID_MassTask(HiggsID_BaseTask):
    ''' HiggsID MLP task
    '''
    def __init__(self, layers=None, activation=None, batch_norm=False, scale_mass=1., **kwargs):
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

    def mass_layer(self, tau_4vec):
        import tensorflow as tf
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras import backend as K
        tau_4vec = K.reshape(tau_4vec, (-1, self._njets, self._n_features))
        pt = K.exp(K.clip(tau_4vec[:, :, 0], -7., 7.)) - 0.1
        eta = tau_4vec[:, :, 1]
        phi = tau_4vec[:, :, 2]
        mass = 1.777

        px = pt * K.cos(phi)
        py = pt * K.sin(phi)
        pz = pt * tf.math.sinh(K.clip(eta, -5, 5))
        epsilon = 0.1  # avoid nan when e=0. sqrt(x)^' = -1/2 * 1/sqrt(x)
        e = K.sqrt(epsilon + px**2 + py**2 + pz**2 + mass**2)
        px = K.reshape(px, (-1, self._njets, 1))
        py = K.reshape(py, (-1, self._njets, 1))
        pz = K.reshape(pz, (-1, self._njets, 1))
        e = K.reshape(e, (-1, self._njets, 1))
        tau_4vec = Concatenate(axis=2)([px, py, pz, e])
        tau_4vec = K.sum(tau_4vec, axis=1)
        px = tau_4vec[:, 0]
        py = tau_4vec[:, 1]
        pz = tau_4vec[:, 2]
        e = tau_4vec[:, 3]
        masssq = e**2 - (px**2 + py**2 + pz**2)
        mass = K.sqrt(epsilon + masssq)
        mass = K.reshape(mass, [-1, 1])
        return mass

    def build_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Lambda
        from multiml.task.keras.modules import MLPBlock

        input = self.get_inputs()[0]
        x = input

        x = Lambda(self.mass_layer, output_shape=(1, ))(x)

        x *= self._scale_mass

        mlp = MLPBlock(layers=self._layers,
                       activation=self._activation,
                       activation_last=self._activation_last,
                       batch_norm=self._batch_norm)
        x = mlp(x)

        self._model = Model(inputs=input, outputs=x)

        self.compile_model()

    def _get_custom_objects(self):
        return {"mass_layer": self.mass_layer}
