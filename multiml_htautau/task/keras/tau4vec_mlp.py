from . import Tau4vec_BaseTask


class Tau4vec_MLPTask(Tau4vec_BaseTask):
    ''' Tau4vec MLP task
    '''
    def __init__(self,
                 layers_images=None,
                 layers_calib=None,
                 activation=None,
                 batch_norm=False,
                 **kwargs):
        """

        Args:
            layers_images (list(int)): the number of nodes in hidden layers in MLP that used for image processing.
            layers_calib (list(int)): the number of nodes in hidden layers in MLP that used for calibration.
            activation (str): activation function for MLP.
            batch_norm (bool): use batch normalization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._layers_images = layers_images
        self._layers_calib = layers_calib
        self._activation = activation
        self._batch_norm = batch_norm

    def build_model(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Concatenate
        from tensorflow.keras.layers import Add
        from tensorflow.keras import backend as K

        input_energy, input_jet = self.get_inputs()
        x = K.reshape(input_energy, [-1] + self._input_shape)
        input_jet_reshape_4 = K.reshape(input_jet, (-1, self._n_features))
        input_jet_reshape_3 = input_jet_reshape_4[:, 0:3]  # mass is not used

        x = Flatten()(x)

        from multiml.task.keras.modules import MLPBlock
        mlp1 = MLPBlock(layers=self._layers_images,
                        activation=self._activation,
                        activation_last='linear',
                        batch_norm=self._batch_norm)
        x = mlp1(x)

        x = Concatenate(axis=1)([x, input_jet_reshape_4])

        if len(self._layers_calib) > 1:
            mlp2 = MLPBlock(layers=self._layers_calib[:-1],
                            activation=self._activation,
                            activation_last=self._activation,
                            batch_norm=self._batch_norm)
            x = mlp2(x)

        x = Dense(self._layers_calib[-1], activation=None)(x)
        x = Activation('linear')(x)

        x = Add()([x, input_jet_reshape_3])

        x = self._set_phi_within_valid_range(x)

        x = K.reshape(x, (-1, len(self._output_var_names)))

        self._model = Model(inputs=[input_energy, input_jet], outputs=[x])

        self.compile_model()
