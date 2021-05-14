from torch.nn import Module
from multiml.task.pytorch.modules import MLPBlock

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
        self._model = _Tau4vec_MLPTask(
            layers_images=self._layers_images,
            layers_calib=self._layers_calib,
            activation=self._activation,
            batch_norm=self._batch_norm
        )

        #self._model_compile()


class _Tau4vec_MLPTask(Module):
    def __init__(self,
                 layers_images=[768, 32, 32, 32, 4],
                 layers_calib=[8, 32, 32, 32, 4],
                 activation='ReLU',
                 batch_norm=False,
                 **kwargs):
        super(_Tau4vec_MLPTask, self).__init__(**kwargs)
        self._mlp1 = MLPBlock(layers=layers_images,
                              activation=activation,
                              activation_last='Identity',
                              batch_norm=batch_norm)
        self._mlp2 = MLPBlock(layers=layers_calib,
                              activation=activation,
                              activation_last='Identity',
                              batch_norm=batch_norm)
        self._layers_calib = layers_calib
        self._len_output_vers = layers_calib[-1] * 2

    def forward(self, x):
        from torch import cat
        fig = x[0].reshape(-1, 3, 16, 16)
        x_1 = fig.reshape(fig.size(0), -1)
        x_1 = self._mlp1(x_1)

        input_jet_reshape_4 = x[1].reshape(-1, 4)
        input_jet_reshape_3 = input_jet_reshape_4[:, :3]  # mass is not used

        x = cat((x_1, input_jet_reshape_4), dim=1)

        x = self._mlp2(x)
        if self._layers_calib[-1] == 4:
            x = x + input_jet_reshape_4
        elif self._layers_calib[-1] == 3:
            x = x + input_jet_reshape_3

        x = Tau4vec_BaseTask.set_phi_within_valid_range(x)
        output = x.reshape(-1, self._layers_calib[-1] * 2)
        return output
