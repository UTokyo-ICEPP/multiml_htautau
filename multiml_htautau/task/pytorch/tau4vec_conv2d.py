from torch.nn import Module
from torch import cat
from multiml.task.pytorch.modules import Conv2DBlock, Conv2DBlock_HPS
from multiml.task.pytorch.modules import MLPBlock, MLPBlock_HPS
from multiml import Hyperparameters

from . import Tau4vec_BaseTask


class Tau4vec_Conv2DTask(Tau4vec_BaseTask):
    ''' Tau4vec Conv2D MLP task
    '''
    def __init__(
        self,
        hps=None,
        #  layers_conv2d=None,
        #  layers_images=None,
        #  layers_calib=None,
        #  activation=None,
        #  batch_norm=False,
        **kwargs):
        """

        Args:
            layers_conv2d (list(tuple(str, dict))): configs of conv2d layer. list of tuple(op_name, op_args).
            layers_images (list(int)): the number of nodes in hidden layers in MLP that used for image processing.
            layers_calib (list(int)): the number of nodes in hidden layers in MLP that used for calibration.
            activation (str): activation function for MLP.
            batch_norm (bool): use batch normalization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._hps = hps

    def build_model(self):

        self._model = _Tau4vec_Conv2DTask(self._hps)
        #self._model_compile()


class _Tau4vec_Conv2DTask(Module):
    def __init__(self, hps, **kwargs):
        super().__init__(**kwargs)
        self._hps = hps

        self._hps_conv2d = Hyperparameters()
        self._hps_conv2d.set_alias(alias={'layers_conv2d': 'layers_conv2d'})
        self._hps_conv2d.add_hp_from_dict(self._hps, is_alias=True)

        self._hps_mlp1 = Hyperparameters()
        self._hps_mlp1.set_alias(
            alias={
                'layers_images': 'layers',
                'activation': 'activation',
                'activation_last': 'activation_last',
                'batch_norm': 'batch_norm'
            })
        self._hps_mlp1.add_hp_from_dict(self._hps, is_alias=True)

        self._hps_mlp2 = Hyperparameters()
        self._hps_mlp2.set_alias(
            alias={
                'layers_calib': 'layers',
                'activation': 'activation',
                'activation_last': 'activation_last',
                'batch_norm': 'batch_norm'
            })
        self._hps_mlp2.add_hp_from_dict(self._hps, is_alias=True)

        self._conv2d = Conv2DBlock_HPS(self._hps_conv2d)
        self._mlp1 = MLPBlock_HPS(self._hps_mlp1)
        self._mlp2 = MLPBlock_HPS(self._hps_mlp2)

        self.hps = Hyperparameters()
        self.hps.add_hp_from_dict(self._hps)

    def get_hps_parameters(self):
        return self.hps.get_hps_parameters()

    def choice(self):
        return self._choice

    def choice(self, choice):
        self._choice = choice
        self._conv2d.set_active_hps(self._choice)
        self._mlp1.set_active_hps(self._choice)
        self._mlp2.set_active_hps(self._choice)

    def forward(self, x):
        fig = x[0].reshape(-1, 3, 16, 16)
        x_1 = self._conv2d(fig)
        x_1 = x_1.reshape(x_1.size(0), -1)  # flatten
        x_1 = self._mlp1(x_1)
        
        
        input_jet_reshape_4 = x[1].reshape(-1, 4)
        input_jet_reshape_3 = input_jet_reshape_4[:, :3]  # mass is not used

        x = cat((x_1, input_jet_reshape_4), dim=1)

        x = self._mlp2(x)

        layers_calib_last = self._hps_mlp2['layers'].active_data[-1]

        if layers_calib_last == 4:
            x = x + input_jet_reshape_4
        elif layers_calib_last == 3:
            x = x + input_jet_reshape_3

        x = Tau4vec_BaseTask.set_phi_within_valid_range(x)
        output = x.reshape(-1, layers_calib_last * 2)
        return output
