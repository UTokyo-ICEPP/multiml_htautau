from torch.nn import Module
from multiml.task.pytorch.modules import MLPBlock, MLPBlock_HPS
from multiml import Hyperparameters

from . import Tau4vec_BaseTask


class Tau4vec_MLPTask(Tau4vec_BaseTask):
    ''' Tau4vec MLP task
    '''
    def __init__(self, hps = None, **kwargs):
        """

        Args:
            layers_images (list(int)): the number of nodes in hidden layers in MLP that used for image processing.
            layers_calib (list(int)): the number of nodes in hidden layers in MLP that used for calibration.
            activation (str): activation function for MLP.
            batch_norm (bool): use batch normalization
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)
        self._hps = hps

    def build_model(self):
        self._model = _Tau4vec_MLPTask( self._hps )
        
        #self._model_compile()


class _Tau4vec_MLPTask(Module):
    def __init__(self, hps, **kwargs):
        super().__init__(**kwargs)
        self._hps = hps
        
        self._hps_mlp1 = Hyperparameters()
        self._hps_mlp1.set_alias( alias =  {'layers_images':'layers', 'activation' : 'activation', 'activation_last':'activation_last', 'batch_norm':'batch_norm' } )
        self._hps_mlp1.add_hp_from_dict(self._hps, is_alias = True )
                
        self._hps_mlp2 = Hyperparameters()
        self._hps_mlp2.set_alias( alias =  {'layers_calib':'layers', 'activation' : 'activation', 'activation_last':'activation_last', 'batch_norm':'batch_norm' } )
        self._hps_mlp2.add_hp_from_dict(self._hps, is_alias = True )
        
        self._mlp1 = MLPBlock_HPS( self._hps_mlp1 )
        self._mlp2 = MLPBlock_HPS( self._hps_mlp2 )
        
        self._len_output_vers = self._hps_mlp2['layers']._data[-1]*2
        
        self.hps = Hyperparameters()
        self.hps.add_hp_from_dict( self._hps )
        
    def get_hps_parameters(self):
        return self.hps.get_hps_parameters()

    def choice(self) : 
        return self._choice
    
    def choice(self, choice):
        self._choice = choice 
        self._mlp1.set_active_hps( self._choice )
        self._mlp2.set_active_hps( self._choice )


    def forward(self, x):
        from torch import cat
        fig = x[0].reshape(-1, 3, 16, 16)
        x_1 = fig.reshape(fig.size(0), -1)
        x_1 = self._mlp1(x_1)

        input_jet_reshape_4 = x[1].reshape(-1, 4)
        input_jet_reshape_3 = input_jet_reshape_4[:, :3]  # mass is not used

        x = cat((x_1, input_jet_reshape_4), dim=1)
        layers_calib_last = self._hps_mlp2['layers'].active_data[-1]

        x = self._mlp2(x)
        
        if layers_calib_last == 4 : 
            x = x + input_jet_reshape_4
        elif layers_calib_last == 3 : 
            x = x + input_jet_reshape_3

        x = Tau4vec_BaseTask.set_phi_within_valid_range(x)
        output = x.reshape(-1, layers_calib_last * 2 )
        return output
