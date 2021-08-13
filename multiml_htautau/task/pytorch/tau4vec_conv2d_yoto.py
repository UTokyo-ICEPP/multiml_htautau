from torch.nn import Module
from torch import cat, tensor_split
from multiml.task.pytorch.modules.yoto import Conv2DBlock_Yoto
from multiml.task.pytorch.modules.yoto import MLPBlock_Yoto
from multiml import Hyperparameters

from . import Tau4vec_BaseTask


class Tau4vec_Conv2D_Yoto_Task(Tau4vec_BaseTask):
    ''' Tau4vec Conv2D MLP task
    '''
    def __init__(self, hps = None, layers_calib_last = None, **kwargs):
        """
        Args:
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__(**kwargs)

        self._hps = hps
        self._layers_calib_last = layers_calib_last

    def build_model(self):
        self._model = _Tau4vec_Conv2D_Yoto_Task( hps = self._hps, layers_calib_last = self._layers_calib_last)
        #self._model_compile()
    def get_n_layers_to_yoto(self):
        self._model.get_n_layers_to_yoto()


class _Tau4vec_Conv2D_Yoto_Task(Module):
    def __init__(self, hps, layers_calib_last, **kwargs):
        super().__init__(**kwargs)
        self._hps = hps
        
        self._conv2d = Conv2DBlock_Yoto( self._hps['conv2d'] )
        self._mlp1   = MLPBlock_Yoto( self._hps['mlp1'] )
        self._mlp2   = MLPBlock_Yoto( self._hps['mlp2'] )
        
        self.layers_calib_last = layers_calib_last
        
        self.n_layers_to_yoto = []
        self.yoto_idx = [0]
        
        self.add_n_layers_to_yoto( self._conv2d.get_n_layers_to_yoto()  )
        self.add_n_layers_to_yoto( self._mlp1.get_n_layers_to_yoto() )
        self.add_n_layers_to_yoto( self._mlp2.get_n_layers_to_yoto() )
        
    def add_n_layers_to_yoto(self, layers):
        self.n_layers_to_yoto += layers 
        self.yoto_idx += [self.yoto_idx[-1] + len(layers)]
                
    def get_n_layers_to_yoto(self):
        return self.n_layers_to_yoto
    
    def set_yoto_layer(self, gamma, beta):
        
        self._conv2d.set_yoto_layer(gamma[self.yoto_idx[0]:self.yoto_idx[1]], beta[self.yoto_idx[0]:self.yoto_idx[1]])
        self._mlp1.set_yoto_layer(gamma[self.yoto_idx[1]:self.yoto_idx[2]], beta[self.yoto_idx[1]:self.yoto_idx[2]])
        self._mlp2.set_yoto_layer(gamma[self.yoto_idx[2]:self.yoto_idx[3]], beta[self.yoto_idx[2]:self.yoto_idx[3]])
        
    def forward(self, x, x_gamma, x_beta):
        
        outputs = []
        #fig = x[0].reshape(-1, 3, 16, 16) 
        figs = tensor_split(x[0], 2, dim = 1)
        input_jet_reshape_4s = tensor_split(x[1], 2, dim = 1)
        
        for fig, input_jet_reshape_4 in zip(figs, input_jet_reshape_4s): 
            

            fig = fig.reshape(-1, 3, 16, 16)


            x_1 = self._conv2d( fig, x_gamma, x_beta )
            x_1 = x_1.reshape(x_1.size(0), -1)  # flatten
            x_1 = self._mlp1( x_1, x_gamma, x_beta )
            
            input_jet_reshape_3 = input_jet_reshape_4[:, :3]  # mass is not used
            
            x_2 = cat((x_1, input_jet_reshape_4), dim=1)
            x_2 = self._mlp2(x_2, x_gamma, x_beta)
            
            if self.layers_calib_last == 4:
                x_2 = x_2 + input_jet_reshape_4
            elif self.layers_calib_last == 3:
                x_2 = x_2 + input_jet_reshape_3

            x_2 = Tau4vec_BaseTask.set_phi_within_valid_range(x_2)
            outputs.append(x_2.reshape(-1, self.layers_calib_last ))
        
        output = cat(outputs, dim = 1)
        
        return output
