from .tau4vec_base import Tau4vec_BaseTask
from .tau4vec_sf import Tau4vec_SFTask
from .tau4vec_mlp import Tau4vec_MLPTask
from .tau4vec_conv2d import Tau4vec_Conv2DTask
from .tau4vec_conv2d_yoto import Tau4vec_Conv2D_Yoto_Task
from .higgsId_base import HiggsID_BaseTask
from .higgsId_mass import HiggsID_MassTask
from .higgsId_lstm import HiggsID_LSTMTask
from .higgsId_mlp import HiggsID_MLPTask
from .higgsId_mlp_yoto import HiggsID_MLP_Yoto_Task

__all__ = [
    "Tau4vec_BaseTask",
    "Tau4vec_SFTask",
    "Tau4vec_MLPTask",
    "Tau4vec_Conv2DTask",
    "Tau4vec_Conv2D_Yoto_Task",
    "HiggsID_BaseTask",
    "HiggsID_MassTask",
    "HiggsID_LSTMTask",
    "HiggsID_MLPTask",
    "HiggsID_MLP_Yoto_Task",
]
