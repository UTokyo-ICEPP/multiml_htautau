from typing import List, Union
from torch import Tensor


def _delta_phi_tf(x, y):
    from tensorflow.math import abs, minimum
    import math
    pi = math.pi
    d = abs(x - y)
    return minimum(d, 2 * pi - d)


def _delta_phi_np(x, y):
    from numpy import abs, minimum
    import math
    pi = math.pi
    d = abs(x - y)
    return minimum(d, 2 * pi - d)


class Tau4vecCalibLoss_tf():
    __name__ = 'Tau4vecCalibLoss_tf'

    def __init__(self, pt_scale, use_pxyz=True):
        self._pt_scale = pt_scale
        self._use_pxyz = use_pxyz

    def _convert_to_pxyz(self, x):
        from tensorflow.math import cos, sin, sinh, exp
        from tensorflow import clip_by_value, stack
        pt = exp(clip_by_value(x[:, 0], -7., 7.)) - 0.1
        pt *= self._pt_scale
        eta = x[:, 1]
        phi = x[:, 2]

        px = pt * cos(phi)
        py = pt * sin(phi)
        pz = pt * sinh(clip_by_value(eta, -5, 5))

        return stack([px, py, pz], axis=1)

    def __call__(self, y_true, y_pred):
        from tensorflow.math import square, reduce_mean
        from tensorflow import reshape, concat, expand_dims
        from tensorflow.keras import backend as K

        assert(K.int_shape(y_true)[-1] == 3 * 2)
        assert(K.int_shape(y_pred)[-1] == 3 * 2)
        y_true = reshape(y_true, [-1, 3])
        y_pred = reshape(y_pred, [-1, 3])

        if self._use_pxyz:
            y_true = self._convert_to_pxyz(y_true)
            y_pred = self._convert_to_pxyz(y_pred)
            return reduce_mean(square(y_true - y_pred))
        else:
            d_sq = square(y_true - y_pred)
            d_phi = square(_delta_phi_tf(y_true[:, 2], y_pred[:, 2]))
            d_phi = expand_dims(d_phi, axis=-1)
            diff = concat([d_sq[:, 0:2], d_phi], axis=-1)
            return reduce_mean(diff)


class Tau4vecCalibLoss_np():
    __name__ = 'Tau4vecCalibLoss_np'

    def __init__(self, pt_scale, use_pxyz=True):
        self._pt_scale = pt_scale
        self._use_pxyz = use_pxyz

    def _convert_to_pxyz(self, x):
        from numpy import cos, sin, sinh, exp, clip, stack
        pt = exp(clip(x[:, 0], -7., 7.)) - 0.1
        pt *= self._pt_scale
        eta = x[:, 1]
        phi = x[:, 2]

        px = pt * cos(phi)
        py = pt * sin(phi)
        pz = pt * sinh(clip(eta, -5, 5))

        return stack([px, py, pz], axis=1)

    def __call__(self, y_true, y_pred):
        from numpy import square, mean, reshape, concatenate, expand_dims

        assert(y_true.shape[-1] == 3 * 2)
        assert(y_pred.shape[-1] == 3 * 2)
        y_true = reshape(y_true, [-1, 3])
        y_pred = reshape(y_pred, [-1, 3])

        if self._use_pxyz:
            y_true = self._convert_to_pxyz(y_true)
            y_pred = self._convert_to_pxyz(y_pred)
            return mean(square(y_true - y_pred))
        else:
            d_sq = square(y_true - y_pred)
            d_phi = square(_delta_phi_np(y_true[:, 2], y_pred[:, 2]))
            d_phi = expand_dims(d_phi, axis=-1)
            diff = concatenate([d_sq[:, 0:2], d_phi], axis=-1)
            return mean(diff)


class Tau4vecCalibLoss_torch(object):
    """

        Examples::
            >>> import torch
            >>> output = torch.randn(10, 6)
            >>> target = torch.rand(10, 6)
            >>> pmse = Tau4vecCalibLoss_torch(0.01, True)
            >>> pmse(output, target)
            tensor(0.0089)
    """
    __name__ = 'Tau4vecCalibLoss_torch_test'

    def __init__(self,
                 pt_scale: float = 1e-2,
                 use_pxyz: bool = True,
                 bound: Union[List[float], None] = None):
        """
        Args:
            pt_scale (float, optional): Defaults to 1e-2.
            use_pxyz (bool, optional): Defaults to True.
            bound (list(float), optional): Defaults to None.
        """
        self._pt_scale = pt_scale
        self._use_pxyz = use_pxyz
        if bound is None:
            import math
            pi = math.pi
            bound = [-pi, pi]
        self._period = abs(bound[0] - bound[1])

    def _convert_to_pxyz(self, x: Tensor) -> Tensor:
        from torch import stack
        pt = x[:, 0].clamp(-7., 7.).exp() - 0.1
        pt *= self._pt_scale
        eta = x[:, 1]
        phi = x[:, 2]

        px = pt * phi.cos()
        py = pt * phi.sin()
        pz = pt * eta.clamp(-5., 5.).sinh()

        return stack([px, py, pz], axis=1)

    def _delta_phi_torch(self,
                         x: Tensor,
                         y: Tensor) -> Tensor:
        from torch import stack
        d = (x - y).abs()
        test = stack([d, self._period - d], axis=1)
        return test.min(axis=1).values

    def __call__(self,
                 output: Tensor,
                 target: Tensor) -> Tensor:
        """
        Args:
            output (Tensor): output of model
            target (Tensor): target
        """
        from torch import cat
        assert(output.shape[-1] == 3 * 2)
        assert(target.shape[-1] == 3 * 2)
        output = output.reshape(-1, 3)
        target = target.reshape(-1, 3)

        if self._use_pxyz:
            output = self._convert_to_pxyz(output)
            target = self._convert_to_pxyz(target)
            return (target - output).pow(2).mean()
            
        else:
            d_sq = (target - output).pow(2)
            d_phi = (
                self._delta_phi_torch(target[:, 2], output[:, 2])
            ).pow(2)
            diff = cat(
                [d_sq[:, 0:2], d_phi.unsqueeze(1)],
                axis=-1
            )
            return diff.mean() 

