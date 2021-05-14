from typing import List
import torch


class PeriodicConditionMSELoss(object):
    """Constrained Loss of Periodic Boundary Conditions

        Examples::
            >>> import torch
            >>> output = torch.randn(10, 3)
            >>> target = torch.rand(10, 3)
            >>> pmse = PeriodicConditionMSELoss([2], [[1, -1]])
            >>> pmse(output, target)
            tensor(0.6376)
    """
    def __init__(self, dim: List, bound: List):
        """
        Args:
            dim (List): axis imposing periodic boundary conditions
            bound (List): boundaries per dim
        """
        self._dim = dim
        self._bound = [[ma, mi] if ma > mi else [mi, ma] for (ma, mi) in bound]

    @staticmethod
    def _periodicse(x_1: torch.Tensor,
                    x_2: torch.Tensor,
                    x_max: float,
                    x_min: float) -> torch.Tensor:
        period = x_max - x_min
        d = torch.abs(x_2 - x_1)
        test = torch.stack([d, period - d], axis=1)
        return torch.min(test, axis=1).values.pow(2)

    @staticmethod
    def _se(x_1: torch.Tensor,
            x_2: torch.Tensor) -> torch.Tensor:
        return (x_2 - x_1)**2

    def __call__(self,
                 output: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output (torch.Tensor): output of model
            target (torch.Tensor): target
        """
        count = 0
        return_val = []
        for i in range(output.shape[1]):
            if i in self._dim:
                return_val.append(
                    self._periodicse(output[:, i],
                                     target[:, i],
                                     self._bound[count][0],
                                     self._bound[count][1])
                )
                count += 1
            else:
                return_val.append(self._se(output[:, i], target[:, i]))
        return torch.mean(torch.stack(return_val, axis=1))
