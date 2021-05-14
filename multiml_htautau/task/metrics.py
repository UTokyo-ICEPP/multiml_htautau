from multiml.agent.metric import BaseMetric


class CustomMSEMetric(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'custom_mse'

    def calculate(self):
        y_true, y_pred = self.get_true_pred_data()

        from .loss import Tau4vecCalibLoss_np
        return Tau4vecCalibLoss_np(pt_scale=1e-2, use_pxyz=True)(y_true, y_pred)
