from sklearn.metrics import (
    r2_score,
    roc_auc_score
)


class Calc_R2:
    def __init__(self, sample_weight=None, multioutput='uniform_average'):
        self.sample_weight = sample_weight
        self.multioutput = multioutput

    def __call__(self, targets, outputs):
        return r2_score(targets, outputs,
                        sample_weight=self.sample_weight,
                        multioutput=self.multioutput)


class Calc_Auc:
    def __init__(self, average='macro',
                 sample_weight=None,
                 max_fpr=None,
                 multi_class='raise',
                 labels=None):
        self.average = average
        self.sample_weight = sample_weight
        self.max_fpr = max_fpr
        self.multi_class = multi_class
        self.labels = labels

    def __call__(self, targets, outputs):
        return roc_auc_score(targets, outputs,
                             average=self.average,
                             sample_weight=self.sample_weight,
                             max_fpr=self.max_fpr,
                             multi_class=self.multi_class,
                             labels=self.labels)
