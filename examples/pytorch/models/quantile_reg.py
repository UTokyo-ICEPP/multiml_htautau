from sklearn import preprocessing
import numpy as np
import lightgbm as lgb

from MyDataset import OnlyDiTauDataset


max_events, data_path, phase = 50000, '../../../../data/raw/onlyDiTau/', 'train'

dataset = OnlyDiTauDataset(max_events, data_path, phase)
x = dataset[:len(dataset)]['inputs'][1]
jet_rscaler = preprocessing.RobustScaler(quantile_range=(25., 75.))
x = jet_rscaler.fit_transform(x)

y = dataset[:len(dataset)]['internal_vec']
tau_rscaler = preprocessing.RobustScaler(quantile_range=(25., 75.))
y = tau_rscaler.fit_transform(y)
alpha = 0.975
params = {
    'objective': 'quantile',
    # 'alpha': alpha,
    'n_estimators': 250,
    'max_depth': 3,
    'learning_rate': 0.1,
    'num_leaves': 9,
    # 'verbose': 0,
}
class quantile_req:
    def __init__(self, dataset, alpha, **params):
        self.dataset = dataset
        self.dataset.train()
        self.alpha = alpha
        params['objective'] = 'quantile'
        params['alpha'] = alpha
        self.params = params
        self.upper = []
        self.pred = []
        self.lower = []
        self._swich_phase()
        self.clf = lgb.LGBMRegressor(**self.params)
    def fit(self):
        for i in range(self.y_len):
            self.clf = lgb.LGBMRegressor(**self.params)
            # upper
            self.train()
            self.clf.fit(self.x, self.y[:, i])
            self.valid()
            up = self.clf.predict(self.x)
            self.upper.append(up)
            # lower
            self.train()
            self.clf.set_params(alpha=1.0 - alpha)
            self.clf.fit(self.x, self.y[:, i])
            self.valid()
            low = self.clf.predict(self.x)
            self.lower.append(low)
            # pred
            self.train()
            self.clf.set_params(objective='regression')
            self.clf.fit(self.x, self.y[:, i])
            self.valid()
            pred = self.clf.predict(self.x)
            self.pred.append(pred)
        self.upper = tau_rscaler.inverse_transform(np.array(self.upper).T)
        self.lower = tau_rscaler.inverse_transform(np.array(self.lower).T)
        self.pred = tau_rscaler.inverse_transform(np.array(self.pred).T)

    def _swich_phase(self):
        self.x = jet_rscaler.transform(self.dataset[:len(self.dataset)]['inputs'][1])
        self.y = tau_rscaler.transform(self.dataset[:len(self.dataset)]['internal_vec'])
        self.y_len = y.shape[1]

    def train(self):
        self.dataset.train()
        self._swich_phase()

    def valid(self):
        self.dataset.valid()
        self._swich_phase()

    def test(self):
        self.dataset.test()
        self._swich_phase()

qu = quantile_req(dataset, alpha, **params)
qu.fit()

# num = 200
# x_val_index = 2
# y_val_index = 0
# dataset.valid()
# input = dataset[:]['inputs'][1][:num, x_val_index]
# output = dataset[:]['internal_vec'][:num, y_val_index]
# index = np.argsort(input)

# import matplotlib.pyplot as plt
# plt.style.use('seaborn-darkgrid')
# plt.figure(figsize=(6, 4))
# plt.scatter(input, output, c='darkgreen', s=5)
# plt.plot(input[index], qu.pred[:, y_val_index][index])
# plt.fill_between(
#     input[index],
#     qu.upper[:, y_val_index][index],
#     qu.lower[:, y_val_index][index],
#     alpha=0.6,
#     label='var'
# )
# plt.show()


d = {0: 'pt', 1: 'eta', 2: 'phi', 3: 'mass'}
num = 200


for x_val_index in (0, 1, 2, 3):
    for y_val_index in (0, 1, 2, 3):
        # x_val_index = 3
        # y_val_index = 3
        dataset.valid()
        input = dataset[:]['inputs'][1][:num, x_val_index]
        output = dataset[:]['internal_vec'][:num, y_val_index]
        index = np.argsort(input)

        import matplotlib.pyplot as plt
        plt.style.use('seaborn-darkgrid')
        plt.figure(figsize=(14, 4))
        plt.scatter(input, output, c='darkgreen', s=5,
                    label="valid data")
        plt.plot(input[index], qu.pred[:, y_val_index][index],
                 label="outputs of traind LightGBM")
        plt.fill_between(
            input[index],
            qu.upper[:, y_val_index][index],
            qu.lower[:, y_val_index][index],
            alpha=0.6,
            label='prediction interval'
        )
        plt.xlabel(f'jet {d[x_val_index]}')
        plt.ylabel(f'tau {d[y_val_index]}')
        plt.savefig(f'lightGBM_traintrain_{d[x_val_index]}-{d[y_val_index]}.png')
