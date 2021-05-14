from copy import copy

import click
import numpy as np
import torch
import gpytorch
from torch.utils.data import DataLoader
from sklearn import preprocessing

from MyDataset import OnlyDiTauDataset_wo_mass
from GPyTorchModels import (
    IndependentMultitaskGPModel,
    MultitaskGPModel,
    tensor_to_array,
)

max_events = 50000
data_path = '../../../../data/raw/onlyDiTau/'
input_dim = 8
num_latents = 6
inducing_points_num = 15
num_tasks = 6
batch_size = 100
epochs = 30
phase = 'train'


def set_seed(seed=200):
    import tensorflow
    import random
    import os
    tensorflow.random.set_seed(seed)
    torch.manual_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(1)


@click.command()
@click.option('--use_pxyz', type=bool, default=False)
def main(use_pxyz: bool):
    dataset = OnlyDiTauDataset_wo_mass(max_events, data_path, phase)
    x = copy(dataset[:len(dataset)]['inputs'][1])
    y = copy(dataset[:len(dataset)]['internal_vec'])
    jet_rscaler = preprocessing.RobustScaler(quantile_range=(25., 75.))
    x = jet_rscaler.fit_transform(x)
    from MyLoss import Tau4vecCalibLoss_torch
    loss_ins = Tau4vecCalibLoss_torch()
    if use_pxyz:
        y = torch.tensor(y).reshape(-1, 3)
        y = loss_ins._convert_to_pxyz(y).reshape(-1, 6)
    tau_rscaler = preprocessing.RobustScaler(quantile_range=(25., 75.))
    y = tau_rscaler.fit_transform(y)
    dataset.test()
    x = copy(dataset[:len(dataset)]['inputs'][1])
    y = copy(dataset[:len(dataset)]['internal_vec'])
    x = jet_rscaler.transform(x)
    if use_pxyz:
        y = torch.tensor(y).reshape(-1, 3)
        y = loss_ins._convert_to_pxyz(y).reshape(-1, 6)
    y = tau_rscaler.transform(y)
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x), torch.tensor(y)
    )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)

    model = MultitaskGPModel(inducing_points_num=inducing_points_num,
                             input_dim=input_dim,
                             num_latents=num_latents,
                             num_tasks=num_tasks)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        rank=num_tasks,
        # noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    )

    if use_pxyz:
        weight = torch.load('../logs/GP_model_xyz.pt')
    else:
        weight = torch.load('../logs/GP_model.pt')
    # weight = torch.load('../logs/GP_model_unique_function.pt')
    model.load_state_dict(weight['model'])
    likelihood.load_state_dict(weight['likelihood'])

    with torch.no_grad():
        pred, lower, upper = [], [], []
        model.eval()
        likelihood.eval()
        for step, (x, y) in enumerate(test_dataloader):
            prediction = likelihood(model(x))
            pred.append(
                tau_rscaler.inverse_transform(prediction.mean)
            )
            lower.append(
                tau_rscaler.inverse_transform(
                    prediction.confidence_region()[0]
                )
            )
            upper.append(
                tau_rscaler.inverse_transform(
                    prediction.confidence_region()[1]
                )
            )
        pred = np.concatenate(pred)
        lower = np.concatenate(lower)
        upper = np.concatenate(upper)

    if use_pxyz:
        p = '_xyz'
    else:
        p = ''
    np.savetxt('../logs/GP_upper'+p+'.csv', upper,
               delimiter=',', newline='\n')
    np.savetxt('../logs/GP_lower'+p+'.csv', lower,
               delimiter=',', newline='\n')
    if use_pxyz:
        d = {0: 'pt', 1: 'eta', 2: 'phi', 3: 'mass'}
        o = {0: 'px', 1: 'py', 2: 'pz'}
    else:
        d = {0: 'pt', 1: 'eta', 2: 'phi', 3: 'mass'}
        o = {0: 'pt', 1: 'eta', 2: 'phi'}
    num = 200
    for x_val_index in range(4):
        for y_val_index in range(3):
            input = jet_rscaler.inverse_transform(
                test_dataset[:, :][0]
            )[:num, x_val_index]
            output = tau_rscaler.inverse_transform(
                test_dataset[:, :][1]
            )[:num, y_val_index]
            index = np.argsort(input)

            import matplotlib.pyplot as plt
            plt.rcParams["font.size"] = 15
            plt.style.use('seaborn-darkgrid')
            plt.figure(figsize=(14, 4))
            plt.scatter(input, output, c='darkgreen',
                        s=10, label="test data")
            plt.plot(input[index], pred[:, y_val_index][index],
                     label="outputs of GP model")
            plt.fill_between(
                input[index],
                upper[:, y_val_index][index],
                lower[:, y_val_index][index],
                alpha=0.6,
                label='2$\sigma$ prediction interval'
            )
            plt.xlabel(f'jet {d[x_val_index]}')
            plt.ylabel(f'tau {o[y_val_index]}')
            plt.legend()
            plt.savefig(f'{d[x_val_index]}-{o[y_val_index]}.png')
            plt.close()


if __name__ == '__main__':
    main()
