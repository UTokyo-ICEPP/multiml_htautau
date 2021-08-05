from copy import copy

import click
import numpy as np
import torch
import gpytorch
from tqdm import tqdm
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
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    dataset.valid()
    x = copy(dataset[:len(dataset)]['inputs'][1])
    y = copy(dataset[:len(dataset)]['internal_vec'])
    x = jet_rscaler.transform(x)
    if use_pxyz:
        y = torch.tensor(y).reshape(-1, 3)
        y = loss_ins._convert_to_pxyz(y).reshape(-1, 6)
    y = tau_rscaler.transform(y)
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=2)

    model = MultitaskGPModel(inducing_points_num=inducing_points_num,
                             input_dim=input_dim,
                             num_latents=num_latents,
                             num_tasks=num_tasks)
    # model = IndependentMultitaskGPModel(
    #     inducing_points_num=inducing_points_num,
    #     input_dim=input_dim,
    #     num_tasks=num_tasks
    # )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        rank=num_tasks,
        # noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    )

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {
            'params': model.parameters()
        },
        {
            'params': likelihood.parameters()
        },
    ],
                                 lr=1e-2)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_dataset))

    for epoch in range(epochs):
        model.train()
        likelihood.train()
        train_data = tqdm(train_dataloader)
        train_data.set_description(f"[Epoch:{epoch+1:04d}/{epochs:04d} " +
                                   f"lr:{optimizer.param_groups[0]['lr']:.5f}]")
        train_loss = 0.0
        for step, (train_x, train_y) in enumerate(train_data):
            # Within each iteration, we will go over each minibatch of data
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            train_loss += loss.item()
            postfix = {'train_loss': f'{(train_loss / (step + 1)):.5f}'}
            train_data.set_postfix(log=postfix)
            # epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            likelihood.eval()
            valid_loss = 0.0
            for step, (x, y) in enumerate(valid_dataloader):
                output = model(x)
                loss = -mll(output, y)
                valid_loss += loss.item()
            print(f'valid_loss: {(valid_loss / (step+1)):.5f} / ' +
                  f'noise: {likelihood.noise.item()}')

    if use_pxyz:
        model_file_name = '../logs/GP_model_xyz.pt'
    else:
        model_file_name = '../logs/GP_model.pt'
    torch.save({
        'model': model.state_dict(),
        'likelihood': likelihood.state_dict()
    }, model_file_name)


if __name__ == '__main__':
    main()
