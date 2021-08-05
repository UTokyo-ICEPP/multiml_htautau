from torch import nn
import numpy as np
import torch

from utils import (
    add_device,
    get_logger,
)

logger = get_logger()


def train(model,
          dataloader,
          input_key,
          target_key,
          optimizer,
          loss_func,
          device=torch.device('cpu')):
    train_loss = 0.0
    for step, data in enumerate(dataloader):
        inputs = add_device(data[input_key], device)
        targets = add_device(data[target_key], device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        postfix = {'train_loss': f'{(train_loss / (step + 1)):.5f}'}
        dataloader.set_postfix(log=postfix)


def valid(model, dataloader, input_key, target_key, device=torch.device('cpu'), activation=None):
    outputs_data = []
    targets_data = []
    for step, data in enumerate(dataloader):
        inputs = add_device(data[input_key], device)
        targets = add_device(data[target_key], device)
        if activation:
            outputs = activation(model(inputs))
        else:
            outputs = model(inputs)
        outputs_data.extend(outputs.tolist())
        targets_data.extend(targets.tolist())
    return outputs_data, targets_data


def pre_train(epochs,
              model,
              dataloader,
              optimizer,
              loss_func,
              input_key,
              target_key,
              device=torch.device('cpu'),
              patience=5,
              metrics=None,
              activation=None):
    from tqdm import tqdm
    logger.info(model)
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    criterion = loss_func
    for epoch in range(epochs):
        model.train()
        dataloader.dataset.train()
        train_data = tqdm(dataloader)
        train_data.set_description(f"[Epoch:{epoch+1:04d}/{epochs:04d} " +
                                   f"lr:{optimizer.param_groups[0]['lr']:.5f}]")
        train(model, train_data, input_key, target_key, optimizer, criterion, device)

        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            dataloader.dataset.valid()
            outputs_data, targets_data = valid(model,
                                               dataloader,
                                               input_key,
                                               target_key,
                                               device=device,
                                               activation=activation)
            valid_loss = criterion(torch.tensor(outputs_data), torch.tensor(targets_data))
            if metrics is None:
                s = f'[Epoch:{epoch+1:04d}|valid| / '\
                    f'loss:{valid_loss:.6f}]'
            else:
                score = metrics(np.array(targets_data), np.array(outputs_data))
                s = f'[Epoch:{epoch+1:04d}|valid| / '\
                    f'loss:{valid_loss:.6f} / '\
                    f'metrics:{score:.6f}]'
            logger.info(s)
            best_model = early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

    return best_model


def wrap_phi_to_2pi_torch(x):
    """Shift input angle x to the range of [-pi, pi]
    """
    import math
    pi = math.pi
    x = torch.fmod(2 * pi + torch.fmod(x + pi, 2 * pi), 2 * pi) - pi
    return x


def wrap_phi_to_2pi_numpy(x):
    """Shift input angle x to the range of [-pi, pi]
    """
    import math
    pi = math.pi
    x = np.fmod(2 * pi + np.fmod(x + pi, 2 * pi), 2 * pi) - pi
    return x


def set_phi_within_valid_range(x):
    if isinstance(x, torch.Tensor):
        x_phi = x[:, 2]
        x_phi = wrap_phi_to_2pi_torch(x_phi)
        x_phi.unsqueeze_(1)
        x = torch.cat([x[:, 0:2], x_phi], axis=1)
    elif isinstance(x, np.ndarray):
        x_phi = x[:, 2]
        x_phi = wrap_phi_to_2pi_numpy(x_phi)
        x_phi = np.expand_dims(x_phi, 1)
        x = np.concatenate([x[:, 0:2], x_phi], axis=1)
    return x


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, save=False, path='./logs'):
        self.patience = patience
        self.verbose = verbose
        self.save = save
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model = None

    def __call__(self, val_loss, model):
        from copy import deepcopy

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = deepcopy(self.save_checkpoint(val_loss, model))
        elif score <= self.best_score:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = deepcopy(self.save_checkpoint(val_loss, model))

        return self.best_model

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  ' +
                'updating model ...')
        if self.save:
            from os.path import join
            save_path = join(self.path, 'checkpoint.pt')
            torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        return model


class MLPBlock(nn.Module):
    def __init__(self,
                 layers,
                 activation,
                 activation_last=None,
                 batch_norm=False,
                 initialize=True,
                 *args,
                 **kwargs):
        super(MLPBlock, self).__init__(*args, **kwargs)
        from utils import get_module

        _layers = []
        for i, node in enumerate(layers):
            if i == len(layers) - 1:
                break
            else:
                _layers.append(nn.Linear(layers[i], layers[i + 1]))

            if batch_norm:
                _layers.append(nn.BatchNorm1d(layers[i + 1]))

            if i == len(layers) - 2:
                if activation_last is None:
                    _layers.append(get_module([nn], 'Identity')())
                else:
                    _layers.append(get_module([nn], activation_last)())
            else:
                _layers.append(get_module([nn], activation)())

        self._layers = nn.Sequential(*_layers)
        if initialize:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self._layers(x)


class Tau4vec_MLPTask(nn.Module):
    def __init__(self,
                 layers_images=[768, 32, 32, 32, 4],
                 layers_calib=[8, 32, 32, 32, 4],
                 activation='ReLU',
                 batch_norm=False,
                 **kwargs):
        super(Tau4vec_MLPTask, self).__init__(**kwargs)
        self._mlp1 = MLPBlock(layers=layers_images,
                              activation=activation,
                              activation_last='Identity',
                              batch_norm=batch_norm)
        self._mlp2 = MLPBlock(layers=layers_calib,
                              activation=activation,
                              activation_last='Identity',
                              batch_norm=batch_norm)
        self._layers_calib = layers_calib
        self._len_output_vers = layers_calib[-1] * 2

    def forward(self, x):
        fig = x[0].reshape(-1, 3, 16, 16)
        x_1 = fig.reshape(fig.size(0), -1)
        x_1 = self._mlp1(x_1)

        input_jet_reshape_4 = x[1].reshape(-1, 4)
        input_jet_reshape_3 = input_jet_reshape_4[:, :3]  # mass is not used

        x = torch.cat((x_1, input_jet_reshape_4), dim=1)

        x = self._mlp2(x)
        if self._layers_calib[-1] == 4:
            x = x + input_jet_reshape_4
        elif self._layers_calib[-1] == 3:
            x = x + input_jet_reshape_3

        x = set_phi_within_valid_range(x)
        output = x.reshape(-1, self._layers_calib[-1] * 2)
        return output


class Conv2DBlock(nn.Module):
    def __init__(self, layers_conv2d=None, initialize=True, *args, **kwargs):
        super(Conv2DBlock, self).__init__(*args, **kwargs)
        from copy import copy
        from utils import get_module
        _layers = []
        conv2d_args = {"stride": 1, "padding": 0, "activation": 'ReLU'}
        maxpooling2d_args = {"kernel_size": 2, "stride": 2}

        for layer, args in layers_conv2d:
            if layer == 'conv2d':
                layer_args = copy(conv2d_args)
                layer_args.update(args)
                activation = layer_args.pop('activation')
                _layers.append(nn.Conv2d(**layer_args))
                _layers.append(get_module([nn], activation)())
            elif layer == 'maxpooling2d':
                layer_args = copy(maxpooling2d_args)
                layer_args.update(args)
                _layers.append(nn.MaxPool2d(**layer_args))
            else:
                raise ValueError(f"{layer} is not implemented")

        self._layers = nn.Sequential(*_layers)
        if initialize:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self._layers(x)


class Tau4vec_Conv2DTask(nn.Module):
    def __init__(self,
                 layers_conv2d=[('conv2d', {
                     'in_channels': 3,
                     'out_channels': 32,
                     'kernel_size': 3
                 }), ('conv2d', {
                     'in_channels': 32,
                     'out_channels': 16,
                     'kernel_size': 3
                 }), ('maxpooling2d', {}),
                                ('conv2d', {
                                    'in_channels': 16,
                                    'out_channels': 16,
                                    'kernel_size': 2
                                }),
                                ('conv2d', {
                                    'in_channels': 16,
                                    'out_channels': 8,
                                    'kernel_size': 2
                                })],
                 layers_images=[128, 16, 16, 16, 4],
                 layers_calib=[8, 64, 64, 64, 4],
                 activation='ReLU',
                 batch_norm=False,
                 **kwargs):
        super(Tau4vec_Conv2DTask, self).__init__(**kwargs)
        self._conv2d = Conv2DBlock(layers_conv2d=layers_conv2d)
        self._mlp1 = MLPBlock(layers=layers_images,
                              activation=activation,
                              activation_last='Identity',
                              batch_norm=batch_norm)
        self._mlp2 = MLPBlock(layers=layers_calib,
                              activation=activation,
                              activation_last='Identity',
                              batch_norm=batch_norm)
        self._layers_calib = layers_calib

    def forward(self, x):
        fig = x[0].reshape(-1, 3, 16, 16)
        x_1 = self._conv2d(fig)

        x_1 = x_1.reshape(x_1.size(0), -1)  # flatten
        x_1 = self._mlp1(x_1)

        input_jet_reshape_4 = x[1].reshape(-1, 4)
        input_jet_reshape_3 = input_jet_reshape_4[:, :3]  # mass is not used

        x = torch.cat((x_1, input_jet_reshape_4), dim=1)

        x = self._mlp2(x)
        if self._layers_calib[-1] == 4:
            x = x + input_jet_reshape_4
        elif self._layers_calib[-1] == 3:
            x = x + input_jet_reshape_3

        x = set_phi_within_valid_range(x)
        output = x.reshape(-1, self._layers_calib[-1] * 2)
        return output


class SF_layer(nn.Module):
    def __init__(self, input_dim):
        super(SF_layer, self).__init__()
        self.sf = nn.Parameter(torch.Tensor(np.ones(input_dim)))
        self.bias = nn.Parameter(torch.Tensor(np.zeros(input_dim)))

    def forward(self, x):
        return x * self.sf + self.bias


class Tau4vec_SFTask(nn.Module):
    def __init__(self, n_input_vars=8, n_output_vars=6, n_jets=2):
        super(Tau4vec_SFTask, self).__init__()
        self.sf_layer = SF_layer(input_dim=(1, n_output_vars // 2))
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars
        self.n_jets = n_jets

    def forward(self, x):
        x = x[1].reshape(-1, self.n_input_vars // self.n_jets)
        if self.n_output_vars == 6:
            x = x[:, :3]  # mass is not used
        x = self.sf_layer(x)

        x = set_phi_within_valid_range(x)
        x = x.reshape(-1, self.n_output_vars)
        return x


class HiggsID_MLPTask(nn.Module):
    def __init__(self,
                 layers=[8, 32, 32, 32, 1],
                 activation='ReLU',
                 activation_last='Identity',
                 batch_norm=False,
                 **kwargs):
        super(HiggsID_MLPTask, self).__init__(**kwargs)
        self.mlp = MLPBlock(layers=layers,
                            activation=activation,
                            activation_last=activation_last,
                            batch_norm=batch_norm)

    def forward(self, x):
        x = self.mlp(x)
        return x


class LSTMBlock(nn.Module):
    def __init__(self,
                 layers,
                 activation=None,
                 batch_norm=False,
                 initialize=True,
                 *args,
                 **kwargs):
        super(LSTMBlock, self).__init__(*args, **kwargs)
        from collections import OrderedDict
        from utils import get_module

        _layers = OrderedDict()
        for i, node in enumerate(layers):
            if i == len(layers) - 1:
                break
            else:
                _layers[f'LSTM{i}'] = nn.LSTM(layers[i], layers[i + 1])

        if batch_norm:
            _layers['batchnorm1d'] = nn.BatchNorm1d(layers[-1])

        if activation is not None:
            _layers[activation] = get_module([nn], activation)()

        self._layers = nn.Sequential(_layers)
        if initialize:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x):
        for layer in self._layers:
            if type(layer) == nn.LSTM:
                x, _ = layer(x)
            else:
                x = layer(x)
        return x


class HiggsID_LSTMTask(nn.Module):
    def __init__(self,
                 layers_lstm=[4, 32, 32, 32, 1],
                 layers_mlp=[1, 1],
                 activation_last='Identity',
                 batch_norm=False,
                 n_jets=2,
                 **kwargs):
        super(HiggsID_LSTMTask, self).__init__(**kwargs)
        self.layers_lstm = layers_lstm
        self.n_jets = n_jets
        self.lstm = LSTMBlock(layers=layers_lstm, batch_norm=batch_norm)
        self.mlp = MLPBlock(layers=layers_mlp,
                            activation='Identity',
                            activation_last=activation_last,
                            batch_norm=batch_norm)

    def forward(self, x):
        x = torch.transpose(x.reshape(-1, self.n_jets, self.layers_lstm[0]), 1, 0)
        x = self.lstm(x)[-1]
        x = self.mlp(x)
        return x


class HiggsID_MassTask(nn.Module):
    def __init__(self,
                 layers=[1, 64, 64, 1],
                 activation='ReLU',
                 activation_last='Identity',
                 batch_norm=False,
                 scale_mass=1. / 125.,
                 n_jets=2,
                 n_input_vars=8,
                 **kwargs):
        super(HiggsID_MassTask, self).__init__(**kwargs)
        self.scale_mass = scale_mass
        self.n_input_vars = n_input_vars
        self.n_jets = n_jets

        self.mlp = MLPBlock(layers=layers,
                            activation=activation,
                            activation_last=activation_last,
                            batch_norm=batch_norm)

    def forward(self, x):
        x = self.mass_layer(x, self.n_jets, self.n_input_vars)
        x = x * self.scale_mass
        x = self.mlp(x)
        return x

    @staticmethod
    def mass_layer(tau_4vec, n_jets, n_input_vars):
        tau_4vec = tau_4vec.reshape(-1, n_jets, n_input_vars // n_jets)
        pt = torch.exp(torch.clamp(tau_4vec[:, :, 0], min=-7., max=7.)) - 0.1
        eta = tau_4vec[:, :, 1]
        phi = tau_4vec[:, :, 2]
        mass = 1.777

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(torch.clamp(eta, min=-5, max=5))
        epsilon = 0.1  # avoid nan when e=0. sqrt(x)^' = -1/2 * 1/sqrt(x)
        e = torch.sqrt(epsilon + px**2 + py**2 + pz**2 + mass**2)

        tau_4vec = torch.stack([px, py, pz, e], dim=2)
        tau_4vec = torch.sum(tau_4vec, dim=1)
        px, py, pz, e = torch.chunk(tau_4vec, chunks=4, dim=1)
        mass = torch.sqrt(epsilon + e**2 - (px**2 + py**2 + pz**2))
        return mass


class SubTask_Gaussian(nn.Module):
    def __init__(self, in_len=8, sigma=1.0):
        super(SubTask_Gaussian, self).__init__()
        self.sigma = sigma
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.sum(axis=1) * 0.0
        if self.sigma != 0:
            sampled_noise = torch.empty_like(x).normal_() * self.sigma
            x = x + sampled_noise
        return self.sigmoid(x).reshape(-1, 1)
