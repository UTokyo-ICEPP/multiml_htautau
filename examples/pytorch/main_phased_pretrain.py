#!/usr/bin/env python3

from itertools import product
from utils import (
    get_logger,
    get_module,
    load_config,
    log,
    set_module,
    set_seed,
    set_task,
)

from torch import nn
from torch.utils.data import DataLoader
import click
import numpy as np
import torch

from multiml.task.pytorch import PytorchBaseTask

from models import MyLoss
from utils import (
    add_device,
    tensor_to_array,
)

log_dir = './logs/connection/'
logger = get_logger(logdir_path=log_dir)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

FIRST_MODEL_NAME = ['MLP', 'CONV2D', 'SF']
SECOND_MODEL_NAME = ['MLP', 'LSTM', 'MASS']


class Tauvec_BaseTask(PytorchBaseTask):
    """Documentation for Tauvec_BaseTask

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self,
                   model,
                   dataloaders,
                   input_key: str,
                   target_key: str,
                   criterion: str,
                   device='cpu',
                   optimizer: str = 'Adam',
                   do_manual_decay: bool = False,
                   hp_epochs: int = 10,
                   hp_lr: float = 1e-3,
                   lr: float = 1e-3,
                   patience: int = 10,
                   hps: dict = {},
                   **kwargs):
        self._model = model
        self._dataloaders = dataloaders
        self._input_key = input_key
        self._target_key = target_key
        if isinstance(criterion, str):
            from models import MyLoss
            self._criterion = get_module([nn, MyLoss], criterion)()
        else:
            self._criterion = criterion

        self._device = device
        from torch import optim
        self._optimizer = get_module([optim], optimizer)(model.parameters())
        self._do_manual_decay = do_manual_decay
        self._hp_epochs = hp_epochs
        self._hp_lr = hp_lr
        self._lr = lr
        self._patience = patience
        self._hps = hps

    def execute(self):
        self._model = self._train_model()

    def get_model(self):
        return self._model

    def _train_model(self):
        torch.backends.cudnn.benchmark = True

        from models.sub_task import EarlyStopping
        early_stopping = EarlyStopping(patience=self._patience, verbose=True)
        for epoch in range(1, self._hp_epochs + 1):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    if self._do_manual_decay:
                        self._lr = self._manual_decay(epoch)
                    else:
                        pass
                    self._model.train()
                else:
                    self._model.eval()

                running_loss = self._train_batch_model(epoch, phase)
            if phase == 'valid':
                best_model = early_stopping(running_loss, self._model)
                if early_stopping.early_stop:
                    break
        return best_model

    def _train_batch_model(self, epoch, phase):
        from tqdm import tqdm
        epoch_loss = 0.0
        total = 0

        getattr(self._dataloaders.dataset, phase)()
        with tqdm(total=len(self._dataloaders), unit="batch", ncols=120) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{self._hp_epochs}] ({phase.ljust(5)})")

            for data in self._dataloaders:
                inputs = add_device(data[self._input_key], self._device)
                labels = add_device(data[self._target_key], self._device)
                self._optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        self._optimizer.step()

                    epoch_loss += loss.item() * inputs[0].size(0)
                    total += inputs[0].size(0)

                    running_loss = epoch_loss / total

                    pbar.set_postfix({"loss": f'{running_loss:.4f}', "lr": f'{self._lr:.4f}'})
                    pbar.update(1)
        if phase == 'valid':
            logger.info(f'Epoch [{epoch}/{self._hp_epochs}] ({phase.ljust(6)})'
                        f'{self._model.__class__.__name__} (Loss_1ST) :'
                        f'{running_loss}')
        return running_loss

    def predict_model(self):
        return self._predict_model()

    def _predict_model(self):
        torch.backends.cudnn.benchmark = True

        results = np.array([])
        epoch_loss = 0.0
        total = 0
        with torch.no_grad():
            for data in self._dataloaders:
                inputs = add_device(data[self._input_key], self._device)
                labels = add_device(data[self._target_key], self._device)
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                epoch_loss += loss.item() * inputs[0].size(0)
                total += inputs[0].size(0)

                running_loss = epoch_loss / total

                preds = tensor_to_array(outputs)
                if not results.any():
                    results = preds
                else:
                    results = np.concatenate([results, preds])

        upper = np.loadtxt('./logs/GP_upper.csv', delimiter=',')
        lower = np.loadtxt('./logs/GP_lower.csv', delimiter=',')

        from models.sub_task import set_phi_within_valid_range

        def reshape3vec(data):
            return data.reshape(-1, 3)

        results = set_phi_within_valid_range(reshape3vec(results))
        upper = set_phi_within_valid_range(reshape3vec(upper))
        lower = set_phi_within_valid_range(reshape3vec(lower))
        ratio = np.sum(
            np.where(((lower < upper)
                      & (results < upper)
                      & (lower < results))
                     | ((upper < lower)
                        & (upper < results)
                        & (lower < results))
                     | ((upper < lower)
                        & (results < upper)
                        & (results < lower)), True, False).all(axis=1)) / (len(results))

        return running_loss, ratio


class HiggsId_BaseTask(PytorchBaseTask):
    """Documentation for HiggsId_BaseTask

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self,
                   model,
                   dataloaders,
                   input_key: str,
                   target_key: str,
                   criterion: str,
                   preprocess=None,
                   device='cpu',
                   optimizer: str = 'Adam',
                   metrics=None,
                   activation: str = 'Sigmoid',
                   do_manual_decay: bool = False,
                   hp_epochs: int = 10,
                   hp_lr: float = 1e-3,
                   lr: float = 1e-3,
                   patience: int = 10,
                   hps: dict = {},
                   **kwargs):
        self._model = model
        self._dataloaders = dataloaders
        self._input_key = input_key
        self._target_key = target_key
        if isinstance(criterion, str):
            from models import MyLoss
            self._criterion = get_module([nn, MyLoss], criterion)()
        else:
            self._criterion = criterion

        from torch.nn import Module
        if isinstance(preprocess, Module):
            preprocess = preprocess.to(device)
        self._preprocess = preprocess
        self._device = device
        from torch import optim
        self._optimizer = get_module([optim], optimizer)(model.parameters())
        self._metrics = metrics
        if isinstance(activation, str):
            self._activation = get_module([nn], activation)()
        else:
            self._activation = activation
        self._do_manual_decay = do_manual_decay
        self._hp_epochs = hp_epochs
        self._hp_lr = hp_lr
        self._lr = lr
        self._patience = patience
        self._hps = hps

    def execute(self):
        self._model = self._train_model()

    def get_model(self):
        return self._model

    def _train_model(self):
        torch.backends.cudnn.benchmark = True

        from models.sub_task import EarlyStopping
        early_stopping = EarlyStopping(patience=self._patience, verbose=True)
        for epoch in range(1, self._hp_epochs + 1):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    if self._do_manual_decay:
                        self._lr = self._manual_decay(epoch)
                    else:
                        pass
                    self._model.train()
                else:
                    self._model.eval()

                running_loss = self._train_batch_model(epoch, phase)
            if phase == 'valid':
                best_model = early_stopping(running_loss, self._model)
                if early_stopping.early_stop:
                    break
        return best_model

    def _train_batch_model(self, epoch, phase):
        from tqdm import tqdm
        epoch_loss = 0.0
        epoch_metrics = 0.0
        total = 0

        getattr(self._dataloaders.dataset, phase)()
        with tqdm(total=len(self._dataloaders), unit="batch", ncols=120) as pbar:
            pbar.set_description(f"Epoch [{epoch}/{self._hp_epochs}] ({phase.ljust(5)})")

            for data in self._dataloaders:
                inputs = add_device(data[self._input_key], self._device)
                with torch.no_grad():
                    inputs = self._preprocess(inputs)
                inputs = inputs.to(device)
                labels = add_device(data[self._target_key], self._device)
                self._optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        self._optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    total += inputs.size(0)

                    running_loss = epoch_loss / total
                    if self._metrics is not None:
                        if self._activation:
                            outputs = self._activation(outputs)
                        outputs = tensor_to_array(outputs)
                        labels = tensor_to_array(labels)
                        metrics = self._metrics(labels, outputs)
                        epoch_metrics += metrics.item() * inputs.size(0)
                        running_metrics = epoch_metrics / total
                        s = {
                            "loss": f'{running_loss:.4f}',
                            "metrics": f'{running_metrics:.4f}',
                            "lr": f'{self._lr:.4f}'
                        }
                        log_s = (f'Epoch [{epoch}/{self._hp_epochs}] ({phase.ljust(6)})'
                                 f'{self._model.__class__.__name__} (Loss_2ND) :'
                                 f'loss/{running_loss}  '
                                 f'metrics/{running_metrics}')
                    else:
                        s = {"loss": f'{running_loss:.4f}', "lr": f'{self._lr:.4f}'}
                        log_s = (f'Epoch [{epoch}/{self._hp_epochs}] ({phase.ljust(6)})'
                                 f'{self._model.__class__.__name__} (Loss_2ND) :'
                                 f'loss/{running_loss}')

                    pbar.set_postfix(s)
                    pbar.update(1)
        if phase == 'valid':
            logger.info(log_s)
        return running_loss

    def predict_model(self):
        return self._predict_model()

    def _predict_model(self):
        torch.backends.cudnn.benchmark = True

        epoch_loss = 0.0
        epoch_metrics = 0.0
        total = 0
        with torch.no_grad():
            for data in self._dataloaders:
                inputs = add_device(data[self._input_key], self._device)
                with torch.no_grad():
                    inputs = self._preprocess(inputs)
                inputs = inputs.to(self._device)
                labels = add_device(data[self._target_key], self._device)
                outputs = self._model(inputs)

                loss = self._criterion(outputs, labels)
                epoch_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)

                running_loss = epoch_loss / total

                if self._activation:
                    outputs = self._activation(outputs)
                preds = tensor_to_array(outputs)
                labels = tensor_to_array(labels)
                metrics = self._metrics(labels, preds)
                epoch_metrics += metrics.item() * inputs.size(0)

                running_metrics = epoch_metrics / total

        return running_loss, running_metrics


def make_output_dict():
    return {
        'AUC': {f'{f}:{s}': []
                for f, s in product(FIRST_MODEL_NAME, SECOND_MODEL_NAME)},
        'LOSS_1ST': {f: []
                     for f in FIRST_MODEL_NAME},
        'LOSS_2ND': {f'{f}:{s}': []
                     for f, s in product(FIRST_MODEL_NAME, SECOND_MODEL_NAME)},
        'RATIO': {f: []
                  for f in FIRST_MODEL_NAME},
    }


def evaluate(tau4vec, pretrained_higgsId, dataloader, conf, device):
    with torch.no_grad():
        logger.info('start eval mode')
        dataloader.dataset.test()
        test_dataset = dataloader.dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        result = make_output_dict()

        pretrain_conf = conf.sub_task_params.tau4vec.pretrain
        loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
        input_key = pretrain_conf.data.input_key
        target_key = pretrain_conf.data.target_key

        for i, model_name in enumerate(FIRST_MODEL_NAME):
            model = tau4vec[i]
            model.to(device)
            tauvec_ins = Tauvec_BaseTask()
            tauvec_ins.initialize(model=model,
                                  dataloaders=test_dataloader,
                                  input_key=input_key,
                                  target_key=target_key,
                                  criterion=loss_func,
                                  device=device)
            (result['LOSS_1ST'][model_name],
             result['RATIO'][model_name]) = tauvec_ins.predict_model()

        for i, j in product(range(3), range(3)):
            pretrain_conf = conf.sub_task_params.higgsId.pretrain
            model = pretrained_higgsId[f'{FIRST_MODEL_NAME[i]}:{SECOND_MODEL_NAME[j]}']
            model.to(device)
            preprocess = tau4vec[i]
            loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
            from models import MyMetrics
            metrics = set_module([nn, MyMetrics], pretrain_conf, 'metrics')
            activation = pretrain_conf.activation.name
            input_key = pretrain_conf.data.input_key
            target_key = pretrain_conf.data.target_key
            higgsid_ins = HiggsId_BaseTask()
            higgsid_ins.initialize(model=model,
                                   dataloaders=test_dataloader,
                                   input_key=input_key,
                                   target_key=target_key,
                                   criterion=loss_func,
                                   preprocess=preprocess,
                                   device=device,
                                   metrics=metrics,
                                   activation=activation)
            (result['LOSS_2ND'][f'{FIRST_MODEL_NAME[i]}:{SECOND_MODEL_NAME[j]}'],
             result['AUC'][f'{FIRST_MODEL_NAME[i]}:{SECOND_MODEL_NAME[j]}']
             ) = higgsid_ins.predict_model()

    logger.info(result)
    return result


@click.command()
@click.option('--conf', '-c', type=str, default="./config/config_phased_pretrain.yaml")
@click.option('--seed', '-s', type=int, default=None)
@click.option('--gpu_index', '-gi', type=int, default=None)
@click.option('--data_path', '-dp', type=str, default=None)
@log(logger)
def main(conf: str, seed: int, gpu_index: int, data_path: str):
    global device
    conf = load_config(conf)
    if seed:
        conf.seed = seed
    if gpu_index and device == torch.device('cuda'):
        device = torch.device(f'cuda:{gpu_index}')
    if data_path is not None:
        conf['dataset']['params']['data_path'] = data_path
    logger.info(device)
    logger.info(conf)

    results = {}
    for add_seed in range(10):
        conf.seed += 1
        set_seed(conf.seed)
        from models import sub_task
        tau4vec = set_task(conf.sub_task_params, 'tau4vec', sub_task)
        logger.info('set_task: tau4vec')
        set_seed(conf.seed)
        higgsId = set_task(conf.sub_task_params, 'higgsId', sub_task)
        logger.info('set_task: higgsId')
        # #####################################################################
        # #####################################################################
        logger.info('copy the pretrain models')
        pre_model = [tau4vec, higgsId]
        task = [tau4vec, higgsId]
        for num_task, sub in enumerate(task):
            for num_model in range(len(sub)):
                pre_model[num_task][num_model].load_state_dict(
                    task[num_task][num_model].state_dict())
        # #####################################################################
        # #####################################################################
        from models import MyDataset
        from models import MyMetrics
        set_seed(conf.seed)
        dataset = set_module([MyDataset], conf, 'dataset')
        set_seed(conf.seed)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        logger.info('set dataloader')
        # #####################################################################
        # pre-train ###########################################################
        # #####################################################################
        logger.info('----- pretrain[0] start -----')
        pretrain_conf = conf.sub_task_params.tau4vec.pretrain
        for i, sub_model in enumerate(tau4vec):
            logger.info(f'pretrain: [0][{i}]')
            set_seed(conf.seed)
            loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
            input_key = pretrain_conf.data.input_key
            target_key = pretrain_conf.data.target_key
            optimizer = pretrain_conf.optimizer.name
            epochs = pretrain_conf.epochs
            patience = pretrain_conf.patience
            lr = pretrain_conf.optimizer.params.lr
            tauvec_base = Tauvec_BaseTask()
            model = sub_model.to(device)
            tauvec_base.initialize(model=model,
                                   dataloaders=dataloader,
                                   input_key=input_key,
                                   target_key=target_key,
                                   criterion=loss_func,
                                   device=device,
                                   optimizer=optimizer,
                                   hp_epochs=epochs,
                                   lr=lr,
                                   patience=patience)
            tauvec_base.execute()
            tau4vec[i] = tauvec_base.get_model()
        logger.info('----- pretrain[0] end -----')

        logger.info('----- pretrain[1] start -----')
        pretrain_conf = conf.sub_task_params.higgsId.pretrain
        pretrained_higgsId = {}
        for i, j in product(range(3), range(3)):
            logger.info(f'pretrain: [{i}][{j}]')
            set_seed(conf.seed)
            model = higgsId[j]
            model.load_state_dict(pre_model[1][j].state_dict())
            model.to(device)
            loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
            metrics = set_module([nn, MyMetrics], pretrain_conf, 'metrics')
            activation = pretrain_conf.activation.name
            input_key = pretrain_conf.data.input_key
            target_key = pretrain_conf.data.target_key
            optimizer = pretrain_conf.optimizer.name
            epochs = pretrain_conf.epochs
            patience = pretrain_conf.patience
            lr = pretrain_conf.optimizer.params.lr
            higgsid_base = HiggsId_BaseTask()
            higgsid_base.initialize(model=model,
                                    dataloaders=dataloader,
                                    input_key=input_key,
                                    target_key=target_key,
                                    criterion=loss_func,
                                    preprocess=tau4vec[i],
                                    device=device,
                                    optimizer=optimizer,
                                    metrics=metrics,
                                    activation=activation,
                                    hp_epochs=epochs,
                                    lr=lr,
                                    patience=patience)
            higgsid_base.execute()
            pretrained_higgsId[
                f'{FIRST_MODEL_NAME[i]}:{SECOND_MODEL_NAME[j]}'] = higgsid_base.get_model()
        logger.info('----- pretrain[1] end -----')
        results[conf.seed] = evaluate(tau4vec, pretrained_higgsId, dataloader, conf, device)
        logger.info(results)

    logger.info(results)
    logger.info('all train and eval step are done')


if __name__ == '__main__':
    main()
