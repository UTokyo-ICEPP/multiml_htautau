import datetime
import os

from torch import nn
from tqdm import tqdm
import numpy as np
import torch

from utils import (
    add_device,
    get_logger,
    tensor_to_array,
)
from .MyMetrics import Calc_Auc
from .sub_task import EarlyStopping

root = "./"
logger = get_logger()


class SPOS(nn.Module):
    def __init__(self, task, loss_func, loss_weight=None,
                 input_key='inputs', target_key='targets',
                 save_dir='',
                 **kwargs):
        super(SPOS, self).__init__(**kwargs)
        self._task = task
        self._loss_func = loss_func
        if loss_weight is None:
            self._loss_weight = np.ones(len(loss_func)) / len(loss_func)
        if (isinstance(loss_weight, list)
           or isinstance(loss_weight, np.ndarray)):
            self._loss_weight = np.array(
                [w if w else None for w in loss_weight]
            )
        self._task_candidate_len = [len(t) for t in task]
        self._choice_block = nn.ModuleList([])
        for t in self._task:
            layer_cb = nn.ModuleList([])
            for sub_task in t:
                layer_cb.append(sub_task)
            self._choice_block.append(layer_cb)
        NOW = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        save_path = os.path.join(root, "logs", save_dir, NOW)
        self._save_path = save_path
        os.makedirs(self._save_path, exist_ok=True)
        self._input_key = input_key
        self._target_key = target_key

    @staticmethod
    def _random_index(task_candidate_len):
        return [np.random.randint(i) for i in task_candidate_len]

    def forward(self, x, choice=None):
        if choice is None:
            choice = self._random_index(self._task_candidate_len)
        outputs = []
        for layer_num, index in enumerate(choice):
            if layer_num == 0:
                outputs.append(self._choice_block[layer_num][index](x))
            else:
                outputs.append(
                    self._choice_block[layer_num][index](outputs[-1])
                )
        return outputs, choice

    def fit(self,
            epochs,
            dataloader,
            device,
            optimizer,
            scheduler,
            patience=3,
            choice=None):
        self.to(device)
        logger.info(f'save at {self._save_path}')
        early_stopping = EarlyStopping(patience=patience,
                                       verbose=True,
                                       path=self._save_path, save=True)
        sigmoid = nn.Sigmoid()
        metrics = Calc_Auc()
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            dataloader.dataset.train()
            train_data = tqdm(dataloader)
            lr = scheduler.get_last_lr()[0]
            train_data.set_description(
                f'[Epoch:{epoch+1:04d}/{epochs:04d} lr:{lr:.5f}]'
            )
            for step, data in enumerate(train_data):
                inputs = add_device(data[self._input_key], device)
                targets = add_device(data[self._target_key], device)
                optimizer.zero_grad()
                outputs, now_choice = self.__call__(inputs, choice)
                loss = 0.0
                for criterion, weight, output, target in zip(self._loss_func,
                                                             self._loss_weight,
                                                             outputs,
                                                             targets):
                    if weight is not None:
                        loss += criterion(output, target) * weight
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                running_loss = train_loss / (step + 1)
                postfix = {'train_loss': f'{running_loss:.5f}',
                           'choice': f'{now_choice}'}
                train_data.set_postfix(log=postfix)
            with torch.no_grad():
                self.eval()
                dataloader.dataset.valid()
                outputs_data = []
                targets_data = []
                valid_loss = 0.0
                for step, data in enumerate(dataloader):
                    inputs = add_device(data[self._input_key], device)
                    targets = add_device(data[self._target_key], device)
                    outputs, now_choice = self.__call__(inputs, choice)
                    loss = 0.0
                    for criterion, weight, output, target in zip(
                            self._loss_func,
                            self._loss_weight,
                            outputs,
                            targets
                    ):
                        if weight is not None:
                            loss += criterion(output, target) * weight
                    outputs_data.extend(
                        tensor_to_array(sigmoid(outputs[1]))
                    )
                    targets_data.extend(
                        tensor_to_array(targets[1])
                    )
                    valid_loss += loss.item()
                targets_data = np.array(targets_data)
                outputs_data = np.array(outputs_data)
                auc_score = metrics(targets_data, outputs_data)
                s = f'[Epoch:{epoch+1:04d}|valid| / '\
                    f'auc:{auc_score:.6f} / '\
                    f'loss:{valid_loss/(step+1):.6f}]'
                logger.info(s)
                _ = early_stopping(valid_loss/(step+1), self)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

        self.load_state_dict(torch.load(
            os.path.join(self._save_path, 'checkpoint.pt')
        ))
