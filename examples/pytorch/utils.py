#!/usr/bin/env python3

from functools import wraps
from typing import Union
import datetime
import inspect
import logging
import os
import random

from attrdict import AttrDict
from torch import Tensor
import numpy as np
import tensorflow
import torch
import yaml


def load_config(config_path: str) -> AttrDict:
    """config(yaml) parser

    Parameters
    ----------
    config_path : string
        path of input config file

    Returns
    -------
    config : attrdict.AttrDict
        config(yaml) converted to attrdict
    """
    with open(config_path, 'r', encoding='utf-8') as fi_:
        return AttrDict(yaml.load(fi_, Loader=yaml.SafeLoader))


def get_module(groups: list, name: Union[str, bool]):
    if name:
        for group in groups:
            if hasattr(group, name):
                return getattr(group, name)
        raise RuntimeError("Module not found:", name)
    else:

        def return_none(**args):
            return None

        return return_none


def set_module(groups: list, config: dict, key: str, **kwargs):
    conf = config[key]
    name = conf['name']
    params = conf.get('params', {})
    params.update(kwargs)
    return get_module(groups, name)(**params)


def set_task(config: dict, key: str, task_module) -> list:
    conf = config[key]
    tasks = conf['tasks']
    task_list = []
    for t in tasks:
        name = t['name']
        params = t.get('params', {})
        task_list.append(get_module([task_module], name)(**params))
    return task_list


def set_seed(seed=200):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    tensorflow.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def tensor_to_array(input_data):
    """ torch.tensor -> np.array
    Parameters
    ----------
    input_data : torch.Tensor
        torch.Tensor data
    Returns
    -------
    output_data : np.ndarray
        input_data converted to array type
    """
    output_data = input_data.to('cpu').detach().numpy().copy()
    return output_data


def _rec_add_device(data, device):
    output = []
    for x in data:
        if isinstance(x, list):
            output.extend(add_device(x, device))
        elif isinstance(x, Tensor):
            output.append(x.to(device))
        elif isinstance(x, np.ndarray):
            output.append(torch.Tensor(x).to(device))
    return output


def add_device(data, device):
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return _rec_add_device(data, device)


class CustomFilter(logging.Filter):
    """fillter for logger"""
    def filter(self, record):
        record.real_filename = getattr(record, 'real_filename', record.filename)
        record.real_funcName = getattr(record, 'real_funcName', record.funcName)
        record.real_lineno = getattr(record, 'real_lineno', record.lineno)
        return True


def get_logger(logdir_path=None):
    """logging.Logger

    Args
    ----
    logdir_path: str
        path of the directory where the log files will be output

    Returns
    -------
    logger (logging.Logger): instance of logging.Logger
    """

    log_format = ('%(levelname)-8s - %(asctime)s - '
                  '[%(real_filename)s %(real_funcName)s %(real_lineno)d] %(message)s')

    sh = logging.StreamHandler()
    sh.addFilter(CustomFilter())
    # sh.setLevel(logging.INFO)

    if logdir_path is None:
        logging.basicConfig(handlers=[sh], format=log_format, level=logging.INFO)
    else:
        if not os.path.exists(logdir_path):
            os.makedirs(logdir_path)
        logfile_path = logdir_path + str(datetime.date.today()) + '.log'
        fh = logging.FileHandler(logfile_path)
        fh.addFilter(CustomFilter())
        logging.basicConfig(handlers=[sh, fh], format=log_format, level=logging.INFO)

    logger = logging.getLogger(__name__)
    return logger


def log(logger):
    """logger function

    Args
    ----
    logger (logging.Logger)

    """
    def _decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            extra = {
                'real_filename': inspect.getfile(func),
                'real_funcName': func_name,
                'real_lineno': inspect.currentframe().f_back.f_lineno
            }

            logger.info(f'[START] {func_name}', extra=extra)

            try:
                return func(*args, **kwargs)
            except Exception as err:
                logging.error(err, exc_info=True, extra=extra)
                logging.error(f'[KILLED] {func_name}', extra=extra)
            else:
                logging.info(f'[END] {func_name}', extra=extra)

        return wrapper

    return _decorator
