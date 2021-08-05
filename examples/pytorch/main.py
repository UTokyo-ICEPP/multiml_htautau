#!/usr/bin/env python3

from itertools import product
from copy import deepcopy

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import click
import numpy as np
import torch

from models import MyLoss
from utils import (
    add_device,
    get_logger,
    get_module,
    load_config,
    log,
    set_module,
    set_seed,
    set_task,
    tensor_to_array,
)

log_dir = './logs/'
logger = get_logger(logdir_path=log_dir)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

FIRST_MODEL_NAME = ['MLP', 'CONV2D', 'SF']
SECOND_MODEL_NAME = ['MLP', 'LSTM', 'MASS']
MODELNAME_CHOICE_INDEX = {
    f'{n1}_{n2}': v
    for (n1, n2), v in zip(product(FIRST_MODEL_NAME, SECOND_MODEL_NAME),
                           product(range(len(FIRST_MODEL_NAME)), range(len(SECOND_MODEL_NAME))))
}


def evaluate(model, conf, dataloader, metrics, result):
    with torch.no_grad():
        logger.info('start eval mode')
        model.eval()
        dataloader.dataset.test()
        test_dataset = dataloader.dataset
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        A = range(len(conf.sub_task_params.tau4vec.tasks))
        B = range(len(conf.sub_task_params.higgsId.tasks))
        count = 0
        num_name_conb = {
            num: f'{f}_{s}'
            for num, (f, s) in enumerate(product(FIRST_MODEL_NAME, SECOND_MODEL_NAME))
        }
        num_name_1st = {
            num: f
            for num, (f, s) in enumerate(product(FIRST_MODEL_NAME, SECOND_MODEL_NAME))
        }
        for choice in product(A, B):
            outputs_data = []
            targets_data = []
            temp_outputs_data = []
            temp_targets_data = []
            for data in test_dataloader:
                inputs = add_device(data['inputs'], DEVICE)
                targets = add_device(data['targets'], DEVICE)
                outputs, choice = model(inputs, choice)
                outputs_data.extend(tensor_to_array(outputs[1]))
                targets_data.extend(tensor_to_array(targets[1]))
                temp_outputs_data.extend(tensor_to_array(outputs[0]))
                temp_targets_data.extend(tensor_to_array(targets[0]))
            targets_data = np.array(targets_data)
            outputs_data = np.array(outputs_data)
            auc_score = metrics(targets_data, outputs_data)
            result['AUC'][num_name_conb[count]].append(auc_score)
            temp_outputs_data = np.array(temp_outputs_data)
            temp_targets_data = np.array(temp_targets_data)
            upper = np.loadtxt('./logs/GP_upper.csv', delimiter=',')
            lower = np.loadtxt('./logs/GP_lower.csv', delimiter=',')

            c_1 = set_module([torch.nn, MyLoss], conf.SPOS_NAS, 'loss_first')
            c_2 = set_module([torch.nn, MyLoss], conf.SPOS_NAS, 'loss_second')
            loss_1st = c_1(torch.tensor(temp_outputs_data), torch.tensor(temp_targets_data))
            loss_2nd = c_2(torch.tensor(outputs_data), torch.tensor(targets_data))

            from models.sub_task import set_phi_within_valid_range

            def reshape3vec(data):
                return data.reshape(-1, 3)

            temp_outputs_data = set_phi_within_valid_range(reshape3vec(temp_outputs_data))
            upper = set_phi_within_valid_range(reshape3vec(upper))
            lower = set_phi_within_valid_range(reshape3vec(lower))
            if count in [0, 3, 6]:
                query = (((lower < upper)
                          & (temp_outputs_data < upper)
                          & (lower < temp_outputs_data))
                         | ((upper < lower)
                            & (upper < temp_outputs_data)
                            & (lower < temp_outputs_data))
                         | ((upper < lower)
                            & (temp_outputs_data < upper)
                            & (temp_outputs_data < lower)))
                ratio = np.sum(np.where(query, True, False).all(axis=1)) / (len(temp_outputs_data))
                result['RATIO'][num_name_1st[count]].append(ratio)
                query = (((lower[:, 0] < upper[:, 0])
                          & (temp_outputs_data[:, 0] < upper[:, 0])
                          & (lower[:, 0] < temp_outputs_data[:, 0]))
                         | ((upper[:, 0] < lower[:, 0])
                            & (upper[:, 0] < temp_outputs_data[:, 0])
                            & (lower[:, 0] < temp_outputs_data[:, 0]))
                         | ((upper[:, 0] < lower[:, 0])
                            & (temp_outputs_data[:, 0] < upper[:, 0])
                            & (temp_outputs_data[:, 0] < lower[:, 0])))
                only_pt_ratio = np.sum(np.where(query, True, False)) / (len(temp_outputs_data))
                result['ONLY_PT_RATIO'][num_name_1st[count]].append(only_pt_ratio)

                result['LOSS_1ST'][num_name_1st[count]].append(loss_1st.item())

            result['LOSS_2ND'][num_name_conb[count]].append(loss_2nd.item())
            logger.info(f'[Choice:{choice} / auc:{auc_score:.6f}] / ' +
                        f'first_loss: {loss_1st:.6f} / ' + f'ratio: {ratio:.6f} / ' +
                        f'only_pt_ratio: {only_pt_ratio:.6f} / ')
            count += 1

    logger.info(result)
    return result


@click.command()
@click.option('--conf', '-c', type=str, default="./config/config.yaml")
@click.option('--seed', '-s', type=int, default=None)
@click.option('--gpu_index', '-gi', type=int, default=None)
@click.option('--data_path', '-dp', type=str, default=None)
@click.option('--event', '-e', type=int, default=None)
@log(logger)
def main(conf: str, seed: int, gpu_index: int, data_path: str, event: int):
    global DEVICE, FIRST_MODEL_NAME, SECOND_MODEL_NAME, MODELNAME_CHOICE_INDEX
    conf = load_config(conf)
    if seed is not None:
        conf.seed = seed
    if gpu_index is not None and DEVICE == torch.device('cuda'):
        DEVICE = torch.device(f'cuda:{gpu_index}')
    if data_path is not None:
        conf['dataset']['params']['data_path'] = data_path
    if event is not None:
        conf['dataset']['params']['max_events'] = event
    logger.info(DEVICE)
    logger.info(conf)

    FIRST_MODEL_NAME = [
        i['name'].split('_')[-1][:-4] + f'-{num}'
        for num, i in enumerate(conf.sub_task_params.tau4vec.tasks)
    ]
    SECOND_MODEL_NAME = [
        i['name'].split('_')[-1][:-4] + f'-{num}'
        for num, i in enumerate(conf.sub_task_params.higgsId.tasks)
    ]
    MODELNAME_CHOICE_INDEX = {
        f'{n1}_{n2}': v
        for (n1,
             n2), v in zip(product(FIRST_MODEL_NAME, SECOND_MODEL_NAME),
                           product(range(len(FIRST_MODEL_NAME)), range(len(SECOND_MODEL_NAME))))
    }

    set_seed(conf.seed)
    from models import sub_task
    tau4vec = set_task(conf.sub_task_params, 'tau4vec', sub_task)
    logger.info('set_task: tau4vec')
    set_seed(conf.seed)
    higgsId = set_task(conf.sub_task_params, 'higgsId', sub_task)
    logger.info('set_task: higgsId')
    from models import MyDataset
    from models import MyMetrics
    set_seed(conf.seed)
    dataset = set_module([MyDataset], conf, 'dataset')
    set_seed(conf.seed)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    logger.info('set dataloader')
    # #########################################################################
    # pre-train ###############################################################
    # #########################################################################
    logger.info('----- pretrain[0] start -----')
    pretrain_conf = conf.sub_task_params.tau4vec.pretrain
    for i, sub_model in enumerate(tau4vec):
        logger.info(f'pretrain: [0][{i}]')
        set_seed(conf.seed)
        optimizer = set_module([optim], pretrain_conf, 'optimizer', params=sub_model.parameters())
        loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
        metrics = set_module([MyMetrics], pretrain_conf, 'metrics')
        activation = set_module([nn], pretrain_conf, 'activation')
        input_key = pretrain_conf.data.input_key
        target_key = pretrain_conf.data.target_key
        patience = pretrain_conf.patience
        tau4vec[i] = sub_task.pre_train(epochs=pretrain_conf.epochs,
                                        model=sub_model,
                                        dataloader=dataloader,
                                        optimizer=optimizer,
                                        loss_func=loss_func,
                                        input_key=input_key,
                                        target_key=target_key,
                                        device=DEVICE,
                                        patience=patience,
                                        metrics=metrics,
                                        activation=activation)
    logger.info('----- pretrain[0] end -----')
    logger.info('----- pretrain[1] start -----')
    pretrain_conf = conf.sub_task_params.higgsId.pretrain
    for i, sub_model in enumerate(higgsId):
        logger.info(f'pretrain: [1][{i}]')
        set_seed(conf.seed)
        optimizer = set_module([optim], pretrain_conf, 'optimizer', params=sub_model.parameters())
        loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
        metrics = set_module([MyMetrics], pretrain_conf, 'metrics')
        activation = set_module([nn], pretrain_conf, 'activation')
        input_key = pretrain_conf.data.input_key
        target_key = pretrain_conf.data.target_key
        patience = pretrain_conf.patience
        higgsId[i] = sub_task.pre_train(epochs=pretrain_conf.epochs,
                                        model=sub_model,
                                        dataloader=dataloader,
                                        optimizer=optimizer,
                                        loss_func=loss_func,
                                        input_key=input_key,
                                        target_key=target_key,
                                        device=DEVICE,
                                        patience=patience,
                                        metrics=metrics,
                                        activation=activation)
    logger.info('----- pretrain[1] end -----')

    # #########################################################################
    # #########################################################################
    logger.info('copy the pretrain models')
    pre_trained_tau4vec = set_task(conf.sub_task_params, 'tau4vec', sub_task)
    pre_trained_higgsId = set_task(conf.sub_task_params, 'higgsId', sub_task)
    pre_trained_model = [pre_trained_tau4vec, pre_trained_higgsId]
    task = [tau4vec, higgsId]
    for num_task, sub in enumerate(task):
        for num_model in range(len(sub)):
            pre_trained_model[num_task][num_model].load_state_dict(
                deepcopy(task[num_task][num_model].state_dict()))
    # #########################################################################
    # #########################################################################

    logger.info('----- SPOS-NAS start -----')
    sposnas_conf = conf.SPOS_NAS

    def make_output_dict():
        return {
            'X': [],
            'AUC': {f'{f}_{s}': []
                    for f, s in product(FIRST_MODEL_NAME, SECOND_MODEL_NAME)},
            'LOSS_1ST': {f: []
                         for f in FIRST_MODEL_NAME},
            'LOSS_2ND': {f'{f}_{s}': []
                         for f, s in product(FIRST_MODEL_NAME, SECOND_MODEL_NAME)},
            'RATIO': {f: []
                      for f in FIRST_MODEL_NAME},
            'ONLY_PT_RATIO': {f: []
                              for f in FIRST_MODEL_NAME},
        }

    # evaluate only pre-train model
    loss_func = [
        set_module([nn, MyLoss], sposnas_conf, 'loss_first'),
        set_module([nn, MyLoss], sposnas_conf, 'loss_second')
    ]
    loss_weight = [0.5, 0.5]
    metrics = get_module([MyMetrics], 'Calc_Auc')()
    from models.SPOS_NAS import SPOS
    model = SPOS(task=task, loss_func=loss_func, loss_weight=loss_weight)
    model.to(DEVICE)
    logger.info('evaluate only pre-train model')
    dummy = make_output_dict()
    evaluate(model, conf, dataloader, metrics, dummy)

    output_dict = make_output_dict()
    X_list = [i for i in range(11)]
    X_list[1:1] = [0.01, 0.1]
    X_list[-1:-1] = [9.9, 9.99]
    for X in (np.array(X_list) * 0.1).round(10):
        output_dict['X'].append(X)
        logger.info(f'loss_ratio: {X:.6f} (loss_1*X + loss_2*(1-X)) start')
        set_seed(conf.seed)
        logger.info('load pretrain models...')
        for num_task, sub in enumerate(task):
            for num_model in range(len(sub)):
                task[num_task][num_model].load_state_dict(
                    deepcopy(pre_trained_model[num_task][num_model].state_dict()))
        logger.info('load pretrain models done')
        logger.info('set model parameters...')
        loss_func = [
            set_module([nn, MyLoss], sposnas_conf, 'loss_first'),
            set_module([nn, MyLoss], sposnas_conf, 'loss_second')
        ]
        loss_weight = [X, 1. - X]
        metrics = get_module([MyMetrics], 'Calc_Auc')()

        model = SPOS(task=task, loss_func=loss_func, loss_weight=loss_weight, save_dir='SPOS')
        model.to(DEVICE)
        optimizer = set_module([optim], sposnas_conf, 'optimizer', params=model.parameters())
        scheduler = set_module([optim.lr_scheduler],
                               sposnas_conf,
                               'scheduler',
                               optimizer=optimizer)
        logger.info('set model parameters done')
        logger.info('fit model...')
        model.fit(epochs=sposnas_conf.epochs,
                  dataloader=dataloader,
                  device=DEVICE,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  patience=sposnas_conf.patience)
        logger.info('fit model done')
        logger.info('eval model...')
        output_dict = evaluate(model, conf, dataloader, metrics, output_dict)
        logger.info('eval model done')

        set_seed(conf.seed)
        logger.info('re-train start')
        selected_model, _ = max({k: v[-1]
                                 for k, v in output_dict['AUC'].items()}.items(),
                                key=lambda x: x[1])
        logger.info(f'selected_model: {selected_model}')
        selected_choice = MODELNAME_CHOICE_INDEX[selected_model]
        model.fit(epochs=sposnas_conf.epochs,
                  dataloader=dataloader,
                  device=DEVICE,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  patience=sposnas_conf.patience,
                  choice=selected_choice)
        logger.info('re-train done')
        dummy = None
        dummy = make_output_dict()
        dummy = evaluate(model, conf, dataloader, metrics, dummy)

        def result_parser(res, selected_model, seed, X):
            AUC = res['AUC'][selected_model][0]
            LOSS_1ST = res['LOSS_1ST'][selected_model.split('_')[0]][0]
            LOSS_2ND = res['LOSS_2ND'][selected_model][0]
            RATIO = res['RATIO'][selected_model.split('_')[0]][0]
            ONLY_PT_RATIO = res['ONLY_PT_RATIO'][selected_model.split('_')[0]][0]
            target_result = dict(seed=seed,
                                 X=X,
                                 AUC=AUC,
                                 LOSS_1ST=LOSS_1ST,
                                 LOSS_2ND=LOSS_2ND,
                                 RATIO=RATIO,
                                 ONLY_PT_RATIO=ONLY_PT_RATIO)
            logger.info(f're-train results: {target_result}')

        result_parser(dummy, selected_model, conf.seed, X)

    logger.info('all train and eval step are done')
    logger.info('plot results done')


if __name__ == '__main__':
    main()
