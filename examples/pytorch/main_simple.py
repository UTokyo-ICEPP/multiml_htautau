#!/usr/bin/env python3

from itertools import product
from copy import deepcopy
import json
import os
import time

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import click
import numpy as np
import torch

from models import MyLoss
from models.SPOS_NAS import SPOS
from utils import (
    add_device,
    get_logger,
    get_module,
    load_config,
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
    f'{n1}_{n2}': v for (n1, n2), v in zip(
        product(FIRST_MODEL_NAME, SECOND_MODEL_NAME),
        product(range(len(FIRST_MODEL_NAME)), range(len(SECOND_MODEL_NAME)))
    )
}


def evaluate(model, conf, dataloader, metrics, result, is_gp_3dim):
    is_gp_check = False
    if conf['dataset']['params']['max_events'] == 50000:
        is_gp_check = True
    with torch.no_grad():
        logger.info('start eval mode')
        model.eval()
        dataloader.dataset.test()
        test_dataset = dataloader.dataset
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=100,
                                     shuffle=False)
        A = range(len(FIRST_MODEL_NAME))
        B = range(len(SECOND_MODEL_NAME))
        count = 0
        num_name_conb = {
            num: f'{f}_{s}' for num, (f, s) in enumerate(
                product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                )
            )
        }
        num_name_1st = {
            num: f for num, (f, s) in enumerate(
                product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                )
            )
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
            loss_1st = c_1(torch.tensor(temp_outputs_data),
                           torch.tensor(temp_targets_data))
            loss_2nd = c_2(torch.tensor(outputs_data),
                           torch.tensor(targets_data))

            if is_gp_check:
                from models.sub_task import set_phi_within_valid_range

                def reshape3vec(data):
                    return data.reshape(-1, 3)

                temp_outputs_data = set_phi_within_valid_range(
                    reshape3vec(temp_outputs_data)
                )
                upper = set_phi_within_valid_range(
                    reshape3vec(upper)
                )
                lower = set_phi_within_valid_range(
                    reshape3vec(lower)
                )
                if not is_gp_3dim:
                    temp_outputs_data = temp_outputs_data.reshape(-1, 6)
                    upper = upper.reshape(-1, 6)
                    lower = lower.reshape(-1, 6)

                query = (
                    ((lower < upper)
                     & (temp_outputs_data < upper)
                     & (lower < temp_outputs_data))
                    | ((upper < lower)
                       & (upper < temp_outputs_data)
                       & (lower < temp_outputs_data))
                    | ((upper < lower)
                       & (temp_outputs_data < upper)
                       & (temp_outputs_data < lower))
                )
                ratio = np.sum(
                    np.where(query, True, False).all(axis=1)
                )/(len(temp_outputs_data))
                result['RATIO'][num_name_1st[count]] = [ratio]

                query = (
                    ((lower[:, 0] < upper[:, 0])
                     & (temp_outputs_data[:, 0] < upper[:, 0])
                     & (lower[:, 0] < temp_outputs_data[:, 0]))
                    | ((upper[:, 0] < lower[:, 0])
                       & (upper[:, 0] < temp_outputs_data[:, 0])
                       & (lower[:, 0] < temp_outputs_data[:, 0]))
                    | ((upper[:, 0] < lower[:, 0])
                       & (temp_outputs_data[:, 0] < upper[:, 0])
                       & (temp_outputs_data[:, 0] < lower[:, 0]))
                )
                if not is_gp_3dim:
                    query = (
                        ((lower[:, [0, 3]] < upper[:, [0, 3]])
                         & (temp_outputs_data[:, [0, 3]] < upper[:, [0, 3]])
                         & (lower[:, [0, 3]] < temp_outputs_data[:, [0, 3]]))
                        | ((upper[:, [0, 3]] < lower[:, [0, 3]])
                           & (upper[:, [0, 3]] < temp_outputs_data[:, [0, 3]])
                           & (lower[:, [0, 3]] < temp_outputs_data[:, [0, 3]]))
                        | ((upper[:, [0, 3]] < lower[:, [0, 3]])
                           & (temp_outputs_data[:, [0, 3]] < upper[:, [0, 3]])
                           & (temp_outputs_data[:, [0, 3]] < lower[:, [0, 3]]))
                    )
                only_pt_ratio = np.sum(
                    np.where(query, True, False)
                )/(len(temp_outputs_data))
                result['ONLY_PT_RATIO'][num_name_1st[count]] = [only_pt_ratio]
            else:
                ratio = -1.0
                only_pt_ratio = -1.0
                result['RATIO'][num_name_1st[count]] = [ratio]
                result['ONLY_PT_RATIO'][num_name_1st[count]] = [only_pt_ratio]

            result['LOSS_1ST'][num_name_1st[count]] = [loss_1st.item()]

            result['LOSS_2ND'][num_name_conb[count]].append(loss_2nd.item())
            logger.info(f'[Choice:{choice} / auc:{auc_score:.6f}] / ' +
                        f'first_loss: {loss_1st:.6f} / ' +
                        f'ratio: {ratio:.6f} / ' +
                        f'only_pt_ratio: {only_pt_ratio:.6f}')
            count += 1

    logger.info(result)
    return result


@click.command()
@click.option('--conf', '-c', type=str, default="./config/config.yaml")
@click.option('--seed', '-s', type=int, default=None)
@click.option('--gpu_index', '-gi', type=int, default=None)
@click.option('--data_path', '-dp', type=str, default=None)
@click.option('--event', '-e', type=int, default=None)
@click.option('--weight', '-w', type=float, default=0.0)
@click.option('--n_times_model', '-nt', type=int, default=1)
@click.option('--prefix', '-p', type=str, default='')
@click.option('--is_gp_3dim', '-idp', type=bool, default=False)
def main(
        conf: str,
        seed: int,
        gpu_index: int,
        data_path: str,
        event: int,
        weight: float,
        n_times_model: int,
        prefix: str,
        is_gp_3dim: bool
):
    global DEVICE, FIRST_MODEL_NAME, SECOND_MODEL_NAME, MODELNAME_CHOICE_INDEX
    start = time.time()
    conf = load_config(conf)
    if seed is not None:
        conf.seed = seed
    if gpu_index is not None and DEVICE == torch.device('cuda'):
        # WARNING: Enable gp_re_index dict in gpu02 only
        gpu_re_index = {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3, 6: 6, 7: 7}
        gpu_index = gpu_re_index[gpu_index]
        DEVICE = torch.device(f'cuda:{gpu_index}')
    if data_path is not None:
        conf['dataset']['params']['data_path'] = data_path
    if event is not None:
        conf['dataset']['params']['max_events'] = event
    conf['is_gp_3dim'] = is_gp_3dim
    logger.info(DEVICE)
    logger.info(conf)

    model_confs_tau4vec = conf.sub_task_params.tau4vec
    model_confs_tau4vec['tasks'] = model_confs_tau4vec['tasks'] * n_times_model
    model_confs_higgsId = conf.sub_task_params.higgsId
    model_confs_higgsId['tasks'] = model_confs_higgsId['tasks'] * n_times_model
    sub_models_conf = {
        'tau4vec': model_confs_tau4vec,
        'higgsId': model_confs_higgsId
    }
    FIRST_MODEL_NAME = [
        i['name'].split('_')[-1][:-4] + f'-{num}'
        for num, i in enumerate(model_confs_tau4vec['tasks'])
    ]
    SECOND_MODEL_NAME = [
        i['name'].split('_')[-1][:-4] + f'-{num}'
        for num, i in enumerate(model_confs_higgsId['tasks'])
    ]
    MODELNAME_CHOICE_INDEX = {
        f'{n1}_{n2}': v
        for (n1, n2), v in zip(
                product(FIRST_MODEL_NAME,
                        SECOND_MODEL_NAME),
                product(range(len(FIRST_MODEL_NAME)),
                        range(len(SECOND_MODEL_NAME)))
        )
    }

    set_seed(conf.seed)
    from models import sub_task
    tau4vec = set_task(sub_models_conf, 'tau4vec', sub_task)
    logger.info('set_task: tau4vec')
    set_seed(conf.seed)
    higgsId = set_task(sub_models_conf, 'higgsId', sub_task)
    logger.info('set_task: higgsId')
    from models import MyDataset
    from models import MyMetrics
    set_seed(conf.seed)
    dataset = set_module([MyDataset], conf, 'dataset')
    set_seed(conf.seed)
    dataloader = DataLoader(dataset,
                            batch_size=100,
                            shuffle=True)
    logger.info('set dataloader')
    # #########################################################################
    # pre-train ###############################################################
    # #########################################################################
    logger.info('----- pretrain[0] start -----')
    pretrain_conf = model_confs_tau4vec['pretrain']
    for i, sub_model in enumerate(tau4vec):
        logger.info(f'pretrain: [0][{i}]')
        set_seed(conf.seed)
        optimizer = set_module([optim],
                               pretrain_conf,
                               'optimizer',
                               params=sub_model.parameters())
        loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
        metrics = set_module([MyMetrics], pretrain_conf, 'metrics')
        activation = set_module([nn], pretrain_conf, 'activation')
        input_key = pretrain_conf['data']['input_key']
        target_key = pretrain_conf['data']['target_key']
        patience = pretrain_conf['patience']
        tau4vec[i] = sub_task.pre_train(epochs=pretrain_conf['epochs'],
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
        optimizer = set_module([optim],
                               pretrain_conf,
                               'optimizer',
                               params=sub_model.parameters())
        loss_func = set_module([nn, MyLoss], pretrain_conf, 'loss_func')
        metrics = set_module([MyMetrics], pretrain_conf, 'metrics')
        activation = set_module([nn], pretrain_conf, 'activation')
        input_key = pretrain_conf['data']['input_key']
        target_key = pretrain_conf['data']['target_key']
        patience = pretrain_conf['patience']
        higgsId[i] = sub_task.pre_train(epochs=pretrain_conf['epochs'],
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
    pre_trained_tau4vec = set_task(sub_models_conf, 'tau4vec', sub_task)
    pre_trained_higgsId = set_task(sub_models_conf, 'higgsId', sub_task)
    pre_trained_model = [pre_trained_tau4vec, pre_trained_higgsId]
    task = [tau4vec, higgsId]
    for num_task, sub in enumerate(task):
        for num_model in range(len(sub)):
            pre_trained_model[num_task][num_model].load_state_dict(
                deepcopy(task[num_task][num_model].state_dict())
            )
    # #########################################################################
    # #########################################################################

    logger.info('----- SPOS-NAS start -----')
    sposnas_conf = conf.SPOS_NAS

    def make_output_dict():
        return {
            'X': [],
            'AUC': {
                f'{f}_{s}': [] for f, s in product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                )
            },
            'LOSS_1ST': {
                f: [] for f in FIRST_MODEL_NAME
            },
            'LOSS_2ND': {
                f'{f}_{s}': [] for f, s in product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                )
            },
            'RATIO': {
                f: [] for f in FIRST_MODEL_NAME
            },
            'ONLY_PT_RATIO': {
                f: [] for f in FIRST_MODEL_NAME
            },
        }

    # SPOS-NAS
    loss_func = [set_module([nn, MyLoss], sposnas_conf, 'loss_first'),
                 set_module([nn, MyLoss], sposnas_conf, 'loss_second')]
    loss_weight = [weight, 1. - weight]
    metrics = get_module([MyMetrics], 'Calc_Auc')()

    model = SPOS(task=task, loss_func=loss_func,
                 loss_weight=loss_weight)
    model.to(DEVICE)

    output_dict = make_output_dict()
    output_dict['X'].append(weight)
    logger.info(f'loss_ratio: {weight:.6f} (loss_1*X + loss_2*(1-X)) start')
    set_seed(conf.seed)
    logger.info('load pretrain models...')
    for num_task, sub in enumerate(task):
        for num_model in range(len(sub)):
            task[num_task][num_model].load_state_dict(
                deepcopy(pre_trained_model[num_task][num_model].state_dict())
            )
    logger.info('load pretrain models done')
    logger.info('set model parameters...')

    optimizer = set_module([optim],
                           sposnas_conf,
                           'optimizer',
                           params=model.parameters())
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
    output_dict = evaluate(model, conf, dataloader, metrics, output_dict, is_gp_3dim)
    logger.info('eval model done')

    set_seed(conf.seed)
    logger.info('re-train start')
    selected_model, _ = max(
        {
            k: v[-1] for k, v in output_dict['AUC'].items()
        }.items(), key=lambda x: x[1]
    )
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

    elapsed_time = time.time() - start
    events = conf.dataset.params.max_events * 2
    if prefix:
        output_file = (f'result.SPOS_NAS-{prefix}_' +
                       f's{seed}_w{weight}_e{events}_' +
                       f'n{n_times_model*3}.json')
    else:
        output_file = (f'result.SPOS_NAS-s{seed}_w{weight}_e{events}_' +
                       f'n{n_times_model*3}.json')

    with open(os.path.join('logs', output_file), 'w') as fo:
        json.dump(
            [{
                'agent': 'SPOS-NAS',
                'tasks': {
                    'tau4vec': {
                        'weight': weight,
                        'loss_test': -1,
                        'mse_test': -1,
                        'ratio_2sigma_GP_test': -1,
                        'models': FIRST_MODEL_NAME,
                        'model_selected': selected_model.split('_')[0]
                    },
                    'higgsId': {
                        'weight': 1. - weight,
                        'loss_test': -1,
                        'auc_test': -1,
                        'models': SECOND_MODEL_NAME,
                        'model_selected': selected_model.split('_')[1]
                    }
                },
                'loss_test': -1,
                'nevents': conf.dataset.params.max_events * 2,
                'seed': conf.seed,
                'walltime': elapsed_time
            }],
            fo,
            indent=2
        )

    dummy = make_output_dict()
    dummy = evaluate(model, conf, dataloader, metrics, dummy, is_gp_3dim)

    def result_parser(res, selected_model, seed, time):
        AUC = res['AUC'][selected_model][0]
        LOSS_1ST = res['LOSS_1ST'][selected_model.split('_')[0]][0]
        LOSS_2ND = res['LOSS_2ND'][selected_model][0]
        RATIO = res['RATIO'][selected_model.split('_')[0]][0]
        ONLY_PT_RATIO = res[
            'ONLY_PT_RATIO'
        ][selected_model.split('_')[0]][0]
        target_result = dict(
            seed=seed,
            AUC=AUC,
            LOSS_1ST=LOSS_1ST,
            LOSS_2ND=LOSS_2ND,
            RATIO=RATIO,
            ONLY_PT_RATIO=ONLY_PT_RATIO
        )
        logger.info(f're-train results: {target_result}')
        return {
            'agent': 'SPOS-NAS',
            'tasks': {
                'tau4vec': {
                    'weight': weight,
                    'loss_test': target_result['LOSS_1ST'],
                    'mse_test': target_result['LOSS_1ST'] * 10000,
                    'ratio_2sigma_GP_test': target_result['RATIO'],
                    'models': FIRST_MODEL_NAME,
                    'model_selected': selected_model.split('_')[0]
                },
                'higgsId': {
                    'weight': 1. - weight,
                    'loss_test': target_result['LOSS_2ND'],
                    'auc_test': target_result['AUC'],
                    'models': SECOND_MODEL_NAME,
                    'model_selected': selected_model.split('_')[1]
                }
            },
            'loss_test': (weight * target_result['LOSS_1ST']
                          + (1. - weight) * target_result['LOSS_2ND']),
            'nevents': conf.dataset.params.max_events * 2,
            'seed': seed,
            'walltime': time
        }

    with open(os.path.join('logs', output_file), 'w') as fo:
        json.dump(
            [result_parser(dummy, selected_model, conf.seed, elapsed_time)],
            fo,
            indent=2
        )

    logger.info('all train and eval step are done')


if __name__ == '__main__':
    main()
