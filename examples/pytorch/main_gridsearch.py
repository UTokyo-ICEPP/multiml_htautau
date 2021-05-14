#!/usr/bin/env python3

from itertools import product
from copy import deepcopy
from utils import (
    get_logger,
    get_module,
    load_config,
    log,
    set_module,
    set_seed,
    set_task,
)

from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import click
import numpy as np
import torch

from models import MyLoss
from utils import (
    add_device,
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


def evaluate(model, conf, dataloader, metrics, result,
             choice=None):
    with torch.no_grad():
        logger.info('start eval mode')
        model.eval()
        dataloader.dataset.test()
        test_dataset = dataloader.dataset
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=100,
                                     shuffle=False)
        A = range(len(conf.sub_task_params.tau4vec.tasks))
        B = range(len(conf.sub_task_params.higgsId.tasks))
        num_name_conb = {
            num: f'{f}_{s}' for num, (f, s) in zip(
                product(A, B),
                product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                )
            )
        }
        outputs_data = []
        targets_data = []
        temp_outputs_data = []
        temp_targets_data = []
        for data in test_dataloader:
            inputs = add_device(data['inputs'], DEVICE)
            targets = add_device(data['targets'], DEVICE)
            outputs, now_choice = model(inputs, choice)
            outputs_data.extend(tensor_to_array(outputs[1]))
            targets_data.extend(tensor_to_array(targets[1]))
            temp_outputs_data.extend(tensor_to_array(outputs[0]))
            temp_targets_data.extend(tensor_to_array(targets[0]))
        targets_data = np.array(targets_data)
        outputs_data = np.array(outputs_data)
        auc_score = metrics(targets_data, outputs_data)
        result['AUC'][num_name_conb[choice]].append(auc_score)
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
        ratio = np.sum(
            np.where(
                ((lower < upper)
                 &
                 (temp_outputs_data < upper)
                 &
                 (lower < temp_outputs_data))
                |
                ((upper < lower)
                 &
                 (upper < temp_outputs_data)
                 &
                 (lower < temp_outputs_data))
                |
                ((upper < lower)
                 &
                 (temp_outputs_data < upper)
                 &
                 (temp_outputs_data < lower)),
                True, False
            ).all(axis=1)
        )/(len(temp_outputs_data))
        only_pt_ratio = np.sum(
            np.where(
                ((lower[:, 0] < upper[:, 0])
                 &
                 (temp_outputs_data[:, 0] < upper[:, 0])
                 &
                 (lower[:, 0] < temp_outputs_data[:, 0]))
                |
                ((upper[:, 0] < lower[:, 0])
                 &
                 (upper[:, 0] < temp_outputs_data[:, 0])
                 &
                 (lower[:, 0] < temp_outputs_data[:, 0]))
                |
                ((upper[:, 0] < lower[:, 0])
                 &
                 (temp_outputs_data[:, 0] < upper[:, 0])
                 &
                 (temp_outputs_data[:, 0] < lower[:, 0])),
                True, False
            )
        )/(len(temp_outputs_data))

        result['RATIO'][num_name_conb[choice]].append(ratio)
        result['ONLY_PT_RATIO'][num_name_conb[choice]].append(only_pt_ratio)
        result['LOSS_1ST'][num_name_conb[choice]].append(loss_1st.item())
        result['LOSS_2ND'][num_name_conb[choice]].append(loss_2nd.item())
        logger.info(f'[Choice:{now_choice} / auc:{auc_score:.6f}] / ' +
                    f'first_loss: {loss_1st:.6f} / ' +
                    f'ratio: {ratio:.6f} / ' +
                    f'only_pt_ratio: {only_pt_ratio:.6f} / ')

    logger.info(result)
    return result


@click.command()
@click.option('--conf', '-c', type=str, default="./config/config.yaml")
@click.option('--seed', '-s', type=int, default=None)
@click.option('--gpu_index', '-gi', type=int, default=None)
@click.option('--data_path', '-dp', type=str, default=None)
@log(logger)
def main(conf: str, seed: int, gpu_index: int, data_path: str):
    global DEVICE
    conf = load_config(conf)
    if seed is not None:
        conf.seed = seed
    if gpu_index is not None and DEVICE == torch.device('cuda'):
        DEVICE = torch.device(f'cuda:{gpu_index}')
    if data_path is not None:
        conf['dataset']['params']['data_path'] = data_path
    logger.info(DEVICE)
    logger.info(conf)

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
    dataloader = DataLoader(dataset,
                            batch_size=100,
                            shuffle=True)
    logger.info('set dataloader')
    # #########################################################################
    # pre-train ###############################################################
    # #########################################################################
    logger.info('----- pretrain[0] start -----')
    pretrain_conf = conf.sub_task_params.tau4vec.pretrain
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
        optimizer = set_module([optim],
                               pretrain_conf,
                               'optimizer',
                               params=sub_model.parameters())
        loss_func = set_module([nn], pretrain_conf, 'loss_func')
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
                f'{f}_{s}': [] for f, s in product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                    )
            },
            'LOSS_2ND': {
                f'{f}_{s}': [] for f, s in product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                )
            },
            'RATIO': {
                f'{f}_{s}': [] for f, s in product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                    )
            },
            'ONLY_PT_RATIO': {
                f'{f}_{s}': [] for f, s in product(
                    FIRST_MODEL_NAME, SECOND_MODEL_NAME
                    )
            },
        }

    # evaluate only pre-train model
    loss_func = [set_module([nn, MyLoss], sposnas_conf, 'loss_first'),
                 set_module([nn, MyLoss], sposnas_conf, 'loss_second')]
    loss_weight = [0.5, 0.5]
    metrics = get_module([MyMetrics], 'Calc_Auc')()
    from models.SPOS_NAS import SPOS
    model = SPOS(task=task, loss_func=loss_func,
                 loss_weight=loss_weight)
    model.to(DEVICE)
    logger.info('evaluate only pre-train model')
    dummy = make_output_dict()
    for now_choice in product(range(3), range(3)):
        pre_train_result = evaluate(model, conf, dataloader, metrics, dummy,
                                    now_choice)

    output_dict = make_output_dict()
    X_list = [0.0, 0.1, 0.5]
    for X in (np.array(X_list)).round(10):
        output_dict['X'].append(X)
        logger.info(f'loss_ratio: {X:.6f} (loss_1*X + loss_2*(1-X)) start')
        set_seed(conf.seed)

        def initialize_pretrain_weight():
            logger.info('load pretrain models...')
            for num_task, sub in enumerate(task):
                for num_model in range(len(sub)):
                    task[num_task][num_model].load_state_dict(
                        deepcopy(pre_trained_model[num_task][num_model].state_dict())
                    )
            logger.info('load pretrain models done')

        logger.info('set model parameters...')
        loss_func = [set_module([nn, MyLoss], sposnas_conf, 'loss_first'),
                     set_module([nn, MyLoss], sposnas_conf, 'loss_second')]
        loss_weight = [X, 1.-X]
        metrics = get_module([MyMetrics], 'Calc_Auc')()

        for now_choice in product(range(3), range(3)):
            initialize_pretrain_weight()
            model = SPOS(task=task, loss_func=loss_func,
                         loss_weight=loss_weight)
            model.to(DEVICE)
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
                      patience=sposnas_conf.patience,
                      choice=now_choice)
            logger.info('fit model done')
            logger.info('eval model...')
            output_dict = evaluate(model,
                                   conf,
                                   dataloader,
                                   metrics,
                                   output_dict,
                                   now_choice)
            logger.info('eval model done')

    logger.info(f'seed: {conf.seed}/ pretrain result: {pre_train_result}')
    logger.info(f'seed: {conf.seed}/ final result: {output_dict}')

    logger.info('all train and eval step are done')

    logger.info('plot results...')
    logger.info('plot auc...')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    import pandas as pd
    df = pd.DataFrame(output_dict['AUC'], index=output_dict['X'])
    df = df.rename(columns={
        f'{f}_{s}': f'{f}:{s}' for f, s in product(
            FIRST_MODEL_NAME, SECOND_MODEL_NAME
        )
    })
    df.plot()
    plt.xlabel('X')
    plt.ylabel('AUC')
    plt.savefig(f'grid_auc_{conf.seed}.png')
    plt.close()

    logger.info('plot loss_2ND...')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    df = pd.DataFrame(output_dict['LOSS_2ND'], index=output_dict['X'])
    df = df.rename(columns={
        f'{f}_{s}': f'{f}:{s}' for f, s in product(
            FIRST_MODEL_NAME, SECOND_MODEL_NAME
        )
    })
    df.plot()
    plt.xlabel('X')
    plt.ylabel('LOSS_2ND')
    plt.savefig(f'grid_loss_2nd_{conf.seed}.png')
    plt.close()

    logger.info('plot loss_1ST...')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    df = pd.DataFrame(output_dict['LOSS_1ST'], index=output_dict['X'])
    df = df.rename(columns={
        f'{f}_{s}': f'{f}:{s}' for f, s in product(
            FIRST_MODEL_NAME, SECOND_MODEL_NAME
        )
    })
    df.plot()
    plt.xlabel('X')
    plt.ylabel('LOSS_1ST')
    plt.savefig(f'grid_loss_1st_{conf.seed}.png')
    plt.close()

    logger.info('plot ratios...')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    df = pd.DataFrame(output_dict['ONLY_PT_RATIO'], index=output_dict['X'])
    df = df.rename(columns={
        f'{f}_{s}': f'{f}:{s}' for f, s in product(
            FIRST_MODEL_NAME, SECOND_MODEL_NAME
        )
    })
    df.plot()
    plt.ylabel('ratio')
    plt.savefig(f'grid_only_pt_ratio_{conf.seed}.png')
    plt.close()
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    df = pd.DataFrame(output_dict['RATIO'], index=output_dict['X'])
    df = df.rename(columns={
        f'{f}_{s}': f'{f}:{s}' for f, s in product(
            FIRST_MODEL_NAME, SECOND_MODEL_NAME
        )
    })
    df.plot()
    plt.ylabel('ratio')
    plt.savefig(f'grid_ratio_{conf.seed}.png')
    plt.close()
    logger.info('plot results done')


if __name__ == '__main__':
    main()
