def preprocessing(save_dir,
                  config,
                  device='cpu',
                  tau4vec_tasks=['MLP', 'conv2D', 'SF'],
                  higgsId_tasks=['mlp', 'lstm', 'mass'],
                  load_weights=False,
                  truth_intermediate_inputs=True):

    from multiml import logger
    logger.set_level(config.log_level)

    from multiml.saver import Saver
    saver = Saver(save_dir, serial_id=config.seed)
    saver.add("seed", config.seed)

    # Storegate
    from my_storegate import get_storegate
    
    storegate = get_storegate(
        data_path=config.dataset.params.data_path,
        max_events=config.dataset.params.max_events
    )

    # Task scheduler
    from multiml.task_scheduler import TaskScheduler

    from my_tasks import get_higgsId_subtasks, get_tau4vec_subtasks
    task_scheduler = TaskScheduler()

    if len(tau4vec_tasks) > 0 and len(higgsId_tasks) > 0:
        subtask1 = get_higgsId_subtasks(config.sub_task_params.higgsId,
                                        saver,
                                        device=device,
                                        subtask_names=higgsId_tasks,
                                        truth_input=truth_intermediate_inputs,
                                        load_weights=load_weights,
                                        use_logits = True)
        task_scheduler.add_task(task_id='higgsId',
                                parents=['tau4vec'],
                                children=[],
                                subtasks=subtask1)

        subtask2 = get_tau4vec_subtasks(config.sub_task_params.tau4vec,
                                        saver,
                                        subtask_names=tau4vec_tasks,
                                        device=device,
                                        load_weights=load_weights)
                                        
        task_scheduler.add_task(task_id='tau4vec',
                                parents=[],
                                children=['higgsId'],
                                subtasks=subtask2)

    elif len(higgsId_tasks) > 0:
        subtask = get_higgsId_subtasks(config.sub_task_params.higgsId,
                                       saver,
                                       device=device,
                                       subtask_names=higgsId_tasks,
                                       load_weights=load_weights,
                                       use_logits = True)
        task_scheduler.add_task(task_id='higgsId', subtasks=subtask)

    elif len(tau4vec_tasks) > 0:
        subtask = get_tau4vec_subtasks(config.sub_task_params.tau4vec,
                                       saver,
                                       subtask_names=tau4vec_tasks,
                                       device=device,
                                       load_weights=load_weights)
        task_scheduler.add_task(task_id='tau4vec', subtasks=subtask)

    else:
        raise ValueError("Strange task combination...")

    # Metric
    if len(tau4vec_tasks) > 0 and len(higgsId_tasks) == 0:
        from multiml_htautau.task.metrics import CustomMSEMetric
        from my_tasks import corr_tau_4vec, truth_tau_4vec
        metric = CustomMSEMetric(
            pred_var_name=corr_tau_4vec,
            true_var_name=truth_tau_4vec,
            phase='test'
        )
    else:
        from multiml.agent.metric import AUCMetric
        metric = AUCMetric(pred_var_name='probability',
                           true_var_name='label',
                           phase='test')

    return saver, storegate, task_scheduler, metric


def get_multi_loss(X: float = None):
    if X is None:
        use_multi_loss = False
        loss_weights = None
    else:
        use_multi_loss = True
        loss_weights = {'higgsId': 1.0 - X, "tau4vec": X}

    return use_multi_loss, loss_weights


def set_seed(seed=1):
    import os
    import random

    import numpy as np
    import tensorflow as tf
    from tensorflow.random import set_seed
    import torch
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
