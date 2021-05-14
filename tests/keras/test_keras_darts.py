from multiml.storegate import StoreGate
from multiml import logger

import os
import numpy as np

run_eagerly = True


def get_storegate(max_events=50000):
    from my_tasks import reco_tau_4vec, truth_tau_4vec
    fourvec_var_list = tuple(reco_tau_4vec + truth_tau_4vec)
    # Toy data
    storegate = StoreGate(backend='numpy', data_id='')
    data0 = np.random.normal(size=(max_events, len(fourvec_var_list)))
    data1 = np.random.uniform(size=(max_events, 2, 16, 16, 3))
    label = np.random.binomial(n=1, p=0.5, size=(max_events, ))
    phase = (0.6, 0.2, 0.2)
    storegate.add_data(var_names=fourvec_var_list, data=data0, phase=phase)
    storegate.add_data(
        var_names=('1stRecoJetEnergyMap', '2ndRecoJetEnergyMap'),
        data=data1,
        phase=phase)
    storegate.add_data(var_names='label', data=label, phase=phase)
    storegate.compile()
    storegate.show_info()

    return storegate


def preprocessing(tau4vec_tasks=['MLP', 'conv2D', 'SF', 'zero', 'noise'],
                  higgsId_tasks=['mlp', 'lstm', 'mass', 'zero', 'noise'],
                  truth_intermediate_inputs=True):

    logger.set_level(logger.DEBUG)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    load_weights = False

    from multiml.saver import Saver
    saver = Saver()

    # Storegate
    storegate = get_storegate(max_events=100)

    # Task scheduler
    from multiml.task_scheduler import TaskScheduler
    from my_tasks import get_higgsId_subtasks, get_tau4vec_subtasks
    task_scheduler = TaskScheduler()

    subtask1 = get_higgsId_subtasks(saver,
                                    subtask_names=higgsId_tasks,
                                    truth_input=truth_intermediate_inputs,
                                    load_weights=load_weights,
                                    run_eagerly=run_eagerly)
    task_scheduler.add_task(task_id='higgsId',
                            parents=['tau4vec'],
                            children=[],
                            subtasks=subtask1)

    subtask2 = get_tau4vec_subtasks(saver,
                                    subtask_names=tau4vec_tasks,
                                    load_weights=load_weights,
                                    run_eagerly=run_eagerly)
    task_scheduler.add_task(task_id='tau4vec',
                            parents=[],
                            children=['higgsId'],
                            subtasks=subtask2)

    # Metric
    from multiml.agent.metric import AUCMetric
    metric = AUCMetric(pred_var_name='probability',
                       true_var_name='label',
                       phase='test')

    return saver, storegate, task_scheduler, metric


def test_keras_darts():
    use_multi_loss = True
    loss_weights = {'higgsId': 0.5, "tau4vec": 0.5}

    # from run_utils import preprocessing
    saver, storegate, task_scheduler, metric = preprocessing(
        tau4vec_tasks=['MLP', 'conv2D', 'SF', 'zero', 'noise'],
        higgsId_tasks=['mlp', 'lstm', 'mass', 'zero', 'noise'],
    )

    # Agent
    from multiml.agent.keras import KerasDartsAgent
    from my_tasks import mapping_truth_corr
    agent = KerasDartsAgent(
        select_one_models=True,
        use_original_darts_optimization=True,
        # DartsAgent
        dartstask_args={
            "num_epochs": 2,
            "max_patience": 1,
            "optimizer_alpha": "adam",
            "optimizer_weight": "adam",
            "learning_rate_alpha": 0.001,
            "learning_rate_weight": 0.001,
            "zeta": 0.001,
            "batch_size": 1200,
            "save_tensorboard": False,
            "phases": None,
            "save_weights": True,
            "run_eagerly": True,
            "use_multi_loss": use_multi_loss,
            "loss_weights": loss_weights,
            "variable_mapping": mapping_truth_corr,
        },
        # EnsembleAgent
        ensembletask_args={
            "dropout_rate": None,
            "individual_loss": True,
            "individual_loss_weights": 1.0,
            "phases": ['test'],
            "save_weights": True,
            "run_eagerly": True,
        },
        # ConnectionSimpleAgent
        freeze_model_weights=False,
        do_pretraining=True,
        connectiontask_name='connection',
        connectiontask_args={
            "num_epochs": 2,
            "max_patience": 1,
            "batch_size": 100,
            "save_weights": True,
            "phases": None,
            "use_multi_loss": use_multi_loss,
            "loss_weights": loss_weights,
            "optimizer": "adam",
            "optimizer_args": dict(learning_rate=1e-3),
            "variable_mapping": mapping_truth_corr,
            "run_eagerly": run_eagerly,
        },
        # BaseAgent
        saver=saver,
        storegate=storegate,
        task_scheduler=task_scheduler,
        metric=metric,
    )

    agent.execute()
    agent.finalize()


if __name__ == '__main__':
    test_keras_darts()
