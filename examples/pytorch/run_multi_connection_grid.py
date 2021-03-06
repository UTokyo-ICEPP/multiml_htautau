import os
import click

from torch import device
from torch.cuda import is_available

save_dir = f'output/{os.path.basename(__file__)[:-3]}'
if is_available():
    DEVICE = device("cuda")
else:
    DEVICE = device("cpu")


@click.command()
@click.option('--conf', '-c', type=str, default="./config/config.yaml")
@click.option('--seed', '-s', type=int, default=None)
@click.option('--gpu_index', '-gi', type=int, default=None)
@click.option('--data_path', '-dp', type=str, default=None)
@click.option('--event', '-e', type=int, default=None)
@click.option('--weight', '-w', type=float, default=0.5)
@click.option('--load_weights', '-lw', type=bool, default=False)
@click.option('--nopretraining', '-np', type=bool, default=False)
def main(conf: str,
         seed: int,
         gpu_index: int,
         data_path: str,
         event: int,
         weight: float,
         load_weights: bool,
         nopretraining: bool):
    global DEVICE
    from utils import load_config
    from run_utils import get_multi_loss, set_seed
    config = load_config(conf)
    if seed is not None:
        config.seed = seed
    if gpu_index is not None and DEVICE == device('cuda'):
        DEVICE = device(f'cuda:{gpu_index}')
    if data_path is not None:
        config['dataset']['params']['data_path'] = data_path
    if event is not None:
        config['dataset']['params']['max_events'] = int(event)
    set_seed(config.seed)

    use_multi_loss, loss_weights = get_multi_loss(weight)

    from run_utils import preprocessing
    saver, storegate, task_scheduler, metric = preprocessing(
        save_dir=save_dir,
        config=config,
        device=DEVICE,
        tau4vec_tasks=['conv2D', 'MLP', 'SF'],
        higgsId_tasks=['lstm', 'mlp', 'mass'],
    )

    # Time measurements
    from timer import timer
    timer_reg = {}

    fix_submodel_weights = False
    load_weights = load_weights
    nopretraining = nopretraining
    phases = ['test'] if load_weights else ['train', 'valid', 'test']
    # Agent
    from multiml.agent.pytorch import PytorchConnectionGridSearchAgent
    with timer(timer_reg, "initialize"):
        from my_tasks import mapping_truth_corr
        agent = PytorchConnectionGridSearchAgent(
            # BaseAgent
            saver=saver,
            storegate=storegate,
            task_scheduler=task_scheduler,
            metric=metric,
            metric_type='max',
            dump_all_results=True,
            # ConnectionGridAgent
            reuse_pretraining=True,
            # ConnectionSimpleAgent
            freeze_model_weights=fix_submodel_weights,
            do_pretraining=not nopretraining,
            connectiontask_args={
                "num_epochs": 100,
                "max_patience": 10,
                "batch_size": 100,
                "save_weights": not load_weights,
                "load_weights": load_weights,
                "phases": phases,
                "loss_weights": loss_weights,
                "optimizer": "Adam",
                "optimizer_args": dict(lr=1e-3),
                "variable_mapping": mapping_truth_corr,
                "device": DEVICE,
            }
        )

    with timer(timer_reg, "execute"):
        agent.execute()

    with timer(timer_reg, "finalize"):
        agent.finalize()

    if not load_weights:
        with open(f"{saver.save_dir}/timer.pkl", 'wb') as f:
            import pickle
            pickle.dump(timer_reg, f)

    # Evaluate the best parameters
    result, config = agent.get_best_result()

    subtasks = []
    job_id = None
    for task_id, subtask_id, params in zip(result['task_ids'],
                                           result['subtask_ids'],
                                           result['subtask_hps']):
        subtask = task_scheduler.get_subtask(task_id=task_id,
                                             subtask_id=subtask_id)
        params.update(load_weights=True, phases=['test'])
        subtask.env.set_hps(params)
        agent._execute_subtask(subtask, is_pretraining=True)
        subtasks.append(subtask.env)
        job_id = params['job_id']

    subtask = agent._build_connected_models(subtasks,
                                            job_id=config["job_id"],
                                            use_task_scheduler=False)
    subtask.env.task_id = 'connection'
    subtask.env.subtask_id = subtask.env.name
    subtask.env.set_hps({"load_weights": True, "phases": ['test']})
    agent._execute_subtask(subtask, is_pretraining=False)

    metric.storegate = storegate
    result_metric = metric.calculate()
    from multiml import logger
    logger.info(f'metric = {result_metric}')


if __name__ == '__main__':
    main()
