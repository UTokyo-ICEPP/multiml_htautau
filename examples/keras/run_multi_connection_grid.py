import os

save_dir = f'output/{os.path.basename(__file__)[:-3]}'

from run_utils import common_parser

parser = common_parser()
parser.add_argument("--fix_submodel_weights",
                    action="store_true",
                    help="Fix submodel weights after pre-training")
args = parser.parse_args()

from run_utils import add_suffix

save_dir = add_suffix(save_dir, args)

from run_utils import get_config_multi_loss

use_multi_loss, loss_weights = get_config_multi_loss(args)

from run_utils import preprocessing

saver, storegate, task_scheduler, metric = preprocessing(
    save_dir=save_dir,
    args=args,
    tau4vec_tasks=['MLP', 'conv2D', 'SF', 'zero', 'noise'],
    higgsId_tasks=['mlp', 'lstm', 'mass', 'zero', 'noise'],
)

# Time measurements
from timer import timer

timer_reg = {}

# Agent
from multiml.agent.keras import KerasConnectionGridSearchAgent
with timer(timer_reg, "initialize"):
    from my_tasks import mapping_truth_corr
    agent = KerasConnectionGridSearchAgent(
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
        freeze_model_weights=args.fix_submodel_weights,
        do_pretraining=not args.nopretraining,
        connectiontask_name='connection',
        connectiontask_args={
            "num_epochs": 100,
            "max_patience": 10,
            "batch_size": 100,
            "save_weights": not args.load_weights,
            "load_weights": args.load_weights,
            "phases": ["test"] if args.load_weights else None,
            "loss_weights": loss_weights,
            "optimizer": "adam",
            "optimizer_args": dict(learning_rate=1e-3),
            "variable_mapping": mapping_truth_corr,
            "run_eagerly": args.run_eagerly,
        },
    )

with timer(timer_reg, "execute"):
    agent.execute()

with timer(timer_reg, "finalize"):
    agent.finalize()

if not args.load_weights:
    with open(f"{saver.save_dir}/timer.pkl", 'wb') as f:
        import pickle
        pickle.dump(timer_reg, f)

# Evaluate the best parameters
result, config = agent.get_best_result()

subtasks = []
job_id = None
for task_id, subtask_id, params in zip(result['task_ids'], result['subtask_ids'],
                                       result['subtask_hps']):
    subtask = task_scheduler.get_subtask(task_id=task_id, subtask_id=subtask_id)
    params.update(save_weights=False, load_weights=True, phases=['test'])
    subtask.env.set_hps(params)
    agent._execute_subtask(subtask, is_pretraining=True)
    subtasks.append(subtask.env)
    job_id = params['job_id']

subtask = agent._build_connected_models(subtasks,
                                        job_id=config["job_id"],
                                        use_task_scheduler=False)
subtask.env.set_hps({"save_weights": False, "load_weights": True, "phases": ['test']})
agent._execute_subtask(subtask, is_pretraining=False)

metric.storegate = storegate
result_metric = metric.calculate()
from multiml import logger

logger.info(f'metric = {result_metric}')

from run_utils import postprocessing

postprocessing(saver, storegate, args, do_probability=True, do_tau4vec=True)
