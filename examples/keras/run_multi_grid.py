import os

save_dir = f'output/{os.path.basename(__file__)[:-3]}'

from run_utils import common_parser

parser = common_parser()
args = parser.parse_args()

from run_utils import add_suffix

save_dir = add_suffix(save_dir, args)

from run_utils import preprocessing

saver, storegate, task_scheduler, metric = preprocessing(
    save_dir=save_dir,
    args=args,
    tau4vec_tasks=['MLP', 'conv2D', 'SF', 'zero', 'noise'],
    higgsId_tasks=['mlp', 'lstm', 'mass', 'zero', 'noise'],
    truth_intermediate_inputs=False,
)

# Time measurements
from timer import timer

timer_reg = {}

# Agent
from multiml.agent.basic import GridSearchAgent
with timer(timer_reg, "initialize"):
    agent = GridSearchAgent(
        # BaseAgent
        saver=saver,
        storegate=storegate,
        task_scheduler=task_scheduler,
        metric=metric,
        metric_type='max',
        dump_all_results=True,
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
result = agent.result

task_scheduler.show_info()
for task_id, subtask_id, params in zip(result['task_ids'], result['subtask_ids'],
                                       result['subtask_hps']):
    subtask = task_scheduler.get_subtask(task_id=task_id, subtask_id=subtask_id)
    params.update(save_weights=False, load_weights=True, phases=['test'])
    subtask.env.set_hps(params)
    agent._execute_subtask(subtask)

metric.storegate = storegate
result_metric = metric.calculate()
from multiml import logger

logger.info(f'metric = {result_metric}')

from run_utils import postprocessing

postprocessing(saver, storegate, args, do_probability=True, do_tau4vec=True)
