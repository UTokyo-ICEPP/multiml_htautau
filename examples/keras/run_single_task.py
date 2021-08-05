import os

save_dir = f'output/{os.path.basename(__file__)[:-3]}'

from run_utils import common_parser

parser = common_parser()
parser.add_argument("--model", dest="model", help="model", type=str, default='MLP')
args = parser.parse_args()

from multiml import logger

logger.set_level(args.log_level)

from setup_tensorflow import setup_tensorflow

setup_tensorflow(args.seed, args.igpu)

from run_utils import add_suffix

save_dir = add_suffix(save_dir, args)
save_dir += f'_{args.model}'

from multiml.saver import Saver

saver = Saver(save_dir, serial_id=args.seed)
saver.add("seed", args.seed)

# Storegate
from my_storegate import get_storegate

storegate = get_storegate(
    data_path=args.data_path,
    max_events=args.max_events,
)

from multiml.task_scheduler import TaskScheduler

task_scheduler = TaskScheduler()

subtask_args = {
    'saver': saver,
    'output_var_names': ('probability', ),
    'true_var_names': 'label',
    'optimizer_args': dict(learning_rate=1e-3),
    'optimizer': 'adam',
    'num_epochs': 100,
    "max_patience": 10,
    'batch_size': 100,
    'phases': None,
    'save_weights': True,
    'run_eagerly': args.run_eagerly,
}
if args.load_weights:
    subtask_args['load_weights'] = True
    subtask_args['save_weights'] = False
    subtask_args['phases'] = ['test']

from my_tasks import reco_tau_4vec

subtask_args['input_var_names'] = reco_tau_4vec

use_logits = True
batch_norm = False

if use_logits:
    from tensorflow.keras.losses import BinaryCrossentropy
    subtask_args['loss'] = BinaryCrossentropy(from_logits=True)
    activation_last = 'linear'
else:
    subtask_args['loss'] = 'binary_crossentropy'
    activation_last = 'sigmoid'

from multiml.hyperparameter import Hyperparameters

subtasks = []
if args.model == 'MLP':
    from multiml.task.keras import MLPTask
    subtask = {}
    subtask['subtask_id'] = 'MLP'
    layers = [
        [128, 128, 128, 1],
        [64, 64, 64, 1],
        [32, 32, 32, 1],
        [16, 16, 16, 1],
        [128, 128, 1],
        [64, 64, 1],
        [32, 32, 1],
        [16, 16, 1],
        [128, 1],
        [64, 1],
        [32, 1],
        [16, 1],
    ]
    subtask['env'] = MLPTask(name='MLP',
                             activation='relu',
                             activation_last=activation_last,
                             batch_norm=batch_norm,
                             **subtask_args)
    subtask['hps'] = Hyperparameters({'layers': layers})

    subtasks.append(subtask)

elif args.model == 'LSTM':
    from multiml_htautau.task.keras import HiggsID_LSTMTask
    subtask = {}
    subtask['subtask_id'] = 'LSTM'
    subtask['env'] = HiggsID_LSTMTask()
    nodes = [
        [128, 128, 128, 1],
        [64, 64, 64, 1],
        [32, 32, 32, 1],
        [16, 16, 16, 1],
        [128, 128, 1],
        [64, 64, 1],
        [32, 32, 1],
        [16, 16, 1],
        [128, 1],
        [64, 1],
        [32, 1],
        [16, 1],
    ]
    subtask['env'].initialize(name='LSTM',
                              input_njets=2,
                              activation_last=activation_last,
                              batch_norm=batch_norm,
                              **subtask_args)
    subtask['hps'] = Hyperparameters({'nodes': nodes})

    subtasks.append(subtask)

else:
    raise ValueError(f'Model should be MLP or LSTM. {args.model} is selected.')

task_scheduler.add_task(task_id='singleTask', subtasks=subtasks)

# Metric
from multiml.agent.metric import AUCMetric

metric = AUCMetric(pred_var_name='probability', true_var_name='label', phase='test')

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

for subtask_id, params in zip(result['subtask_ids'], result['subtask_hps']):
    subtask = task_scheduler.get_subtask(task_id='singleTask', subtask_id=subtask_id)
    params.update(save_weights=False, load_weights=True, phases=['test'])
    subtask.env.set_hps(params)
    agent._execute_subtask(subtask)

metric.storegate = storegate
result_metric = metric.calculate()
logger.info(f'metric = {result_metric}')

from run_utils import postprocessing

postprocessing(saver, storegate, args, do_probability=True, do_tau4vec=False)
