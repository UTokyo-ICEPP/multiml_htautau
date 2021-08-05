import os

save_dir = f'output/{os.path.basename(__file__)[:-3]}'

from run_utils import common_parser

parser = common_parser()
args = parser.parse_args()

from run_utils import add_suffix

save_dir = add_suffix(save_dir, args)

from run_utils import get_config_multi_loss

use_multi_loss, loss_weights = get_config_multi_loss(args)

from run_utils import preprocessing

saver, storegate, task_scheduler, metric = preprocessing(
    save_dir=save_dir,
    args=args,
    tau4vec_tasks=['MLP'],
    higgsId_tasks=['lstm'],
)

# Time measurements
from timer import timer

timer_reg = {}

# Agent
from multiml.agent.keras import KerasConnectionRandomSearchAgent
with timer(timer_reg, "initialize"):
    from my_tasks import mapping_truth_corr
    agent = KerasConnectionRandomSearchAgent(
        # BaseAgent
        saver=saver,
        storegate=storegate,
        task_scheduler=task_scheduler,
        metric=metric,
        # ConnectionSimpleAgent
        freeze_model_weights=False,
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

from run_utils import postprocessing

postprocessing(saver, storegate, args, do_probability=True, do_tau4vec=True)
