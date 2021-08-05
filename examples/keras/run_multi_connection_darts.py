import os

save_dir = f'output/{os.path.basename(__file__)[:-3]}'

from run_utils import common_parser

parser = common_parser()
parser.add_argument("--no_model_selection", action="store_true", help="Not select one model")
args = parser.parse_args()

from run_utils import add_suffix

save_dir = add_suffix(save_dir, args)
if args.no_model_selection:
    save_dir += '_ensemble'

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
from multiml.agent.keras import KerasDartsAgent
with timer(timer_reg, "initialize"):
    from my_tasks import mapping_truth_corr
    agent = KerasDartsAgent(
        # BaseAgent
        saver=saver,
        storegate=storegate,
        task_scheduler=task_scheduler,
        metric=metric,
        # DartsAgent
        select_one_models=not args.no_model_selection,
        use_original_darts_optimization=True,
        dartstask_args={
            "num_epochs": 100,
            "max_patience": 20,
            "optimizer_alpha": "adam",
            "optimizer_weight": "adam",
            "learning_rate_alpha": 0.001,
            "learning_rate_weight": 0.001,
            "zeta": 0.001,
            "batch_size": 1200,
            "save_tensorboard": False,
            "phases": None,
            "save_weights": True,
            "run_eagerly": args.run_eagerly,
            "loss_weights": loss_weights,
            "variable_mapping": mapping_truth_corr,
        },
        # EnsembleAgent
        ensembletask_args={
            "dropout_rate": args.dropout_rate,
            "individual_loss": True,
            "individual_loss_weights": args.individual_loss_weights,
            "phases": ['test'],
            "save_weights": True,
            "run_eagerly": args.run_eagerly,
        },
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
