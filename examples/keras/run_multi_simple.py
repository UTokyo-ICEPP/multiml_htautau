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
    tau4vec_tasks=['SF'],
    higgsId_tasks=['mlp'],
)

# Time measurements
from timer import timer

timer_reg = {}

# Agent
from multiml.agent.basic import RandomSearchAgent
with timer(timer_reg, "initialize"):
    agent = RandomSearchAgent(
        # BaseAgent
        saver=saver,
        storegate=storegate,
        task_scheduler=task_scheduler,
        metric=metric,
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
