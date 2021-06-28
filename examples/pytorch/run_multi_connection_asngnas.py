import os
import click
import json

import numpy as np
from torch import device
from torch.cuda import is_available

if is_available():
    DEVICE = device("cuda")
else:
    DEVICE = device("cpu")

from multiml import logger




@click.command()
@click.option('--conf', '-c', type=str, default="./config/config.yaml")
@click.option('--seed', '-s', type=int, default=None)
@click.option('--gpu_index', '-gi', type=int, default=None)
@click.option('--data_path', '-dp', type=str, default=None)
@click.option('--event', '-ev', type=int, default=None)
@click.option('--weight', '-w', type=float, default=0.0)
@click.option('--lam', '-l', type=float, default=2.0)
@click.option('--delta', '-del', type=float, default=1.0)
@click.option('--alpha', '-alp', type=float, default=1.0)
@click.option('--load_weights', '-lw', type=bool, default=False)
@click.option('--do_pretraining', '-dp', type=bool, default=False)
@click.option('--epoch', '-ep', type=int, default=100)
@click.option('--jobid', '-ji', type=str, default='default')
@click.option('--loglevel', '-ll', type=click.Choice(['DEBUG', 'INFO', 'WARN', 'ERROR', 'DISABLED', 'MIN_LEVEL']))
def main(conf: str,
         seed: int,
         gpu_index: int,
         data_path: str,
         event: int,
         weight: float,
         lam:float,
         delta:float,
         alpha:float,
         load_weights: bool,
         do_pretraining: bool,
         epoch: int,
         jobid: str,
         loglevel ):
    logger.set_level(loglevel)
    global DEVICE
    from utils import load_config
    from run_utils import get_multi_loss, set_seed
    config = load_config(conf)
    
    verbose = 1
    
    if seed is not None:
        config.seed = seed
    if gpu_index is not None and DEVICE == device('cuda'):
        DEVICE = device(f'cuda:{gpu_index}')
    if data_path is not None:
        config['dataset']['params']['data_path'] = data_path
    if event is not None:
        config['dataset']['params']['max_events'] = int(event)
    else :
        event = config['dataset']['params']['max_events']
    set_seed(config.seed)
    
    if do_pretraining : 
        jobid = 'pretrain_'+jobid
    else : 
        jobid = 'no_train_'+jobid
                
    save_dir = f'output/{os.path.basename(__file__)[:-3]}_{event}evt_weight{weight}_{jobid}'

    use_multi_loss, loss_weights = get_multi_loss(weight)

    from run_utils import preprocessing
    saver, storegate, task_scheduler, metric = preprocessing(
        save_dir=save_dir,
        config=config,
        device=DEVICE,
        tau4vec_tasks=['MLP', 'conv2D', 'SF'],
        higgsId_tasks=['mlp', 'mass', 'lstm'],
    )

    # Time measurements
    from timer import timer
    timer_reg = {}

    load_weights = load_weights
    do_pretraining = do_pretraining
    phases = ['test'] if load_weights else ['train', 'valid', 'test']
    
    
    # Agent
    from multiml.agent.pytorch import PytorchASNGNASAgent
    with timer(timer_reg, "initialize"):
        from my_tasks import mapping_truth_corr
        agent = PytorchASNGNASAgent(
            verbose = verbose,
            num_epochs = epoch,
            batch_size = {'type':'equal_length', 'length':1000, 'test' : 100},
            lam = lam,
            delta_init_factor = delta,
            #alpha = alpha,
            # BaseAgent
            saver=saver,
            storegate=storegate,
            task_scheduler=task_scheduler,
            metric=metric,
            
            # EnsembleAgent
            # ConnectionSimpleAgent
            freeze_model_weights=False,
            do_pretraining = do_pretraining,
            connectiontask_args={
                "num_epochs": epoch,
                "max_patience": 100,
                "batch_size": 128, 
                "load_weights": load_weights,
                "phases": phases,
                "loss_weights": loss_weights,
                "optimizer": "Adam",
                "optimizer_args": dict(lr=1e-3),
                "variable_mapping": mapping_truth_corr,
                "device": DEVICE,
                "verbose":verbose,
                'metrics':['loss', 'subloss', 'auc'],
                
            }
        )

    with timer(timer_reg, "execute"):
        agent.execute()

    with timer(timer_reg, "finalize"):
        agent.finalize()
        
        
    
    results = agent.results_json
    results['walltime'] = timer_reg['execute'][1]
    results['timer_reg'] = timer_reg
    results['seed'] = seed
    results['nevents'] = event*2
        
    def print_dict(key, val) : 
        if type(val) is dict :
            for k, v in val.items():
                print_dict( f'{key} {k}', v)
        else : 
            logger.info(f'{key: <30} : {val}')
        
    
    for key, val in results.items() : 
        print_dict(key, val)
    
    with open(f'{saver.save_dir}/result.run_connection_asngnas_{event}evt_weight{weight}.json', 'w') as fo : 
        json.dump([results], fo, indent=2)
    
    if not load_weights:
        with open(f"{saver.save_dir}/timer.pkl", 'wb') as f:
            import pickle
            pickle.dump(timer_reg, f)
            
    ### post processing 
    variables = []
    from my_tasks import corr_tau_4vec
    variables.extend(corr_tau_4vec)
    variables.extend(['probability'])
    
    for phase in phases : 
        # dump prediction
        storegate.set_data_id("")
        y_pred = np.array( storegate.get_data(phase = phase, var_names = variables ) )
        
        os.makedirs(f'{saver.save_dir}/pred/{phase}', exist_ok = True )
        
        for i, v in enumerate(variables):
            np.save(f'{saver.save_dir}/pred/{phase}/{v}', y_pred[i])
    
    


if __name__ == '__main__':
    main()
