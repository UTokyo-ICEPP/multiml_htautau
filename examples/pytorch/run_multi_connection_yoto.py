import os
import argparse
import json

import numpy as np
from torch import device
from torch.cuda import is_available
import torch

if is_available():
    DEVICE = device("cuda")
else:
    DEVICE = device("cpu")

from multiml import logger


def main(opts):
    logger.set_level(opts.loglevel)
    global DEVICE
    from utils import load_config
    from run_utils import get_multi_loss, set_seed
    print(opts.config)
    config = load_config(opts.config)
    
    verbose = 1
    
    if opts.seed is not None:
        config['seed'] = opts.seed
        
    if opts.gpu_index is not None and DEVICE == device('cuda'):
        DEVICE = device(f'cuda:{opts.gpu_index}')
        
    if opts.data_path is not None:
        config['dataset']['params']['data_path'] = opts.data_path
        
    if opts.event is not None:
        config['dataset']['params']['max_events'] = int(opts.event)
        
    
    
    set_seed(config.seed)
    
    if opts.do_pretrain : 
        jobid = 'pretrain_' + opts.jobid
    else : 
        jobid = 'no_train_' + opts.jobid
                
    save_dir = f'output/{os.path.basename(__file__)[:-3]}_{opts.event}evt_{jobid}'

    use_multi_loss = True
    

    from run_utils import preprocessing
    saver, storegate, task_scheduler, metric = preprocessing(
        save_dir=save_dir,
        config=config,
        device=DEVICE,
        tau4vec_tasks = config.Yoto.tau4vec_tasks,
        higgsId_tasks = config.Yoto.higgsId_tasks,
        is_yoto = True, 
        # truth_intermediate_inputs = False, # this is ok, this is for only pre-training, once we create connection_task, this will be replaced by output of task1
    )

    # Time measurements
    from timer import timer
    timer_reg = {}

    phases = ['test'] if opts.load_weights else ['train', 'valid']
    
    def lambda_to_weight(lam):
        return lam, 1.0 - lam
    
    from multiml.agent.pytorch import PytorchYotoConnectionAgent
    with timer(timer_reg, "initialize"):
        from my_tasks import mapping_truth_corr
        config['Yoto']['connectiontask_args']['phases'] = phases
        config['Yoto']['connectiontask_args']['variable_mapping'] = mapping_truth_corr
        config['Yoto']['connectiontask_args']['device'] = DEVICE
        
        agent = PytorchYotoConnectionAgent(
            loss_merge_f = torch.mean, 
            eval_lambdas = config.Yoto.eval_lambdas,
            lambda_to_weight = lambda_to_weight, 
            yoto_model_args = config.Yoto.model_args, 
            verbose = verbose,
            
            num_epochs = config.Yoto.epochs,
            max_patience = config.Yoto.patience,
            batch_size = config.Yoto.batch_size,
            optimizer = config.Yoto.optimizer.name, 
            optimizer_args = config.Yoto.optimizer.params, 
            scheduler = config.Yoto.scheduler.name,
            scheduler_args = config.Yoto.scheduler.params,
            amp = config.amp, 
            num_workers = config.num_workers, 
            
            # BaseAgent
            saver=saver,
            storegate=storegate,
            task_scheduler=task_scheduler,
            metric=metric,
            
            # EnsembleAgent
            # ConnectionSimpleAgent
            freeze_model_weights = False,
            do_pretraining = opts.do_pretrain,
            connectiontask_args= config.Yoto.connectiontask_args,
        )

    with timer(timer_reg, "execute"):
        agent.execute()

    with timer(timer_reg, "finalize"):
        agent.finalize()
        
        
    
    results = agent.results_json
    results['walltime'] = timer_reg['execute'][1]
    results['timer_reg'] = timer_reg
    results['seed'] = opts.seed
    results['nevents'] = opts.event*2
        
    def print_dict(key, val) : 
        if type(val) is dict :
            for k, v in val.items():
                print_dict( f'{key} {k}', v)
        else : 
            logger.info(f'{key: <50} : {val}')
    
    for key, val in results.items() : 
        print_dict(key, val)
    
    with open(f'{saver.save_dir}/result.run_connection_yoto_{opts.event}evt.json', 'w') as fo : 
        json.dump([results], fo, indent=2)
    
    if not opts.load_weights:
        with open(f"{saver.save_dir}/timer.pkl", 'wb') as f:
            import pickle
            pickle.dump(timer_reg, f)
            
    ### post processing 
    variables = []
    from my_tasks import corr_tau_4vec
    variables.extend(corr_tau_4vec)
    variables.extend(['probability'])
    
    # for phase in phases : 
    #     # dump prediction
    #     storegate.set_data_id("")
    #     y_pred = np.array( storegate.get_data(phase = phase, var_names = variables ) )
        
    #     os.makedirs(f'{saver.save_dir}/pred/{phase}', exist_ok = True )
        
    #     for i, v in enumerate(variables):
    #         np.save(f'{saver.save_dir}/pred/{phase}/{v}', y_pred[i])
    
    

if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = '' )
    parser.add_argument('-c', '--config',        action = 'store', dest = 'config',       required = True,  type = str,   default = None, help = 'text path for config file(yaml)')
    parser.add_argument('-s', '--seed',          action = 'store', dest = 'seed',         required = False, type = int,   default = None, help = 'seed integer')
    parser.add_argument('-g', '--gpu_index',     action = 'store', dest = 'gpu_index',    required = False, type = int,   default = None, help = 'gpu index')
    parser.add_argument('-ev', '--event',        action = 'store', dest = 'event',        required = False, type = int,   default = None, help = 'number of event ')
    parser.add_argument('-ep', '--epochs',       action = 'store', dest = 'epochs',       required = False, type = int,   default = None, help = 'number of epochs ')    
    parser.add_argument("-j", '--jobid',         action = 'store', dest = 'jobid',        required = True,  type = str,   default = 'default', help = 'job id ')
    parser.add_argument('-dp', '--data_path',    action = 'store', dest = 'data_path',    required = False, type = str,   default = None, help = 'data path')
    parser.add_argument('-p', '--do_pretrain',   action = 'store', dest = 'do_pretrain',  required = False, type = strtobool,  help = 'do pretraining')
    parser.add_argument('-lw', '--load_weights', action = 'store', dest = 'load_weights', required = False, type = bool,  help = 'load weight')
    parser.add_argument('-ll', '--loglevel',     action = 'store', dest = 'loglevel',     required = False, choices = ('DEBUG', 'INFO', 'WARN', 'ERROR', 'DISABLED'), default = 'INFO', help = 'msg level to use. Valid choices are [""], default is "INFO"')
    
    # parser.add_argument('-p', '--property',    action = 'store', dest = 'properties',  required = False, nargs = '*' )
    opts = parser.parse_args()
    main( opts )
