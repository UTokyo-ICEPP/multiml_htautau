import os
import argparse
import json

import numpy as np
from torch import device
from torch.cuda import is_available

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
        
    if opts.clip_value is not None : 
        config['ASNG']['clip'] = opts.clip_value
        
    if opts.alpha is not None : 
        config['ASNG']['alpha'] = opts.alpha
    
        
    if opts.lam is not None : 
        config['ASNG']['lam'] = opts.lam
        
    if opts.delta is not None : 
        config['ASNG']['delta'] = opts.delta
        
    if opts.epochs is not None : 
        config['ASNG']['epochs'] = opts.epochs
        
    
    set_seed(config.seed)
    
    if opts.do_pretrain : 
        jobid = 'pretrain_' + opts.jobid
    else : 
        jobid = 'no_train_' + opts.jobid
                
    save_dir = f'output/{os.path.basename(__file__)[:-3]}_{opts.event}evt_weight{opts.weight}_{jobid}'

    use_multi_loss, loss_weights = get_multi_loss(opts.weight)

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

    phases = ['test'] if opts.load_weights else ['train', 'valid', 'test']
    
    # Agent
    logger.info(f'lambda / alpha / delta is {config.ASNG.lam} / {config.ASNG.alpha} / {config.ASNG.delta}')
    
    
    from multiml.agent.pytorch import PytorchASNGNASAgent
    with timer(timer_reg, "initialize"):
        from my_tasks import mapping_truth_corr
        config['ASNG']['connectiontask_args']['phases'] = phases
        config['ASNG']['connectiontask_args']['variable_mapping'] = mapping_truth_corr
        config['ASNG']['connectiontask_args']['device'] = DEVICE
        config['ASNG']['connectiontask_args']['loss_weights'] = loss_weights
        

        agent = PytorchASNGNASAgent(
            verbose = verbose,
            num_epochs = config.ASNG.epochs,
            max_patience = config.ASNG.patience,
            batch_size = config.ASNG.batch_size,
            asng_args = config.ASNG.asng_args, 
            optimizer = config.ASNG.optimizer.name, 
            optimizer_args = config.ASNG.optimizer.params, 
            scheduler = config.ASNG.scheduler,
            # BaseAgent
            saver=saver,
            storegate=storegate,
            task_scheduler=task_scheduler,
            metric=metric,
            
            # EnsembleAgent
            # ConnectionSimpleAgent
            freeze_model_weights=False,
            do_pretraining = opts.do_pretrain,
            connectiontask_args= config.ASNG.connectiontask_args,
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
            logger.info(f'{key: <30} : {val}')
    
    for key, val in results.items() : 
        print_dict(key, val)
    
    with open(f'{saver.save_dir}/result.run_connection_asngnas_{opts.event}evt_weight{opts.weight}.json', 'w') as fo : 
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
    
    for phase in phases : 
        # dump prediction
        storegate.set_data_id("")
        y_pred = np.array( storegate.get_data(phase = phase, var_names = variables ) )
        
        os.makedirs(f'{saver.save_dir}/pred/{phase}', exist_ok = True )
        
        for i, v in enumerate(variables):
            np.save(f'{saver.save_dir}/pred/{phase}/{v}', y_pred[i])
    
    

if __name__ == '__main__':
    from distutils.util import strtobool
    parser = argparse.ArgumentParser( description = 'This is a script to run xtrk_ntuple_maker' )
    parser.add_argument('-c', '--config',        action = 'store', dest = 'config',       required = True,  type = str,   default = None, help = 'text path for config file(yaml)')
    parser.add_argument('-s', '--seed',          action = 'store', dest = 'seed',         required = False, type = int,   default = None, help = 'seed integer')
    parser.add_argument('-g', '--gpu_index',     action = 'store', dest = 'gpu_index',    required = False, type = int,   default = None, help = 'gpu index')
    parser.add_argument('-ev', '--event',        action = 'store', dest = 'event',        required = False, type = int,   default = None, help = 'number of event ')
    parser.add_argument('-ep', '--epochs',       action = 'store', dest = 'epochs',       required = False, type = int,   default = None, help = 'number of epochs ')    
    parser.add_argument('-cl', '--clip_value',   action = 'store', dest = 'clip_value',   required = False, type = int,   default = None, help = 'clip value of grad ')
    parser.add_argument('-d', '--delta',         action = 'store', dest = 'delta',        required = False, type = float, default = None, help = 'clip value of grad ')
    parser.add_argument('-a', '--alpha',         action = 'store', dest = 'alpha',        required = False, type = float, default = None, help = 'alpha of ASNG ') 
    parser.add_argument('-w', '--weight',        action = 'store', dest = 'weight',       required = False, type = float, default = None, help = 'weight of task1 ')     
    parser.add_argument('-l',  '--lam',          action = 'store', dest = 'lam',          required = False, type = int,   default = None, help = 'lambda value ')    
    parser.add_argument("-j", '--jobid',         action = 'store', dest = 'jobid',        required = True,  type = str,   default = 'default', help = 'job id ')
    parser.add_argument('-dp', '--data_path',    action = 'store', dest = 'data_path',    required = False, type = str,   default = None, help = 'data path')
    parser.add_argument('-p', '--do_pretrain',   action = 'store', dest = 'do_pretrain',  required = False, type = strtobool,  help = 'do pretraining')
    parser.add_argument('-lw', '--load_weights', action = 'store', dest = 'load_weights', required = False, type = bool,  help = 'load weight')
    parser.add_argument('-ll', '--loglevel',     action = 'store', dest = 'loglevel',     required = False, choices = ('DEBUG', 'INFO', 'WARN', 'ERROR', 'DISABLED'), default = 'INFO', help = 'msg level to use. Valid choices are [""], default is "INFO"')
    
    # parser.add_argument('-p', '--property',    action = 'store', dest = 'properties',  required = False, nargs = '*' )
    opts = parser.parse_args()
    main( opts )
