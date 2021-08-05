def common_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", help="seed", type=int, default=42)
    parser.add_argument("--load_weights", action="store_true", help="Load pretrained models")
    parser.add_argument("--data_path",
                        dest="data_path",
                        help="input data path",
                        type=str,
                        default='/tmp/onlyDiTau/')
    parser.add_argument("-n",
                        "--max_events",
                        dest="max_events",
                        help="maximum number of events in input samples",
                        type=int,
                        default=50000)
    parser.add_argument("--output_destname",
                        dest="output_destname",
                        help="output_destname",
                        type=str,
                        default=None)
    parser.add_argument("-w",
                        "--tau4vec_weights",
                        dest="tau4vec_weights",
                        help="tau4vec_weights",
                        type=float,
                        default=None)
    parser.add_argument("--individual_loss_weights",
                        dest="individual_loss_weights",
                        help="individual_loss_weights",
                        type=float,
                        default=0.0)
    parser.add_argument("--dropout_rate",
                        dest="dropout_rate",
                        help="dropout_rate",
                        type=float,
                        default=None)
    parser.add_argument("--nopretraining", action="store_true", help="do not pretraining")
    parser.add_argument("--remove_dummy_models",
                        action="store_true",
                        help="Do not use dummy models (zero and noise)")
    parser.add_argument("--run_eagerly", action="store_true", help="Run by eager mode")
    parser.add_argument("--log_level", dest="log_level", help="log level", type=int, default=10)
    parser.add_argument("--igpu", dest="igpu", help="GPU index", type=int, default=0)
    return parser


def add_suffix(save_dir, args):
    if args.tau4vec_weights is not None:
        save_dir += f'_w{int(args.tau4vec_weights * 10000):05}'

    if args.dropout_rate is not None:
        save_dir += f'_d{int(args.dropout_rate * 100):03}'

    if args.nopretraining:
        save_dir += '_nopretrain'

    if args.individual_loss_weights != 0.0:
        save_dir += f'_il{int(args.individual_loss_weights * 100):03}'

    if args.run_eagerly:
        save_dir += '_eager'

    return save_dir


def plot_performance(saver, storegate, do_probability=True, do_tau4vec=True):
    import os
    save_dir = saver.save_dir + "/plot"
    os.makedirs(save_dir, exist_ok=True)

    if do_probability:
        from plot_utils import plot_classification
        plot_classification(storegate=storegate,
                            var_pred=('probability', ),
                            var_target=('label'),
                            data_id="",
                            phase="test",
                            save_dir=save_dir)

    if do_tau4vec:
        from my_tasks import truth_tau_4vec, corr_tau_4vec
        from plot_utils import plot_regression
        plot_regression(storegate=storegate,
                        var_pred=corr_tau_4vec,
                        var_target=truth_tau_4vec,
                        data_id="",
                        phase="train",
                        save_dir=save_dir)
        plot_regression(storegate=storegate,
                        var_pred=corr_tau_4vec,
                        var_target=truth_tau_4vec,
                        data_id="",
                        phase="valid",
                        save_dir=save_dir)
        plot_regression(storegate=storegate,
                        var_pred=corr_tau_4vec,
                        var_target=truth_tau_4vec,
                        data_id="",
                        phase="test",
                        save_dir=save_dir)

        from plot_utils import plot_system_mass
        plot_system_mass(storegate=storegate,
                         var_pred=corr_tau_4vec,
                         label='label',
                         data_id="",
                         phase="train",
                         save_dir=save_dir)


def copy_outputfiles(output_destname):
    # compress output files
    import tarfile
    archive = tarfile.open('output.tar.gz', mode='w:gz')
    archive.add('output')
    archive.close()

    # Copy output files
    if output_destname is not None:
        import subprocess
        subprocess.check_call(['gsutil', 'cp', 'output.tar.gz', output_destname])

    # For debug
    import os
    files = os.listdir('/')
    for fname in files:
        print('/' + fname)
    files = os.listdir('/output')
    for fname in files:
        print('/output/' + fname)


def preprocessing(save_dir,
                  args,
                  tau4vec_tasks=['MLP', 'conv2D', 'SF', 'zero', 'noise'],
                  higgsId_tasks=['mlp', 'lstm', 'mass', 'zero', 'noise'],
                  truth_intermediate_inputs=True):
    from multiml import logger
    logger.set_level(args.log_level)

    from setup_tensorflow import setup_tensorflow
    setup_tensorflow(args.seed, args.igpu)

    load_weights = args.load_weights

    from multiml.saver import Saver
    saver = Saver(save_dir, serial_id=args.seed)
    saver.add("seed", args.seed)

    # Storegate
    from my_storegate import get_storegate
    storegate = get_storegate(
        data_path=args.data_path,
        max_events=args.max_events,
    )

    # Task scheduler
    from multiml.task_scheduler import TaskScheduler
    from my_tasks import get_higgsId_subtasks, get_tau4vec_subtasks
    task_scheduler = TaskScheduler()

    if args.remove_dummy_models:
        tau4vec_tasks = [v for v in tau4vec_tasks if v not in ['zero', 'noise']]
        higgsId_tasks = [v for v in higgsId_tasks if v not in ['zero', 'noise']]

    if len(tau4vec_tasks) > 0 and len(higgsId_tasks) > 0:
        subtask1 = get_higgsId_subtasks(saver,
                                        subtask_names=higgsId_tasks,
                                        truth_input=truth_intermediate_inputs,
                                        load_weights=load_weights,
                                        run_eagerly=args.run_eagerly)
        task_scheduler.add_task(task_id='higgsId',
                                parents=['tau4vec'],
                                children=[],
                                subtasks=subtask1)

        subtask2 = get_tau4vec_subtasks(saver,
                                        subtask_names=tau4vec_tasks,
                                        load_weights=load_weights,
                                        run_eagerly=args.run_eagerly)
        task_scheduler.add_task(task_id='tau4vec',
                                parents=[],
                                children=['higgsId'],
                                subtasks=subtask2)

    elif len(higgsId_tasks) > 0:
        subtask = get_higgsId_subtasks(saver,
                                       subtask_names=higgsId_tasks,
                                       load_weights=load_weights,
                                       run_eagerly=args.run_eagerly)
        task_scheduler.add_task(task_id='higgsId', subtasks=subtask)

    elif len(tau4vec_tasks) > 0:
        subtask = get_tau4vec_subtasks(saver,
                                       subtask_names=tau4vec_tasks,
                                       load_weights=load_weights,
                                       run_eagerly=args.run_eagerly)
        task_scheduler.add_task(task_id='tau4vec', subtasks=subtask)

    else:
        raise ValueError("Strange task combination...")

    # Metric
    if len(tau4vec_tasks) > 0 and len(higgsId_tasks) == 0:
        from multiml_htautau.task.metrics import CustomMSEMetric
        from my_tasks import corr_tau_4vec, truth_tau_4vec
        metric = CustomMSEMetric(pred_var_name=corr_tau_4vec,
                                 true_var_name=truth_tau_4vec,
                                 phase='test')
    else:
        from multiml.agent.metric import AUCMetric
        metric = AUCMetric(pred_var_name='probability', true_var_name='label', phase='test')

    return saver, storegate, task_scheduler, metric


def postprocessing(saver, storegate, args, do_probability=True, do_tau4vec=True):
    # Performance check
    from run_utils import plot_performance
    plot_performance(saver, storegate, do_probability=do_probability, do_tau4vec=do_tau4vec)

    # Compress and copy output files
    if args.output_destname is not None:
        from run_utils import copy_outputfiles
        copy_outputfiles(args.output_destname)

    # Dump prediction
    variables = []
    if do_tau4vec:
        from my_tasks import corr_tau_4vec
        variables.extend(corr_tau_4vec)
    if do_probability:
        variables.extend(['probability'])

    from plot_utils import dump_predictions
    for phase in ['train', 'valid', 'test']:
        dump_predictions(storegate=storegate,
                         variables=tuple(variables),
                         data_id="",
                         phase=phase,
                         save_dir=saver.save_dir)


def get_config_multi_loss(args):
    if args.tau4vec_weights is None:
        use_multi_loss = False
        loss_weights = None
    else:
        use_multi_loss = True
        loss_weights = {'higgsId': 1.0 - args.tau4vec_weights, "tau4vec": args.tau4vec_weights}

    return use_multi_loss, loss_weights
