reco_tau_4vec = (
    '1stRecoJetPt',
    '1stRecoJetEta',
    '1stRecoJetPhi',
    '1stRecoJetMass',
    '2ndRecoJetPt',
    '2ndRecoJetEta',
    '2ndRecoJetPhi',
    '2ndRecoJetMass',
)
truth_tau_4vec = (
    '1stTruthTauJetPt',
    '1stTruthTauJetEta',
    '1stTruthTauJetPhi',
    '2ndTruthTauJetPt',
    '2ndTruthTauJetEta',
    '2ndTruthTauJetPhi',
)
corr_tau_4vec = (
    '1stCorrTauJetPt',
    '1stCorrTauJetEta',
    '1stCorrTauJetPhi',
    '2ndCorrTauJetPt',
    '2ndCorrTauJetEta',
    '2ndCorrTauJetPhi',
)

mapping_truth_corr = [(v1, v2)
                      for v1, v2 in zip(truth_tau_4vec, corr_tau_4vec)]


def get_higgsId_subtasks(config,
                         saver,
                         device='cpu',
                         subtask_names=[],
                         truth_input=True,
                         batch_norm=False,
                         load_weights=True,
                         use_logits=True):

    subtasks = []

    save_dir = saver.save_dir
    conf = config.pretrain
    higgsId_args = {
        'saver': saver,
        'output_var_names': ('probability',),
        'true_var_names': ('label',),
        'optimizer': conf.optimizer.name,
        'optimizer_args': dict( **conf.optimizer.params ),
        'num_epochs': conf.epochs,
        "max_patience": conf.patience,
        'batch_size': conf.batch_size,
        'save_weights': True,
        'load_weights': False,
        'device': device,
        'verbose':conf.verbose,
    }
    if load_weights:
        higgsId_args['load_weights'] = True
        higgsId_args['phases'] = ['test']

    if use_logits:
        higgsId_args['loss'] = 'BCEWithLogitsLoss'
        activation_last = 'Identity'
    else:
        higgsId_args['loss'] = 'BCELoss'
        activation_last = 'Sigmoid'

    if truth_input:
        higgsId_args['input_var_names'] = truth_tau_4vec
    else:
        higgsId_args['input_var_names'] = corr_tau_4vec

    from multiml.hyperparameter import Hyperparameters
    for subtask_name in subtask_names:
        subtask = {}
        if subtask_name == 'mlp':
            from multiml_htautau.task.pytorch import HiggsID_MLPTask
            conf = config.tasks.HiggsID_MLPTask
            subtask['subtask_id'] = 'higgsId-mlp'
            subtask['env'] = HiggsID_MLPTask(activation=conf.params.activation,
                                             activation_last=activation_last,
                                             batch_norm=conf.params.batch_norm,
                                             **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers': [ conf.params.layers]})

        elif subtask_name == 'lstm':
            from multiml_htautau.task.pytorch import HiggsID_LSTMTask
            conf = config.tasks.HiggsID_LSTMTask
            subtask['subtask_id'] = 'higgsId-lstm'
            subtask['env'] = HiggsID_LSTMTask(layers_mlp=conf.params.layers_mlp,
                                              n_jets=conf.params.n_jets,
                                              activation_last=activation_last,
                                              batch_norm=conf.params.batch_norm,
                                              **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers_lstm': [ conf.params.layers_lstm ]})

        elif subtask_name == 'mass':
            from multiml_htautau.task.pytorch import HiggsID_MassTask
            conf = config.tasks.HiggsID_MassTask
            subtask['subtask_id'] = 'higgsId-mass'
            subtask['env'] = HiggsID_MassTask(n_jets=conf.params.n_jets,
                                              n_input_vars=conf.params.n_input_vars,
                                              activation=conf.params.activation,
                                              activation_last=activation_last,
                                              batch_norm=conf.params.batch_norm,
                                              scale_mass=conf.params.scale_mass,
                                              **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers': [conf.params.layers ]})

        else:
            raise KeyError(f"subtask_name = {subtask_name} is not defined.")

        subtasks.append(subtask)
    return subtasks


def get_tau4vec_subtasks(config,
                         saver,
                         subtask_names=[],
                         device='cpu',
                         batch_norm=False,
                         load_weights=True):
    subtasks = []

    save_dir = saver.save_dir
    conf = config.pretrain

    from multiml_htautau.task.loss import Tau4vecCalibLoss_torch
    tau4vec_args = {
        'saver': saver,
        'true_var_names': truth_tau_4vec,
        'input_vars_energy': ('1stRecoJetEnergyMap', '2ndRecoJetEnergyMap'),
        'input_vars_jet': reco_tau_4vec,
        'output_var_names': corr_tau_4vec,
        'input_njets': 2,
        'optimizer': conf.optimizer.name,
        'optimizer_args': dict( **conf.optimizer.params ),
        'loss': Tau4vecCalibLoss_torch(pt_scale=1e-2, use_pxyz=True),
        'num_epochs': conf.epochs,
        "max_patience": conf.patience,
        'batch_size': conf.batch_size,
        'save_weights': True,
        'load_weights': False,
        'device': device,
        'verbose':conf.verbose,
    }
    if load_weights:
        tau4vec_args['load_weights'] = True
        tau4vec_args['phases'] = ['test']

    from multiml import Hyperparameters
    for subtask_name in subtask_names:
        subtask = {}
        if subtask_name == 'MLP':
            from multiml_htautau.task.pytorch import Tau4vec_MLPTask
            conf = config.tasks.Tau4vec_MLPTask
            subtask['subtask_id'] = 'tau4vec-MLP'
            subtask['env'] = Tau4vec_MLPTask(batch_norm=conf.params.batch_norm,
                                             activation=conf.params.activation,
                                             **tau4vec_args)
            subtask['hps'] = Hyperparameters({
                'layers_images': [conf.params.layers_images],
                'layers_calib': [conf.params.layers_calib],
            })

        elif subtask_name == 'conv2D':
            from multiml_htautau.task.pytorch import Tau4vec_Conv2DTask
            conf = config.tasks.Tau4vec_Conv2DTask

            subtask['subtask_id'] = 'tau4vec-conv2D'
            subtask['env'] = Tau4vec_Conv2DTask(batch_norm=conf.params.batch_norm,
                                                activation=conf.params.activation,
                                                **tau4vec_args)
            subtask['hps'] = Hyperparameters({
                'layers_conv2d': [ conf.params.layers_conv2d ],
                'layers_images': [conf.params.layers_images ],
                'layers_calib': [conf.params.layers_calib ],
            })

        elif subtask_name == 'SF':
            from multiml_htautau.task.pytorch import Tau4vec_SFTask
            subtask['subtask_id'] = 'tau4vec-SF'
            subtask['env'] = Tau4vec_SFTask(**tau4vec_args)
            subtask['hps'] = None
        else:
            raise KeyError(f"subtask_name = {subtask_name} is not defined.")

        subtasks.append(subtask)
    return subtasks
