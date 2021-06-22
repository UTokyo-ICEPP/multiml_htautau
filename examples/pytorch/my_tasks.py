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


def get_higgsId_subtasks(saver,
                         device='cpu',
                         subtask_names=[],
                         truth_input=True,
                         batch_norm=False,
                         load_weights=True,
                         use_logits=True):

    subtasks = []

    save_dir = saver.save_dir

    higgsId_args = {
        'saver': saver,
        'output_var_names': ('probability',),
        'true_var_names': ('label',),
        'optimizer': 'Adam',
        'optimizer_args': dict(lr=1e-3),
        'num_epochs': 100,
        "max_patience": 10,
        'batch_size': 64,
        'save_weights': True,
        'load_weights': False,
        'device': device,
        'verbose':2,        
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
            subtask['subtask_id'] = 'higgsId-mlp'
            subtask['env'] = HiggsID_MLPTask(activation='ReLU',
                                             activation_last=activation_last,
                                             batch_norm=batch_norm,
                                             **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers': [[6, 32, 32, 32, 1]]})

        elif subtask_name == 'lstm':
            from multiml_htautau.task.pytorch import HiggsID_LSTMTask
            subtask['subtask_id'] = 'higgsId-lstm'
            subtask['env'] = HiggsID_LSTMTask(layers_mlp=[1, 1],
                                              n_jets=2,
                                              activation_last=activation_last,
                                              batch_norm=batch_norm,
                                              **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers_lstm': [[3, 32, 32, 32, 1]]})

        elif subtask_name == 'mass':
            from multiml_htautau.task.pytorch import HiggsID_MassTask
            subtask['subtask_id'] = 'higgsId-mass'
            subtask['env'] = HiggsID_MassTask(n_jets=2,
                                              n_input_vars=6,
                                              activation='ReLU',
                                              activation_last=activation_last,
                                              batch_norm=batch_norm,
                                              scale_mass=1. / 125.,
                                              **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers': [[1, 64, 64, 1]]})

        else:
            raise KeyError(f"subtask_name = {subtask_name} is not defined.")

        subtasks.append(subtask)
    return subtasks


def get_tau4vec_subtasks(saver,
                         subtask_names=[],
                         device='cpu',
                         batch_norm=False,
                         load_weights=True):
    subtasks = []

    save_dir = saver.save_dir

    from multiml_htautau.task.loss import Tau4vecCalibLoss_torch
    tau4vec_args = {
        'saver': saver,
        'true_var_names': truth_tau_4vec,
        'input_vars_energy': ('1stRecoJetEnergyMap', '2ndRecoJetEnergyMap'),
        'input_vars_jet': reco_tau_4vec,
        'output_var_names': corr_tau_4vec,
        'input_njets': 2,
        'optimizer': 'Adam',
        'optimizer_args': dict(lr=1e-3),
        'loss': Tau4vecCalibLoss_torch(pt_scale=1e-2, use_pxyz=True),
        'num_epochs': 100,
        "max_patience": 10,
        'batch_size': 64,
        'save_weights': True,
        'load_weights': False,
        'device': device,
        'verbose':2,
    }
    if load_weights:
        tau4vec_args['load_weights'] = True
        tau4vec_args['phases'] = ['test']

    from multiml import Hyperparameters
    for subtask_name in subtask_names:
        subtask = {}
        if subtask_name == 'MLP':
            from multiml_htautau.task.pytorch import Tau4vec_MLPTask
            subtask['subtask_id'] = 'tau4vec-MLP'
            subtask['env'] = Tau4vec_MLPTask(batch_norm=batch_norm,
                                             activation='ReLU',
                                             **tau4vec_args)
            subtask['hps'] = Hyperparameters({
                'layers_images': [[768, 16, 16, 16, 4]],
                'layers_calib': [[8, 32, 32, 3]],
            })

        elif subtask_name == 'conv2D':
            from multiml_htautau.task.pytorch import Tau4vec_Conv2DTask
            subtask['subtask_id'] = 'tau4vec-conv2D'
            layers_conv2d = [
                ('conv2d', {
                    'in_channels': 3,
                    'out_channels': 32,
                    'kernel_size': 3
                }),
                ('conv2d', {
                    'in_channels': 32,
                    'out_channels': 16,
                    'kernel_size': 3
                }),
                ('maxpooling2d', {
                }),
                ('conv2d', {
                    'in_channels': 16,
                    'out_channels': 16,
                    'kernel_size': 2
                }),
                ('conv2d', {
                    'in_channels': 16,
                    'out_channels': 8,
                    'kernel_size': 2
                }),
            ]
            subtask['env'] = Tau4vec_Conv2DTask(batch_norm=batch_norm,
                                                activation='ReLU',
                                                **tau4vec_args)
            subtask['hps'] = Hyperparameters({
                'layers_conv2d': [layers_conv2d],
                'layers_images': [[128, 16, 16, 16, 4]],
                'layers_calib': [[8, 64, 64, 64, 3]],
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
