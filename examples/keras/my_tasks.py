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
                         subtask_names=[],
                         truth_input=True,
                         batch_norm=False,
                         load_weights=True,
                         use_logits=True,
                         run_eagerly=None):

    subtasks = []

    higgsId_args = {
        'saver': saver,
        'output_var_names': ('probability',),
        'true_var_names': 'label',
        'optimizer_args': dict(learning_rate=1e-3),
        'optimizer': 'adam',
        'num_epochs': 100,
        "max_patience": 10,
        'batch_size': 100,
        'phases': None,
        'save_weights': True,
        'run_eagerly': run_eagerly,
    }
    if load_weights:
        higgsId_args['load_weights'] = True
        higgsId_args['save_weights'] = False
        higgsId_args['phases'] = ['test']

    if use_logits:
        from tensorflow.keras.losses import BinaryCrossentropy
        higgsId_args['loss'] = BinaryCrossentropy(from_logits=True)
        activation_last = 'linear'
    else:
        higgsId_args['loss'] = 'binary_crossentropy'
        activation_last = 'sigmoid'

    if truth_input:
        higgsId_args['input_var_names'] = truth_tau_4vec
    else:
        higgsId_args['input_var_names'] = corr_tau_4vec

    from multiml import Hyperparameters
    for subtask_name in subtask_names:
        subtask = {}
        if subtask_name == 'mlp':
            from multiml.task.keras import MLPTask
            subtask['subtask_id'] = 'higgsId-mlp'
            subtask['env'] = MLPTask(name='higgsId-mlp',
                                     activation='relu',
                                     activation_last=activation_last,
                                     batch_norm=batch_norm,
                                     **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers': [[32, 32, 32, 1]]})

        elif subtask_name == 'lstm':
            from multiml_htautau.task.keras import HiggsID_LSTMTask
            subtask['subtask_id'] = 'higgsId-lstm'
            subtask['env'] = HiggsID_LSTMTask(
                name='higgsId-lstm',
                input_njets=2,
                activation_last=activation_last,
                batch_norm=batch_norm,
                **higgsId_args)
            subtask['hps'] = Hyperparameters({'nodes': [[32, 32, 32, 1]]})

        elif subtask_name == 'mass':
            from multiml_htautau.task.keras import HiggsID_MassTask
            subtask['subtask_id'] = 'higgsId-mass'
            subtask['env'] = HiggsID_MassTask(
                name='higgsId-mass',
                input_njets=2,
                activation='relu',
                activation_last=activation_last,
                batch_norm=batch_norm,
                scale_mass=1. / 125.,
                **higgsId_args)
            subtask['hps'] = Hyperparameters({'layers': [[64, 64, 1]]})

        elif subtask_name == 'zero':
            from multiml_htautau.task.keras import HiggsID_ZeroTask
            subtask['subtask_id'] = 'higgsId-zero'
            subtask['env'] = HiggsID_ZeroTask(name='higgsId-zero', input_njets=2, **higgsId_args)
            subtask['hps'] = None

        elif subtask_name == 'noise':
            subtask['subtask_id'] = 'higgsId-noise'
            from multiml_htautau.task.keras import HiggsID_NoiseTask
            subtask['env'] = HiggsID_NoiseTask(name='higgsId-noise', input_njets=2, **higgsId_args)
            subtask['hps'] = None

        else:
            raise KeyError(f"subtask_name = {subtask_name} is not defined.")

        subtasks.append(subtask)
    return subtasks


def get_tau4vec_subtasks(saver,
                         subtask_names=[],
                         batch_norm=False,
                         load_weights=True,
                         run_eagerly=None):
    subtasks = []

    from multiml_htautau.task.loss import Tau4vecCalibLoss_tf
    tau4vec_args = {
        'saver': saver,
        'true_var_names': truth_tau_4vec,
        'input_vars_energy': ('1stRecoJetEnergyMap', '2ndRecoJetEnergyMap'),
        'input_vars_jet': reco_tau_4vec,
        'output_var_names': corr_tau_4vec,
        'input_njets': 2,
        'optimizer_args': dict(learning_rate=1e-3),
        'optimizer': 'adam',
        'loss': Tau4vecCalibLoss_tf(pt_scale=1e-2, use_pxyz=True),
        'num_epochs': 100,
        "max_patience": 10,
        'batch_size': 100,
        'phases': None,
        'save_weights': True,
        'run_eagerly': run_eagerly,
    }
    if load_weights:
        tau4vec_args['load_weights'] = True
        tau4vec_args['save_weights'] = False
        tau4vec_args['phases'] = ['test']

    from multiml import Hyperparameters
    for subtask_name in subtask_names:
        subtask = {}
        if subtask_name == 'MLP':
            from multiml_htautau.task.keras import Tau4vec_MLPTask
            subtask['subtask_id'] = 'tau4vec-MLP'
            subtask['env'] = Tau4vec_MLPTask(
                name='tau4vec-MLP',
                batch_norm=batch_norm,
                activation='relu',
                **tau4vec_args)
            subtask['hps'] = Hyperparameters({'layers_images': [[16, 16, 16, 4]],
                                              'layers_calib': [[32, 32, 3]],
                                             })

        elif subtask_name == 'conv2D':
            from multiml_htautau.task.keras import Tau4vec_Conv2DTask
            subtask['subtask_id'] = 'tau4vec-conv2D'
            layers_conv2d = [
                ('conv2d', {
                    'filters': 32,
                    'kernel_size': (3, 3)
                }),
                ('conv2d', {
                    'filters': 16,
                    'kernel_size': (3, 3)
                }),
                ('maxpooling2d', {
                    'pool_size': (2, 2)
                }),
                ('conv2d', {
                    'filters': 16,
                    'kernel_size': (2, 2)
                }),
                ('conv2d', {
                    'filters': 8,
                    'kernel_size': (2, 2)
                }),
            ]
            subtask['env'] = Tau4vec_Conv2DTask(
                name='tau4vec-conv2D',
                batch_norm=batch_norm,
                activation='relu',
                **tau4vec_args)
            subtask['hps'] = Hyperparameters({'layers_conv2d': [layers_conv2d],
                                              'layers_images': [[16, 16, 16, 4]],
                                              'layers_calib': [[64, 64, 64, 3]],
                                             })

        elif subtask_name == 'SF':
            from multiml_htautau.task.keras import Tau4vec_SFTask
            subtask['subtask_id'] = 'tau4vec-SF'
            subtask['env'] = Tau4vec_SFTask(name='tau4vec-SF',
                                            **tau4vec_args)
            subtask['hps'] = None

        elif subtask_name == 'zero':
            from multiml_htautau.task.keras import Tau4vec_ZeroTask
            subtask['subtask_id'] = 'tau4vec-zero'
            subtask['env'] = Tau4vec_ZeroTask(name='tau4vec-zero', **tau4vec_args)
            subtask['hps'] = None

        elif subtask_name == 'noise':
            from multiml_htautau.task.keras import Tau4vec_NoiseTask
            subtask['subtask_id'] = 'tau4vec-noise'
            subtask['env'] = Tau4vec_NoiseTask(name='tau4vec-noise', **tau4vec_args)
            subtask['hps'] = None

        else:
            raise KeyError(f"subtask_name = {subtask_name} is not defined.")

        subtasks.append(subtask)
    return subtasks
