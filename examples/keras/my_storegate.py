from multiml.storegate import StoreGate

import numpy as np


def get_storegate(data_path='/tmp/onlyDiTau/', max_events=50000):
    # Index for signal/background shuffle
    cur_seed = np.random.get_state()
    np.random.seed(1)
    permute = np.random.permutation(2 * max_events)
    np.random.set_state(cur_seed)

    storegate = StoreGate(backend='numpy', data_id='')

    for path, var_names in [
        ("jet.npy", ('1stRecoJetPt', '1stRecoJetEta', '1stRecoJetPhi', '1stRecoJetMass',
                     '2ndRecoJetPt', '2ndRecoJetEta', '2ndRecoJetPhi', '2ndRecoJetMass')),
        ("tau.npy",
         ('1stTruthTauJetPt', '1stTruthTauJetEta', '1stTruthTauJetPhi', '1stTruthTauJetMass',
          '2ndTruthTauJetPt', '2ndTruthTauJetEta', '2ndTruthTauJetPhi', '2ndTruthTauJetMass')),
        ("istau.npy", ('tauFlag1stJet', 'tauFlag2ndJet')),
        ("energy.npy", ('1stRecoJetEnergyMap', '2ndRecoJetEnergyMap')),
    ]:
        data_list = []
        for label in ['Htautau', 'Zpure_tau']:
            data_loaded = np.load(data_path + f"{label}_{path}")
            data_loaded = data_loaded[:max_events]
            data_list.append(data_loaded)
        data_loaded = np.concatenate(data_list)
        data_loaded = data_loaded[permute]

        storegate.update_data(data=data_loaded, var_names=var_names, phase=(0.6, 0.2, 0.2))

    # # Added TauMass
    # tau_mass = np.full(shape=2 * max_events, fill_value=1.777)
    # storegate.update_data(
    #     data_id='',
    #     data=tau_mass,
    #     var_names=['TauMass'],
    # )
    # storegate._num_events['TauMass'] += len(tau_mass)

    # Setting labels
    labels = np.concatenate([
        np.ones(max_events),
        np.zeros(max_events),
    ])[permute]

    storegate.update_data(data=labels, var_names='label', phase=(0.6, 0.2, 0.2))

    storegate.compile()
    storegate.show_info()

    return storegate
