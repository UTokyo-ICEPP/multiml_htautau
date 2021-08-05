from torch.utils.data import Dataset
import numpy as np


class OnlyDiTauDataset_wo_mass(Dataset):
    def __init__(self, max_events, data_path, phase=None):
        from multiml.storegate import StoreGate
        from multiml.data.numpy import NumpyFlatData
        cur_seed = np.random.get_state()
        np.random.seed(1)
        permute = np.random.permutation(2 * max_events)
        np.random.set_state(cur_seed)

        self.storegate = StoreGate(backend='numpy', data_id='')
        self.jet_vals = [
            '1stRecoJetPt', '1stRecoJetEta', '1stRecoJetPhi', '1stRecoJetMass', '2ndRecoJetPt',
            '2ndRecoJetEta', '2ndRecoJetPhi', '2ndRecoJetMass'
        ]
        self.tau_vals = [
            '1stTruthTauJetPt', '1stTruthTauJetEta', '1stTruthTauJetPhi', '1stTruthTauJetMass',
            '2ndTruthTauJetPt', '2ndTruthTauJetEta', '2ndTruthTauJetPhi', '2ndTruthTauJetMass'
        ]
        self.tau_vals_wo_mass = [
            '1stTruthTauJetPt',
            '1stTruthTauJetEta',
            '1stTruthTauJetPhi',
            '2ndTruthTauJetPt',
            '2ndTruthTauJetEta',
            '2ndTruthTauJetPhi',
        ]
        self.istau_vals = ['tauFlag1stJet', 'tauFlag2ndJet']
        self.label_vals = ['label']
        self.energy_vals = ['1stRecoJetEnergyMap', '2ndRecoJetEnergyMap']
        for path, var_names in [
            ("jet.npy", self.jet_vals),
            ("tau.npy", self.tau_vals),
            ("istau.npy", self.istau_vals),
            ("energy.npy", self.energy_vals),
        ]:
            data_list = []
            for label in ['Htautau', 'Zpure_tau']:
                data_loaded = NumpyFlatData().load_file(data_path + f"{label}_{path}")
                data_loaded = data_loaded[:max_events]
                data_list.append(data_loaded)
            data_loaded = np.concatenate(data_list)
            data_loaded = data_loaded[permute]

            self.storegate.update_data(data_id='',
                                       data=data_loaded,
                                       var_names=var_names,
                                       phase=(0.6, 0.2, 0.2))

        labels = np.concatenate([
            np.ones(max_events),
            np.zeros(max_events),
        ])[permute]

        self.storegate.update_data(data_id='',
                                   data=labels,
                                   var_names='label',
                                   phase=(0.6, 0.2, 0.2))
        self.storegate.compile()
        if phase is None:
            phase = 'train'
        self.phase = phase
        self._swich_phase()

    def _swich_phase(self):
        self.energy = np.transpose(self.storegate.get_data(
            self.energy_vals,
            self.phase,
            '',
        ), (0, 1, 4, 2, 3))
        self.jet_4vec = self.storegate.get_data(
            self.jet_vals,
            self.phase,
            '',
        )
        self.tau4vec_target = self.storegate.get_data(
            self.tau_vals_wo_mass,
            self.phase,
            '',
        ).astype(np.float32)
        self.label = self.storegate.get_data(
            self.label_vals,
            self.phase,
            '',
        ).astype(np.float32)

    def train(self):
        self.phase = 'train'
        self._swich_phase()

    def valid(self):
        self.phase = 'valid'
        self._swich_phase()

    def test(self):
        self.phase = 'test'
        self._swich_phase()

    def __len__(self):
        return self.storegate.get_metadata()['sizes'][self.phase]

    def __getitem__(self, idx):
        energy = self.energy[idx]
        jet_4vec = self.jet_4vec[idx]
        tau4vec_target = self.tau4vec_target[idx]
        label = self.label[idx]
        return {
            'inputs': [energy, jet_4vec],
            'targets': [tau4vec_target, label],
            'internal_vec': tau4vec_target,
            'internal_label': label
        }
