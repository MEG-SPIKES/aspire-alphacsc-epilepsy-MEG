from pathlib import Path
import os

import mne
import yaml

mne.set_log_level("ERROR")

ceses_file = os.path.join(os.path.dirname(__file__), 'case_info.yml')
with open(ceses_file, 'rt') as f:
    cases = yaml.safe_load(f.read())

data_path = Path(cases['data_path'])
ffwd = data_path / 'forward_model'
meg_data = data_path / 'meg_data'
old_results = data_path / 'old_results'
new_results = data_path / 'new_results'
free_surfer = data_path / 'FreeSurfer'
fresecton = data_path / 'resections'
stcs = data_path / 'stcs'


class CaseManager():
    def __init__(self, subject: int = 1) -> None:

        self.freesurfer_dir = free_surfer
        self.case = cases['case_name'][f'subject{subject}']

        fwd = mne.read_forward_solution(ffwd / f'{subject}_forward_ico5.fif')
        self.fwd = {'ico5': fwd}
        self.bem = {'ico5': None}
        self.trans = {'ico5': None}

        self.fif_file = str(meg_data / f'{subject}_raw_tsss_mc_art_corr.fif')

        self.old_results = str(old_results / f"{subject}_results.h5")

        self.cluster_dataset = new_results / f'{subject}_clusters.nc'
        self.dataset = new_results / f'{subject}_alphacsc_results.nc'
        self.resection = fresecton / f'{subject}_resection.npy'
        self.info = mne.io.read_info(self.fif_file)
        self.manual_stc = stcs / f'{subject}_manual_stc.npy'
        self.resection_stc = stcs / f'{subject}_resection_stc.npy'
        self.peak_stc = stcs / f'{subject}_peak_stc.npy'
        self.slope_stc = stcs / f'{subject}_slope_stc.npy'
