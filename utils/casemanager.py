import os
import warnings
from pathlib import Path
from typing import List, Union

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

        # fwd = mne.read_forward_solution(ffwd / f'{subject}_forward_ico5.fif')
        # self.fwd = {'ico5': fwd}
        # self.bem = {'ico5': None}
        # self.trans = {'ico5': None}
        self.fwd_folder = ffwd / f'{subject}_subject'
        self.fwd_folder.mkdir(exist_ok=True)

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
        self.prepare_forward_model()

    def prepare_forward_model(self, spacings: List[str] = ['ico5', 'oct5'],
                              sensors: Union[str, bool] = True) -> None:
        info = mne.io.read_info(self.fif_file)
        self.info = mne.pick_info(info, mne.pick_types(info, meg=sensors))
        self.fwd = {}
        self.bem, self.src, self.trans = {}, {}, {}
        for spacing in spacings:
            fwd_name = self.fwd_folder
            fwd_name = fwd_name / f'forward_{spacing}.fif'
            fwd, bem, src, trans = self._prepare_forward_model(
                fwd_name, self.info, spacing=spacing, n_jobs=7)

            if isinstance(sensors, str):
                fwd = mne.pick_types_forward(fwd, meg=sensors)

            self.fwd[spacing] = fwd
            self.bem[spacing] = bem
            self.src[spacing] = src
            self.trans[spacing] = trans

    def _prepare_forward_model(self, fwd_name, info, spacing='ico5',
                               n_jobs=7):
        """Make forwad solution

        NOTE: Coregistration was done in Brainstorm and affine
        from MRI srucute in BrainStorm was used

        Parameters
        ----------
        freesurfer_dir : str
            FreeSurfer folder (including bem)
        spacing : str, optional
            'oct5' - 1026*2 sources, by default 'ico5' - 10242*2 sources
        """
        fsrc = fwd_name.with_name(f'source_spaces_{spacing}.fif')
        if not fsrc.is_file():
            try:
                src = mne.setup_source_space(
                    self.case, spacing=spacing, add_dist='patch',
                    subjects_dir=self.freesurfer_dir, n_jobs=n_jobs)
            except Exception:
                warnings.warn(f'Using ico4 instead of {spacing}')
                # traceback.print_exc()
                src = mne.setup_source_space(
                    self.case, spacing='ico4', add_dist='patch',
                    subjects_dir=self.freesurfer_dir, n_jobs=n_jobs)
            mne.write_source_spaces(
                fsrc, src, overwrite=True, verbose='error')
        else:
            src = mne.read_source_spaces(fsrc, verbose='error')

        fbem = fwd_name.with_name('bem_solution.fif')
        if not fbem.is_file():
            # (0.3, 0.006, 0.3)  # for three layers
            conductivity = (0.3, )
            try:
                model = mne.make_bem_model(
                    subject=self.case, ico=5, conductivity=conductivity,
                    subjects_dir=self.freesurfer_dir)
            except Exception:
                warnings.warn('Using ico4 instead of ico5 for BEM model')
                # traceback.print_exc()
                model = mne.make_bem_model(
                    subject=self.case, ico=4, conductivity=conductivity,
                    subjects_dir=self.freesurfer_dir)
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(fwd_name.with_name('bem_solution.fif'), bem)
        else:
            bem = mne.read_bem_solution(fbem, verbose='error')

        ftrans = fwd_name.with_name('checked_visually_trans.fif')
        trans = mne.read_trans(ftrans)

        # info = mne.io.read_info(self.fif_path)
        if not fwd_name.is_file():
            fwd = mne.make_forward_solution(
                info, trans=trans, src=src, bem=bem,
                meg=True, eeg=False, mindist=5.0)
            mne.write_forward_solution(fwd_name, fwd, overwrite=True)
        else:
            fwd = mne.read_forward_solution(
                str(fwd_name), verbose='error')  # 306 sensors x 20000 dipoles
        return fwd, bem, src, trans
