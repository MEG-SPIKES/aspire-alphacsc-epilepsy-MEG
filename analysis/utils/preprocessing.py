import gc
import math

import matplotlib.pyplot as plt
import mne
import numpy as np
from megspikes.casemanager.casemanager import CaseManager


class Preprocessing(CaseManager):
    """ Here we run all preprocessing steps"""

    def __init__(self, root=None, case=None, free_surfer=None):
        super().__init__(root=root, case=case, free_surfer=free_surfer)
        self.set_basic_folders()
        self.select_fif_file(self.run)
        self.prepare_forward_model()
        self.basic_folders["ICA"] = self.case_meg / 'ICA'
        self.basic_folders["ICA"].mkdir(exist_ok=True)

    def maxwell_filtering(self, bad_channels):
        crosstalk_file = str(self.root / 'ct_sparse.fif')
        fine_cal_file = str(self.root / 'sss_cal.dat')
        for fif_file in self.basic_folders["MEG_data"].glob('*.fif'):
            raw = mne.io.read_raw_fif(fif_file, preload=True,
                                      verbose='error')
            raw.info['bads'] = bad_channels
            raw_tsss = mne.preprocessing.maxwell_filter(
                raw, st_duration=10,  st_correlation=0.9,
                destination=(0, 0, 0.04),
                cross_talk=crosstalk_file,
                calibration=fine_cal_file, verbose=True)

            fname = f'{fif_file.stem}_tsss_mc.fif'
            self.tsss_file = self.basic_folders["tsss_fif_files"] / fname
            raw_tsss.save(self.tsss_file, overwrite=True, verbose='error')

    def plot_psd(self):
        """ Plot PSD for all files"""
        print('Plotting PSD ...')
        for fif_file in self.basic_folders["tsss_fif_files"].glob('*.fif'):

            data = mne.io.read_raw_fif(fif_file, preload=False,
                                       verbose='error')
            self._plot_psd_one_file(fif_file, data)
        del data
        print('DONE')

    def _plot_psd_one_file(self, fif_file, data):
        """ Plot PSD for one file"""

        picks = mne.pick_types(data.info, meg=True, eeg=False,
                               eog=False, stim=False)
        fig = data.plot_psd(tmax=np.inf, picks=picks, fmax=150,
                            show=False, verbose=False)
        fig.suptitle("{}".format(fif_file.stem), fontsize="x-large")
        fig.axes[0].set_xticks(np.arange(0, 150, step=10))
        fig.axes[1].set_xticks(np.arange(0, 150, step=10))

        fig.subplots_adjust(wspace=0.5, hspace=0.5, left=0.125,
                            right=0.9, top=0.85, bottom=0.2)
        fname = self.basic_folders["PSD"] / '{}.png'.format(fif_file.stem)
        fig.savefig(fname, format='png', dpi=300)
        fig.clf()
        plt.close(fig)
        del data, fname
        gc.collect()

    def ICA_plots(self):
        """ Fit ICA and detect automatic bad components
        Plot components"""

        comp_prop = self.basic_folders["ICA"] / 'component_properties'
        comp_prop.mkdir(exist_ok=True)
        sources = self.basic_folders["ICA"] / 'sources'
        sources.mkdir(exist_ok=True)
        print('ICA fitting, artefscts detecting, properties and sources plotting...')
        for fif_file in self.basic_folders["tsss_fif_files"].glob(f"{self.run}.fif"):

            ica_save = self.basic_folders["ICA"] / \
                '{}_ica.fif'.format(fif_file.stem)
            comp_prop_pic = comp_prop / fif_file.stem
            comp_prop_pic.mkdir(exist_ok=True)
            sources_pic = sources / fif_file.stem
            sources_pic.mkdir(exist_ok=True)

            data = mne.io.read_raw_fif(
                fif_file, preload=True, verbose='error')
            data.filter(1., None, fir_design='firwin')
            ica = Preprocessing._ICA_fit(self, data)
            if len(data.times)/1000 > 600:
                data_0 = data.copy().crop(10, 600)
                ica = self._ICA_artifacts(data_0, ica, ica_save)
            else:
                ica = self._ICA_artifacts(data, ica, ica_save)
            print('Bad components for {}: {}'.format(
                fif_file.stem, ica.exclude))
            if len(data.times)/1000 > 100:
                data_0 = data.copy().crop(10, 100)
                self._plot_prop(comp_prop_pic, ica, data_0)
                self._plot_sources(sources_pic, ica, data_0, iterations=6)
            else:
                print('File {} is too small ({} s)'.format(fif_file.stem,
                                                           len(data.times)/1000))
                data_0 = data.copy()
                self._plot_prop(comp_prop_pic, ica, data_0)
            del data_0, data, ica
            gc.collect()
        gc.collect()
        print('DONE')

    def _ICA_fit(self, data):
        """ Fit ICA """
        ica = mne.preprocessing.ICA(n_components=0.999, method='fastica',
                                    random_state=0, max_iter=100, verbose=False)
        picks = mne.pick_types(data.info, meg=True, eeg=False, eog=False,
                               stim=False, exclude='bads')
        ica.fit(data, picks=picks, reject_by_annotation=True, verbose=False)
        return ica

    def _ICA_artifacts(self, data, ica, save_fname):
        """ Automatic ICA detection """

        ecg_channels = mne.pick_types(data.info, ecg=True, meg=False,
                                      eeg=False, eog=False)
        if ecg_channels != []:
            ecg_channels = data.info['ch_names'][ecg_channels[0]]
        else:
            ecg_channels = None
        eog_channels = mne.pick_types(data.info, ecg=False, meg=False,
                                      eeg=False, eog=True)
        if eog_channels != []:
            eog_channels = [data.info['ch_names'][eog_channels[0]],
                            data.info['ch_names'][eog_channels[1]]]
        else:
            eog_channels = None

        ica.detect_artifacts(data, ecg_ch=ecg_channels, eog_ch=eog_channels,
                             ecg_criterion=0.5, eog_criterion=0.5)
        ica.save(save_fname)
        return ica

    def _plot_prop(self, directory, ica, data):
        for i in range(ica.n_components_):
            fig = ica.plot_properties(data, picks=i, show=False,
                                      psd_args={'fmax': 200.})
            name = 'ICA_component_{}_properties.png'.format(i)
            fname = directory / name
            fig[0].savefig(fname, format='png', dpi=150)
            [f.clf() for f in fig]
            [plt.close(f) for f in fig]
        del data, ica, fname
        gc.collect()

    def _plot_sources(self, directory, ica, data, iterations=4, window=5):
        for i in range(math.ceil(ica.n_components_/10)):
            for ii in range(iterations):
                m = 0 + i*10
                n = 10 + i*10
                if (ica.n_components_ - n) <= 0:
                    n = ica.n_components_
                start = 0+window*ii
                stop = window+window*ii
                fig = ica.plot_sources(data, picks=range(m, n), start=start,
                                       stop=stop, show=False)
                name = 'ICA_ts_{}_{}_time_{}_{}_s.png'.format(
                    m, n, start, stop)
                fname = directory / name
                fig.savefig(fname, format='png', dpi=150)
                fig.clf()
                plt.close(fig)
        del data, ica, fname
        gc.collect()

    def apply_ica(self, bad_components=[]):
        """ Apply ICA """
        import mne
        import gc

        if len(bad_components) == 0:
            bad_components = self.bad_ica_components

        # bad_components = self.path.get_bad_components_from_airtable()
        print('Bad components {}'.format(bad_components))
        if bad_components == []:
            print('Empty field Bad components')
        elif bad_components[0] == -2:
            print('Bad components were not selected')
        elif bad_components[0] == -1:
            print('File has not any bad components')

            data = mne.io.read_raw_fif(self.tsss_file, preload=True,
                                       verbose='error')
            fname = '{}_art_corr.fif'.format(self.tsss_file.stem)
            self.fif_file = self.basic_folders["art_cor_fif_files"] / fname
            data.save(self.fif_file, overwrite=True, verbose='error')
        else:
            ica_fif = '{}_ica.fif'.format(self.tsss_file.stem)
            ica_path = self.basic_folders["ICA"] / ica_fif
            ica = mne.preprocessing.read_ica(ica_path, verbose='error')
            ica.exclude = bad_components

            data = mne.io.read_raw_fif(self.tsss_file, preload=True,
                                       verbose='error')
            ica.apply(data)
            fname = '{}_art_corr.fif'.format(self.tsss_file.stem)
            self.fif_file = self.basic_folders["art_cor_fif_files"] / fname
            data.save(self.fif_file, overwrite=True, verbose='error')
            del ica, data
            print('Components {} were deleted from fif {}'.format(bad_components,
                                                                  self.case))
        gc.collect()

    def bad_annotation(self):
        # Bad annot from CaseManager
        bad_annot = self.bad_annot
        if bad_annot:
            data = mne.io.read_raw_fif(self.fif_file, preload=True,
                                       verbose='error')
            bad_annot = mne.Annotations(bad_annot['onsets'],
                                        bad_annot['durations'],
                                        bad_annot['descriptions'])
            data.set_annotations(bad_annot)
            data.save(self.fif_file, overwrite=True, verbose='error')
            print('Bad annotations were added to fif {}'.format(self.case))

    def delete_bad_annotations(self):
        self.bad_annotation()
        data = mne.io.read_raw_fif(self.fif_file, preload=True,
                                   verbose='error')
        print(data)
        data_arr = data.get_data(reject_by_annotation='omit')
        data_new = mne.io.RawArray(data_arr, data.info)
        data_new.save(self.fif_file, overwrite=True, verbose='error')
        print(data_new)

        print('Bad data were deleted from fif {}'.format(self.case))
