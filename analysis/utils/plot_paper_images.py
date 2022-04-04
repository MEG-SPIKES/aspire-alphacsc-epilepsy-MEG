import matplotlib.pylab as plt
import mne
import numpy as np
import pandas as pd
from megspikes.casemanager.casemanager import CaseManager


def fig4_one_row(n: int, case: CaseManager, fig4_table: pd.DataFrame, subj: int,
                 detection_type: str, fwd_mni: np.ndarray,
                 resection_stc: np.ndarray, detection_stc: np.ndarray,
                 dist_to_resection: float):
    fig4_table.loc[n, 'case_name'] = case.case
    fig4_table.loc[n, 'case'] = subj
    fig4_table.loc[n, 'detection_type'] = detection_type
    fig4_table.loc[n, 'n_sources_resection'] = fwd_mni[resection_stc > 0].shape[
        0]
    fig4_table.loc[n, 'n_sources'] = fwd_mni[detection_stc > 0].shape[0]
    fig4_table.loc[n, 'n_sources_to_n_sources_resection'] = (
            fig4_table.loc[n, 'n_sources'] / fig4_table.loc[n, 'n_sources_resection'])
    fig4_table.loc[n, 'distance_resection'] = dist_to_resection
    n += 1
    return fig4_table, n


def stc_surfaces(subjects_dir, stc_resection, stc_manual=None, stc_peak=None, stc_slope=None):
    # pipeline roc
    fig, ax = plt.subplots(nrows=4, ncols=4*2, figsize=(25, 15), dpi=300)
    surfer_kwargs = dict(hemi='split',  surface='inflated',  # 'white'
                         views=['lat'], spacing='ico4',
                         time_viewer=False, #  backend='mayavi',
                         subjects_dir=subjects_dir, colormap='Greens', colorbar=False,
                         background='w', foreground='k',  # , size=(500, 2000),
                         smoothing_steps='nearest', alpha=1, verbose='error',
                         add_data_kwargs={"fmin": 0, "fmid": 0.9, "fmax": 1})
    col = -1
    cmaps = ['Blues', 'Reds', 'Reds', 'Reds']
    titles = ['Resection', 'Manual', 'Automated (peak)', 'Automated (slope)']
    stcs = [stc_resection, stc_manual, stc_peak, stc_slope]
    for stc, cmap, title in zip(stcs, cmaps, titles):
        for hemi in ['rh', 'lh']:
            col += 1
            for row, view in enumerate(['lat', 'med', 'fro', 'par']):
                ax[row, col].axis('off')
                ax[row, col].get_xaxis().set_visible(False)
                ax[row, col].get_yaxis().set_visible(False)
                if not isinstance(stc, mne.SourceEstimate):
                    continue
                surfer_kwargs['hemi'] = hemi
                surfer_kwargs['views'] = view
                surfer_kwargs['colormap'] = cmap
                brain = stc.plot(**surfer_kwargs)
                im = brain.screenshot()
                brain.close()
                ax[row, col].imshow(im)
                if view == 'lat':
                    ax[row, col].set_title(f"{title} {hemi}")
    plt.tight_layout()
    return fig


def stc_surfaces_slope_peak(subjects_dir, stc_peak=None, stc_slope=None):
    # pipeline roc - same as above but without resection and manual
    fig, ax = plt.subplots(nrows=4, ncols=2*2, figsize=(25, 15), dpi=300)
    surfer_kwargs = dict(hemi='split',  surface='inflated',  # 'white'
                         views=['lat'], spacing='ico4',
                         time_viewer=False, #  backend='mayavi',
                         subjects_dir=subjects_dir, colormap='Greens', colorbar=False,
                         background='w', foreground='k',  # , size=(500, 2000),
                         smoothing_steps='nearest', alpha=1, verbose='error',
                         add_data_kwargs={"fmin": 0.0, "fmid": 0.95, "fmax": 1.})
    col = -1
    cmaps = [ 'Reds', 'Reds']
    titles = [ 'Automated (peak)', 'Automated (slope)']
    stcs = [ stc_peak, stc_slope]
    for stc, cmap, title in zip(stcs, cmaps, titles):
        for hemi in ['rh', 'lh']:
            col += 1
            for row, view in enumerate(['lat', 'med', 'fro', 'par']):
                ax[row, col].axis('off')
                ax[row, col].get_xaxis().set_visible(False)
                ax[row, col].get_yaxis().set_visible(False)
                if not isinstance(stc, mne.SourceEstimate):
                    continue
                surfer_kwargs['hemi'] = hemi
                surfer_kwargs['views'] = view
                surfer_kwargs['colormap'] = cmap
                brain = stc.plot(**surfer_kwargs)
                im = brain.screenshot()
                brain.close()
                ax[row, col].imshow(im)
                if view == 'lat':
                    ax[row, col].set_title(f"{title} {hemi}")
    plt.tight_layout()
    return fig

