import mne
import numpy as np
import xarray as xr

from megspikes.pipeline import (iz_prediction_pipeline,
                                read_detection_iz_prediction_pipeline)
from megspikes.visualization.visualization import ClusterSlopeViewer
from sklearn import set_config

from utils.utils import setup_case_manager


params = {
    'PrepareClustersDataset': {'detection_sfreq': 200.}
}

subj = 1

case = setup_case_manager(subj)

pipe = read_detection_iz_prediction_pipeline(case, params)
detection_results = xr.open_dataset(case.dataset)

raw = mne.io.read_raw_fif(case.fif_file)
clusters, _ = pipe.fit_transform((detection_results, raw.copy()))

pc = ClusterSlopeViewer(clusters, case)

