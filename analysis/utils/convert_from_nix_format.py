import nixio
import mne
import numpy as np
import xarray as xr

from megspikes.pipeline import iz_prediction_pipeline
from megspikes.database.database import check_and_write_to_dataset
from megspikes.visualization.visualization import ClusterSlopeViewer


def convert_clusters_from_nix_format(case, raw):
    # read nix file
    nixfile = nixio.File.open(str(case.old_results), nixio.FileMode.ReadOnly)

    det_grad = nixfile.blocks['AtomsLibrary_grad'].data_frames[
        'Clustering results']['timestamps']
    det_clust_grad = nixfile.blocks['AtomsLibrary_grad'].data_frames[
        'Clustering results']['atom']
    det_mag = nixfile.blocks['AtomsLibrary_mag'].data_frames[
        'Clustering results']['timestamps']
    det_clust_mag = nixfile.blocks['AtomsLibrary_mag'].data_frames[
        'Clustering results']['atom']

    spikes = np.hstack([det_grad, det_mag])
    spikes, ind = np.unique(spikes, return_index=True)
    spike_clusters = np.hstack(
        [det_clust_grad, det_clust_mag + det_clust_grad.max() + 1])
    spike_clusters = spike_clusters[ind]

    sensors = np.hstack([np.zeros_like(det_grad), np.ones_like(det_mag)])
    sensors = sensors[ind]

    detections = {
        'spikes': (spikes / 200) * 1000,  # NOTE: spikes sfreq should be 1000Hz
        'spike_clusters': spike_clusters,
        'spike_sensors': sensors
    }

    baseline = nixfile.blocks['ROC'].data_frames['Atoms']['t1']
    slope = nixfile.blocks['ROC'].data_frames['Atoms']['t2']
    peak = nixfile.blocks['ROC'].data_frames['Atoms']['t3']
    selected = nixfile.blocks['ROC'].data_frames['Atoms']['selected']
    sensors = (nixfile.blocks['ROC'].data_frames['Atoms']['grad'] - 1) * -1 # because grad is 0 and mag is 1
    atoms = nixfile.blocks['ROC'].data_frames['Atoms']['atom']


    params = {
        'PrepareClustersDataset': {'detection_sfreq': 1000.}
    }
    pipe = iz_prediction_pipeline(case, params, rewrite_previous_results=True)

    # create new clusters dataset
    case.cluster_dataset = case.cluster_dataset.with_name(
        f'{case.case}_clusters_results_converted.nc')
    clusters, _ = pipe.fit_transform((detections, raw.copy()))

    # add slope information
    check_and_write_to_dataset(
        clusters, 'cluster_properties', slope,
        dict(cluster_property='time_slope'))
    check_and_write_to_dataset(
        clusters, 'cluster_properties', baseline,
        dict(cluster_property='time_baseline'))
    check_and_write_to_dataset(
        clusters, 'cluster_properties', peak,
        dict(cluster_property='time_peak'))
    check_and_write_to_dataset(
        clusters, 'cluster_properties', selected,
        dict(cluster_property='selected_for_iz_prediction'))
    check_and_write_to_dataset(
        clusters, 'cluster_properties', atoms,
        dict(cluster_property='atom'))

    pc = ClusterSlopeViewer(clusters, case)
    pc.fname_save_ds = str(pc.data.case.cluster_dataset.with_name(
        f"{pc.data.case_name}_clusters_converted_manually_checked.nc"))

    pc._rerun_iz_prediction()
    pc._save_dataset()
    return clusters, pc
