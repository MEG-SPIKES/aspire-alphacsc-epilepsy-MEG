import megspikes.database.database as db
import mne
import nixio
import numpy as np
from megspikes.localization.localization import ForwardToMNI, PredictIZClusters
from sklearn.pipeline import Pipeline

mne.set_log_level("ERROR")


def convert_results(case):
    nixfile = nixio.File.open(
        case.old_results,
        nixio.FileMode.ReadWrite)

    fif_file = mne.io.read_raw_fif(case.fif_file).copy()

    pcds = db.PrepareClustersDataset(
        fif_file=case.fif_file, fwd=case.fwd['ico5'])

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
        'spikes': spikes,
        'spike_clusters': spike_clusters,
        'spike_sensors': sensors
    }

    clusters_dataset, _ = pcds.fit_transform((detections, fif_file))
    cluster = 0
    for sens, atoms in zip(['grad', 'mag'],
                           [np.unique(det_clust_grad),
                            np.unique(det_clust_mag)]):
        for atom in np.int32(atoms):

            for sens_loc in ['grad', 'mag']:
                db.check_and_write_to_dataset(
                    clusters_dataset, 'mne_localization',
                    nixfile.blocks['ROC'].data_arrays[
                        f'MNE_{sens}_{atom}_{sens_loc}'][:, :1000],
                    dict(sensors=sens_loc, cluster=cluster)
                    )
                db.check_and_write_to_dataset(
                    clusters_dataset, 'evoked',
                    nixfile.blocks['ROC'].data_arrays[
                        f'evoked_{sens}_{atom}_{sens_loc}'][:, :1000],
                    dict(cluster=cluster,
                         channel=clusters_dataset.attrs[sens_loc])
                    )
            cluster += 1
    for prop1, prop2 in zip(['t1', 't2', 't3', 'selected'],
                            ['time_baseline', 'time_slope', 'time_peak',
                             'selected_for_iz_prediction']):
        db.check_and_write_to_dataset(
            clusters_dataset, 'cluster_properties',
            nixfile.blocks['ROC'].data_frames['Atoms'][prop1],
            dict(cluster_property=prop2))

    pipe = Pipeline([
        ('convert_forward_to_mni', ForwardToMNI(case=case)),
        ('predict_IZ',
         PredictIZClusters(case=case)),
        ('save_dataset', db.SaveDataset(dataset=case.cluster_dataset))
    ])
    clusters_dataset, _ = pipe.fit_transform((clusters_dataset, _))
    np.save(case.resection,
            nixfile.blocks['Resection'].data_arrays['resection_mni'][:])
    np.save(case.manual_stc,
            nixfile.blocks['ROC'].data_arrays['surface_manual'][:])
    np.save(case.resection_stc,
            nixfile.blocks['ROC'].data_arrays['surface_resection'][:])
    np.save(case.peak_stc,
            clusters_dataset.iz_prediction.loc[:, 'peak'].values)
    np.save(case.slope_stc,
            clusters_dataset.iz_prediction.loc[:, 'slope'].values)
    return clusters_dataset
