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