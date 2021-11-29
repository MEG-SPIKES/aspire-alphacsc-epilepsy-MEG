

def fig4_one_row(n, case, fig4_table, subj, detection_type, fwd_mni,
                 resection_stc, detection_stc, dist_to_resection):
    fig4_table.loc[n, 'case_name'] = case.case
    fig4_table.loc[n, 'case'] = subj
    fig4_table.loc[n, 'detection_type'] = detection_type
    fig4_table.loc[n, 'n_sources_resection'] = fwd_mni[resection_stc > 0].shape[
        0]
    fig4_table.loc[n, 'n_sources'] = fwd_mni[detection_stc > 0].shape[0]
    fig4_table.loc[n, 'n_sources_to_n_sources_resection'] = (
            fig4_table.loc[n, 'n_sources'] / fig4_table.loc[
        n, 'n_sources_resection'])
    fig4_table.loc[n, 'distance_resection'] = dist_to_resection
    n += 1
    return fig4_table, n