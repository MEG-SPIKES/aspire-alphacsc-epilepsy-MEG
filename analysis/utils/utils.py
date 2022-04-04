import yaml
from megspikes.casemanager.casemanager import CaseManager
from pathlib import Path

with open('case_info.yml', 'rt') as f:
    cases = yaml.safe_load(f.read())


def setup_case_manager(subject: int) -> CaseManager:
    case_name = cases['case_name'][subject]
    case = CaseManager(root=cases['cases_path'],
                       case=case_name,
                       free_surfer=cases['free_surfer_path'])

    case.set_basic_folders()
    case.select_fif_file(case.run)
    case.prepare_forward_model()

    case.manual_detections = (
            case.basic_folders['MANUAL'] / f"{case_name}_manual_detections.npy")

    if 'old_results_path' in cases.keys():
        case.old_results = Path(cases['old_results_path']) / \
                           f"{case_name}_results.h5"

    case.detection_pdf_reports = case.case_meg / 'REPORTS'
    case.detection_pdf_reports.mkdir(exist_ok=True)
    return case
