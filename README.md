# aspire-alphacsc-epilepsy-MEG
Plots and analysis for the paper "Data-driven approach for the delineation of the irritative zone in epilepsy in MEG"

## Quick start

Create a new Anaconda environment:

```bash
conda create -n megspikes-pipeline python=3.7
conda activate megspikes-pipeline
```

The easiest way to install the `megspikes` package is using pip. You should clone the repository and install all dependencies:

```bash
git clone https://github.com/MEG-SPIKES/megspikes.git
cd megspikes/
pip install .
```

To install the correct kernel for Jupyter Notebook use the following command:

```bash
python -m ipykernel install --user --name megspikes-pipeline
```

Additionally, to plot paper images `Seaborn` package is needed.

```bash
pip install seaborn
```

All the pipelines are in the [analysis](analysis) folder. Additionally, `case_info.yml` file in the analysis folder is required. The structure of the `case_info.yml` file:

```yaml
case_name:
  1: SAMPLE1
  2: SAMPLE2
cases_path: /aspire-alphacsc/Cases
free_surfer_path: /aspire-alphacsc/FreeSurfer

```
