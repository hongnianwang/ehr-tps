# EHR-TPS

Code for temporal pattern mining from EHR data for early acute kidney injury prediction.

This repository contains the analysis code used to build MIMIC-IV cohorts, mine temporal patterns with S3M, integrate pattern-distance features, train models, run eICU-CRD external validation, and summarise outputs.

## Contents

- `src/`: preprocessing, temporal pattern mining, modeling, validation, and figure/table scripts
- `src/utils/`: shapelet distance and evaluation utilities
- `scripts/`: shell wrappers for common pipeline steps
- `sql/`: MIMIC-IV and eICU-CRD cohort extraction queries
- `data/`: placeholders for local restricted data

Generated outputs are not tracked. This includes patient-level files, processed datasets, S3M outputs, shapelet distance files, model artifacts, aggregate result tables, and figures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

S3M is required for temporal pattern mining:

```bash
wget https://github.com/BorgwardtLab/S3M/releases/download/v1.0.0-alpha/s3m-1.0.0-alpha.deb
sudo apt-get update
sudo dpkg -i s3m-1.0.0-alpha.deb || sudo apt --fix-broken install -y
```

## Data

The analysis uses MIMIC-IV and eICU-CRD. Both require PhysioNet credentialing and data use agreements. Place local data files under `data/mimiciv/` and `data/eicu/`.

Patient-level data are not included in this repository.

## Pipeline

A typical run order is:

```bash
# MIMIC preprocessing
bash scripts/run_mimic_preprocessing.sh

# Shapelet mining
python src/02_run_s3m_vital.py
python src/02_run_s3m_lab.py

# Feature integration and modeling
python src/03_integrate_shapelet_features.py
python src/03_model_machine_learning.py

# Current modeling cohort and validation
bash scripts/run_mimic_modeling_validation.sh

# eICU feature alignment and external validation
python src/03_integrate_shapelet_features_eicu.py
python src/04_external_validate_eicu.py
```

SQL cohort scripts are in `sql/mimic/` and `sql/eicu/`. Run them in an isolated PostgreSQL schema with the relevant MIMIC-IV or eICU-CRD schemas available.

## Reproducing Tables and Figures

- Table 1: `src/table1_combined_mimic_eicu.r`
- Model performance tables: `src/generate_table2_table3_performance.py`
- Figure 2: `src/figure2_revised_panels.R` and related `figure2*` scripts
- Figure 3: `src/build_figure3_data.py` and `src/figure3_internal_external_validation.R`
- Figure 4: `src/figure4_shap_analysis.py`, `src/figure4_shap_panels.py`, `src/figure4_shapelet_examples.py`, and `src/figure4_interpretability.py`

The generated files are written to local `results/` and `shapelets/` directories, which are ignored by git.

## Citation

Citation will be added after publication.

## Acknowledgements

We acknowledge BorgwardtLab for releasing S3M (https://github.com/BorgwardtLab/S3M), which is used as the temporal pattern mining backend. Parts of `src/utils/` are adapted from the S3M implementation.

## License

MIT
