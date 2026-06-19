#!/usr/bin/env bash
set -euo pipefail

# 1) Build balanced intersection raw files from long time-series tables.
python src/01_make_mimic_intersection_raw.py \
  --positive-ts data/mimiciv/data-aki_ts.csv \
  --negative-ts data/mimiciv/data-control_ts.csv \
  --output-dir data/mimiciv/raw \
  --min-count 5

# 2) Convert raw intersection files into S3M-ready Train/Test matrices.
#    Vital-sign matrices feed src/02_run_s3m_vital.py; lab matrices feed
#    src/02_run_s3m_lab.py. Labels are written under data/mimiciv/.
#    The locked split keeps 168 cases per class in the test set when the
#    balanced development cohort has 835 cases per class.
python src/01_process_mimic_raw_to_s3m.py \
  --raw-dir data/mimiciv/raw \
  --processed-dir data/mimiciv/processed \
  --vital-dir data/mimiciv/ts_vital \
  --lab-dir data/mimiciv/ts_lab \
  --split-anchor-file dbp_combined_min5_intersection.csv \
  --test-size 0.20119760479041916 \
  --random-state 42 \
  --save-flat-labels
