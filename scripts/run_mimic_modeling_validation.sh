#!/usr/bin/env bash
set -euo pipefail

python src/build_mimic_modeling_dataset.py \
  --data-out-dir data/mimiciv/processed_modeling_cohort \
  --out-dir results/mimic_modeling_cohort_experiment

python src/run_mimic_internal_external_validation.py \
  --train-file data/mimiciv/processed_modeling_cohort/processed_train_with_shapelets.csv \
  --internal-test-file data/mimiciv/processed_modeling_cohort/processed_test_with_shapelets.csv \
  --external-test-file data/eicu/processed/processed_test_with_shapelets.csv \
  --out-dir results/mimic_internal_external_validation_selected
