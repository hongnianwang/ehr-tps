#!/usr/bin/env bash
set -euo pipefail

python src/04_external_validate_eicu.py \
  --train_file data/mimiciv/processed/processed_train_with_shapelets.csv \
  --test_file data/eicu/processed/processed_test_with_shapelets.csv \
  --outdir results/external_validation_eicu
