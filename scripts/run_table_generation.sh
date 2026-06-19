#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <internal_pred_dir> <external_pred_dir>"
  exit 1
fi

python src/generate_table2_table3_performance.py \
  --internal_pred_dir "$1" \
  --external_pred_dir "$2" \
  --outdir results/paper_tables
