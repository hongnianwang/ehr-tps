#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${DB_NAME:-mimiciv3.0}"
OUT_DIR="${OUT_DIR:-results/cohort_audit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}"

psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -f sql/mimic/cohort.sql
psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -f sql/mimic/mimic_flow_statistics.sql

psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -c "\copy (
  SELECT step_number, step_description, patient_count, excluded_count, exclusion_reason
  FROM exclusion_statistics
  ORDER BY step_number
) TO '${OUT_DIR}/mimic_exclusion_statistics_from_cohort_sql.csv' WITH (FORMAT CSV, HEADER true)"

psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -c "\copy (
  SELECT step_number, step_description, remaining_n, excluded_n, exclusion_reason
  FROM mimic_flow_statistics
  ORDER BY step_number
) TO '${OUT_DIR}/mimic_flow_statistics.csv' WITH (FORMAT CSV, HEADER true)"

psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -c "\copy (
  SELECT metric, n
  FROM mimic_flow_step5_detail
  ORDER BY metric
) TO '${OUT_DIR}/mimic_flow_step5_detail.csv' WITH (FORMAT CSV, HEADER true)"

psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -c "\copy (
  SELECT 'aki_cohort_final' AS table_name, COUNT(DISTINCT stay_id) AS n
  FROM aki_cohort_final
  UNION ALL
  SELECT 'control_cohort_final', COUNT(DISTINCT stay_id)
  FROM control_cohort_final
) TO '${OUT_DIR}/mimic_final_candidate_counts_from_cohort_sql.csv' WITH (FORMAT CSV, HEADER true)"

psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -c "\copy (
  SELECT
    COUNT(*) AS post_los_stays,
    COUNT(*) FILTER (WHERE ckd_stage4to5_flag = 1) AS ckd45,
    COUNT(*) FILTER (WHERE esrd_flag = 1) AS esrd,
    COUNT(*) FILTER (WHERE dialysis_flag = 1) AS dialysis,
    COUNT(*) FILTER (WHERE early_rrt_flag = 1) AS early_rrt,
    COUNT(*) FILTER (WHERE renal_exclusion_flag = 1) AS excluded_esrd_dialysis_early_rrt
  FROM flagged_patients
) TO '${OUT_DIR}/mimic_renal_flag_counts_from_cohort_sql.csv' WITH (FORMAT CSV, HEADER true)"

echo "MIMIC SQL cohort audit written to ${OUT_DIR}"
