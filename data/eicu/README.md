# eICU Inputs

Required access: PhysioNet credentialed eICU-CRD.

Typical processed files used by this project:
- `processed_test_with_shapelets.csv`
- `data-test_tabular.csv`
- `ts_stay_order.csv`
- `shapelet_csv_results/Test_*_metrics.csv`

Key SQL for external validation cohort/index and extraction:
- `sql/eicu/eicu_flow_statistics.sql`
- `sql/eicu/eicu_build_cohort_index.sql`
- `sql/eicu/eicu_external_validation.sql`

Run SQL in PostgreSQL with the `eicu_crd` schema available.

Do not commit patient-level files to git.
