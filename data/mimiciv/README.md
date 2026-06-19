# MIMIC-IV Inputs

Required access: PhysioNet credentialed MIMIC-IV.

Current MIMIC-IV data layers:

- `data-aki_tabular.csv`, `data-control_tabular.csv`, `data-aki_ts.csv`, `data-control_ts.csv`: local SQL-extracted candidate cohort files. Do not commit these patient-level files.
- `processed/`: 835 AKI / 835 non-AKI shapelet-discovery layer.
- `processed_modeling_cohort/`: current modeling layer, created with `src/build_mimic_modeling_dataset.py`.

Key modeling-cohort output files:

- `processed_modeling_cohort/processed_data_modeling_cohort.csv`
- `processed_modeling_cohort/processed_train_with_shapelets.csv`
- `processed_modeling_cohort/processed_test_with_shapelets.csv`
- `processed_modeling_cohort/split_counts.csv`
- `processed_modeling_cohort/eligible_control_ids.csv`

Key SQL for cohort construction and flow statistics:
- `sql/mimic/cohort.sql`
- `sql/mimic/mimic_flow_statistics.sql`

`sql/mimic/cohort.sql` builds the full AKI and non-AKI candidate cohorts. It does not directly output the final modeling set. The modeling layer is built downstream from the locked shapelet-discovery files, with all AKI cases from that layer retained and non-AKI controls filtered by the locked temporal-data requirements and ICU LOS. The resulting sample count is audited in `split_counts.csv`.

Run SQL in PostgreSQL with MIMIC-IV schemas (`mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`) available.

Do not commit patient-level files to git.
