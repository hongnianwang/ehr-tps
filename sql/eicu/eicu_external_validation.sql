-- ============================================================================
-- eICU external validation extraction (aligned to MIMIC feature set)
--
-- Required input table:
--   eicu_cohort_index(patientunitstayid, label, prediction_offset_min)
--
-- Column definitions:
--   patientunitstayid      : ICU stay id
--   label                  : outcome label (e.g., AKI=1 / control=0)
--   prediction_offset_min  : minutes from ICU admit to prediction timepoint
--                            - AKI: first AKI onset offset (24h-72h)
--                            - Control: typically 72*60
--
-- Observation window:
--   [ICU admit - 6h, prediction_time - 12h]
--
-- Outputs:
--   1) eicu_external_validation_tabular
--      gender, age, race + last/max/min for all requested variables
--   2) eicu_external_validation_ts_long
--      potassium/bun/creatinine/spo2/dbp/sbp/heart_rate time series
--      resampling: vitals=1h, labs=4h
--      inclusion: each of the 7 variables has >=5 measurements in window
-- ============================================================================

SET search_path TO public, eicu_crd;

-- -----------------------------
-- 0) Clean old outputs
-- -----------------------------
DROP TABLE IF EXISTS public.eicu_external_validation_tabular;
DROP TABLE IF EXISTS public.eicu_external_validation_ts_long;

-- -----------------------------
-- 1) Cohort + demographics
-- -----------------------------
CREATE OR REPLACE TEMP VIEW ev_cohort AS
SELECT
    c.patientunitstayid::bigint AS stay_id,
    c.label::int AS label,
    c.prediction_offset_min::int AS prediction_offset_min,
    (-6 * 60) AS window_start_min,
    (c.prediction_offset_min::int - 12 * 60) AS window_end_min
FROM public.eicu_cohort_index c
JOIN patient p
    ON p.patientunitstayid = c.patientunitstayid
WHERE
    -- Keep the predefined prediction window
    c.prediction_offset_min BETWEEN 24 * 60 AND 72 * 60
    -- ICU LOS >= 24h
    AND p.unitdischargeoffset >= 24 * 60
    -- Adult only (eICU age is varchar)
    AND (
        CASE
            WHEN trim(p.age) = '> 89' THEN 90
            WHEN trim(p.age) ~ '^[0-9]+$' THEN trim(p.age)::int
            ELSE NULL
        END
    ) >= 18
    AND (c.prediction_offset_min::int - 12 * 60) >= (-6 * 60);

CREATE OR REPLACE TEMP VIEW ev_demographics AS
SELECT
    p.patientunitstayid::bigint AS stay_id,
    COALESCE(NULLIF(trim(p.gender), ''), 'Unknown') AS gender,
    CASE
        WHEN trim(p.age) = '> 89' THEN 90
        WHEN trim(p.age) ~ '^[0-9]+$' THEN trim(p.age)::int
        ELSE NULL
    END AS age,
    CASE
        -- IMPORTANT: check Caucasian before Asian, otherwise "Caucasian"
        -- would be incorrectly matched by "%asian%".
        WHEN p.ethnicity ILIKE '%caucasian%' OR p.ethnicity ILIKE '%white%' THEN 'White'
        WHEN p.ethnicity ILIKE '%asian%' THEN 'Asian'
        WHEN p.ethnicity ILIKE '%african%' THEN 'Black'
        WHEN p.ethnicity ILIKE '%hispanic%' THEN 'Hispanic'
        WHEN p.ethnicity ILIKE '%other%' AND p.ethnicity NOT ILIKE '%unknown%' THEN 'Other'
        ELSE 'Unknown'
    END AS race
FROM patient p
JOIN ev_cohort c
    ON c.stay_id = p.patientunitstayid;

-- -----------------------------
-- 2) Vital events (heart_rate/sbp/dbp/resp/spo2/temp)
--    Source: vitalperiodic + vitalaperiodic
--    Note: nursecharting fallback is intentionally disabled to reduce
--    temporary disk usage on local PostgreSQL instances.
-- -----------------------------
CREATE OR REPLACE TEMP VIEW ev_vital_events AS
WITH vp_long AS (
    SELECT c.stay_id, vp.observationoffset::int AS offset_min, 'heart_rate' AS variable, vp.heartrate::numeric AS value
    FROM ev_cohort c
    JOIN vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min

    UNION ALL
    SELECT c.stay_id, vp.observationoffset::int, 'sbp', vp.systemicsystolic::numeric
    FROM ev_cohort c
    JOIN vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min

    UNION ALL
    SELECT c.stay_id, vp.observationoffset::int, 'dbp', vp.systemicdiastolic::numeric
    FROM ev_cohort c
    JOIN vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min

    UNION ALL
    SELECT c.stay_id, vp.observationoffset::int, 'respiratory_rate', vp.respiration::numeric
    FROM ev_cohort c
    JOIN vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min

    UNION ALL
    SELECT c.stay_id, vp.observationoffset::int, 'o2_saturation', vp.sao2::numeric
    FROM ev_cohort c
    JOIN vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min

    UNION ALL
    SELECT c.stay_id, vp.observationoffset::int, 'temperature', vp.temperature::numeric
    FROM ev_cohort c
    JOIN vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
),
va_long AS (
    SELECT c.stay_id, va.observationoffset::int AS offset_min, 'sbp' AS variable, va.noninvasivesystolic::numeric AS value
    FROM ev_cohort c
    JOIN vitalaperiodic va ON va.patientunitstayid = c.stay_id
    WHERE va.observationoffset BETWEEN c.window_start_min AND c.window_end_min

    UNION ALL
    SELECT c.stay_id, va.observationoffset::int, 'dbp', va.noninvasivediastolic::numeric
    FROM ev_cohort c
    JOIN vitalaperiodic va ON va.patientunitstayid = c.stay_id
    WHERE va.observationoffset BETWEEN c.window_start_min AND c.window_end_min
),
all_vital AS (
    SELECT * FROM vp_long
    UNION ALL
    SELECT * FROM va_long
),
filtered AS (
    SELECT
        stay_id,
        offset_min,
        variable,
        CASE
            WHEN variable = 'heart_rate' AND value BETWEEN 25 AND 225 THEN value
            WHEN variable = 'sbp' AND value BETWEEN 25 AND 250 THEN value
            WHEN variable = 'dbp' AND value BETWEEN 1 AND 200 THEN value
            WHEN variable = 'respiratory_rate' AND value BETWEEN 0 AND 70 THEN value
            WHEN variable = 'o2_saturation' AND value BETWEEN 0 AND 100 THEN value
            WHEN variable = 'temperature' AND value BETWEEN 25 AND 46 THEN value
            ELSE NULL
        END AS value
    FROM all_vital
)
SELECT
    stay_id,
    offset_min,
    variable,
    AVG(value)::numeric AS value
FROM filtered
WHERE variable IS NOT NULL AND value IS NOT NULL
GROUP BY stay_id, offset_min, variable;

-- -----------------------------
-- 3) Lab events (BUN/WBC/K/...
-- -----------------------------
CREATE OR REPLACE TEMP VIEW ev_lab_events AS
WITH lab_dedup AS (
    SELECT
        c.stay_id,
        l.labresultoffset::int AS offset_min,
        l.labresult::numeric AS raw_value,
        lower(l.labname) AS labname_lc,
        ROW_NUMBER() OVER (
            PARTITION BY c.stay_id, lower(l.labname), l.labresultoffset
            ORDER BY l.labresultrevisedoffset DESC, l.labid DESC
        ) AS rn
    FROM ev_cohort c
    JOIN lab l
        ON l.patientunitstayid = c.stay_id
    WHERE
        l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
        AND l.labresult IS NOT NULL
),
lab_mapped AS (
    SELECT
        stay_id,
        offset_min,
        CASE
            WHEN labname_lc = 'bun' THEN 'bun'
            WHEN labname_lc = 'wbc x 1000' THEN 'wbc'
            WHEN labname_lc = 'potassium' THEN 'potassium'
            WHEN labname_lc = 'calcium' THEN 'calcium'
            WHEN labname_lc = 'creatinine' THEN 'creatinine'
            WHEN labname_lc IN ('glucose', 'bedside glucose') THEN 'glucose'
            WHEN labname_lc = 'magnesium' THEN 'magnesium'
            WHEN labname_lc = 'sodium' THEN 'sodium'
            WHEN labname_lc = 'hgb' THEN 'hemoglobin'
            WHEN labname_lc = 'platelets x 1000' THEN 'platelet'
            WHEN labname_lc IN ('bicarbonate', 'hco3', 'total co2') THEN 'bicarbonate'
            WHEN labname_lc = 'chloride' THEN 'chloride'
            WHEN labname_lc = 'lactate' THEN 'lactate'
            WHEN labname_lc = 'hct' THEN 'hematocrit'
            WHEN labname_lc = 'rbc' THEN 'rbc'
            ELSE NULL
        END AS variable,
        raw_value AS value
    FROM lab_dedup
    WHERE rn = 1
),
filtered AS (
    SELECT
        stay_id,
        offset_min,
        variable,
        CASE
            WHEN variable = 'bun' AND value > 0 AND value <= 300 THEN value
            WHEN variable = 'wbc' AND value > 0 AND value <= 1000 THEN value
            WHEN variable = 'potassium' AND value > 0 AND value <= 30 THEN value
            WHEN variable = 'calcium' AND value > 0 AND value <= 10000 THEN value
            WHEN variable = 'creatinine' AND value > 0 AND value <= 30 THEN value
            WHEN variable = 'glucose' AND value > 0 AND value <= 30000 THEN value
            WHEN variable = 'magnesium' AND value > 0 AND value <= 10000 THEN value
            WHEN variable = 'sodium' AND value > 0 AND value <= 200 THEN value
            WHEN variable = 'hemoglobin' AND value > 0 AND value <= 50 THEN value
            WHEN variable = 'platelet' AND value > 0 AND value <= 10000 THEN value
            WHEN variable = 'bicarbonate' AND value > 0 AND value <= 10000 THEN value
            WHEN variable = 'chloride' AND value > 0 AND value <= 10000 THEN value
            WHEN variable = 'lactate' AND value > 0 AND value <= 10000 THEN value
            WHEN variable = 'hematocrit' AND value > 0 AND value <= 100 THEN value
            WHEN variable = 'rbc' AND value > 0 AND value <= 10 THEN value
            ELSE NULL
        END AS value
    FROM lab_mapped
)
SELECT
    stay_id,
    offset_min,
    variable,
    AVG(value)::numeric AS value
FROM filtered
WHERE variable IS NOT NULL AND value IS NOT NULL
GROUP BY stay_id, offset_min, variable;

-- -----------------------------
-- 4) Union all cleaned events
-- -----------------------------
CREATE OR REPLACE TEMP VIEW ev_all_events AS
SELECT stay_id, offset_min, variable, value FROM ev_vital_events
UNION ALL
SELECT stay_id, offset_min, variable, value FROM ev_lab_events;

-- -----------------------------
-- 5) Tabular feature output
-- -----------------------------
CREATE TABLE public.eicu_external_validation_tabular AS
WITH last_vals AS (
    SELECT stay_id, variable, value AS last_value
    FROM (
        SELECT
            stay_id,
            variable,
            value,
            offset_min,
            ROW_NUMBER() OVER (PARTITION BY stay_id, variable ORDER BY offset_min DESC) AS rn
        FROM ev_all_events
    ) t
    WHERE rn = 1
),
stat_vals AS (
    SELECT
        stay_id,
        variable,
        MAX(value) AS max_value,
        MIN(value) AS min_value
    FROM ev_all_events
    GROUP BY stay_id, variable
)
SELECT
    c.stay_id,
    c.label,
    d.gender,
    d.age,
    d.race,

    MAX(CASE WHEN lv.variable = 'heart_rate' THEN lv.last_value END) AS heart_rate_last,
    MAX(CASE WHEN lv.variable = 'sbp' THEN lv.last_value END) AS sbp_last,
    MAX(CASE WHEN lv.variable = 'dbp' THEN lv.last_value END) AS dbp_last,
    MAX(CASE WHEN lv.variable = 'respiratory_rate' THEN lv.last_value END) AS respiratory_rate_last,
    MAX(CASE WHEN lv.variable = 'o2_saturation' THEN lv.last_value END) AS o2_saturation_last,
    MAX(CASE WHEN lv.variable = 'temperature' THEN lv.last_value END) AS temperature_last,

    MAX(CASE WHEN lv.variable = 'bun' THEN lv.last_value END) AS bun_last,
    MAX(CASE WHEN lv.variable = 'wbc' THEN lv.last_value END) AS wbc_last,
    MAX(CASE WHEN lv.variable = 'potassium' THEN lv.last_value END) AS potassium_last,
    MAX(CASE WHEN lv.variable = 'calcium' THEN lv.last_value END) AS calcium_last,
    MAX(CASE WHEN lv.variable = 'creatinine' THEN lv.last_value END) AS creatinine_last,
    MAX(CASE WHEN lv.variable = 'glucose' THEN lv.last_value END) AS glucose_last,
    MAX(CASE WHEN lv.variable = 'magnesium' THEN lv.last_value END) AS magnesium_last,
    MAX(CASE WHEN lv.variable = 'sodium' THEN lv.last_value END) AS sodium_last,
    MAX(CASE WHEN lv.variable = 'hemoglobin' THEN lv.last_value END) AS hemoglobin_last,
    MAX(CASE WHEN lv.variable = 'platelet' THEN lv.last_value END) AS platelet_last,
    MAX(CASE WHEN lv.variable = 'bicarbonate' THEN lv.last_value END) AS bicarbonate_last,
    MAX(CASE WHEN lv.variable = 'chloride' THEN lv.last_value END) AS chloride_last,
    MAX(CASE WHEN lv.variable = 'lactate' THEN lv.last_value END) AS lactate_last,
    MAX(CASE WHEN lv.variable = 'hematocrit' THEN lv.last_value END) AS hematocrit_last,
    MAX(CASE WHEN lv.variable = 'rbc' THEN lv.last_value END) AS rbc_last,

    MAX(CASE WHEN sv.variable = 'heart_rate' THEN sv.max_value END) AS heart_rate_max,
    MAX(CASE WHEN sv.variable = 'heart_rate' THEN sv.min_value END) AS heart_rate_min,
    MAX(CASE WHEN sv.variable = 'sbp' THEN sv.max_value END) AS sbp_max,
    MAX(CASE WHEN sv.variable = 'sbp' THEN sv.min_value END) AS sbp_min,
    MAX(CASE WHEN sv.variable = 'dbp' THEN sv.max_value END) AS dbp_max,
    MAX(CASE WHEN sv.variable = 'dbp' THEN sv.min_value END) AS dbp_min,
    MAX(CASE WHEN sv.variable = 'respiratory_rate' THEN sv.max_value END) AS respiratory_rate_max,
    MAX(CASE WHEN sv.variable = 'respiratory_rate' THEN sv.min_value END) AS respiratory_rate_min,
    MAX(CASE WHEN sv.variable = 'o2_saturation' THEN sv.max_value END) AS o2_saturation_max,
    MAX(CASE WHEN sv.variable = 'o2_saturation' THEN sv.min_value END) AS o2_saturation_min,
    MAX(CASE WHEN sv.variable = 'temperature' THEN sv.max_value END) AS temperature_max,
    MAX(CASE WHEN sv.variable = 'temperature' THEN sv.min_value END) AS temperature_min,

    MAX(CASE WHEN sv.variable = 'bun' THEN sv.max_value END) AS bun_max,
    MAX(CASE WHEN sv.variable = 'bun' THEN sv.min_value END) AS bun_min,
    MAX(CASE WHEN sv.variable = 'wbc' THEN sv.max_value END) AS wbc_max,
    MAX(CASE WHEN sv.variable = 'wbc' THEN sv.min_value END) AS wbc_min,
    MAX(CASE WHEN sv.variable = 'potassium' THEN sv.max_value END) AS potassium_max,
    MAX(CASE WHEN sv.variable = 'potassium' THEN sv.min_value END) AS potassium_min,
    MAX(CASE WHEN sv.variable = 'calcium' THEN sv.max_value END) AS calcium_max,
    MAX(CASE WHEN sv.variable = 'calcium' THEN sv.min_value END) AS calcium_min,
    MAX(CASE WHEN sv.variable = 'creatinine' THEN sv.max_value END) AS creatinine_max,
    MAX(CASE WHEN sv.variable = 'creatinine' THEN sv.min_value END) AS creatinine_min,
    MAX(CASE WHEN sv.variable = 'glucose' THEN sv.max_value END) AS glucose_max,
    MAX(CASE WHEN sv.variable = 'glucose' THEN sv.min_value END) AS glucose_min,
    MAX(CASE WHEN sv.variable = 'magnesium' THEN sv.max_value END) AS magnesium_max,
    MAX(CASE WHEN sv.variable = 'magnesium' THEN sv.min_value END) AS magnesium_min,
    MAX(CASE WHEN sv.variable = 'sodium' THEN sv.max_value END) AS sodium_max,
    MAX(CASE WHEN sv.variable = 'sodium' THEN sv.min_value END) AS sodium_min,
    MAX(CASE WHEN sv.variable = 'hemoglobin' THEN sv.max_value END) AS hemoglobin_max,
    MAX(CASE WHEN sv.variable = 'hemoglobin' THEN sv.min_value END) AS hemoglobin_min,
    MAX(CASE WHEN sv.variable = 'platelet' THEN sv.max_value END) AS platelet_max,
    MAX(CASE WHEN sv.variable = 'platelet' THEN sv.min_value END) AS platelet_min,
    MAX(CASE WHEN sv.variable = 'bicarbonate' THEN sv.max_value END) AS bicarbonate_max,
    MAX(CASE WHEN sv.variable = 'bicarbonate' THEN sv.min_value END) AS bicarbonate_min,
    MAX(CASE WHEN sv.variable = 'chloride' THEN sv.max_value END) AS chloride_max,
    MAX(CASE WHEN sv.variable = 'chloride' THEN sv.min_value END) AS chloride_min,
    MAX(CASE WHEN sv.variable = 'lactate' THEN sv.max_value END) AS lactate_max,
    MAX(CASE WHEN sv.variable = 'lactate' THEN sv.min_value END) AS lactate_min,
    MAX(CASE WHEN sv.variable = 'hematocrit' THEN sv.max_value END) AS hematocrit_max,
    MAX(CASE WHEN sv.variable = 'hematocrit' THEN sv.min_value END) AS hematocrit_min,
    MAX(CASE WHEN sv.variable = 'rbc' THEN sv.max_value END) AS rbc_max,
    MAX(CASE WHEN sv.variable = 'rbc' THEN sv.min_value END) AS rbc_min

FROM ev_cohort c
LEFT JOIN ev_demographics d ON d.stay_id = c.stay_id
LEFT JOIN last_vals lv ON lv.stay_id = c.stay_id
LEFT JOIN stat_vals sv ON sv.stay_id = c.stay_id
GROUP BY c.stay_id, c.label, d.gender, d.age, d.race
ORDER BY c.stay_id;

-- -----------------------------
-- 6) Time series output (7 variables)
--    vital: heart_rate/sbp/dbp/o2_saturation -> 1h
--    lab  : bun/creatinine/potassium         -> 4h
-- -----------------------------
CREATE TABLE public.eicu_external_validation_ts_long AS
WITH ts_events AS (
    SELECT stay_id, offset_min, variable, value
    FROM ev_all_events
    WHERE variable IN (
        'heart_rate', 'sbp', 'dbp', 'o2_saturation',
        'bun', 'creatinine', 'potassium'
    )
),
counts AS (
    SELECT stay_id, variable, COUNT(*) AS n_obs
    FROM ts_events
    GROUP BY stay_id, variable
),
eligible_stay AS (
    SELECT stay_id
    FROM counts
    GROUP BY stay_id
    HAVING COUNT(*) = 7 AND MIN(n_obs) >= 5
),
resampled AS (
    SELECT
        c.stay_id,
        c.label,
        t.variable,
        CASE
            WHEN t.variable IN ('heart_rate', 'sbp', 'dbp', 'o2_saturation') THEN 1
            ELSE 4
        END AS resample_hours,
        CASE
            WHEN t.variable IN ('heart_rate', 'sbp', 'dbp', 'o2_saturation')
                THEN FLOOR(t.offset_min / 60.0)::int * 60
            ELSE FLOOR(t.offset_min / 240.0)::int * 240
        END AS bin_offset_min,
        AVG(t.value)::numeric AS value
    FROM ts_events t
    JOIN eligible_stay e ON e.stay_id = t.stay_id
    JOIN ev_cohort c ON c.stay_id = t.stay_id
    GROUP BY
        c.stay_id,
        c.label,
        t.variable,
        CASE
            WHEN t.variable IN ('heart_rate', 'sbp', 'dbp', 'o2_saturation') THEN 1
            ELSE 4
        END,
        CASE
            WHEN t.variable IN ('heart_rate', 'sbp', 'dbp', 'o2_saturation')
                THEN FLOOR(t.offset_min / 60.0)::int * 60
            ELSE FLOOR(t.offset_min / 240.0)::int * 240
        END
)
SELECT
    stay_id,
    label,
    variable,
    resample_hours,
    bin_offset_min,
    (bin_offset_min / 60.0)::numeric(8,2) AS bin_hour_from_icu,
    value
FROM resampled
ORDER BY stay_id, variable, bin_offset_min;

-- -----------------------------
-- 7) Quick sanity checks
-- -----------------------------
-- SELECT COUNT(*) AS n_tabular FROM public.eicu_external_validation_tabular;
-- SELECT variable, resample_hours, COUNT(*) AS n_points
-- FROM public.eicu_external_validation_ts_long
-- GROUP BY variable, resample_hours
-- ORDER BY variable;
