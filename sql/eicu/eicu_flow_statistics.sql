-- ============================================================================
-- eICU cohort flow statistics (external validation)
-- Also rebuilds public.eicu_cohort_index for downstream extraction.
-- Unit of counting: patientunitstayid
-- ============================================================================

SET search_path TO eicu_crd, public;

DROP TABLE IF EXISTS public.eicu_flow_statistics;
CREATE TABLE public.eicu_flow_statistics (
    step_number INT,
    step_description VARCHAR(200),
    remaining_n BIGINT,
    excluded_n BIGINT,
    exclusion_reason VARCHAR(200)
);

-- Step 0: all ICU stays in eICU
DROP TABLE IF EXISTS tmp_eicu_s0;
CREATE TEMP TABLE tmp_eicu_s0 AS
SELECT
    p.patientunitstayid,
    p.unitdischargeoffset,
    CASE
        WHEN trim(p.age) = '> 89' THEN 90
        WHEN trim(p.age) ~ '^[0-9]+$' THEN trim(p.age)::int
        ELSE NULL
    END AS age_num
FROM patient p;

INSERT INTO public.eicu_flow_statistics
SELECT
    0,
    'All ICU stays in eICU',
    COUNT(*),
    0,
    NULL
FROM tmp_eicu_s0;

-- Step 1: adult + ICU LOS >=24h
DROP TABLE IF EXISTS tmp_eicu_s1;
CREATE TEMP TABLE tmp_eicu_s1 AS
SELECT *
FROM tmp_eicu_s0
WHERE age_num >= 18
  AND unitdischargeoffset >= 24 * 60;

INSERT INTO public.eicu_flow_statistics
SELECT
    1,
    'Adult and ICU LOS >=24h',
    COUNT(*),
    (SELECT remaining_n FROM public.eicu_flow_statistics WHERE step_number = 0) - COUNT(*),
    'Age <18 or ICU LOS <24h or invalid age'
FROM tmp_eicu_s1;

-- Step 2: exclude ESRD / dialysis dependence / early RRT (48h)
DROP TABLE IF EXISTS tmp_eicu_renal_flags;
CREATE TEMP TABLE tmp_eicu_renal_flags AS
SELECT
    s.patientunitstayid,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM diagnosis d
            WHERE d.patientunitstayid = s.patientunitstayid
              AND (
                    d.diagnosisstring ILIKE '%chronic kidney disease|Stage 4%'
                 OR d.diagnosisstring ILIKE '%chronic kidney disease|Stage 5%'
              )
        ) THEN 1 ELSE 0
    END AS ckd45_dx_flag,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM diagnosis d
            WHERE d.patientunitstayid = s.patientunitstayid
              AND (
                    d.diagnosisstring ILIKE '%ESRD (end stage renal disease)%'
                 OR d.diagnosisstring ILIKE '%end stage renal disease%'
                 OR d.diagnosisstring ILIKE '%esrd%'
              )
        ) THEN 1 ELSE 0
    END AS esrd_dx_flag,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM pasthistory ph
            WHERE ph.patientunitstayid = s.patientunitstayid
              AND (
                    ph.pasthistorypath ILIKE '%renal failure - hemodialysis%'
                 OR ph.pasthistorypath ILIKE '%renal failure - peritoneal dialysis%'
                 OR ph.pasthistoryvalue ILIKE '%hemodialysis%'
                 OR ph.pasthistoryvalue ILIKE '%peritoneal dialysis%'
                 OR ph.pasthistoryvalue ILIKE '%ESRD%'
              )
        ) THEN 1 ELSE 0
    END AS dialysis_dependency_flag,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM treatment t
            WHERE t.patientunitstayid = s.patientunitstayid
              AND t.treatmentoffset <= 48 * 60
              AND (
                    t.treatmentstring ILIKE '%dialysis%'
                 OR t.treatmentstring ILIKE '%CVVH%'
                 OR t.treatmentstring ILIKE '%CVVHD%'
                 OR t.treatmentstring ILIKE '%CRRT%'
                 OR t.treatmentstring ILIKE '%SLED%'
                 OR t.treatmentstring ILIKE '%ultrafiltration%'
              )
        ) THEN 1 ELSE 0
    END AS early_rrt_flag
FROM tmp_eicu_s1 s;

DROP TABLE IF EXISTS tmp_eicu_s2;
CREATE TEMP TABLE tmp_eicu_s2 AS
SELECT
    f.patientunitstayid
FROM tmp_eicu_renal_flags f
WHERE f.esrd_dx_flag = 0
  AND f.dialysis_dependency_flag = 0
  AND f.early_rrt_flag = 0;

INSERT INTO public.eicu_flow_statistics
SELECT
    2,
    'Exclude ESRD / dialysis dependence / early RRT within 48h',
    COUNT(*),
    (SELECT remaining_n FROM public.eicu_flow_statistics WHERE step_number = 1) - COUNT(*),
    'ESRD or dialysis dependency or early RRT'
FROM tmp_eicu_s2;

-- Creatinine sequence in eligible adult ICU stays
DROP TABLE IF EXISTS tmp_eicu_creat_seq;
CREATE TEMP TABLE tmp_eicu_creat_seq AS
WITH creat_raw AS (
    SELECT
        l.patientunitstayid,
        l.labresultoffset::int AS offset_min,
        l.labresult::numeric AS creat,
        COALESCE(l.labresultrevisedoffset, l.labresultoffset)::int AS revised_offset,
        l.labid
    FROM lab l
    JOIN tmp_eicu_s2 s
        ON s.patientunitstayid = l.patientunitstayid
    WHERE lower(l.labname) = 'creatinine'
      AND l.labresult IS NOT NULL
      AND l.labresult > 0
      AND l.labresult <= 30
),
creat_dedup AS (
    SELECT patientunitstayid, offset_min, creat
    FROM (
        SELECT
            r.*,
            ROW_NUMBER() OVER (
                PARTITION BY r.patientunitstayid, r.offset_min
                ORDER BY r.revised_offset DESC, r.labid DESC
            ) AS rn
        FROM creat_raw r
    ) t
    WHERE rn = 1
)
SELECT
    d.patientunitstayid,
    d.offset_min,
    d.creat,
    ROW_NUMBER() OVER (
        PARTITION BY d.patientunitstayid
        ORDER BY d.offset_min
    ) AS seq
FROM creat_dedup d;

DROP TABLE IF EXISTS tmp_eicu_s3;
CREATE TEMP TABLE tmp_eicu_s3 AS
WITH summary AS (
    SELECT
        s.patientunitstayid,
        COUNT(*) AS n_creat,
        MIN(CASE WHEN s.seq = 1 THEN s.creat END) AS first_creat
    FROM tmp_eicu_creat_seq s
    GROUP BY s.patientunitstayid
)
SELECT
    patientunitstayid
FROM summary
WHERE n_creat >= 2
  AND first_creat <= 3.0;

INSERT INTO public.eicu_flow_statistics
SELECT
    3,
    'Creatinine n>=2 and first creatinine <=3.0',
    COUNT(*),
    (SELECT remaining_n FROM public.eicu_flow_statistics WHERE step_number = 2) - COUNT(*),
    'Creatinine missing or failed creatinine eligibility'
FROM tmp_eicu_s3;

-- KDIGO-like AKI rule on creatinine trajectories
DROP TABLE IF EXISTS tmp_eicu_first_aki;
CREATE TEMP TABLE tmp_eicu_first_aki AS
WITH aki_hits AS (
    SELECT DISTINCT
        c2.patientunitstayid,
        c2.offset_min AS aki_offset
    FROM tmp_eicu_creat_seq c1
    JOIN tmp_eicu_creat_seq c2
        ON c1.patientunitstayid = c2.patientunitstayid
       AND c2.offset_min > c1.offset_min
       AND (
            (c2.offset_min - c1.offset_min <= 48 * 60 AND c2.creat - c1.creat >= 0.3)
            OR
            (c2.offset_min - c1.offset_min <= 7 * 24 * 60 AND c2.creat >= 1.5 * c1.creat)
       )
    JOIN tmp_eicu_s3 e
        ON e.patientunitstayid = c2.patientunitstayid
)
SELECT
    patientunitstayid,
    MIN(aki_offset) AS first_aki_offset
FROM aki_hits
GROUP BY patientunitstayid;

DROP TABLE IF EXISTS tmp_eicu_aki_any;
CREATE TEMP TABLE tmp_eicu_aki_any AS
SELECT *
FROM tmp_eicu_first_aki;

DROP TABLE IF EXISTS tmp_eicu_non_aki;
CREATE TEMP TABLE tmp_eicu_non_aki AS
SELECT s2.patientunitstayid
FROM tmp_eicu_s3 s2
LEFT JOIN tmp_eicu_first_aki a
    ON a.patientunitstayid = s2.patientunitstayid
WHERE a.patientunitstayid IS NULL;

INSERT INTO public.eicu_flow_statistics
SELECT
    4,
    'AKI by creatinine rule (any time)',
    (SELECT COUNT(*) FROM tmp_eicu_aki_any),
    (SELECT COUNT(*) FROM tmp_eicu_non_aki),
    'Non-AKI'
;

-- AKI within 24-72h
DROP TABLE IF EXISTS tmp_eicu_aki_24_72;
CREATE TEMP TABLE tmp_eicu_aki_24_72 AS
SELECT
    patientunitstayid,
    first_aki_offset
FROM tmp_eicu_first_aki
WHERE first_aki_offset BETWEEN 24 * 60 AND 72 * 60;

INSERT INTO public.eicu_flow_statistics
SELECT
    5,
    'AKI onset between 24h and 72h',
    COUNT(*),
    (SELECT remaining_n FROM public.eicu_flow_statistics WHERE step_number = 4) - COUNT(*),
    'AKI onset <24h or >72h'
FROM tmp_eicu_aki_24_72;

-- Non-AKI with ICU LOS >=72h
DROP TABLE IF EXISTS tmp_eicu_non_aki_72h;
CREATE TEMP TABLE tmp_eicu_non_aki_72h AS
SELECT
    n.patientunitstayid
FROM tmp_eicu_non_aki n
JOIN tmp_eicu_s1 s1
    ON s1.patientunitstayid = n.patientunitstayid
WHERE s1.unitdischargeoffset >= 72 * 60;

INSERT INTO public.eicu_flow_statistics
SELECT
    6,
    'Non-AKI and ICU LOS >=72h',
    COUNT(*),
    (SELECT COUNT(*) FROM tmp_eicu_non_aki) - COUNT(*),
    'Non-AKI but ICU LOS <72h'
FROM tmp_eicu_non_aki_72h;

-- Final external validation cohort index
DROP TABLE IF EXISTS public.eicu_cohort_index;
CREATE TABLE public.eicu_cohort_index AS
SELECT
    a.patientunitstayid,
    1::int AS label,
    a.first_aki_offset::int AS prediction_offset_min
FROM tmp_eicu_aki_24_72 a
UNION ALL
SELECT
    n.patientunitstayid,
    0::int AS label,
    (72 * 60)::int AS prediction_offset_min
FROM tmp_eicu_non_aki_72h n;

CREATE INDEX IF NOT EXISTS idx_eicu_cohort_index_stay
    ON public.eicu_cohort_index (patientunitstayid);

ANALYZE public.eicu_cohort_index;

INSERT INTO public.eicu_flow_statistics
SELECT
    7,
    'Final eICU external validation index cohort',
    COUNT(*),
    0,
    NULL
FROM public.eicu_cohort_index;

-- Output for reproducibility checks
SELECT
    step_number,
    step_description,
    remaining_n,
    excluded_n,
    exclusion_reason
FROM public.eicu_flow_statistics
ORDER BY step_number;

-- Optional downstream checks after running eicu_external_validation*.sql:
-- SELECT COUNT(*) AS n_tabular FROM public.eicu_external_validation_tabular;
-- SELECT COUNT(DISTINCT stay_id) AS n_ts_stay FROM public.eicu_external_validation_ts_long;
