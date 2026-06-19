-- ============================================================================
-- Build eicu_cohort_index for external validation extraction
-- Output table: public.eicu_cohort_index
--   patientunitstayid, label, prediction_offset_min
--
-- Rules (KDIGO creatinine-based, ICU timeline approximation):
--   1) adult (age >=18), ICU LOS >=24h
--   2) exclude ESRD / dialysis dependence / early RRT within 48h
--   3) creatinine measurements >=2, first creatinine <=3.0 mg/dL
--   3) AKI event if either:
--      a) creat(t2) - creat(t1) >= 0.3 within 48h
--      b) creat(t2) >= 1.5 * creat(t1) within 7d
--   4) AKI cohort: first AKI offset in [24h,72h]
--      prediction_offset_min = first_aki_offset
--   5) Non-AKI cohort: no AKI event, ICU LOS >=72h
--      prediction_offset_min = 72h
-- ============================================================================

SET search_path TO eicu_crd, public;

DROP TABLE IF EXISTS public.eicu_cohort_index;

CREATE TABLE public.eicu_cohort_index AS
WITH adult_icu AS (
    SELECT
        p.patientunitstayid,
        p.unitdischargeoffset,
        CASE
            WHEN trim(p.age) = '> 89' THEN 90
            WHEN trim(p.age) ~ '^[0-9]+$' THEN trim(p.age)::int
            ELSE NULL
        END AS age_num
    FROM patient p
    WHERE p.unitdischargeoffset >= 24 * 60
),
renal_flags AS (
    SELECT
        a.patientunitstayid,
        a.unitdischargeoffset,
        a.age_num,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM diagnosis d
                WHERE d.patientunitstayid = a.patientunitstayid
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
                WHERE d.patientunitstayid = a.patientunitstayid
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
                WHERE ph.patientunitstayid = a.patientunitstayid
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
                WHERE t.patientunitstayid = a.patientunitstayid
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
    FROM adult_icu a
),
adult_icu_eligible AS (
    SELECT
        rf.patientunitstayid,
        rf.unitdischargeoffset,
        rf.age_num
    FROM renal_flags rf
    WHERE rf.age_num >= 18
      AND rf.esrd_dx_flag = 0
      AND rf.dialysis_dependency_flag = 0
      AND rf.early_rrt_flag = 0
),
creat_raw AS (
    SELECT
        l.patientunitstayid,
        l.labresultoffset::int AS offset_min,
        l.labresult::numeric AS creat,
        COALESCE(l.labresultrevisedoffset, l.labresultoffset)::int AS revised_offset,
        l.labid
    FROM lab l
    JOIN adult_icu_eligible a
        ON a.patientunitstayid = l.patientunitstayid
    WHERE
        lower(l.labname) = 'creatinine'
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
),
creat_seq AS (
    SELECT
        d.patientunitstayid,
        d.offset_min,
        d.creat,
        ROW_NUMBER() OVER (
            PARTITION BY d.patientunitstayid ORDER BY d.offset_min
        ) AS seq
    FROM creat_dedup d
),
creat_summary AS (
    SELECT
        s.patientunitstayid,
        COUNT(*) AS n_creat,
        MIN(CASE WHEN s.seq = 1 THEN s.creat END) AS first_creat
    FROM creat_seq s
    GROUP BY s.patientunitstayid
),
eligible AS (
    SELECT
        a.patientunitstayid,
        a.unitdischargeoffset
    FROM adult_icu_eligible a
    JOIN creat_summary cs
        ON cs.patientunitstayid = a.patientunitstayid
    WHERE
        cs.n_creat >= 2
        AND cs.first_creat <= 3.0
),
aki_hits AS (
    SELECT DISTINCT
        c2.patientunitstayid,
        c2.offset_min AS aki_offset
    FROM creat_seq c1
    JOIN creat_seq c2
        ON c1.patientunitstayid = c2.patientunitstayid
       AND c2.offset_min > c1.offset_min
       AND (
            (c2.offset_min - c1.offset_min <= 48 * 60 AND c2.creat - c1.creat >= 0.3)
            OR
            (c2.offset_min - c1.offset_min <= 7 * 24 * 60 AND c2.creat >= 1.5 * c1.creat)
       )
    JOIN eligible e
        ON e.patientunitstayid = c2.patientunitstayid
),
first_aki AS (
    SELECT
        patientunitstayid,
        MIN(aki_offset) AS first_aki_offset
    FROM aki_hits
    GROUP BY patientunitstayid
),
aki_cohort AS (
    SELECT
        e.patientunitstayid,
        1::int AS label,
        f.first_aki_offset::int AS prediction_offset_min
    FROM eligible e
    JOIN first_aki f
        ON f.patientunitstayid = e.patientunitstayid
    WHERE f.first_aki_offset BETWEEN 24 * 60 AND 72 * 60
),
non_aki_cohort AS (
    SELECT
        e.patientunitstayid,
        0::int AS label,
        (72 * 60)::int AS prediction_offset_min
    FROM eligible e
    LEFT JOIN first_aki f
        ON f.patientunitstayid = e.patientunitstayid
    WHERE
        f.patientunitstayid IS NULL
        AND e.unitdischargeoffset >= 72 * 60
)
SELECT * FROM aki_cohort
UNION ALL
SELECT * FROM non_aki_cohort;

CREATE INDEX IF NOT EXISTS idx_eicu_cohort_index_stay
    ON public.eicu_cohort_index (patientunitstayid);

ANALYZE public.eicu_cohort_index;

-- Quick summary
SELECT label, COUNT(*) AS n
FROM public.eicu_cohort_index
GROUP BY label
ORDER BY label;
