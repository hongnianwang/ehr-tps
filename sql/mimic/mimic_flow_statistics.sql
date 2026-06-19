-- ============================================================================
-- MIMIC cohort flow statistics (cohort.sql-consistent version)
-- Unit of counting:
--   - Step 0: subject_id
--   - Step 1+: ICU stay_id
--
-- NOTE:
-- This script intentionally follows cohort.sql behavior for reproducibility,
-- including Step 2 counting logic and the downstream carry-over.
-- ============================================================================

DROP TABLE IF EXISTS public.mimic_flow_statistics;
CREATE TABLE public.mimic_flow_statistics (
    step_number INT,
    step_description VARCHAR(200),
    remaining_n BIGINT,
    excluded_n BIGINT,
    exclusion_reason VARCHAR(200)
);

DROP TABLE IF EXISTS public.mimic_flow_step5_detail;
CREATE TABLE public.mimic_flow_step5_detail (
    metric VARCHAR(120),
    n BIGINT
);

-- Step 0: all adult subjects
INSERT INTO public.mimic_flow_statistics
SELECT
    0,
    'MIMIC-IV数据库所有成年患者',
    COUNT(DISTINCT p.subject_id),
    0,
    NULL
FROM mimiciv_hosp.patients p
WHERE p.anchor_age >= 18;

-- Base adult ICU stays
DROP TABLE IF EXISTS tmp_mimic_base_patients;
CREATE TEMP TABLE tmp_mimic_base_patients AS
SELECT
    pat.subject_id,
    adm.hadm_id,
    icu.stay_id,
    icu.intime AS icu_intime,
    icu.outtime AS icu_outtime
FROM mimiciv_hosp.patients pat
JOIN mimiciv_hosp.admissions adm
    ON adm.subject_id = pat.subject_id
JOIN mimiciv_icu.icustays icu
    ON icu.hadm_id = adm.hadm_id
WHERE pat.anchor_age >= 18;

INSERT INTO public.mimic_flow_statistics
SELECT
    1,
    '成年患者ICU入住记录',
    COUNT(DISTINCT stay_id),
    (SELECT remaining_n FROM public.mimic_flow_statistics WHERE step_number = 0) - COUNT(DISTINCT subject_id),
    '无ICU入住记录'
FROM tmp_mimic_base_patients;

-- Step 2 follows cohort.sql counting:
-- eligible: LOS>=24h; excluded: LOS<24h (NULL LOS counted in neither branch)
INSERT INTO public.mimic_flow_statistics
WITH icu_duration_stats AS (
    SELECT
        COUNT(DISTINCT CASE WHEN (icu_outtime - icu_intime) >= INTERVAL '24 hours' THEN stay_id END) AS eligible_count,
        COUNT(DISTINCT CASE WHEN (icu_outtime - icu_intime) < INTERVAL '24 hours' THEN stay_id END) AS excluded_count
    FROM tmp_mimic_base_patients
)
SELECT
    2,
    'ICU停留时间≥24小时',
    eligible_count,
    excluded_count,
    'ICU停留时间<24小时'
FROM icu_duration_stats;

DELETE FROM tmp_mimic_base_patients
WHERE (icu_outtime - icu_intime) < INTERVAL '24 hours';

-- Renal flags (eligibility excludes ESRD / dialysis dependence / early RRT;
-- CKD stage 4-5 is recorded for audit only and is not an exclusion criterion)
DROP TABLE IF EXISTS tmp_mimic_flagged_patients;
CREATE TEMP TABLE tmp_mimic_flagged_patients AS
WITH exclusion_flags AS (
    SELECT
        bp.subject_id,
        bp.hadm_id,
        bp.stay_id,
        MAX(CASE WHEN EXISTS (
            SELECT 1
            FROM mimiciv_hosp.diagnoses_icd d
            WHERE d.hadm_id = bp.hadm_id
              AND (
                    (d.icd_version = 9 AND SUBSTR(d.icd_code, 1, 4) IN ('5854', '5855'))
                 OR (d.icd_version = 10 AND SUBSTR(d.icd_code, 1, 4) IN ('N184', 'N185'))
              )
        ) THEN 1 ELSE 0 END) AS ckd_stage4to5_flag,
        MAX(CASE WHEN EXISTS (
            SELECT 1
            FROM mimiciv_hosp.diagnoses_icd d
            WHERE d.hadm_id = bp.hadm_id
              AND (
                    (d.icd_version = 9 AND SUBSTR(d.icd_code, 1, 4) = '5856')
                 OR (d.icd_version = 10 AND SUBSTR(d.icd_code, 1, 4) = 'N186')
              )
        ) THEN 1 ELSE 0 END) AS esrd_flag,
        MAX(CASE WHEN EXISTS (
            SELECT 1
            FROM mimiciv_hosp.diagnoses_icd d
            WHERE d.hadm_id = bp.hadm_id
              AND (
                    (d.icd_version = 9 AND (d.icd_code = 'V4511' OR d.icd_code LIKE 'V56%'))
                 OR (d.icd_version = 10 AND d.icd_code = 'Z992')
              )
        ) THEN 1 ELSE 0 END) AS dialysis_flag,
        MAX(CASE WHEN EXISTS (
            SELECT 1
            FROM mimiciv_icu.procedureevents pe
            JOIN mimiciv_icu.icustays ic
                ON ic.stay_id = pe.stay_id
            WHERE pe.stay_id = bp.stay_id
              AND pe.itemid IN (225802, 225803, 225805, 224270, 225809, 225955)
              AND pe.starttime <= (ic.intime + INTERVAL '48 hours')
        ) THEN 1 ELSE 0 END) AS early_rrt_flag
    FROM tmp_mimic_base_patients bp
    GROUP BY bp.subject_id, bp.hadm_id, bp.stay_id
)
SELECT
    bp.*,
    ef.ckd_stage4to5_flag,
    ef.esrd_flag,
    ef.dialysis_flag,
    ef.early_rrt_flag,
    CASE
        WHEN COALESCE(ef.esrd_flag, 0) = 1
          OR COALESCE(ef.dialysis_flag, 0) = 1
          OR COALESCE(ef.early_rrt_flag, 0) = 1
        THEN 1 ELSE 0
    END AS renal_exclusion_flag,
    CASE
        WHEN COALESCE(ef.esrd_flag, 0) = 0
          AND COALESCE(ef.dialysis_flag, 0) = 0
          AND COALESCE(ef.early_rrt_flag, 0) = 0
        THEN 1 ELSE 0
    END AS is_eligible
FROM tmp_mimic_base_patients bp
JOIN exclusion_flags ef
    ON ef.subject_id = bp.subject_id
   AND ef.hadm_id = bp.hadm_id
   AND ef.stay_id = bp.stay_id;

INSERT INTO public.mimic_flow_statistics
SELECT
    3,
    '排除ESRD/透析/早期RRT患者',
    SUM(CASE WHEN renal_exclusion_flag = 0 THEN 1 ELSE 0 END),
    SUM(CASE WHEN renal_exclusion_flag = 1 THEN 1 ELSE 0 END),
    'ESRD/透析依赖/早期RRT（48小时内）；CKD4-5仅记录不排除'
FROM tmp_mimic_flagged_patients;

-- Step 4: creatinine availability among eligible stays
INSERT INTO public.mimic_flow_statistics
WITH creat_data AS (
    SELECT
        COUNT(DISTINCT fp.stay_id) AS total_eligible,
        COUNT(DISTINCT k.stay_id) AS with_creat
    FROM tmp_mimic_flagged_patients fp
    LEFT JOIN mimiciv_derived.kdigo_stages_1 k
        ON k.stay_id = fp.stay_id
    WHERE fp.is_eligible = 1
)
SELECT
    4,
    '有肌酐测量数据',
    with_creat,
    total_eligible - with_creat,
    '无肌酐测量'
FROM creat_data;

DROP TABLE IF EXISTS tmp_mimic_eligible_stays;
CREATE TEMP TABLE tmp_mimic_eligible_stays AS
SELECT
    stay_id,
    subject_id,
    hadm_id,
    icu_intime,
    icu_outtime
FROM tmp_mimic_flagged_patients
WHERE is_eligible = 1;

DROP TABLE IF EXISTS tmp_mimic_first_creat_times;
CREATE TEMP TABLE tmp_mimic_first_creat_times AS
SELECT
    k.stay_id,
    MIN(k.charttime) AS first_charttime
FROM mimiciv_derived.kdigo_stages_1 k
WHERE k.stay_id IN (SELECT stay_id FROM tmp_mimic_eligible_stays)
GROUP BY k.stay_id;

DROP TABLE IF EXISTS tmp_mimic_creatinine_summary;
CREATE TEMP TABLE tmp_mimic_creatinine_summary AS
SELECT
    k.stay_id,
    COUNT(k.creat) AS creatinine_count,
    MAX(CASE WHEN k.charttime = fct.first_charttime THEN k.creat ELSE NULL END) AS first_creatinine
FROM mimiciv_derived.kdigo_stages_1 k
JOIN tmp_mimic_first_creat_times fct
    ON fct.stay_id = k.stay_id
   AND k.charttime >= fct.first_charttime
GROUP BY k.stay_id;

DROP TABLE IF EXISTS tmp_mimic_creatinine_eligible;
CREATE TEMP TABLE tmp_mimic_creatinine_eligible AS
SELECT stay_id
FROM tmp_mimic_creatinine_summary
WHERE first_creatinine <= 3.0
  AND creatinine_count >= 2;

DROP TABLE IF EXISTS tmp_mimic_b_patients_kdigo;
CREATE TEMP TABLE tmp_mimic_b_patients_kdigo AS
SELECT
    e.stay_id,
    e.subject_id,
    e.hadm_id,
    k.charttime,
    k.creat AS serum_creatinine,
    k.aki_stage_creat AS aki_stage
FROM tmp_mimic_eligible_stays e
JOIN tmp_mimic_creatinine_eligible ce
    ON ce.stay_id = e.stay_id
JOIN mimiciv_derived.kdigo_stages_1 k
    ON k.stay_id = e.stay_id
WHERE k.charttime BETWEEN e.icu_intime AND e.icu_outtime;

DROP TABLE IF EXISTS tmp_mimic_step5_stays;
CREATE TEMP TABLE tmp_mimic_step5_stays AS
SELECT DISTINCT stay_id
FROM tmp_mimic_b_patients_kdigo;

INSERT INTO public.mimic_flow_step5_detail
SELECT 'first_creatinine_gt_3.0', COUNT(*)
FROM tmp_mimic_creatinine_summary
WHERE first_creatinine > 3.0
UNION ALL
SELECT 'creatinine_count_lt_2', COUNT(*)
FROM tmp_mimic_creatinine_summary
WHERE creatinine_count < 2
UNION ALL
SELECT 'overlap_gt3_and_lt2', COUNT(*)
FROM tmp_mimic_creatinine_summary
WHERE first_creatinine > 3.0
  AND creatinine_count < 2
UNION ALL
SELECT 'excluded_union_gt3_or_lt2', COUNT(*)
FROM tmp_mimic_creatinine_summary
WHERE first_creatinine > 3.0
   OR creatinine_count < 2
UNION ALL
SELECT
    'pass_threshold_but_no_in_icu_creatinine_row',
    (SELECT COUNT(*) FROM tmp_mimic_creatinine_eligible) - (SELECT COUNT(*) FROM tmp_mimic_step5_stays);

INSERT INTO public.mimic_flow_statistics
SELECT
    5,
    '首次肌酐<=3.0且测量>=2次',
    COUNT(DISTINCT stay_id),
    (SELECT remaining_n FROM public.mimic_flow_statistics WHERE step_number = 4) - COUNT(DISTINCT stay_id),
    '首次肌酐>3.0或测量<2次（含ICU窗内无肌酐行）'
FROM tmp_mimic_step5_stays;

INSERT INTO public.mimic_flow_statistics
SELECT
    6,
    'KDIGO分期完成',
    COUNT(DISTINCT stay_id),
    0,
    '进入AKI分期'
FROM tmp_mimic_b_patients_kdigo;

DROP TABLE IF EXISTS tmp_mimic_patient_aki_status;
CREATE TEMP TABLE tmp_mimic_patient_aki_status AS
SELECT
    stay_id,
    MAX(aki_stage) AS max_aki_stage
FROM tmp_mimic_b_patients_kdigo
GROUP BY stay_id;

INSERT INTO public.mimic_flow_statistics
SELECT
    7,
    '发生AKI的患者',
    COUNT(*) FILTER (WHERE COALESCE(max_aki_stage, 0) >= 1),
    COUNT(*) FILTER (WHERE COALESCE(max_aki_stage, 0) = 0),
    '未发生AKI（进入对照组候选）'
FROM tmp_mimic_patient_aki_status;

DROP TABLE IF EXISTS tmp_mimic_aki_time_analysis;
CREATE TEMP TABLE tmp_mimic_aki_time_analysis AS
WITH first_aki_events AS (
    SELECT
        p.stay_id,
        p.charttime AS aki_time,
        i.intime AS icu_intime,
        EXTRACT(EPOCH FROM (p.charttime - i.intime)) / 3600.0 AS hours_from_admission_to_aki,
        ROW_NUMBER() OVER (PARTITION BY p.stay_id ORDER BY p.charttime) AS aki_rank
    FROM tmp_mimic_b_patients_kdigo p
    JOIN mimiciv_icu.icustays i
        ON i.stay_id = p.stay_id
    WHERE p.aki_stage >= 1
)
SELECT *
FROM first_aki_events
WHERE aki_rank = 1;

INSERT INTO public.mimic_flow_statistics
SELECT
    8,
    'AKI发生在24-72小时内',
    COUNT(*) FILTER (WHERE hours_from_admission_to_aki BETWEEN 24 AND 72),
    COUNT(*) FILTER (WHERE hours_from_admission_to_aki < 24 OR hours_from_admission_to_aki > 72),
    'AKI<24h或>72h'
FROM tmp_mimic_aki_time_analysis;

INSERT INTO public.mimic_flow_statistics
SELECT
    9,
    '最终对照队列（未发生AKI）',
    COUNT(*),
    0,
    NULL
FROM tmp_mimic_patient_aki_status
WHERE COALESCE(max_aki_stage, 0) = 0;

-- Output
SELECT
    step_number,
    step_description,
    remaining_n,
    excluded_n,
    exclusion_reason
FROM public.mimic_flow_statistics
ORDER BY step_number;

SELECT
    metric,
    n
FROM public.mimic_flow_step5_detail
ORDER BY metric;
