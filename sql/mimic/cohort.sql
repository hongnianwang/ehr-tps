-- ===================================================================
-- SQL数据处理流程 - AKI病例对照研究（含详细纳排统计）
-- ===================================================================

-- 创建纳排统计表
DROP TABLE IF EXISTS exclusion_statistics;
CREATE TABLE exclusion_statistics (
    step_number INT,
    step_description VARCHAR(200),
    patient_count INT,
    excluded_count INT,
    exclusion_reason VARCHAR(200)
);

-- ===================================================================
-- 步骤0: 统计MIMIC-IV数据库中所有成年患者
-- ===================================================================
INSERT INTO exclusion_statistics 
SELECT 
    0 as step_number,
    'MIMIC-IV数据库所有成年患者' as step_description,
    COUNT(DISTINCT subject_id) as patient_count,
    0 as excluded_count,
    NULL as exclusion_reason
FROM mimiciv_hosp.patients 
WHERE anchor_age >= 18;

-- ===================================================================
-- 步骤1: 创建基础患者表和标志表
-- ===================================================================

-- 1.1 创建基础患者表（含人口统计学和住院信息）
DROP TABLE IF EXISTS base_patients;
CREATE TABLE base_patients AS
SELECT 
    pat.subject_id,
    pat.gender,
    pat.anchor_age AS age,
    adm.hadm_id,
    adm.admittime,
    adm.dischtime,
    icu.stay_id,
    icu.intime AS icu_intime,
    icu.outtime AS icu_outtime
FROM 
    mimiciv_hosp.patients pat
JOIN mimiciv_hosp.admissions adm 
    ON pat.subject_id = adm.subject_id
JOIN mimiciv_icu.icustays icu 
    ON adm.hadm_id = icu.hadm_id
WHERE 
    pat.anchor_age >= 18;

-- 统计ICU入住记录
INSERT INTO exclusion_statistics 
SELECT 
    1 as step_number,
    '成年患者ICU入住记录' as step_description,
    COUNT(DISTINCT stay_id) as patient_count,
    (SELECT patient_count FROM exclusion_statistics WHERE step_number = 0) - COUNT(DISTINCT subject_id) as excluded_count,
    '无ICU入住记录' as exclusion_reason
FROM base_patients;

-- 统计ICU停留时间筛选
INSERT INTO exclusion_statistics 
WITH icu_duration_stats AS (
    SELECT 
        COUNT(DISTINCT CASE WHEN (icu_outtime - icu_intime) >= INTERVAL '24 hours' THEN stay_id END) as eligible_count,
        COUNT(DISTINCT CASE WHEN (icu_outtime - icu_intime) < INTERVAL '24 hours' THEN stay_id END) as excluded_count
    FROM base_patients
)
SELECT 
    2 as step_number,
    'ICU停留时间≥24小时' as step_description,
    eligible_count as patient_count,
    excluded_count as excluded_count,
    'ICU停留时间<24小时' as exclusion_reason
FROM icu_duration_stats;

-- 更新base_patients表
DELETE FROM base_patients WHERE (icu_outtime - icu_intime) < INTERVAL '24 hours';
CREATE INDEX idx_base ON base_patients (subject_id, hadm_id);

-- 1.2 创建包含所有标志位的患者表
DROP TABLE IF EXISTS flagged_patients;
CREATE TABLE flagged_patients AS
WITH exclusion_flags AS (
    SELECT 
        bp.subject_id,
        bp.hadm_id,
        bp.stay_id,
        -- 慢性肾病4-5期标志
        MAX(CASE WHEN EXISTS (
            SELECT 1 FROM mimiciv_hosp.diagnoses_icd d 
            WHERE d.hadm_id = bp.hadm_id 
            AND (
                (d.icd_version = 9 AND SUBSTR(d.icd_code,1,4) IN ('5854','5855')) OR
                (d.icd_version = 10 AND SUBSTR(d.icd_code,1,4) IN ('N184','N185'))
            )
        ) THEN 1 ELSE 0 END) AS ckd_stage4to5_flag,
        
        -- 终末期肾病标志
        MAX(CASE WHEN EXISTS (
            SELECT 1 FROM mimiciv_hosp.diagnoses_icd d 
            WHERE d.hadm_id = bp.hadm_id 
            AND (
                (d.icd_version = 9 AND SUBSTR(d.icd_code,1,4) = '5856') OR
                (d.icd_version = 10 AND SUBSTR(d.icd_code,1,4) = 'N186')
            )
        ) THEN 1 ELSE 0 END) AS esrd_flag,
        
        -- 透析依赖标志
        MAX(CASE WHEN EXISTS (
            SELECT 1 FROM mimiciv_hosp.diagnoses_icd d 
            WHERE d.hadm_id = bp.hadm_id 
            AND (
                (d.icd_version = 9 AND (d.icd_code = 'V4511' OR d.icd_code LIKE 'V56%')) OR
                (d.icd_version = 10 AND d.icd_code = 'Z992')
            )
        ) THEN 1 ELSE 0 END) AS dialysis_flag,
        
        -- 入院前AKI标志
        MAX(CASE WHEN EXISTS (
            SELECT 1 FROM mimiciv_hosp.diagnoses_icd d 
            WHERE d.hadm_id = bp.hadm_id 
            AND (
                (d.icd_version = 9 AND d.icd_code LIKE '584%') OR
                (d.icd_version = 10 AND d.icd_code LIKE 'N17%')
            )
        ) THEN 1 ELSE 0 END) AS pre_admission_aki_flag,
        
        -- RRT治疗标志
        MAX(CASE WHEN EXISTS (
            SELECT 1 FROM mimiciv_icu.procedureevents pe
            JOIN mimiciv_icu.icustays ic ON pe.stay_id = ic.stay_id
            WHERE pe.stay_id = bp.stay_id
            AND pe.itemid IN (225802,225803,225805,224270,225809,225955)
            AND pe.starttime <= (ic.intime + INTERVAL '48' HOUR)
        ) THEN 1 ELSE 0 END) AS early_rrt_flag
    FROM 
        base_patients bp
    GROUP BY 
        bp.subject_id, bp.hadm_id, bp.stay_id
)
SELECT 
    bp.*,
    ef.ckd_stage4to5_flag,
    ef.esrd_flag,
    ef.dialysis_flag,
    ef.pre_admission_aki_flag,
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
FROM 
    base_patients bp
JOIN 
    exclusion_flags ef ON bp.subject_id = ef.subject_id AND bp.hadm_id = ef.hadm_id AND bp.stay_id = ef.stay_id;

CREATE INDEX idx_flagged ON flagged_patients (subject_id, hadm_id);

-- 统计排除的患者数
INSERT INTO exclusion_statistics 
SELECT 
    3 as step_number,
    '排除ESRD/透析/早期RRT患者' as step_description,
    SUM(CASE WHEN renal_exclusion_flag = 0 THEN 1 ELSE 0 END) as patient_count,
    SUM(CASE WHEN renal_exclusion_flag = 1 THEN 1 ELSE 0 END) as excluded_count,
    'ESRD/透析依赖/早期RRT（48小时内）；CKD4-5仅记录不排除' as exclusion_reason
FROM flagged_patients;

-- ===================================================================
-- 步骤2: 创建KDIGO肌酐数据表
-- ===================================================================

-- 先统计有肌酐数据的患者
INSERT INTO exclusion_statistics 
WITH creat_data AS (
    SELECT 
        COUNT(DISTINCT fp.stay_id) as total_eligible,
        COUNT(DISTINCT k.stay_id) as with_creat
    FROM flagged_patients fp
    LEFT JOIN mimiciv_derived.kdigo_stages_1 k 
        ON fp.stay_id = k.stay_id
    WHERE fp.is_eligible = 1
)
SELECT 
    4 as step_number,
    '有肌酐测量数据' as step_description,
    with_creat as patient_count,
    total_eligible - with_creat as excluded_count,
    '无肌酐测量' as exclusion_reason
FROM creat_data;

DROP TABLE IF EXISTS b_patients_kdigo;
CREATE TABLE b_patients_kdigo AS
WITH eligible_stays AS (
    SELECT 
        stay_id,
        subject_id,
        hadm_id,
        icu_intime,
        icu_outtime
    FROM 
        flagged_patients
    WHERE 
        is_eligible = 1
),
first_creat_times AS (
    SELECT 
        stay_id,
        MIN(charttime) AS first_charttime
    FROM 
        mimiciv_derived.kdigo_stages_1
    WHERE 
        stay_id IN (SELECT stay_id FROM eligible_stays)
    GROUP BY 
        stay_id
),
creatinine_metrics AS (
    SELECT 
        k.stay_id,
        COUNT(k.creat) AS creatinine_count,
        MAX(CASE WHEN k.charttime = fct.first_charttime THEN k.creat ELSE NULL END) AS first_creatinine
    FROM 
        mimiciv_derived.kdigo_stages_1 k
    JOIN
        first_creat_times fct ON k.stay_id = fct.stay_id AND k.charttime >= fct.first_charttime
    GROUP BY 
        k.stay_id
    HAVING 
        MAX(CASE WHEN k.charttime = fct.first_charttime THEN k.creat ELSE NULL END) <= 3.0
        AND COUNT(k.creat) >= 2
)
SELECT 
    e.stay_id,
    e.subject_id,
    e.hadm_id,
    k.charttime,
    k.creat AS serum_creatinine,
    k.aki_stage_creat AS aki_stage,
    cm.first_creatinine,
    cm.creatinine_count
FROM 
    eligible_stays e
JOIN 
    creatinine_metrics cm ON e.stay_id = cm.stay_id
JOIN 
    mimiciv_derived.kdigo_stages_1 k ON e.stay_id = k.stay_id
WHERE 
    k.charttime BETWEEN e.icu_intime AND e.icu_outtime
ORDER BY 
    e.stay_id, k.charttime;

-- 统计肌酐筛选结果
INSERT INTO exclusion_statistics 
SELECT 
    5 as step_number,
    '首次肌酐≤3.0且测量≥2次' as step_description,
    COUNT(DISTINCT stay_id) as patient_count,
    (SELECT patient_count FROM exclusion_statistics WHERE step_number = 4) - COUNT(DISTINCT stay_id) as excluded_count,
    '首次肌酐>3.0或测量<2次' as exclusion_reason
FROM b_patients_kdigo;

-- 统计AKI发生情况
INSERT INTO exclusion_statistics 
SELECT 
    6 as step_number,
    'KDIGO分期完成' as step_description,
    COUNT(DISTINCT stay_id) as patient_count,
    0 as excluded_count,
    '进入AKI分期' as exclusion_reason
FROM b_patients_kdigo;

-- ===================================================================
-- 步骤3A: 创建AKI队列
-- ===================================================================

-- 统计AKI发生情况（修正版）
INSERT INTO exclusion_statistics 
WITH patient_aki_status AS (
    SELECT 
        stay_id,
        MAX(aki_stage) as max_aki_stage
    FROM b_patients_kdigo
    GROUP BY stay_id
),
aki_stats AS (
    SELECT 
        COUNT(CASE WHEN max_aki_stage >= 1 THEN 1 END) as aki_count,
        COUNT(CASE WHEN max_aki_stage = 0 OR max_aki_stage IS NULL THEN 1 END) as non_aki_count
    FROM patient_aki_status
)
SELECT 
    7 as step_number,
    '发生AKI的患者' as step_description,
    aki_count as patient_count,
    non_aki_count as excluded_count,
    '未发生AKI（进入对照组候选）' as exclusion_reason
FROM aki_stats;

-- 统计AKI时间窗
DROP TABLE IF EXISTS aki_time_analysis;
CREATE TABLE aki_time_analysis AS
WITH first_aki_events AS (
    SELECT 
        p.stay_id,
        p.subject_id,
        p.hadm_id,
        p.charttime AS aki_time,
        i.intime AS icu_intime,
        EXTRACT(EPOCH FROM (p.charttime - i.intime))/3600 AS hours_from_admission_to_aki,
        p.aki_stage,
        ROW_NUMBER() OVER (PARTITION BY p.stay_id ORDER BY p.charttime) AS aki_rank
    FROM 
        b_patients_kdigo p
    JOIN 
        mimiciv_icu.icustays i ON p.stay_id = i.stay_id
    WHERE 
        p.aki_stage >= 1
)
SELECT * FROM first_aki_events WHERE aki_rank = 1;

INSERT INTO exclusion_statistics 
WITH time_stats AS (
    SELECT 
        COUNT(DISTINCT CASE WHEN hours_from_admission_to_aki < 24 THEN stay_id END) as before_24h,
        COUNT(DISTINCT CASE WHEN hours_from_admission_to_aki BETWEEN 24 AND 72 THEN stay_id END) as in_window,
        COUNT(DISTINCT CASE WHEN hours_from_admission_to_aki > 72 THEN stay_id END) as after_72h
    FROM aki_time_analysis
)
SELECT 
    8 as step_number,
    'AKI发生在24-72小时内' as step_description,
    in_window as patient_count,
    before_24h + after_72h as excluded_count,
    'AKI<24h或>72h' as exclusion_reason
FROM time_stats;

-- ===================================================================
-- 【新增】对照组的详细排除统计
-- ===================================================================

-- ===================================================================
-- 对照组的纳排统计（修正版）
-- ===================================================================

-- 步骤9: 对照组候选（未发生AKI）
INSERT INTO exclusion_statistics 
WITH patient_aki_status AS (
    SELECT 
        stay_id,
        MAX(aki_stage) as max_aki_stage
    FROM b_patients_kdigo
    GROUP BY stay_id
)
SELECT 
    9 as step_number,
    '对照组候选（未发生AKI）' as step_description,
    COUNT(*) as patient_count,
    0 as excluded_count,
    NULL as exclusion_reason
FROM patient_aki_status
WHERE max_aki_stage = 0 OR max_aki_stage IS NULL;

-- 步骤10: 排除缺少icustay_detail记录的患者
INSERT INTO exclusion_statistics 
WITH patient_aki_status AS (
    SELECT 
        stay_id,
        MAX(aki_stage) as max_aki_stage
    FROM b_patients_kdigo
    GROUP BY stay_id
),
non_aki_patients AS (
    SELECT stay_id
    FROM patient_aki_status
    WHERE max_aki_stage = 0 OR max_aki_stage IS NULL
),
icustay_detail_check AS (
    SELECT COUNT(DISTINCT n.stay_id) as with_detail
    FROM non_aki_patients n
    JOIN mimiciv_derived.icustay_detail det ON n.stay_id = det.stay_id
)
SELECT 
    10 as step_number,
    '对照组有ICU详细记录' as step_description,
    with_detail as patient_count,
    (SELECT patient_count FROM exclusion_statistics WHERE step_number = 9) - with_detail as excluded_count,
    '缺少icustay_detail表记录' as exclusion_reason
FROM icustay_detail_check;

-- ===================================================================
-- 创建最终AKI队列（继续原来的代码）
-- ===================================================================

DROP TABLE IF EXISTS aki_cohort_final;
CREATE TABLE aki_cohort_final AS
WITH first_aki_events AS (
    SELECT 
        p.stay_id,
        p.subject_id,
        p.hadm_id,
        p.charttime AS aki_time,
        i.intime AS icu_intime,
        EXTRACT(EPOCH FROM (p.charttime - i.intime))/3600 AS hours_from_admission_to_aki,
        p.aki_stage,
        ROW_NUMBER() OVER (PARTITION BY p.stay_id ORDER BY p.charttime) AS aki_rank
    FROM 
        b_patients_kdigo p
    JOIN 
        mimiciv_icu.icustays i ON p.stay_id = i.stay_id
    WHERE 
        p.aki_stage >= 1
),
aki_cohort AS (
    SELECT
        stay_id,
        subject_id,
        hadm_id,
        aki_time,
        icu_intime,
        hours_from_admission_to_aki,
        aki_stage
    FROM
        first_aki_events
    WHERE
        aki_rank = 1
        AND hours_from_admission_to_aki <= 72
        AND hours_from_admission_to_aki >= 24
    ORDER BY
        hours_from_admission_to_aki
),
demographics AS (
    SELECT
        a.stay_id,
        a.subject_id,
        a.hadm_id,
        a.aki_time,
        a.icu_intime,
        a.hours_from_admission_to_aki,
        a.aki_stage,
        p.gender,
        p.anchor_age AS age,
        CASE WHEN p.anchor_age < 18 THEN '< 18'
             WHEN p.anchor_age < 30 THEN '18-29'
             WHEN p.anchor_age < 50 THEN '30-49'
             WHEN p.anchor_age < 70 THEN '50-69'
             ELSE '≥ 70' END AS age_group,
        adm.admission_type,
        adm.insurance,
        adm.language,
        adm.marital_status,
        det.race,
        i.los AS icu_los_days
    FROM
        aki_cohort a
    JOIN
        mimiciv_hosp.patients p ON a.subject_id = p.subject_id
    JOIN
        mimiciv_hosp.admissions adm ON a.hadm_id = adm.hadm_id
    JOIN
        mimiciv_icu.icustays i ON a.stay_id = i.stay_id
    JOIN
        mimiciv_derived.icustay_detail det ON a.stay_id = det.stay_id
),
vital_signs AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.aki_time,
        c.icu_intime,
        c.hours_from_admission_to_aki,
        c.aki_stage,
        v.charttime,
        AVG(CASE WHEN v.itemid IN (220045)
                AND v.valuenum > 0
                AND v.valuenum < 300
                THEN v.valuenum END
        ) AS heart_rate,
        AVG(CASE WHEN v.itemid IN (220179, 220050, 225309)
                AND v.valuenum > 0
                AND v.valuenum < 400
                THEN v.valuenum END
        ) AS sbp,
        AVG(CASE WHEN v.itemid IN (220180, 220051, 225310)
                AND v.valuenum > 0
                AND v.valuenum < 300
                THEN v.valuenum END
        ) AS dbp,
        AVG(CASE WHEN v.itemid IN (220052, 220181, 225312)
                AND v.valuenum > 0
                AND v.valuenum < 300
                THEN v.valuenum END
        ) AS mbp,
        AVG(CASE WHEN v.itemid IN (220210, 224690)
                AND v.valuenum > 0
                AND v.valuenum < 70
                THEN v.valuenum END
        ) AS respiratory_rate,
        AVG(CASE WHEN v.itemid IN (220277)
                AND v.valuenum > 0
                AND v.valuenum <= 100
                THEN v.valuenum END
        ) AS o2_saturation,
        ROUND(CAST(
            AVG(CASE
                WHEN v.itemid IN (223761)
                    AND v.valuenum > 70
                    AND v.valuenum < 120
                    THEN (v.valuenum - 32) / 1.8
                WHEN v.itemid IN (223762)
                    AND v.valuenum > 10
                    AND v.valuenum < 50
                    THEN v.valuenum END)
            AS NUMERIC), 2) AS temperature,
        MAX(CASE WHEN v.itemid = 224642 THEN v.value END
        ) AS temperature_site,
        AVG(CASE WHEN v.itemid IN (225664, 220621, 226537)
                AND v.valuenum > 0
                THEN v.valuenum END
        ) AS glucose
    FROM
        aki_cohort c
    JOIN
        mimiciv_icu.chartevents v
    ON
        c.stay_id = v.stay_id
    WHERE
        v.charttime BETWEEN (c.icu_intime - INTERVAL '6' HOUR) AND (c.aki_time - INTERVAL '12' HOUR)
        AND v.itemid IN (
            220045, 225309, 225310, 225312, 220050, 220051, 220052, 220179, 220180, 220181,
            220210, 224690, 220277, 225664, 220621, 226537, 223762, 223761, 224642
        )
    GROUP BY
        c.stay_id, c.subject_id, c.hadm_id, c.aki_time, c.icu_intime, 
        c.hours_from_admission_to_aki, c.aki_stage, v.charttime
),
lab_data AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.aki_time,
        c.icu_intime,
        c.hours_from_admission_to_aki,
        c.aki_stage,
        l.charttime,
        AVG(CASE WHEN l.itemid = 51006 AND l.valuenum <= 300 THEN l.valuenum END) AS bun,
        AVG(CASE WHEN l.itemid IN (51300, 51301) AND l.valuenum <= 1000 THEN l.valuenum END) AS wbc,
        AVG(CASE WHEN l.itemid = 50971 AND l.valuenum <= 30 THEN l.valuenum END) AS potassium,
        AVG(CASE WHEN l.itemid = 50893 AND l.valuenum <= 10000 THEN l.valuenum END) AS calcium,
        AVG(CASE WHEN l.itemid = 50912 AND l.valuenum <= 30 THEN l.valuenum END) AS creatinine,
        AVG(CASE WHEN l.itemid = 50931 AND l.valuenum <= 30000 THEN l.valuenum END) AS glucose,
        AVG(CASE WHEN l.itemid = 50960 AND l.valuenum <= 10000 THEN l.valuenum END) AS magnesium,
        AVG(CASE WHEN l.itemid = 50983 AND l.valuenum <= 200 THEN l.valuenum END) AS sodium,
        AVG(CASE WHEN l.itemid = 51222 AND l.valuenum <= 50 THEN l.valuenum END) AS hemoglobin,
        AVG(CASE WHEN l.itemid = 51265 AND l.valuenum <= 10000 THEN l.valuenum END) AS platelet,
        AVG(CASE WHEN l.itemid = 50882 AND l.valuenum <= 10000 THEN l.valuenum END) AS bicarbonate,
        AVG(CASE WHEN l.itemid = 50902 AND l.valuenum <= 10000 THEN l.valuenum END) AS chloride,
        AVG(CASE WHEN l.itemid = 50813 AND l.valuenum <= 10000 THEN l.valuenum END) AS lactate,
        AVG(CASE WHEN l.itemid = 51221 AND l.valuenum > 0 AND l.valuenum <= 100 THEN l.valuenum END) AS hematocrit,
        AVG(CASE WHEN l.itemid = 51279 AND l.valuenum > 0 AND l.valuenum <= 10 THEN l.valuenum END) AS rbc
    FROM
        aki_cohort c
    JOIN
        mimiciv_hosp.labevents l
    ON
        c.subject_id = l.subject_id
    WHERE
        l.charttime BETWEEN (c.icu_intime - INTERVAL '6' HOUR) AND (c.aki_time - INTERVAL '12' HOUR)
        AND l.itemid IN (
            51006, 51300, 51301, 50971, 50893, 50912, 50931, 50960, 50983, 51222, 51265,
            50882, 50902, 50813, 51221, 51279
        )
    GROUP BY
        c.stay_id, c.subject_id, c.hadm_id, c.aki_time, c.icu_intime, 
        c.hours_from_admission_to_aki, c.aki_stage, l.charttime
),
combined_data AS (
    SELECT
        COALESCE(v.stay_id, l.stay_id) AS stay_id,
        COALESCE(v.subject_id, l.subject_id) AS subject_id,
        COALESCE(v.hadm_id, l.hadm_id) AS hadm_id,
        COALESCE(v.aki_time, l.aki_time) AS aki_time,
        COALESCE(v.icu_intime, l.icu_intime) AS icu_intime,
        COALESCE(v.hours_from_admission_to_aki, l.hours_from_admission_to_aki) AS hours_from_admission_to_aki,
        COALESCE(v.aki_stage, l.aki_stage) AS aki_stage,
        COALESCE(v.charttime, l.charttime) AS charttime,
        v.heart_rate, v.sbp, v.dbp, v.respiratory_rate, v.o2_saturation, v.temperature,
        l.bun, l.wbc, l.potassium, l.calcium, l.creatinine, l.glucose, l.magnesium, l.sodium,
        l.hemoglobin, l.platelet, l.bicarbonate, l.chloride, l.lactate, l.hematocrit, l.rbc
    FROM
        vital_signs v
    FULL OUTER JOIN
        lab_data l
    ON
        v.stay_id = l.stay_id
        AND v.charttime = l.charttime
),
vital_stats AS (
    SELECT
        stay_id,
        MAX(heart_rate) AS heart_rate_max, MIN(heart_rate) AS heart_rate_min,
        MAX(sbp) AS sbp_max, MIN(sbp) AS sbp_min,
        MAX(dbp) AS dbp_max, MIN(dbp) AS dbp_min,
        MAX(respiratory_rate) AS respiratory_rate_max, MIN(respiratory_rate) AS respiratory_rate_min,
        MAX(o2_saturation) AS o2_saturation_max, MIN(o2_saturation) AS o2_saturation_min,
        MAX(temperature) AS temperature_max, MIN(temperature) AS temperature_min
    FROM combined_data GROUP BY stay_id
),
lab_stats AS (
    SELECT
        stay_id,
        MAX(bun) AS bun_max, MIN(bun) AS bun_min,
        MAX(wbc) AS wbc_max, MIN(wbc) AS wbc_min,
        MAX(potassium) AS potassium_max, MIN(potassium) AS potassium_min,
        MAX(calcium) AS calcium_max, MIN(calcium) AS calcium_min,
        MAX(creatinine) AS creatinine_max, MIN(creatinine) AS creatinine_min,
        MAX(glucose) AS glucose_max, MIN(glucose) AS glucose_min,
        MAX(magnesium) AS magnesium_max, MIN(magnesium) AS magnesium_min,
        MAX(sodium) AS sodium_max, MIN(sodium) AS sodium_min,
        MAX(hemoglobin) AS hemoglobin_max, MIN(hemoglobin) AS hemoglobin_min,
        MAX(platelet) AS platelet_max, MIN(platelet) AS platelet_min,
        MAX(bicarbonate) AS bicarbonate_max, MIN(bicarbonate) AS bicarbonate_min,
        MAX(chloride) AS chloride_max, MIN(chloride) AS chloride_min,
        MAX(lactate) AS lactate_max, MIN(lactate) AS lactate_min,
        MAX(hematocrit) AS hematocrit_max, MIN(hematocrit) AS hematocrit_min,
        MAX(rbc) AS rbc_max, MIN(rbc) AS rbc_min
    FROM combined_data GROUP BY stay_id
),
vital_last AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.heart_rate IS NOT NULL THEN cd.charttime END) AS heart_rate_last_time,
        MAX(CASE WHEN cd.sbp IS NOT NULL THEN cd.charttime END) AS sbp_last_time,
        MAX(CASE WHEN cd.dbp IS NOT NULL THEN cd.charttime END) AS dbp_last_time,
        MAX(CASE WHEN cd.respiratory_rate IS NOT NULL THEN cd.charttime END) AS respiratory_rate_last_time,
        MAX(CASE WHEN cd.o2_saturation IS NOT NULL THEN cd.charttime END) AS o2_saturation_last_time,
        MAX(CASE WHEN cd.temperature IS NOT NULL THEN cd.charttime END) AS temperature_last_time
    FROM combined_data cd GROUP BY cd.stay_id
),
lab_last AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.bun IS NOT NULL THEN cd.charttime END) AS bun_last_time,
        MAX(CASE WHEN cd.wbc IS NOT NULL THEN cd.charttime END) AS wbc_last_time,
        MAX(CASE WHEN cd.potassium IS NOT NULL THEN cd.charttime END) AS potassium_last_time,
        MAX(CASE WHEN cd.calcium IS NOT NULL THEN cd.charttime END) AS calcium_last_time,
        MAX(CASE WHEN cd.creatinine IS NOT NULL THEN cd.charttime END) AS creatinine_last_time,
        MAX(CASE WHEN cd.glucose IS NOT NULL THEN cd.charttime END) AS glucose_last_time,
        MAX(CASE WHEN cd.magnesium IS NOT NULL THEN cd.charttime END) AS magnesium_last_time,
        MAX(CASE WHEN cd.sodium IS NOT NULL THEN cd.charttime END) AS sodium_last_time,
        MAX(CASE WHEN cd.hemoglobin IS NOT NULL THEN cd.charttime END) AS hemoglobin_last_time,
        MAX(CASE WHEN cd.platelet IS NOT NULL THEN cd.charttime END) AS platelet_last_time,
        MAX(CASE WHEN cd.bicarbonate IS NOT NULL THEN cd.charttime END) AS bicarbonate_last_time,
        MAX(CASE WHEN cd.chloride IS NOT NULL THEN cd.charttime END) AS chloride_last_time,
        MAX(CASE WHEN cd.lactate IS NOT NULL THEN cd.charttime END) AS lactate_last_time,
        MAX(CASE WHEN cd.hematocrit IS NOT NULL THEN cd.charttime END) AS hematocrit_last_time,
        MAX(CASE WHEN cd.rbc IS NOT NULL THEN cd.charttime END) AS rbc_last_time
    FROM combined_data cd GROUP BY cd.stay_id
),
vital_last_values AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.charttime = vl.heart_rate_last_time THEN cd.heart_rate END) AS heart_rate_last,
        MAX(CASE WHEN cd.charttime = vl.sbp_last_time THEN cd.sbp END) AS sbp_last,
        MAX(CASE WHEN cd.charttime = vl.dbp_last_time THEN cd.dbp END) AS dbp_last,
        MAX(CASE WHEN cd.charttime = vl.respiratory_rate_last_time THEN cd.respiratory_rate END) AS respiratory_rate_last,
        MAX(CASE WHEN cd.charttime = vl.o2_saturation_last_time THEN cd.o2_saturation END) AS o2_saturation_last,
        MAX(CASE WHEN cd.charttime = vl.temperature_last_time THEN cd.temperature END) AS temperature_last
    FROM combined_data cd
    JOIN vital_last vl ON cd.stay_id = vl.stay_id
    GROUP BY cd.stay_id
),
lab_last_values AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.charttime = ll.bun_last_time THEN cd.bun END) AS bun_last,
        MAX(CASE WHEN cd.charttime = ll.wbc_last_time THEN cd.wbc END) AS wbc_last,
        MAX(CASE WHEN cd.charttime = ll.potassium_last_time THEN cd.potassium END) AS potassium_last,
        MAX(CASE WHEN cd.charttime = ll.calcium_last_time THEN cd.calcium END) AS calcium_last,
        MAX(CASE WHEN cd.charttime = ll.creatinine_last_time THEN cd.creatinine END) AS creatinine_last,
        MAX(CASE WHEN cd.charttime = ll.glucose_last_time THEN cd.glucose END) AS glucose_last,
        MAX(CASE WHEN cd.charttime = ll.magnesium_last_time THEN cd.magnesium END) AS magnesium_last,
        MAX(CASE WHEN cd.charttime = ll.sodium_last_time THEN cd.sodium END) AS sodium_last,
        MAX(CASE WHEN cd.charttime = ll.hemoglobin_last_time THEN cd.hemoglobin END) AS hemoglobin_last,
        MAX(CASE WHEN cd.charttime = ll.platelet_last_time THEN cd.platelet END) AS platelet_last,
        MAX(CASE WHEN cd.charttime = ll.bicarbonate_last_time THEN cd.bicarbonate END) AS bicarbonate_last,
        MAX(CASE WHEN cd.charttime = ll.chloride_last_time THEN cd.chloride END) AS chloride_last,
        MAX(CASE WHEN cd.charttime = ll.lactate_last_time THEN cd.lactate END) AS lactate_last,
        MAX(CASE WHEN cd.charttime = ll.hematocrit_last_time THEN cd.hematocrit END) AS hematocrit_last,
        MAX(CASE WHEN cd.charttime = ll.rbc_last_time THEN cd.rbc END) AS rbc_last
    FROM combined_data cd
    JOIN lab_last ll ON cd.stay_id = ll.stay_id
    GROUP BY cd.stay_id
)
SELECT
    d.*,
    vlv.heart_rate_last, vlv.sbp_last, vlv.dbp_last, vlv.respiratory_rate_last, vlv.o2_saturation_last, vlv.temperature_last,
    llv.bun_last, llv.wbc_last, llv.potassium_last, llv.calcium_last, llv.creatinine_last, llv.glucose_last, llv.magnesium_last, llv.sodium_last,
    llv.hemoglobin_last, llv.platelet_last, llv.bicarbonate_last, llv.chloride_last, llv.lactate_last, llv.hematocrit_last, llv.rbc_last,
    vs.heart_rate_max, vs.heart_rate_min, vs.sbp_max, vs.sbp_min, vs.dbp_max, vs.dbp_min,
    vs.respiratory_rate_max, vs.respiratory_rate_min, vs.o2_saturation_max, vs.o2_saturation_min, vs.temperature_max, vs.temperature_min,
    ls.bun_max, ls.bun_min, ls.wbc_max, ls.wbc_min, ls.potassium_max, ls.potassium_min, ls.calcium_max, ls.calcium_min,
    ls.creatinine_max, ls.creatinine_min, ls.glucose_max, ls.glucose_min, ls.magnesium_max, ls.magnesium_min, ls.sodium_max, ls.sodium_min,
    ls.hemoglobin_max, ls.hemoglobin_min, ls.platelet_max, ls.platelet_min, ls.bicarbonate_max, ls.bicarbonate_min,
    ls.chloride_max, ls.chloride_min, ls.lactate_max, ls.lactate_min, ls.hematocrit_max, ls.hematocrit_min, ls.rbc_max, ls.rbc_min
FROM demographics d
LEFT JOIN vital_last_values vlv ON d.stay_id = vlv.stay_id
LEFT JOIN lab_last_values llv ON d.stay_id = llv.stay_id
LEFT JOIN vital_stats vs ON d.stay_id = vs.stay_id
LEFT JOIN lab_stats ls ON d.stay_id = ls.stay_id
ORDER BY d.stay_id;

-- ===================================================================
-- 步骤3B: 创建对照组队列
-- ===================================================================

DROP TABLE IF EXISTS control_cohort_final;
CREATE TABLE control_cohort_final AS
WITH aki_patients AS (
    SELECT DISTINCT stay_id
    FROM b_patients_kdigo
    WHERE aki_stage > 0
),
non_aki_patients AS (
    SELECT
        p.stay_id,
        p.subject_id,
        p.hadm_id,
        i.intime AS icu_intime,
        i.intime + INTERVAL '72 hour' AS observation_time,
        0 AS aki_stage
    FROM
        b_patients_kdigo p
    JOIN
        mimiciv_icu.icustays i ON p.stay_id = i.stay_id
    WHERE
        NOT EXISTS (
            SELECT 1
            FROM aki_patients a
            WHERE a.stay_id = p.stay_id
        )
    GROUP BY
        p.stay_id, p.subject_id, p.hadm_id, i.intime
),
control_cohort AS (
    SELECT
        stay_id,
        subject_id,
        hadm_id,
        observation_time AS aki_time,
        icu_intime,
        72 AS hours_from_admission_to_aki,
        aki_stage
    FROM
        non_aki_patients
),
demographics AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.aki_time,
        c.icu_intime,
        c.hours_from_admission_to_aki,
        c.aki_stage,
        p.gender,
        p.anchor_age AS age,
        CASE WHEN p.anchor_age < 18 THEN '< 18'
             WHEN p.anchor_age < 30 THEN '18-29'
             WHEN p.anchor_age < 50 THEN '30-49'
             WHEN p.anchor_age < 70 THEN '50-69'
             ELSE '≥ 70' END AS age_group,
        adm.admission_type,
        adm.insurance,
        adm.language,
        adm.marital_status,
        det.race,
        i.los AS icu_los_days
    FROM
        control_cohort c
    JOIN
        mimiciv_hosp.patients p ON c.subject_id = p.subject_id
    JOIN
        mimiciv_hosp.admissions adm ON c.hadm_id = adm.hadm_id
    JOIN
        mimiciv_icu.icustays i ON c.stay_id = i.stay_id
    JOIN
        mimiciv_derived.icustay_detail det ON c.stay_id = det.stay_id
),
vital_signs AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.aki_time,
        c.icu_intime,
        c.hours_from_admission_to_aki,
        c.aki_stage,
        v.charttime,
        AVG(CASE WHEN v.itemid IN (220045) AND v.valuenum > 0 AND v.valuenum < 300 THEN v.valuenum END) AS heart_rate,
        AVG(CASE WHEN v.itemid IN (220179, 220050, 225309) AND v.valuenum > 0 AND v.valuenum < 400 THEN v.valuenum END) AS sbp,
        AVG(CASE WHEN v.itemid IN (220180, 220051, 225310) AND v.valuenum > 0 AND v.valuenum < 300 THEN v.valuenum END) AS dbp,
        AVG(CASE WHEN v.itemid IN (220052, 220181, 225312) AND v.valuenum > 0 AND v.valuenum < 300 THEN v.valuenum END) AS mbp,
        AVG(CASE WHEN v.itemid IN (220210, 224690) AND v.valuenum > 0 AND v.valuenum < 70 THEN v.valuenum END) AS respiratory_rate,
        AVG(CASE WHEN v.itemid IN (220277) AND v.valuenum > 0 AND v.valuenum <= 100 THEN v.valuenum END) AS o2_saturation,
        ROUND(CAST(AVG(CASE
                WHEN v.itemid IN (223761) AND v.valuenum > 70 AND v.valuenum < 120 THEN (v.valuenum - 32) / 1.8
                WHEN v.itemid IN (223762) AND v.valuenum > 10 AND v.valuenum < 50 THEN v.valuenum END)
            AS NUMERIC), 2) AS temperature,
        MAX(CASE WHEN v.itemid = 224642 THEN v.value END) AS temperature_site,
        AVG(CASE WHEN v.itemid IN (225664, 220621, 226537) AND v.valuenum > 0 THEN v.valuenum END) AS glucose
    FROM
        control_cohort c
    JOIN
        mimiciv_icu.chartevents v
    ON
        c.stay_id = v.stay_id
    WHERE
        v.charttime BETWEEN (c.icu_intime - INTERVAL '6' HOUR) AND (c.aki_time - INTERVAL '12' HOUR)
        AND v.itemid IN (
            220045, 225309, 225310, 225312, 220050, 220051, 220052, 220179, 220180, 220181,
            220210, 224690, 220277, 225664, 220621, 226537, 223762, 223761, 224642
        )
    GROUP BY
        c.stay_id, c.subject_id, c.hadm_id, c.aki_time, c.icu_intime, 
        c.hours_from_admission_to_aki, c.aki_stage, v.charttime
),
lab_data AS (
    SELECT
        c.stay_id,
        c.subject_id,
        c.hadm_id,
        c.aki_time,
        c.icu_intime,
        c.hours_from_admission_to_aki,
        c.aki_stage,
        l.charttime,
        AVG(CASE WHEN l.itemid = 51006 AND l.valuenum <= 300 THEN l.valuenum END) AS bun,
        AVG(CASE WHEN l.itemid IN (51300, 51301) AND l.valuenum <= 1000 THEN l.valuenum END) AS wbc,
        AVG(CASE WHEN l.itemid = 50971 AND l.valuenum <= 30 THEN l.valuenum END) AS potassium,
        AVG(CASE WHEN l.itemid = 50893 AND l.valuenum <= 10000 THEN l.valuenum END) AS calcium,
        AVG(CASE WHEN l.itemid = 50912 AND l.valuenum <= 30 THEN l.valuenum END) AS creatinine,
        AVG(CASE WHEN l.itemid = 50931 AND l.valuenum <= 30000 THEN l.valuenum END) AS glucose,
        AVG(CASE WHEN l.itemid = 50960 AND l.valuenum <= 10000 THEN l.valuenum END) AS magnesium,
        AVG(CASE WHEN l.itemid = 50983 AND l.valuenum <= 200 THEN l.valuenum END) AS sodium,
        AVG(CASE WHEN l.itemid = 51222 AND l.valuenum <= 50 THEN l.valuenum END) AS hemoglobin,
        AVG(CASE WHEN l.itemid = 51265 AND l.valuenum <= 10000 THEN l.valuenum END) AS platelet,
        AVG(CASE WHEN l.itemid = 50882 AND l.valuenum <= 10000 THEN l.valuenum END) AS bicarbonate,
        AVG(CASE WHEN l.itemid = 50902 AND l.valuenum <= 10000 THEN l.valuenum END) AS chloride,
        AVG(CASE WHEN l.itemid = 50813 AND l.valuenum <= 10000 THEN l.valuenum END) AS lactate,
        AVG(CASE WHEN l.itemid = 51221 AND l.valuenum > 0 AND l.valuenum <= 100 THEN l.valuenum END) AS hematocrit,
        AVG(CASE WHEN l.itemid = 51279 AND l.valuenum > 0 AND l.valuenum <= 10 THEN l.valuenum END) AS rbc
    FROM
        control_cohort c
    JOIN
        mimiciv_hosp.labevents l
    ON
        c.subject_id = l.subject_id
    WHERE
        l.charttime BETWEEN (c.icu_intime - INTERVAL '6' HOUR) AND (c.aki_time - INTERVAL '12' HOUR)
        AND l.itemid IN (
            51006, 51300, 51301, 50971, 50893, 50912, 50931, 50960, 50983, 51222, 51265,
            50882, 50902, 50813, 51221, 51279
        )
    GROUP BY
        c.stay_id, c.subject_id, c.hadm_id, c.aki_time, c.icu_intime, 
        c.hours_from_admission_to_aki, c.aki_stage, l.charttime
),
combined_data AS (
    SELECT
        COALESCE(v.stay_id, l.stay_id) AS stay_id,
        COALESCE(v.subject_id, l.subject_id) AS subject_id,
        COALESCE(v.hadm_id, l.hadm_id) AS hadm_id,
        COALESCE(v.aki_time, l.aki_time) AS aki_time,
        COALESCE(v.icu_intime, l.icu_intime) AS icu_intime,
        COALESCE(v.hours_from_admission_to_aki, l.hours_from_admission_to_aki) AS hours_from_admission_to_aki,
        COALESCE(v.aki_stage, l.aki_stage) AS aki_stage,
        COALESCE(v.charttime, l.charttime) AS charttime,
        v.heart_rate, v.sbp, v.dbp, v.respiratory_rate, v.o2_saturation, v.temperature,
        l.bun, l.wbc, l.potassium, l.calcium, l.creatinine, l.glucose, l.magnesium, l.sodium,
        l.hemoglobin, l.platelet, l.bicarbonate, l.chloride, l.lactate, l.hematocrit, l.rbc
    FROM
        vital_signs v
    FULL OUTER JOIN
        lab_data l
    ON
        v.stay_id = l.stay_id
        AND v.charttime = l.charttime
),
vital_stats AS (
    SELECT
        stay_id,
        MAX(heart_rate) AS heart_rate_max, MIN(heart_rate) AS heart_rate_min,
        MAX(sbp) AS sbp_max, MIN(sbp) AS sbp_min,
        MAX(dbp) AS dbp_max, MIN(dbp) AS dbp_min,
        MAX(respiratory_rate) AS respiratory_rate_max, MIN(respiratory_rate) AS respiratory_rate_min,
        MAX(o2_saturation) AS o2_saturation_max, MIN(o2_saturation) AS o2_saturation_min,
        MAX(temperature) AS temperature_max, MIN(temperature) AS temperature_min
    FROM combined_data GROUP BY stay_id
),
lab_stats AS (
    SELECT
        stay_id,
        MAX(bun) AS bun_max, MIN(bun) AS bun_min,
        MAX(wbc) AS wbc_max, MIN(wbc) AS wbc_min,
        MAX(potassium) AS potassium_max, MIN(potassium) AS potassium_min,
        MAX(calcium) AS calcium_max, MIN(calcium) AS calcium_min,
        MAX(creatinine) AS creatinine_max, MIN(creatinine) AS creatinine_min,
        MAX(glucose) AS glucose_max, MIN(glucose) AS glucose_min,
        MAX(magnesium) AS magnesium_max, MIN(magnesium) AS magnesium_min,
        MAX(sodium) AS sodium_max, MIN(sodium) AS sodium_min,
        MAX(hemoglobin) AS hemoglobin_max, MIN(hemoglobin) AS hemoglobin_min,
        MAX(platelet) AS platelet_max, MIN(platelet) AS platelet_min,
        MAX(bicarbonate) AS bicarbonate_max, MIN(bicarbonate) AS bicarbonate_min,
        MAX(chloride) AS chloride_max, MIN(chloride) AS chloride_min,
        MAX(lactate) AS lactate_max, MIN(lactate) AS lactate_min,
        MAX(hematocrit) AS hematocrit_max, MIN(hematocrit) AS hematocrit_min,
        MAX(rbc) AS rbc_max, MIN(rbc) AS rbc_min
    FROM combined_data GROUP BY stay_id
),
vital_last AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.heart_rate IS NOT NULL THEN cd.charttime END) AS heart_rate_last_time,
        MAX(CASE WHEN cd.sbp IS NOT NULL THEN cd.charttime END) AS sbp_last_time,
        MAX(CASE WHEN cd.dbp IS NOT NULL THEN cd.charttime END) AS dbp_last_time,
        MAX(CASE WHEN cd.respiratory_rate IS NOT NULL THEN cd.charttime END) AS respiratory_rate_last_time,
        MAX(CASE WHEN cd.o2_saturation IS NOT NULL THEN cd.charttime END) AS o2_saturation_last_time,
        MAX(CASE WHEN cd.temperature IS NOT NULL THEN cd.charttime END) AS temperature_last_time
    FROM combined_data cd GROUP BY cd.stay_id
),
lab_last AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.bun IS NOT NULL THEN cd.charttime END) AS bun_last_time,
        MAX(CASE WHEN cd.wbc IS NOT NULL THEN cd.charttime END) AS wbc_last_time,
        MAX(CASE WHEN cd.potassium IS NOT NULL THEN cd.charttime END) AS potassium_last_time,
        MAX(CASE WHEN cd.calcium IS NOT NULL THEN cd.charttime END) AS calcium_last_time,
        MAX(CASE WHEN cd.creatinine IS NOT NULL THEN cd.charttime END) AS creatinine_last_time,
        MAX(CASE WHEN cd.glucose IS NOT NULL THEN cd.charttime END) AS glucose_last_time,
        MAX(CASE WHEN cd.magnesium IS NOT NULL THEN cd.charttime END) AS magnesium_last_time,
        MAX(CASE WHEN cd.sodium IS NOT NULL THEN cd.charttime END) AS sodium_last_time,
        MAX(CASE WHEN cd.hemoglobin IS NOT NULL THEN cd.charttime END) AS hemoglobin_last_time,
        MAX(CASE WHEN cd.platelet IS NOT NULL THEN cd.charttime END) AS platelet_last_time,
        MAX(CASE WHEN cd.bicarbonate IS NOT NULL THEN cd.charttime END) AS bicarbonate_last_time,
        MAX(CASE WHEN cd.chloride IS NOT NULL THEN cd.charttime END) AS chloride_last_time,
        MAX(CASE WHEN cd.lactate IS NOT NULL THEN cd.charttime END) AS lactate_last_time,
        MAX(CASE WHEN cd.hematocrit IS NOT NULL THEN cd.charttime END) AS hematocrit_last_time,
        MAX(CASE WHEN cd.rbc IS NOT NULL THEN cd.charttime END) AS rbc_last_time
    FROM combined_data cd GROUP BY cd.stay_id
),
vital_last_values AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.charttime = vl.heart_rate_last_time THEN cd.heart_rate END) AS heart_rate_last,
        MAX(CASE WHEN cd.charttime = vl.sbp_last_time THEN cd.sbp END) AS sbp_last,
        MAX(CASE WHEN cd.charttime = vl.dbp_last_time THEN cd.dbp END) AS dbp_last,
        MAX(CASE WHEN cd.charttime = vl.respiratory_rate_last_time THEN cd.respiratory_rate END) AS respiratory_rate_last,
        MAX(CASE WHEN cd.charttime = vl.o2_saturation_last_time THEN cd.o2_saturation END) AS o2_saturation_last,
        MAX(CASE WHEN cd.charttime = vl.temperature_last_time THEN cd.temperature END) AS temperature_last
    FROM combined_data cd
    JOIN vital_last vl ON cd.stay_id = vl.stay_id
    GROUP BY cd.stay_id
),
lab_last_values AS (
    SELECT
        cd.stay_id,
        MAX(CASE WHEN cd.charttime = ll.bun_last_time THEN cd.bun END) AS bun_last,
        MAX(CASE WHEN cd.charttime = ll.wbc_last_time THEN cd.wbc END) AS wbc_last,
        MAX(CASE WHEN cd.charttime = ll.potassium_last_time THEN cd.potassium END) AS potassium_last,
        MAX(CASE WHEN cd.charttime = ll.calcium_last_time THEN cd.calcium END) AS calcium_last,
        MAX(CASE WHEN cd.charttime = ll.creatinine_last_time THEN cd.creatinine END) AS creatinine_last,
        MAX(CASE WHEN cd.charttime = ll.glucose_last_time THEN cd.glucose END) AS glucose_last,
        MAX(CASE WHEN cd.charttime = ll.magnesium_last_time THEN cd.magnesium END) AS magnesium_last,
        MAX(CASE WHEN cd.charttime = ll.sodium_last_time THEN cd.sodium END) AS sodium_last,
        MAX(CASE WHEN cd.charttime = ll.hemoglobin_last_time THEN cd.hemoglobin END) AS hemoglobin_last,
        MAX(CASE WHEN cd.charttime = ll.platelet_last_time THEN cd.platelet END) AS platelet_last,
        MAX(CASE WHEN cd.charttime = ll.bicarbonate_last_time THEN cd.bicarbonate END) AS bicarbonate_last,
        MAX(CASE WHEN cd.charttime = ll.chloride_last_time THEN cd.chloride END) AS chloride_last,
        MAX(CASE WHEN cd.charttime = ll.lactate_last_time THEN cd.lactate END) AS lactate_last,
        MAX(CASE WHEN cd.charttime = ll.hematocrit_last_time THEN cd.hematocrit END) AS hematocrit_last,
        MAX(CASE WHEN cd.charttime = ll.rbc_last_time THEN cd.rbc END) AS rbc_last
    FROM combined_data cd
    JOIN lab_last ll ON cd.stay_id = ll.stay_id
    GROUP BY cd.stay_id
)
SELECT
    d.*,
    vlv.heart_rate_last, vlv.sbp_last, vlv.dbp_last, vlv.respiratory_rate_last, vlv.o2_saturation_last, vlv.temperature_last,
    llv.bun_last, llv.wbc_last, llv.potassium_last, llv.calcium_last, llv.creatinine_last, llv.glucose_last, llv.magnesium_last, llv.sodium_last,
    llv.hemoglobin_last, llv.platelet_last, llv.bicarbonate_last, llv.chloride_last, llv.lactate_last, llv.hematocrit_last, llv.rbc_last,
    vs.heart_rate_max, vs.heart_rate_min, vs.sbp_max, vs.sbp_min, vs.dbp_max, vs.dbp_min,
    vs.respiratory_rate_max, vs.respiratory_rate_min, vs.o2_saturation_max, vs.o2_saturation_min, vs.temperature_max, vs.temperature_min,
    ls.bun_max, ls.bun_min, ls.wbc_max, ls.wbc_min, ls.potassium_max, ls.potassium_min, ls.calcium_max, ls.calcium_min,
    ls.creatinine_max, ls.creatinine_min, ls.glucose_max, ls.glucose_min, ls.magnesium_max, ls.magnesium_min, ls.sodium_max, ls.sodium_min,
    ls.hemoglobin_max, ls.hemoglobin_min, ls.platelet_max, ls.platelet_min, ls.bicarbonate_max, ls.bicarbonate_min,
    ls.chloride_max, ls.chloride_min, ls.lactate_max, ls.lactate_min, ls.hematocrit_max, ls.hematocrit_min, ls.rbc_max, ls.rbc_min
FROM demographics d
LEFT JOIN vital_last_values vlv ON d.stay_id = vlv.stay_id
LEFT JOIN lab_last_values llv ON d.stay_id = llv.stay_id
LEFT JOIN vital_stats vs ON d.stay_id = vs.stay_id
LEFT JOIN lab_stats ls ON d.stay_id = ls.stay_id
ORDER BY d.stay_id;

-- ===================================================================
-- 最终统计
-- ===================================================================

-- 统计最终AKI队列（步骤11）
INSERT INTO exclusion_statistics 
SELECT 
    11 as step_number,
    '最终AKI队列' as step_description,
    COUNT(DISTINCT stay_id) as patient_count,
    0 as excluded_count,
    NULL as exclusion_reason
FROM aki_cohort_final;

-- 统计最终对照组（步骤12）
INSERT INTO exclusion_statistics 
SELECT 
    12 as step_number,
    '最终对照组（数据完整）' as step_description,
    COUNT(DISTINCT stay_id) as patient_count,
    (SELECT patient_count FROM exclusion_statistics WHERE step_number = 10) - COUNT(DISTINCT stay_id) as excluded_count,
    '缺少必要的临床数据' as exclusion_reason
FROM control_cohort_final;

-- ===================================================================
-- 输出纳排流程统计结果
-- ===================================================================

-- 显示完整的纳排流程
SELECT 
    step_number AS "步骤",
    step_description AS "描述",
    patient_count AS "剩余患者数",
    excluded_count AS "排除患者数",
    exclusion_reason AS "排除原因"
FROM exclusion_statistics
ORDER BY step_number;

-- 生成CONSORT流程图数据
WITH consort_data AS (
    SELECT 
        0 as sort_order,
        '===== CONSORT流程图数据 =====' AS title
    UNION ALL
    SELECT 
        step_number as sort_order,
        '步骤' || step_number || ': ' || step_description || 
        ' | 剩余: ' || patient_count || 
        ' | 排除: ' || excluded_count || 
        CASE WHEN exclusion_reason IS NOT NULL 
             THEN ' (' || exclusion_reason || ')' 
             ELSE '' 
        END AS title
    FROM exclusion_statistics
)
SELECT title 
FROM consort_data
ORDER BY sort_order;
