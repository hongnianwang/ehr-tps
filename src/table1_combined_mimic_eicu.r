#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(gtsummary)
  library(dplyr)
  library(stringr)
  library(gt)
  library(flextable)
  library(writexl)
})

theme_gtsummary_compact()

args <- commandArgs(trailingOnly = TRUE)
full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- full_args[grepl("^--file=", full_args)]
default_root <- if (length(file_arg) >= 1) {
  normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), ".."))
} else {
  normalizePath(getwd())
}
root_dir <- if (length(args) >= 1) normalizePath(args[[1]]) else default_root
out_dir <- file.path(root_dir, "results", "tableone")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# Utilities
# -----------------------------
format_p <- function(x) {
  case_when(
    is.na(x) ~ "",
    x < 0.001 ~ "<0.001",
    x < 0.01 ~ sprintf("%.3f", x),
    TRUE ~ sprintf("%.2f", x)
  )
}

normalize_gender <- function(x) {
  x <- as.character(x)
  x <- ifelse(x %in% c("Female", "Male"), x, "Unknown")
  factor(x, levels = c("Female", "Male", "Unknown"))
}

normalize_race <- function(x) {
  x <- as.character(x)
  x <- case_when(
    x %in% c("Asian", "Black", "Hispanic", "White", "Other", "Unknown") ~ x,
    TRUE ~ "Unknown"
  )
  factor(x, levels = c("Asian", "Black", "Hispanic", "Other", "Unknown", "White"))
}

build_tbl <- function(df) {
  df2 <- df %>%
    mutate(
      AKI = factor(AKI, levels = c("No", "Yes")),
      gender = normalize_gender(gender),
      race_grouped = normalize_race(race_grouped)
    ) %>%
    select(
      AKI, age, gender, race_grouped,
      heart_rate_last, sbp_last, dbp_last,
      respiratory_rate_last, o2_saturation_last, temperature_last,
      bun_last, creatinine_last,
      sodium_last, potassium_last, chloride_last, bicarbonate_last,
      calcium_last, magnesium_last,
      hemoglobin_last, hematocrit_last, platelet_last,
      wbc_last, rbc_last,
      glucose_last, lactate_last
    )

  df2 %>%
    tbl_summary(
      by = AKI,
      type = all_continuous() ~ "continuous",
      statistic = list(
        all_continuous() ~ "{median} [{p25}, {p75}]",
        all_categorical() ~ "{n} ({p}%)"
      ),
      digits = list(
        all_continuous() ~ 1,
        all_categorical() ~ c(0, 1)
      ),
      label = list(
        age ~ "Age, years",
        gender ~ "Sex",
        race_grouped ~ "Race/ethnicity",
        heart_rate_last ~ "Heart rate, beats/min",
        sbp_last ~ "Systolic blood pressure, mmHg",
        dbp_last ~ "Diastolic blood pressure, mmHg",
        respiratory_rate_last ~ "Respiratory rate, breaths/min",
        o2_saturation_last ~ "Oxygen saturation, %",
        temperature_last ~ "Temperature, °C",
        bun_last ~ "Blood urea nitrogen, mg/dL",
        creatinine_last ~ "Serum creatinine, mg/dL",
        glucose_last ~ "Glucose, mg/dL",
        sodium_last ~ "Sodium, mEq/L",
        potassium_last ~ "Potassium, mEq/L",
        chloride_last ~ "Chloride, mEq/L",
        bicarbonate_last ~ "Bicarbonate, mEq/L",
        calcium_last ~ "Calcium, mg/dL",
        magnesium_last ~ "Magnesium, mg/dL",
        hemoglobin_last ~ "Hemoglobin, g/dL",
        hematocrit_last ~ "Hematocrit, %",
        platelet_last ~ "Platelet count, ×10^3/μL",
        wbc_last ~ "White blood cell count, ×10^3/μL",
        rbc_last ~ "Red blood cell count, ×10^6/μL",
        lactate_last ~ "Lactate, mmol/L"
      ),
      missing = "no"
    ) %>%
    add_p(
      test = list(
        all_continuous() ~ "wilcox.test",
        all_categorical() ~ "chisq.test"
      ),
      pvalue_fun = format_p
    ) %>%
    modify_header(
      label = "**Characteristic**",
      all_stat_cols() ~ "**{level}**<br>(N = {n})",
      p.value = "**P value**"
    ) %>%
    bold_labels()
}

# -----------------------------
# Read data
# -----------------------------
mimic_path <- file.path(root_dir, "data", "mimiciv", "processed_modeling_cohort", "processed_data_modeling_cohort.csv")
if (!file.exists(mimic_path)) {
  stop(
    "Retained MIMIC modeling-cohort file not found: ",
    mimic_path,
    "\nRun src/build_mimic_modeling_dataset.py before generating Table 1."
  )
}
eicu_path <- file.path(root_dir, "data", "eicu", "processed", "processed_test_with_shapelets.csv")

mimic_df <- read.csv(mimic_path, stringsAsFactors = FALSE) %>%
  mutate(
    AKI = as.character(AKI),
    race_grouped = as.character(race_grouped)
  )

eicu_df <- read.csv(eicu_path, stringsAsFactors = FALSE) %>%
  mutate(
    AKI = ifelse(label == 1, "Yes", "No"),
    race_grouped = as.character(race)
  )

# -----------------------------
# Build separate tables
# -----------------------------
tbl_mimic <- build_tbl(mimic_df)
tbl_eicu <- build_tbl(eicu_df)

# Merge into one overall Table 1
# Keep eICU in original prevalence (no downsampling) for methodological consistency
# with external validation metrics and calibration interpretation.
tbl_combined <- tbl_merge(
  tbls = list(tbl_mimic, tbl_eicu),
  tab_spanner = c(
    "**MIMIC-IV (Development; retained cohort)**",
    "**eICU-CRD (External; original prevalence)**"
  )
) %>%
  modify_caption("**Table 1. Baseline Characteristics in MIMIC-IV and eICU-CRD Cohorts**") %>%
  modify_footnote(
    everything() ~ "Median [IQR] for continuous variables; n (%) for categorical variables. P values from Mann-Whitney U test (continuous) and Pearson chi-square test (categorical)."
  )

# Save outputs
out_docx <- file.path(out_dir, "Table1_combined_mimic_eicu.docx")
out_xlsx <- file.path(out_dir, "Table1_combined_mimic_eicu.xlsx")
out_html <- file.path(out_dir, "Table1_combined_mimic_eicu.html")

out_mimic_docx <- file.path(out_dir, "Table1_main_mimic_updated.docx")
out_eicu_docx <- file.path(out_dir, "TableS_external_eicu_updated.docx")

# Main combined table
tbl_combined %>% as_flex_table() %>% save_as_docx(path = out_docx)
tbl_combined %>% as_tibble() %>% write_xlsx(out_xlsx)
tbl_combined %>% as_gt() %>% gtsave(out_html)

# Also save per-dataset tables for supplementary use
tbl_mimic %>%
  modify_caption("**Table 1A. Baseline Characteristics of Development Cohort (MIMIC-IV)**") %>%
  as_flex_table() %>%
  save_as_docx(path = out_mimic_docx)

tbl_eicu %>%
  modify_caption("**Table 1B. Baseline Characteristics of External Validation Cohort (eICU-CRD)**") %>%
  as_flex_table() %>%
  save_as_docx(path = out_eicu_docx)

# Save concise cohort summary
summary_df <- bind_rows(
  tibble(dataset = "MIMIC-IV", AKI = sum(mimic_df$AKI == "Yes"), non_AKI = sum(mimic_df$AKI == "No"), total = nrow(mimic_df)),
  tibble(dataset = "eICU-CRD", AKI = sum(eicu_df$AKI == "Yes"), non_AKI = sum(eicu_df$AKI == "No"), total = nrow(eicu_df))
) %>%
  mutate(prevalence_AKI = round(100 * AKI / total, 2))

write.csv(summary_df, file.path(out_dir, "Table1_dataset_counts.csv"), row.names = FALSE)

cat("Saved files:\n")
cat(out_docx, "\n")
cat(out_xlsx, "\n")
cat(out_html, "\n")
cat(out_mimic_docx, "\n")
cat(out_eicu_docx, "\n")
cat(file.path(out_dir, "Table1_dataset_counts.csv"), "\n")
