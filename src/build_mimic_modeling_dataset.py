#!/usr/bin/env python3
"""
Build the MIMIC-IV modeling cohort with locked shapelet features.

The current shapelet-discovery cohort is kept unchanged:
- existing 835/835 train/test rows remain in their original split;
- only previously unused eligible non-AKI admissions are added;
- added controls are split with the same test fraction as the original controls;
- the locked 10 shapelet features are computed directly from the fixed patterns.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mimiciv"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
EXPERIMENT_DATA_DIR = DATA_DIR / "processed_modeling_cohort"
EXPERIMENT_RESULTS_DIR = RESULTS_DIR / "mimic_modeling_cohort_experiment"

TS_VARS = ["heart_rate", "sbp", "dbp", "spo2", "bun", "creatinine", "potassium"]
PREFIX_TO_RAW = {
    "heart": "heart_rate",
    "sbp": "sbp",
    "dbp": "dbp",
    "spo2": "spo2",
    "bun": "bun",
    "creatinine": "creatinine",
    "potassium": "potassium",
}
VITAL_PREFIXES = {"heart", "sbp", "dbp", "spo2"}
LAB_PREFIXES = {"bun", "creatinine", "potassium"}


@dataclass(frozen=True)
class SplitInfo:
    added_train_ids: list[int]
    added_test_ids: list[int]
    eligible_control_ids: list[int]
    current_control_ids_original: list[int]
    retained_current_control_ids: list[int]
    excluded_current_control_ids: list[int]
    current_aki_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the MIMIC-IV modeling cohort and locked shapelet features"
    )
    parser.add_argument(
        "--data-out-dir",
        type=Path,
        default=EXPERIMENT_DATA_DIR,
        help="Directory for generated modeling-cohort dataset files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_RESULTS_DIR,
        help="Directory for model metrics and predictions when --skip-models is not used.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Only build datasets and shapelet features; do not train models.",
    )
    return parser.parse_args()


def is_shapelet_col(col: str) -> bool:
    if "_" not in col or "-" not in col:
        return False
    return col.split("_", 1)[0] in PREFIX_TO_RAW


def normalize_gender_code(v) -> int:
    if pd.isna(v):
        return -1
    s = str(v).strip().lower()
    if s in {"0", "0.0", "female", "f"}:
        return 0
    if s in {"1", "1.0", "male", "m"}:
        return 1
    return -1


def normalize_gender_label(v) -> str:
    code = normalize_gender_code(v)
    if code == 0:
        return "Female"
    if code == 1:
        return "Male"
    return "Unknown"


def normalize_race_label(v) -> str:
    if pd.isna(v):
        return "Unknown"
    s = str(v).strip().lower()
    code_map = {
        "0": "Asian",
        "0.0": "Asian",
        "1": "Black",
        "1.0": "Black",
        "2": "Hispanic",
        "2.0": "Hispanic",
        "3": "Other",
        "3.0": "Other",
        "4": "Unknown",
        "4.0": "Unknown",
        "5": "White",
        "5.0": "White",
    }
    if s in code_map:
        return code_map[s]
    if "asian" in s:
        return "Asian"
    if "black" in s:
        return "Black"
    if "hisp" in s or "latino" in s:
        return "Hispanic"
    if "white" in s:
        return "White"
    if s in {
        "",
        "unknown",
        "unable to obtain",
        "patient declined to answer",
        "declined",
    }:
        return "Unknown"
    return "Other"


def normalize_race_code(v) -> int:
    label = normalize_race_label(v)
    return {
        "Asian": 0,
        "Black": 1,
        "Hispanic": 2,
        "Other": 3,
        "Unknown": 4,
        "White": 5,
    }[label]


def clean_row_float(values: pd.Series | np.ndarray, max_length: int) -> np.ndarray:
    row = pd.Series(values).fillna(0)
    if (row == 0).all():
        return np.zeros(max_length)

    first_non_zero = row.ne(0).idxmax()
    last_non_zero = row.ne(0)[::-1].idxmax()
    row_trimmed = row.loc[first_non_zero:last_non_zero]
    row_filled = row_trimmed.replace(0, np.nan).ffill().bfill()
    result = np.round(row_filled.values.astype(float), 1)

    if len(result) < max_length:
        padding = np.full(max_length - len(result), result[0] if len(result) else 0)
        result = np.concatenate([padding, result])
    elif len(result) > max_length:
        result = result[-max_length:]
    return result


def min_subsequence_distance(time_series: np.ndarray, shapelet: np.ndarray) -> float:
    ts = np.asarray(time_series, dtype=float)
    shp = np.asarray(shapelet, dtype=float)
    if len(ts) < len(shp):
        ts, shp = shp, ts

    best = math.inf
    n = len(shp)
    for start in range(0, len(ts) - n + 1):
        cur = 0.0
        window = ts[start : start + n]
        for i in range(n):
            diff = window[i] - shp[i]
            cur += diff * diff
            if cur >= best:
                break
        if cur < best:
            best = cur
    return float(best)


def parse_shapelet_from_col(col: str) -> np.ndarray:
    return np.array([float(x) for x in col.split("_", 1)[1].split("-")], dtype=float)


def get_eligible_control_ids() -> list[int]:
    ts = pd.read_csv(
        DATA_DIR / "data-control_ts.csv",
        usecols=["stay_id"] + TS_VARS,
        na_values=["NULL"],
    )
    counts = ts.groupby("stay_id")[TS_VARS].count()
    min5_ids = set(counts.index[(counts[TS_VARS] >= 5).all(axis=1)].astype(int))

    control_tab = pd.read_csv(
        DATA_DIR / "data-control_tabular.csv",
        usecols=["stay_id", "icu_los_days"],
        na_values=["NULL"],
    )
    los_ids = set(
        control_tab.loc[
            pd.to_numeric(control_tab["icu_los_days"], errors="coerce") >= 3,
            "stay_id",
        ].astype(int)
    )
    return sorted(min5_ids & los_ids)


def make_splits(random_state: int, out_dir: Path) -> SplitInfo:
    current_train = pd.read_csv(PROCESSED_DIR / "processed_train_with_shapelets.csv")
    current_test = pd.read_csv(PROCESSED_DIR / "processed_test_with_shapelets.csv")
    current_all = pd.concat([current_train, current_test], ignore_index=True)

    current_aki_ids = sorted(current_all.loc[current_all["label"] == 1, "stay_id"].astype(int))
    current_control_ids = sorted(
        current_all.loc[current_all["label"] == 0, "stay_id"].astype(int)
    )
    eligible_control_ids = get_eligible_control_ids()
    eligible_set = set(eligible_control_ids)

    original_current_set = set(current_control_ids)
    retained_current_control_ids = sorted(original_current_set & eligible_set)
    excluded_current_control_ids = sorted(original_current_set - eligible_set)
    added_control_ids = sorted(eligible_set - original_current_set)

    original_train_control_n = int((current_train["label"] == 0).sum())
    original_test_control_n = int((current_test["label"] == 0).sum())
    original_control_n = len(current_control_ids)
    target_test_control_n = math.ceil(
        len(eligible_control_ids) * original_test_control_n / original_control_n
    )
    current_retained_test_n = int(
        (
            (current_test["label"] == 0)
            & (current_test["stay_id"].astype(int).isin(eligible_set))
        ).sum()
    )
    added_test_n = target_test_control_n - current_retained_test_n
    if added_test_n <= 0 or added_test_n >= len(added_control_ids):
        raise RuntimeError(
            f"Invalid added-test count: {added_test_n} for {len(added_control_ids)} added controls"
        )
    added_train_ids, added_test_ids = train_test_split(
        added_control_ids,
        test_size=added_test_n,
        random_state=random_state,
        shuffle=True,
    )

    pd.DataFrame({"stay_id": eligible_control_ids}).to_csv(
        out_dir / "eligible_control_ids.csv", index=False
    )
    pd.DataFrame({"stay_id": added_train_ids}).to_csv(
        out_dir / "added_control_train_ids.csv", index=False
    )
    pd.DataFrame({"stay_id": added_test_ids}).to_csv(
        out_dir / "added_control_test_ids.csv", index=False
    )
    pd.DataFrame({"stay_id": excluded_current_control_ids}).to_csv(
        out_dir / "excluded_current_controls_outside_modeling_layer.csv", index=False
    )
    pd.DataFrame({"stay_id": retained_current_control_ids}).to_csv(
        out_dir / "retained_current_controls_inside_modeling_layer.csv", index=False
    )

    return SplitInfo(
        added_train_ids=[int(x) for x in added_train_ids],
        added_test_ids=[int(x) for x in added_test_ids],
        eligible_control_ids=eligible_control_ids,
        current_control_ids_original=current_control_ids,
        retained_current_control_ids=retained_current_control_ids,
        excluded_current_control_ids=excluded_current_control_ids,
        current_aki_ids=current_aki_ids,
    )


def build_added_tabular(ids: list[int], base_cols: list[str]) -> pd.DataFrame:
    control_tab = pd.read_csv(DATA_DIR / "data-control_tabular.csv", na_values=["NULL"])
    control_tab["stay_id"] = control_tab["stay_id"].astype(int)
    tab = control_tab.loc[control_tab["stay_id"].isin(ids)].copy()
    tab = tab.set_index("stay_id").loc[ids].reset_index()

    missing = [c for c in base_cols if c not in tab.columns]
    if missing:
        raise ValueError(f"Missing tabular columns in data-control_tabular.csv: {missing}")

    tab = tab[base_cols].copy()
    tab["gender"] = tab["gender"].apply(normalize_gender_code)
    tab["race"] = tab["race"].apply(normalize_race_code)
    for col in tab.columns:
        if col != "stay_id":
            tab[col] = pd.to_numeric(tab[col], errors="coerce")
    return tab


def read_added_control_ts(all_added_ids: list[int]) -> pd.DataFrame:
    ts = pd.read_csv(
        DATA_DIR / "data-control_ts.csv",
        usecols=["stay_id", "icu_intime", "charttime"] + TS_VARS,
        na_values=["NULL"],
    )
    ts["stay_id"] = ts["stay_id"].astype(int)
    ts = ts.loc[ts["stay_id"].isin(all_added_ids)].copy()
    ts["charttime"] = pd.to_datetime(ts["charttime"])
    ts["icu_intime"] = pd.to_datetime(ts["icu_intime"])
    ts["hours_from_admission"] = (
        (ts["charttime"] - ts["icu_intime"]).dt.total_seconds() / 3600
    ).round().astype(int)
    return ts


def build_cleaned_series(
    ts: pd.DataFrame,
    ids: list[int],
    raw_var: str,
    bins: list[int],
    bin_hours: int,
) -> np.ndarray:
    sub = ts[["stay_id", "hours_from_admission", raw_var]].dropna(subset=[raw_var]).copy()
    if bin_hours > 1:
        sub["time_bin"] = (sub["hours_from_admission"] // bin_hours) * bin_hours
    else:
        sub["time_bin"] = sub["hours_from_admission"]

    wide = sub.pivot_table(
        index="stay_id", columns="time_bin", values=raw_var, aggfunc="mean"
    )
    wide = wide.reindex(index=ids, columns=bins).fillna(0)
    max_length = len(bins)
    return np.vstack(
        [clean_row_float(wide.loc[stay_id].values, max_length) for stay_id in ids]
    )


def compute_shapelet_features_for_added(
    all_added_ids: list[int], shapelet_cols: list[str], out_dir: Path
) -> pd.DataFrame:
    ts = read_added_control_ts(all_added_ids)
    by_prefix: dict[str, list[str]] = {}
    for col in shapelet_cols:
        by_prefix.setdefault(col.split("_", 1)[0], []).append(col)

    shapelet_features = pd.DataFrame({"stay_id": all_added_ids})
    vital_bins = list(range(-6, 61))
    lab_bins = list(range(-8, 61, 4))

    for prefix, cols in by_prefix.items():
        raw_var = PREFIX_TO_RAW[prefix]
        if prefix in VITAL_PREFIXES:
            bins = vital_bins
            bin_hours = 1
        elif prefix in LAB_PREFIXES:
            bins = lab_bins
            bin_hours = 4
        else:
            raise ValueError(f"Unknown shapelet prefix: {prefix}")

        cleaned = build_cleaned_series(ts, all_added_ids, raw_var, bins, bin_hours)
        for col in cols:
            shapelet = parse_shapelet_from_col(col)
            shapelet_features[col] = [
                min_subsequence_distance(row, shapelet) for row in cleaned
            ]

    shapelet_features.to_csv(out_dir / "added_control_shapelet_features.csv", index=False)
    return shapelet_features


def validate_current_shapelets(shapelet_cols: list[str], out_dir: Path, n_check: int = 50) -> dict:
    current_train = pd.read_csv(PROCESSED_DIR / "processed_train_with_shapelets.csv")
    check = current_train.head(n_check).copy()
    raw_file = {
        "heart": "heart_rate_combined_min5_intersection.csv",
        "sbp": "sbp_combined_min5_intersection.csv",
        "dbp": "dbp_combined_min5_intersection.csv",
        "spo2": "spo2_combined_min5_intersection.csv",
        "bun": "bun_combined_min5_bin4h_intersection.csv",
        "creatinine": "creatinine_combined_min5_bin4h_intersection.csv",
        "potassium": "potassium_combined_min5_bin4h_intersection.csv",
    }
    raw_cache = {
        prefix: pd.read_csv(DATA_DIR / "raw" / filename).fillna(0).set_index("stay_id")
        for prefix, filename in raw_file.items()
    }

    max_abs_diff = 0.0
    checked = 0
    for _, row in check.iterrows():
        stay_id = int(row["stay_id"])
        for col in shapelet_cols:
            prefix = col.split("_", 1)[0]
            raw_row = raw_cache[prefix].loc[stay_id]
            cleaned = clean_row_float(raw_row.iloc[1:].values, len(raw_row) - 1)
            calc = min_subsequence_distance(cleaned, parse_shapelet_from_col(col))
            diff = abs(calc - float(row[col]))
            max_abs_diff = max(max_abs_diff, diff)
            checked += 1

    result = {"n_rows_checked": int(len(check)), "n_distances_checked": checked, "max_abs_diff": max_abs_diff}
    (out_dir / "shapelet_reproduction_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def build_processed_modeling_data(split_info: SplitInfo, out_dir: Path) -> None:
    processed_cols = pd.read_csv(PROCESSED_DIR / "processed_data.csv", nrows=0).columns.tolist()
    aki_tab = pd.read_csv(DATA_DIR / "data-aki_tabular.csv", na_values=["NULL"])
    control_tab = pd.read_csv(DATA_DIR / "data-control_tabular.csv", na_values=["NULL"])
    aki_tab["stay_id"] = aki_tab["stay_id"].astype(int)
    control_tab["stay_id"] = control_tab["stay_id"].astype(int)

    aki = aki_tab.loc[aki_tab["stay_id"].isin(split_info.current_aki_ids)].copy()
    control = control_tab.loc[
        control_tab["stay_id"].isin(split_info.eligible_control_ids)
    ].copy()
    aki["AKI"] = "Yes"
    control["AKI"] = "No"
    combined = pd.concat([aki, control], ignore_index=True)
    combined["gender"] = combined["gender"].apply(normalize_gender_label)
    combined["race_grouped"] = combined["race"].apply(normalize_race_label)

    missing = [c for c in processed_cols if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing processed_data columns: {missing}")
    combined = combined[processed_cols].copy()
    combined.to_csv(out_dir / "processed_data_modeling_cohort.csv", index=False)


def build_full_datasets(split_info: SplitInfo, out_dir: Path) -> tuple[Path, Path]:
    current_train = pd.read_csv(PROCESSED_DIR / "processed_train_with_shapelets.csv")
    current_test = pd.read_csv(PROCESSED_DIR / "processed_test_with_shapelets.csv")

    train_cols = current_train.columns.tolist()
    shapelet_cols = [c for c in train_cols if is_shapelet_col(c)]
    base_cols = [c for c in train_cols if c not in shapelet_cols + ["label"]]

    reproduction = validate_current_shapelets(shapelet_cols, out_dir)
    if reproduction["max_abs_diff"] > 1e-8:
        raise RuntimeError(f"Shapelet reproduction check failed: {reproduction}")

    all_added_ids = split_info.added_train_ids + split_info.added_test_ids
    added_tab = build_added_tabular(all_added_ids, base_cols)
    added_shapelets = compute_shapelet_features_for_added(
        all_added_ids, shapelet_cols, out_dir
    )
    added = added_tab.merge(added_shapelets, on="stay_id", how="left")
    added["label"] = 0
    added = added[train_cols]

    added_train = added.loc[added["stay_id"].isin(split_info.added_train_ids)].copy()
    added_test = added.loc[added["stay_id"].isin(split_info.added_test_ids)].copy()
    added_train = added_train.set_index("stay_id").loc[split_info.added_train_ids].reset_index()
    added_test = added_test.set_index("stay_id").loc[split_info.added_test_ids].reset_index()

    eligible_controls = set(split_info.eligible_control_ids)
    current_train_kept = current_train.loc[
        (current_train["label"] == 1)
        | (
            (current_train["label"] == 0)
            & current_train["stay_id"].astype(int).isin(eligible_controls)
        )
    ].copy()
    current_test_kept = current_test.loc[
        (current_test["label"] == 1)
        | (
            (current_test["label"] == 0)
            & current_test["stay_id"].astype(int).isin(eligible_controls)
        )
    ].copy()

    full_train = pd.concat([current_train_kept[train_cols], added_train[train_cols]], ignore_index=True)
    full_test = pd.concat([current_test_kept[train_cols], added_test[train_cols]], ignore_index=True)

    train_path = out_dir / "processed_train_with_shapelets.csv"
    test_path = out_dir / "processed_test_with_shapelets.csv"
    full_train.to_csv(train_path, index=False)
    full_test.to_csv(test_path, index=False)

    split_counts = pd.DataFrame(
        [
            {
                "split": "train",
                "total": len(full_train),
                "AKI": int((full_train["label"] == 1).sum()),
                "non_AKI": int((full_train["label"] == 0).sum()),
                "AKI_prevalence": float(full_train["label"].mean()),
                "existing_rows_retained": len(current_train_kept),
                "existing_controls_excluded_outside_modeling_layer": int(
                    ((current_train["label"] == 0) & ~current_train["stay_id"].astype(int).isin(eligible_controls)).sum()
                ),
                "added_non_AKI": len(added_train),
            },
            {
                "split": "test",
                "total": len(full_test),
                "AKI": int((full_test["label"] == 1).sum()),
                "non_AKI": int((full_test["label"] == 0).sum()),
                "AKI_prevalence": float(full_test["label"].mean()),
                "existing_rows_retained": len(current_test_kept),
                "existing_controls_excluded_outside_modeling_layer": int(
                    ((current_test["label"] == 0) & ~current_test["stay_id"].astype(int).isin(eligible_controls)).sum()
                ),
                "added_non_AKI": len(added_test),
            },
            {
                "split": "combined",
                "total": len(full_train) + len(full_test),
                "AKI": int((full_train["label"] == 1).sum() + (full_test["label"] == 1).sum()),
                "non_AKI": int((full_train["label"] == 0).sum() + (full_test["label"] == 0).sum()),
                "AKI_prevalence": float(
                    (full_train["label"].sum() + full_test["label"].sum())
                    / (len(full_train) + len(full_test))
                ),
                "existing_rows_retained": len(current_train_kept) + len(current_test_kept),
                "existing_controls_excluded_outside_modeling_layer": len(
                    split_info.excluded_current_control_ids
                ),
                "added_non_AKI": len(added_train) + len(added_test),
            },
        ]
    )
    split_counts.to_csv(out_dir / "split_counts.csv", index=False)
    return train_path, test_path


def get_feature_sets(train_df: pd.DataFrame) -> dict[int, list[str]]:
    demographic_cols = ["gender", "age", "race"]
    last_cols = [col for col in train_df.columns if "_last" in col]
    min_cols = [col for col in train_df.columns if "_min" in col]
    max_cols = [col for col in train_df.columns if "_max" in col]
    shapelet_cols = [
        col
        for col in train_df.columns
        if col not in demographic_cols
        and "_last" not in col
        and "_min" not in col
        and "_max" not in col
        and col not in {"stay_id", "label"}
        and "_" in col
    ]
    return {
        1: demographic_cols + last_cols,
        2: demographic_cols + last_cols + min_cols + max_cols,
        3: demographic_cols + last_cols + min_cols + max_cols + shapelet_cols,
    }


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], model_type: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[int] | list[str]]:
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    categorical_cols = [col for col in ["gender", "race"] if col in features]

    if model_type == "catboost":
        for col in categorical_cols:
            X_train[col] = X_train[col].apply(
                normalize_gender_label if col == "gender" else normalize_race_label
            )
            X_test[col] = X_test[col].apply(
                normalize_gender_label if col == "gender" else normalize_race_label
            )
            X_train[col] = X_train[col].astype(str).fillna("NA")
            X_test[col] = X_test[col].astype(str).fillna("NA")
        categorical_info = [X_train.columns.get_loc(col) for col in categorical_cols]
    elif model_type == "lightgbm":
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")
        categorical_info = categorical_cols
    else:
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(pd.concat([X_train[col], X_test[col]], ignore_index=True).astype(str))
            X_train[col] = encoder.transform(X_train[col].astype(str))
            X_test[col] = encoder.transform(X_test[col].astype(str))
        categorical_info = []

    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
    if numeric_cols:
        train_means = X_train[numeric_cols].apply(pd.to_numeric, errors="coerce").mean()
        X_train[numeric_cols] = X_train[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(train_means)
        X_test[numeric_cols] = X_test[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(train_means)
    return X_train, X_test, categorical_info


def build_model(model_type: str):
    if model_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=1000,
            learning_rate=0.08,
            max_depth=6,
            min_child_weight=4,
            subsample=0.70,
            colsample_bytree=0.65,
            reg_alpha=0.30,
            reg_lambda=0.25,
            gamma=0.02,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            verbosity=-1,
            random_state=42,
            force_row_wise=True,
            deterministic=True,
            learning_rate=0.05,
            num_leaves=96,
            max_depth=6,
            n_estimators=1400,
            min_child_samples=30,
            subsample=0.82,
            colsample_bytree=0.72,
            reg_lambda=0.3,
            reg_alpha=0.001,
            min_split_gain=0.05,
            n_jobs=-1,
        )
    if model_type == "catboost":
        return cb.CatBoostClassifier(
            iterations=650,
            learning_rate=0.18,
            depth=7,
            l2_leaf_reg=7,
            random_strength=1.5,
            bagging_temperature=0.75,
            random_state=42,
            verbose=False,
            thread_count=-1,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def auprc_trapz(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_fn(y_true[idx], y_prob[idx]))
    if not values:
        return (np.nan, np.nan)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    pos = y_true == 1
    neg = y_true == 0
    a_pos, a_neg = pred_a[pos], pred_a[neg]
    b_pos, b_neg = pred_b[pos], pred_b[neg]
    m, n = len(a_pos), len(a_neg)
    if m < 2 or n < 2:
        return (np.nan, np.nan)

    def structural_components(pos_scores, neg_scores):
        comp = (
            (pos_scores[:, None] > neg_scores[None, :]).astype(float)
            + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).astype(float)
        )
        return comp.mean(axis=1), comp.mean(axis=0), comp.mean()

    v10_a, v01_a, auc_a = structural_components(a_pos, a_neg)
    v10_b, v01_b, auc_b = structural_components(b_pos, b_neg)
    sx = np.cov(np.vstack([v10_a, v10_b]), ddof=1) / m
    sy = np.cov(np.vstack([v01_a, v01_b]), ddof=1) / n
    cov = sx + sy
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return (np.nan, np.nan)
    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return (float(z), float(p))


def paired_auprc_bootstrap_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    observed = auprc_trapz(y_true, pred_a) - auprc_trapz(y_true, pred_b)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(auprc_trapz(y_true[idx], pred_a[idx]) - auprc_trapz(y_true[idx], pred_b[idx]))
    if not diffs:
        return (observed, np.nan, np.nan, np.nan)
    diffs = np.asarray(diffs)
    p = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return (
        float(observed),
        float(p),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = auprc_trapz(y_true, y_prob)
    auroc_ci = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap, random_state)
    auprc_ci = bootstrap_ci(y_true, y_prob, auprc_trapz, n_bootstrap, random_state + 17)
    return {
        "AUROC": auroc,
        "AUROC_ci_low": auroc_ci[0],
        "AUROC_ci_high": auroc_ci[1],
        "AUPRC": auprc,
        "AUPRC_ci_low": auprc_ci[0],
        "AUPRC_ci_high": auprc_ci[1],
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Accuracy_0.5": float(accuracy_score(y_true, y_pred)),
        "Precision_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1_0.5": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_models(
    train_path: Path,
    test_path: Path,
    out_dir: Path,
    metadata_dir: Path,
    n_bootstrap: int,
    random_state: int,
) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_sets = get_feature_sets(train_df)
    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    metrics_rows = []
    delong_rows = []
    auprc_test_rows = []

    for model_type in ["xgboost", "lightgbm", "catboost"]:
        pred_frame = pd.DataFrame({"stay_id": test_df["stay_id"], "y_true": y_test})
        version_probs = {}

        for version, features in feature_sets.items():
            X_train, X_test, categorical_info = prepare_features(
                train_df, test_df, features, model_type
            )
            model = build_model(model_type)
            if model_type == "lightgbm":
                model.fit(
                    X_train,
                    y_train,
                    categorical_feature=categorical_info if categorical_info else None,
                )
            elif model_type == "catboost":
                model.fit(
                    X_train,
                    y_train,
                    cat_features=categorical_info if categorical_info else None,
                )
            else:
                model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            version_probs[version] = y_prob
            pred_frame[f"v{version}_prob"] = y_prob
            metrics = evaluate_predictions(
                y_test, y_prob, n_bootstrap=n_bootstrap, random_state=random_state + version
            )
            metrics_rows.append(
                {
                    "model": model_type,
                    "version": f"V{version}",
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "n_test_AKI": int(y_test.sum()),
                    "n_test_non_AKI": int((y_test == 0).sum()),
                    "n_features": len(features),
                    **metrics,
                }
            )

        pred_frame.to_csv(out_dir / f"{model_type}_predictions.csv", index=False)

        comparisons = [("V2_vs_V1", 2, 1), ("V3_vs_V1", 3, 1), ("V3_vs_V2", 3, 2)]
        for name, a, b in comparisons:
            z, p = delong_test(y_test, version_probs[a], version_probs[b])
            delong_rows.append(
                {
                    "model": model_type,
                    "comparison": name,
                    "AUROC_a": roc_auc_score(y_test, version_probs[a]),
                    "AUROC_b": roc_auc_score(y_test, version_probs[b]),
                    "AUROC_diff": roc_auc_score(y_test, version_probs[a])
                    - roc_auc_score(y_test, version_probs[b]),
                    "z": z,
                    "p_value": p,
                }
            )
            diff, p_prc, ci_low, ci_high = paired_auprc_bootstrap_test(
                y_test,
                version_probs[a],
                version_probs[b],
                n_bootstrap=n_bootstrap,
                random_state=random_state + a * 100 + b,
            )
            auprc_test_rows.append(
                {
                    "model": model_type,
                    "comparison": name,
                    "AUPRC_diff": diff,
                    "p_value": p_prc,
                    "diff_ci_low": ci_low,
                    "diff_ci_high": ci_high,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    delong_df = pd.DataFrame(delong_rows)
    auprc_tests_df = pd.DataFrame(auprc_test_rows)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    delong_df.to_csv(out_dir / "delong_tests.csv", index=False)
    auprc_tests_df.to_csv(out_dir / "auprc_paired_bootstrap_tests.csv", index=False)

    lines = [
        "# MIMIC-IV Modeling Cohort Experiment",
        "",
        "## Split Counts",
        "",
        pd.read_csv(metadata_dir / "split_counts.csv").to_markdown(index=False),
        "",
        "## AUROC and AUPRC",
        "",
        metrics_df[
            [
                "model",
                "version",
                "n_features",
                "AUROC",
                "AUROC_ci_low",
                "AUROC_ci_high",
                "AUPRC",
                "AUPRC_ci_low",
                "AUPRC_ci_high",
                "Brier",
            ]
        ].to_markdown(index=False),
        "",
        "## DeLong Tests",
        "",
        delong_df.to_markdown(index=False),
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_out_dir = Path(args.data_out_dir)
    out_dir = Path(args.out_dir)
    data_out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_info = make_splits(args.random_state, data_out_dir)
    run_config = {
        "target_layer": "modeling_cohort",
        "random_state": args.random_state,
        "bootstrap": args.bootstrap,
        "eligible_controls": len(split_info.eligible_control_ids),
        "current_aki": len(split_info.current_aki_ids),
        "current_controls_original": len(split_info.current_control_ids_original),
        "current_controls_retained_inside_modeling_layer": len(
            split_info.retained_current_control_ids
        ),
        "current_controls_excluded_outside_modeling_layer": len(
            split_info.excluded_current_control_ids
        ),
        "added_controls": len(split_info.added_train_ids) + len(split_info.added_test_ids),
        "added_controls_train": len(split_info.added_train_ids),
        "added_controls_test": len(split_info.added_test_ids),
    }
    (data_out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    build_processed_modeling_data(split_info, data_out_dir)
    train_path, test_path = build_full_datasets(split_info, data_out_dir)
    if not args.skip_models:
        run_models(
            train_path,
            test_path,
            out_dir,
            data_out_dir,
            args.bootstrap,
            args.random_state,
        )

    print(f"data_out_dir={data_out_dir}")
    print(f"results_out_dir={out_dir}")


if __name__ == "__main__":
    main()
