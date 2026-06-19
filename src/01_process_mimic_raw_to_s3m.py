#!/usr/bin/env python3
"""Convert MIMIC raw intersection CSVs to S3M Train/Test matrices.

"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


MODEL_COLUMNS = [
    "gender",
    "age",
    "race",
    "heart_rate_last",
    "sbp_last",
    "dbp_last",
    "respiratory_rate_last",
    "o2_saturation_last",
    "temperature_last",
    "bun_last",
    "wbc_last",
    "potassium_last",
    "calcium_last",
    "creatinine_last",
    "glucose_last",
    "magnesium_last",
    "sodium_last",
    "hemoglobin_last",
    "platelet_last",
    "bicarbonate_last",
    "chloride_last",
    "lactate_last",
    "hematocrit_last",
    "rbc_last",
    "heart_rate_max",
    "heart_rate_min",
    "sbp_max",
    "sbp_min",
    "dbp_max",
    "dbp_min",
    "respiratory_rate_max",
    "respiratory_rate_min",
    "o2_saturation_max",
    "o2_saturation_min",
    "temperature_max",
    "temperature_min",
    "bun_max",
    "bun_min",
    "wbc_max",
    "wbc_min",
    "potassium_max",
    "potassium_min",
    "calcium_max",
    "calcium_min",
    "creatinine_max",
    "creatinine_min",
    "glucose_max",
    "glucose_min",
    "magnesium_max",
    "magnesium_min",
    "sodium_max",
    "sodium_min",
    "hemoglobin_max",
    "hemoglobin_min",
    "platelet_max",
    "platelet_min",
    "bicarbonate_max",
    "bicarbonate_min",
    "chloride_max",
    "chloride_min",
    "lactate_max",
    "lactate_min",
    "hematocrit_max",
    "hematocrit_min",
    "rbc_max",
    "rbc_min",
    "label",
]

SUMMARY_COLUMNS = [
    "AKI",
    "age",
    "gender",
    "race_grouped",
    "heart_rate_last",
    "sbp_last",
    "dbp_last",
    "respiratory_rate_last",
    "o2_saturation_last",
    "temperature_last",
    "bun_last",
    "creatinine_last",
    "glucose_last",
    "sodium_last",
    "potassium_last",
    "chloride_last",
    "bicarbonate_last",
    "calcium_last",
    "magnesium_last",
    "hemoglobin_last",
    "hematocrit_last",
    "platelet_last",
    "wbc_last",
    "rbc_last",
    "lactate_last",
    "icu_los_days",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/mimiciv/raw")
    parser.add_argument("--processed-dir", default="data/mimiciv/processed")
    parser.add_argument("--positive-tabular", default="data/mimiciv/data-aki_tabular.csv")
    parser.add_argument("--negative-tabular", default="data/mimiciv/data-control_tabular.csv")
    parser.add_argument(
        "--skip-tabular",
        action="store_true",
        help="Only create S3M matrices and labels; skip processed tabular train/test files.",
    )
    parser.add_argument(
        "--vital-dir",
        default="data/mimiciv/ts_vital",
        help="Output directory for vital-sign S3M Train/Test matrices.",
    )
    parser.add_argument(
        "--lab-dir",
        default="data/mimiciv/ts_lab",
        help="Output directory for laboratory S3M Train/Test matrices.",
    )
    parser.add_argument(
        "--split-anchor-file",
        default="dbp_combined_min5_intersection.csv",
        help="File used to derive train/test split.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--save-flat-labels",
        action="store_true",
        help="Also save y_train_2.csv/y_test_2.csv with stay_id,label columns.",
    )
    return parser.parse_args()


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
    if "asian" in s or s in {"0", "0.0"}:
        return "Asian"
    if "black" in s or s in {"1", "1.0"}:
        return "Black"
    if "hisp" in s or "latino" in s or s in {"2", "2.0"}:
        return "Hispanic"
    if "white" in s or s in {"5", "5.0"}:
        return "White"
    if s in {"", "unknown", "unable to obtain", "patient declined to answer", "4", "4.0"}:
        return "Unknown"
    return "Other"


def normalize_race_code(v) -> int:
    return {
        "Asian": 0,
        "Black": 1,
        "Hispanic": 2,
        "Other": 3,
        "Unknown": 4,
        "White": 5,
    }[normalize_race_label(v)]


def clean_row_float(row: pd.Series, max_length: int | None = None) -> np.ndarray:
    first_non_zero = row.ne(0).idxmax() if not (row == 0).all() else None
    last_non_zero = row.ne(0)[::-1].idxmax() if not (row == 0).all() else None

    if first_non_zero is None or last_non_zero is None:
        return np.zeros(max_length) if max_length is not None else np.array([])

    row_trimmed = row.loc[first_non_zero:last_non_zero]
    row_filled = row_trimmed.replace(0, np.nan).ffill().bfill()
    result = np.round(row_filled.values, 1)

    if max_length is not None:
        if len(result) < max_length:
            pad_len = max_length - len(result)
            first_value = result[0] if len(result) > 0 else 0
            padding = np.full(pad_len, first_value)
            result = np.concatenate([padding, result])
        elif len(result) > max_length:
            result = result[-max_length:]
    return result


def calculate_max_lengths(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    max_lengths: Dict[str, int] = {}
    for key, df in data_dict.items():
        max_length = 0
        for _, row in df.iterrows():
            # Legacy-compatible behavior from notebook: skip stay_id only.
            data_row = row.iloc[1:]
            if not (data_row == 0).all():
                non_zero_indices = data_row.reset_index(drop=True).to_numpy().nonzero()[0]
                if len(non_zero_indices) > 0:
                    first_idx = non_zero_indices[0]
                    last_idx = non_zero_indices[-1]
                    max_length = max(max_length, int(last_idx - first_idx + 1))
        max_lengths[key] = max_length
    return max_lengths


def load_tabular_cohort(args: argparse.Namespace) -> pd.DataFrame:
    positive = pd.read_csv(args.positive_tabular, na_values=["NULL"])
    negative = pd.read_csv(args.negative_tabular, na_values=["NULL"])
    positive["label"] = 1
    negative["label"] = 0
    tabular = pd.concat([positive, negative], ignore_index=True)
    tabular["stay_id"] = tabular["stay_id"].astype(int)
    return tabular.set_index("stay_id", drop=False)


def build_tabular_outputs(
    args: argparse.Namespace,
    anchor_df: pd.DataFrame,
    indices_train: set[int],
    indices_test: set[int],
) -> None:
    tabular = load_tabular_cohort(args)
    anchor_order = anchor_df[["stay_id", "label"]].copy()
    anchor_order["stay_id"] = anchor_order["stay_id"].astype(int)
    missing_ids = sorted(set(anchor_order["stay_id"]) - set(tabular.index))
    if missing_ids:
        raise ValueError(f"Missing tabular rows for stay_id values: {missing_ids[:10]}")

    selected = tabular.loc[anchor_order["stay_id"].tolist()].copy()
    selected["label"] = anchor_order["label"].to_numpy()

    processed_data = selected.copy()
    processed_data["AKI"] = processed_data["label"].map({1: "Yes", 0: "No"})
    processed_data["gender"] = processed_data["gender"].apply(normalize_gender_label)
    processed_data["race_grouped"] = processed_data["race"].apply(normalize_race_label)
    missing_summary = [c for c in SUMMARY_COLUMNS if c not in processed_data.columns]
    if missing_summary:
        raise ValueError(f"Missing columns for processed_data.csv: {missing_summary}")
    processed_data[SUMMARY_COLUMNS].to_csv(
        os.path.join(args.processed_dir, "processed_data.csv"), index=False
    )

    model_data = selected.copy()
    model_data["gender"] = model_data["gender"].apply(normalize_gender_code)
    model_data["race"] = model_data["race"].apply(normalize_race_code)
    missing_model = [c for c in MODEL_COLUMNS if c not in model_data.columns]
    if missing_model:
        raise ValueError(f"Missing columns for processed_train/test.csv: {missing_model}")
    model_data = model_data[MODEL_COLUMNS].copy()
    for col in model_data.columns:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    train_order = anchor_order.loc[anchor_order["stay_id"].isin(indices_train), "stay_id"]
    test_order = anchor_order.loc[anchor_order["stay_id"].isin(indices_test), "stay_id"]
    model_data.loc[train_order.to_list()].to_csv(
        os.path.join(args.processed_dir, "processed_train.csv"), index=False
    )
    model_data.loc[test_order.to_list()].to_csv(
        os.path.join(args.processed_dir, "processed_test.csv"), index=False
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.vital_dir, exist_ok=True)
    os.makedirs(args.lab_dir, exist_ok=True)

    files = sorted(
        f
        for f in os.listdir(args.raw_dir)
        if f.endswith(".csv") and "med" not in f and "combined_min" in f and "intersection" in f
    )
    if not files:
        raise FileNotFoundError(f"No intersection csv files found in {args.raw_dir}")

    data_dict: Dict[str, pd.DataFrame] = {}
    for f in files:
        key = f.split("_")[0]
        df = pd.read_csv(os.path.join(args.raw_dir, f)).fillna(0)
        if "stay_id" not in df.columns or "label" not in df.columns:
            continue
        data_dict[key] = df

    if not data_dict:
        raise ValueError("No valid data files with stay_id/label found.")

    split_key = args.split_anchor_file.split("_")[0]
    if split_key not in data_dict:
        available = ", ".join(sorted(data_dict.keys()))
        raise ValueError(
            f"Split anchor key '{split_key}' not found. Available: {available}"
        )

    base_df = data_dict[split_key]
    X = base_df.drop("label", axis=1)
    y = base_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    train_labels = pd.DataFrame({"stay_id": X_train["stay_id"].values, "label": y_train.values})
    test_labels = pd.DataFrame({"stay_id": X_test["stay_id"].values, "label": y_test.values})

    # Main labels: index = stay_id (compatible with legacy read_csv(index_col=0))
    y_train_out = train_labels.set_index("stay_id")
    y_test_out = test_labels.set_index("stay_id")
    y_train_path = os.path.join(args.processed_dir, "..", "y_train.csv")
    y_test_path = os.path.join(args.processed_dir, "..", "y_test.csv")
    y_train_out.to_csv(y_train_path)
    y_test_out.to_csv(y_test_path)

    if args.save_flat_labels:
        train_labels.to_csv(os.path.join(args.processed_dir, "..", "y_train_2.csv"), index=False)
        test_labels.to_csv(os.path.join(args.processed_dir, "..", "y_test_2.csv"), index=False)

    indices_train = set(y_train_out.index.tolist())
    indices_test = set(y_test_out.index.tolist())

    max_lengths = calculate_max_lengths(data_dict)
    vital_keys = {"heart", "sbp", "dbp", "spo2"}
    lab_keys = {"bun", "creatinine", "potassium"}

    for key, df in data_dict.items():
        if key in vital_keys:
            matrix_dir = args.vital_dir
        elif key in lab_keys:
            matrix_dir = args.lab_dir
        else:
            matrix_dir = args.processed_dir

        train_df = df[df["stay_id"].isin(indices_train)].set_index("stay_id")
        test_df = df[df["stay_id"].isin(indices_test)].set_index("stay_id")
        current_max_length = max_lengths[key]

        train_filename = os.path.join(matrix_dir, f"Train_{key}.csv")
        with open(train_filename, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", lineterminator="\n")
            for _, rows in train_df.iterrows():
                cleaned = clean_row_float(rows.iloc[1:], current_max_length)
                if len(cleaned) > 1:
                    row_cleaned = np.concatenate([[rows.iloc[0]], cleaned])
                    writer.writerow([str(x).strip() for x in row_cleaned])
                else:
                    writer.writerow([rows.iloc[0]])

        test_filename = os.path.join(matrix_dir, f"Test_{key}.csv")
        with open(test_filename, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", lineterminator="\n")
            for _, rows in test_df.iterrows():
                cleaned = clean_row_float(rows.iloc[1:], current_max_length)
                if len(cleaned) > 1:
                    row_cleaned = np.concatenate([[rows.iloc[0]], cleaned])
                    writer.writerow([str(x).strip() for x in row_cleaned])
                else:
                    writer.writerow([rows.iloc[0]])

        print(f"Saved {train_filename}")
        print(f"Saved {test_filename}")

    print("Done.")
    print(f"Split sizes: train={len(indices_train)}, test={len(indices_test)}")
    print(f"Labels: {y_train_path}, {y_test_path}")

    if not args.skip_tabular:
        build_tabular_outputs(args, base_df, indices_train, indices_test)
        print(f"Tabular files written under {args.processed_dir}")


if __name__ == "__main__":
    main()
