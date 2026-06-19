#!/usr/bin/env python3
"""
Integrate eICU shapelet distance CSVs into final tabular features.

Input:
1) Test_*_metrics.csv from ShapeletEvaluation.py
2) eICU tabular test file
3) eICU ts_stay_order.csv (keeps row-order alignment for distance vectors)
4) Optional MIMIC reference feature file for shapelet-column alignment

Output:
- X_test_with_shapelets.csv
- processed_test_with_shapelets.csv
- shapelet_test_features_raw.csv
"""

import argparse
import ast
import os
import re
from typing import List

import numpy as np
import pandas as pd


ALLOWED_PREFIX = {"heart", "sbp", "dbp", "spo2", "bun", "creatinine", "potassium"}


def parse_numeric_list(value) -> List[float]:
    if isinstance(value, list):
        return [float(x) if x is not None else np.nan for x in value]
    if pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                out = []
                for x in obj:
                    if x is None:
                        out.append(np.nan)
                    else:
                        try:
                            out.append(float(x))
                        except Exception:
                            out.append(np.nan)
                return out
        except Exception:
            pass

        tokens = re.split(r"[,\s]+", s[1:-1].strip())
        out = []
        for tok in tokens:
            if tok == "":
                continue
            if tok.lower() in {"nan", "none"}:
                out.append(np.nan)
            else:
                try:
                    out.append(float(tok))
                except Exception:
                    out.append(np.nan)
        return out

    try:
        return [float(s)]
    except Exception:
        return []


def shapelet_to_feature_key(shapelet_value) -> str:
    arr = parse_numeric_list(shapelet_value)
    if len(arr) == 0:
        return ""
    return "-".join(str(float(x)) for x in arr)


def build_shapelet_feature_df(metrics_dir: str, n_rows: int, top_k_each_file: int) -> pd.DataFrame:
    columns_dict = {}

    files = sorted(
        f
        for f in os.listdir(metrics_dir)
        if f.startswith("Test_") and f.endswith("_metrics.csv")
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No Test_*_metrics.csv found in {metrics_dir}")

    for filename in files:
        file_path = os.path.join(metrics_dir, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        if df.empty:
            continue
        if "shapelet" not in df.columns or "distances" not in df.columns:
            continue

        parts = filename.split("_")
        if len(parts) < 2:
            continue
        prefix = parts[1]
        if prefix not in ALLOWED_PREFIX:
            continue

        if "p_val" in df.columns:
            df = df.sort_values("p_val", ascending=True)
        df = df.reset_index(drop=True)

        limit = min(top_k_each_file, len(df))
        for i in range(limit):
            key = shapelet_to_feature_key(df.loc[i, "shapelet"])
            if key == "":
                continue
            col_name = f"{prefix}_{key}"
            if col_name in columns_dict:
                # avoid duplicate shapelet feature names
                continue

            dists = parse_numeric_list(df.loc[i, "distances"])
            if len(dists) == 0:
                continue
            if len(dists) > n_rows:
                dists = dists[:n_rows]
            elif len(dists) < n_rows:
                dists.extend([np.nan] * (n_rows - len(dists)))
            columns_dict[col_name] = dists

    return pd.DataFrame(columns_dict, index=range(n_rows))


def get_reference_shapelet_columns(reference_feature_file: str) -> List[str]:
    ref = pd.read_csv(reference_feature_file, nrows=1)
    cols = []
    for c in ref.columns:
        if "-" not in c:
            continue
        if "_" not in c:
            continue
        pref = c.split("_", 1)[0]
        if pref in ALLOWED_PREFIX:
            cols.append(c)
    return cols


def min_subsequence_distance(time_series: np.ndarray, shapelet: np.ndarray) -> float:
    ts = np.asarray(time_series, dtype=float)
    shp = np.asarray(shapelet, dtype=float)
    if len(ts) < len(shp):
        ts, shp = shp, ts

    min_dist = np.inf
    n = len(shp)
    for start in range(0, len(ts) - n + 1):
        cur = 0.0
        for i in range(n):
            diff = shp[i] - ts[start + i]
            cur += diff * diff
            if cur >= min_dist:
                break
        if cur < min_dist:
            min_dist = cur
    return float(min_dist)


def parse_shapelet_from_feature_name(feature_name: str) -> np.ndarray:
    # e.g. "heart_88.0-98.0-82.0-86.3"
    arr_str = feature_name.split("_", 1)[1]
    vals = [float(x) for x in arr_str.split("-") if x != ""]
    return np.array(vals, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Integrate eICU shapelet distance features")
    parser.add_argument(
        "--metrics_dir",
        default="data/eicu/shapelet_csv_results",
        help="Directory containing Test_*_metrics.csv",
    )
    parser.add_argument(
        "--tabular_file",
        default="data/eicu/data-test_tabular.csv",
        help="eICU tabular test csv (must include stay_id,label)",
    )
    parser.add_argument(
        "--ts_order_file",
        default="data/eicu/ts_stay_order.csv",
        help="ts row-order mapping file with stay_id,label",
    )
    parser.add_argument(
        "--reference_feature_file",
        default="data/mimiciv/processed/X_train_with_shapelets.csv",
        help="MIMIC reference feature csv to align shapelet columns",
    )
    parser.add_argument(
        "--top_k_each_file",
        type=int,
        default=20,
        help="Top-k shapelets kept from each Test_*_metrics.csv by p_val",
    )
    parser.add_argument(
        "--output_dir",
        default="data/eicu/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--compute_missing_from_test_ts",
        action="store_true",
        help="If set, compute missing aligned shapelet columns directly from eICU Test_*.csv",
    )
    parser.add_argument(
        "--eicu_ts_vital_dir",
        default="data/eicu/ts_vital",
        help="Directory containing eICU Test_heart/sbp/dbp/spo2.csv",
    )
    parser.add_argument(
        "--eicu_ts_lab_dir",
        default="data/eicu/ts_lab",
        help="Directory containing eICU Test_bun/creatinine/potassium.csv",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ts_order = pd.read_csv(args.ts_order_file)
    if "stay_id" not in ts_order.columns:
        raise ValueError(f"Missing stay_id in {args.ts_order_file}")
    if "label" not in ts_order.columns:
        raise ValueError(f"Missing label in {args.ts_order_file}")

    tabular = pd.read_csv(args.tabular_file)
    if "stay_id" not in tabular.columns:
        raise ValueError(f"Missing stay_id in {args.tabular_file}")

    # Keep TS row order (distances are generated in this order)
    merged_tabular = ts_order[["stay_id", "label"]].merge(
        tabular, on="stay_id", how="left", suffixes=("_ts", "")
    )
    if "label" in merged_tabular.columns and "label_ts" in merged_tabular.columns:
        merged_tabular["label"] = merged_tabular["label_ts"]
    elif "label_ts" in merged_tabular.columns:
        merged_tabular = merged_tabular.rename(columns={"label_ts": "label"})

    # Build shapelet distance features
    shapelet_raw = build_shapelet_feature_df(
        metrics_dir=args.metrics_dir,
        n_rows=len(ts_order),
        top_k_each_file=args.top_k_each_file,
    )

    # Align to MIMIC training shapelet columns so downstream model features match
    reference_cols = get_reference_shapelet_columns(args.reference_feature_file)
    shapelet_aligned = pd.DataFrame(index=shapelet_raw.index)
    for col in reference_cols:
        if col in shapelet_raw.columns:
            shapelet_aligned[col] = shapelet_raw[col]
        else:
            shapelet_aligned[col] = np.nan

    if args.compute_missing_from_test_ts:
        ts_file_map = {
            "heart": os.path.join(args.eicu_ts_vital_dir, "Test_heart.csv"),
            "sbp": os.path.join(args.eicu_ts_vital_dir, "Test_sbp.csv"),
            "dbp": os.path.join(args.eicu_ts_vital_dir, "Test_dbp.csv"),
            "spo2": os.path.join(args.eicu_ts_vital_dir, "Test_spo2.csv"),
            "bun": os.path.join(args.eicu_ts_lab_dir, "Test_bun.csv"),
            "creatinine": os.path.join(args.eicu_ts_lab_dir, "Test_creatinine.csv"),
            "potassium": os.path.join(args.eicu_ts_lab_dir, "Test_potassium.csv"),
        }
        cache = {}
        missing_cols = [c for c in shapelet_aligned.columns if shapelet_aligned[c].isna().all()]
        for col in missing_cols:
            prefix = col.split("_", 1)[0]
            ts_file = ts_file_map.get(prefix)
            if ts_file is None or (not os.path.exists(ts_file)):
                continue
            if prefix not in cache:
                mat = pd.read_csv(ts_file, header=None).iloc[:, 1:].to_numpy(dtype=float)
                cache[prefix] = mat
            mat = cache[prefix]
            shapelet = parse_shapelet_from_feature_name(col)
            dists = [min_subsequence_distance(ts, shapelet) for ts in mat]
            if len(dists) == len(shapelet_aligned):
                shapelet_aligned[col] = dists

    base_features = merged_tabular.drop(
        columns=[c for c in ["stay_id", "label", "label_ts"] if c in merged_tabular.columns]
    )
    X_test_with_shapelets = pd.concat(
        [base_features.reset_index(drop=True), shapelet_aligned.reset_index(drop=True)],
        axis=1,
    )
    processed_test_with_shapelets = X_test_with_shapelets.copy()
    processed_test_with_shapelets.insert(0, "stay_id", merged_tabular["stay_id"].reset_index(drop=True))
    processed_test_with_shapelets["label"] = merged_tabular["label"].reset_index(drop=True)

    # save
    shapelet_raw.to_csv(
        os.path.join(args.output_dir, "shapelet_test_features_raw.csv"), index=False
    )
    shapelet_aligned.to_csv(
        os.path.join(args.output_dir, "shapelet_test_features_aligned.csv"), index=False
    )
    X_test_with_shapelets.to_csv(
        os.path.join(args.output_dir, "X_test_with_shapelets.csv"), index=False
    )
    processed_test_with_shapelets.to_csv(
        os.path.join(args.output_dir, "processed_test_with_shapelets.csv"), index=False
    )

    print("Done.")
    print(f"rows: {len(X_test_with_shapelets)}")
    print(f"shapelet_raw_cols: {shapelet_raw.shape[1]}")
    print(f"shapelet_aligned_cols: {shapelet_aligned.shape[1]}")
    print(f"X_test_with_shapelets shape: {X_test_with_shapelets.shape}")
    print(
        "label distribution:",
        processed_test_with_shapelets["label"].value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()
