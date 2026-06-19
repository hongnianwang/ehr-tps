#!/usr/bin/env python3
"""Build balanced MIMIC raw time-grid files from AKI/Non-AKI long TS tables.

"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Set

import numpy as np
import pandas as pd


VITAL_VARIABLES = ["heart_rate", "sbp", "dbp", "spo2"]
LAB_VARIABLES = ["creatinine", "bun", "potassium"]
ALL_VARIABLES = VITAL_VARIABLES + LAB_VARIABLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive-ts",
        default="data/mimiciv/data-aki_ts.csv",
        help="AKI long-format TS csv.",
    )
    parser.add_argument(
        "--negative-ts",
        default="data/mimiciv/data-control_ts.csv",
        help="Non-AKI long-format TS csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mimiciv/raw",
        help="Output folder for *_combined_min*_intersection.csv files.",
    )
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="If set, sample this many stays per class. If unset, use min(pos,neg).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vital-bin-hours", type=int, default=1)
    parser.add_argument("--lab-bin-hours", type=int, default=4)
    return parser.parse_args()


def filter_by_measurement_count(df: pd.DataFrame, feature: str, min_count: int) -> Set[int]:
    data_non_null = df.dropna(subset=[feature])
    feat_counts = data_non_null.groupby("stay_id")[feature].count()
    return set(feat_counts[feat_counts >= min_count].index.tolist())


def intersect_qualified_ids(df: pd.DataFrame, min_count: int) -> Set[int]:
    qualified = [filter_by_measurement_count(df, feat, min_count) for feat in ALL_VARIABLES]
    return set.intersection(*qualified)


def pick_ids(ids: Iterable[int], n: int, rng: np.random.Generator) -> List[int]:
    ids_list = list(ids)
    if len(ids_list) <= n:
        return ids_list
    return list(rng.choice(ids_list, size=n, replace=False))


def add_hours_from_admission(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hours_from_admission"] = (
        pd.to_datetime(out["charttime"]) - pd.to_datetime(out["icu_intime"])
    ).dt.total_seconds() / 3600.0
    out["hours_from_admission"] = out["hours_from_admission"].round().astype(int)
    return out


def build_time_bins(min_hour: int, max_hour: int, bin_hours: int) -> List[int]:
    if bin_hours > 1:
        start = (min_hour // bin_hours) * bin_hours
        end = ((max_hour // bin_hours) + 1) * bin_hours
        return list(range(start, end, bin_hours))
    return list(range(min_hour, max_hour + 1))


def pivot_feature(
    df: pd.DataFrame,
    feature: str,
    time_bins: List[int],
    bin_hours: int,
    label: int,
) -> pd.DataFrame:
    out = add_hours_from_admission(df)
    if bin_hours > 1:
        out["time_bin"] = (out["hours_from_admission"] // bin_hours) * bin_hours
    else:
        out["time_bin"] = out["hours_from_admission"]

    pivot = out.pivot_table(
        index="stay_id",
        columns="time_bin",
        values=feature,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=time_bins)

    prefix = "b_" if bin_hours > 1 else "h_"
    pivot.columns = [f"{prefix}{c}" for c in pivot.columns]
    pivot.insert(0, "label", label)
    return pivot


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    positive_data = pd.read_csv(args.positive_ts)
    negative_data = pd.read_csv(args.negative_ts)

    positive_data["label"] = 1
    negative_data["label"] = 0

    pos_ids = intersect_qualified_ids(positive_data, args.min_count)
    neg_ids = intersect_qualified_ids(negative_data, args.min_count)

    if args.target_per_class is None:
        sample_count = min(len(pos_ids), len(neg_ids))
    else:
        sample_count = min(args.target_per_class, len(pos_ids), len(neg_ids))

    rng = np.random.default_rng(args.seed)
    pos_ids = pick_ids(pos_ids, sample_count, rng)
    neg_ids = pick_ids(neg_ids, sample_count, rng)

    pos_filtered_all = add_hours_from_admission(positive_data[positive_data["stay_id"].isin(pos_ids)])
    neg_filtered_all = add_hours_from_admission(negative_data[negative_data["stay_id"].isin(neg_ids)])
    all_hours = sorted(
        set(pos_filtered_all["hours_from_admission"].tolist())
        | set(neg_filtered_all["hours_from_admission"].tolist())
    )
    min_hour, max_hour = min(all_hours), max(all_hours)

    print(f"Qualified AKI stays: {len(pos_ids)}")
    print(f"Qualified Non-AKI stays: {len(neg_ids)}")
    print(f"Unified hour range: {min_hour} to {max_hour}")

    for feature in ALL_VARIABLES:
        bin_hours = args.lab_bin_hours if feature in LAB_VARIABLES else args.vital_bin_hours
        time_bins = build_time_bins(min_hour, max_hour, bin_hours)

        pos_filtered = positive_data[positive_data["stay_id"].isin(pos_ids)].copy()
        neg_filtered = negative_data[negative_data["stay_id"].isin(neg_ids)].copy()

        pos_pivot = pivot_feature(pos_filtered, feature, time_bins, bin_hours, label=1)
        neg_pivot = pivot_feature(neg_filtered, feature, time_bins, bin_hours, label=0)
        combined = pd.concat([pos_pivot, neg_pivot]).sort_index()

        bin_suffix = f"_bin{bin_hours}h" if bin_hours > 1 else ""
        output_file = os.path.join(
            args.output_dir,
            f"{feature}_combined_min{args.min_count}{bin_suffix}_intersection.csv",
        )
        combined.to_csv(output_file)

        print(
            f"Saved {output_file} "
            f"(n={len(combined)}, pos={int(combined['label'].sum())}, "
            f"neg={len(combined) - int(combined['label'].sum())})"
        )


if __name__ == "__main__":
    main()

