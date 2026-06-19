#!/usr/bin/env python3
"""
Internal and external validation for the MIMIC-IV modeling cohort.

This script trains V1/V2/V3 models on the locked MIMIC-IV training split,
evaluates them on the internal MIMIC holdout and the external eICU cohort, and
draws ROC, PRC, calibration, and DCA panels for three model families.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder


MODEL_ORDER = ["xgboost", "lightgbm", "catboost"]
VERSION_ORDER = [1, 2, 3]
VERSION_LABELS = {1: "V1", 2: "V2", 3: "V3"}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
COLORS = {
    1: "#7A7A7A",
    2: "#0072B2",
    3: "#009E73",
    "all": "#D55E00",
    "none": "#555555",
}


SELECTED_PARAMS = {
    "xgboost": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 4,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
        "gamma": 0.02,
    },
    "lightgbm": {
        "learning_rate": 0.04,
        "num_leaves": 80,
        "max_depth": 6,
        "n_estimators": 1500,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_lambda": 0.50,
        "reg_alpha": 0.01,
        "min_split_gain": 0.02,
    },
    "catboost": {
        "iterations": 900,
        "learning_rate": 0.12,
        "depth": 6,
        "l2_leaf_reg": 6,
        "random_strength": 1.0,
        "bagging_temperature": 0.50,
    },
}


def default_paths() -> dict[str, Path]:
    script = Path(__file__).resolve()
    root = script.parents[1]
    return {
        "root": root,
        "train": root
        / "data"
        / "mimiciv"
        / "processed_modeling_cohort"
        / "processed_train_with_shapelets.csv",
        "internal": root
        / "data"
        / "mimiciv"
        / "processed_modeling_cohort"
        / "processed_test_with_shapelets.csv",
        "external": root / "data" / "eicu" / "processed" / "processed_test_with_shapelets.csv",
        "out": root / "results" / "mimic_internal_external_validation_selected",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=paths["train"])
    parser.add_argument("--internal-test-file", type=Path, default=paths["internal"])
    parser.add_argument("--external-test-file", type=Path, default=paths["external"])
    parser.add_argument("--out-dir", type=Path, default=paths["out"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--dca-focus-max", type=float, default=0.60)
    return parser.parse_args()


def normalize_gender(v) -> str:
    if pd.isna(v):
        return "Unknown"
    s = str(v).strip().lower()
    if s in {"0", "0.0", "female", "f"}:
        return "Female"
    if s in {"1", "1.0", "male", "m"}:
        return "Male"
    return "Unknown"


def normalize_race(v) -> str:
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
    if s in {"", "unknown", "unk", "nan", "unable to obtain", "declined"}:
        return "Unknown"
    return "Other"


def build_feature_sets(df: pd.DataFrame) -> dict[int, list[str]]:
    demographic_cols = [c for c in ["gender", "age", "race"] if c in df.columns]
    last_cols = [c for c in df.columns if "_last" in c]
    min_cols = [c for c in df.columns if "_min" in c]
    max_cols = [c for c in df.columns if "_max" in c]
    shapelet_cols = [
        c
        for c in df.columns
        if c not in demographic_cols
        and "_last" not in c
        and "_min" not in c
        and "_max" not in c
        and c not in {"stay_id", "label"}
        and "_" in c
    ]
    return {
        1: demographic_cols + last_cols,
        2: demographic_cols + last_cols + min_cols + max_cols,
        3: demographic_cols + last_cols + min_cols + max_cols + shapelet_cols,
    }


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int] | list[str]]:
    x_train = train_df[features].copy()
    x_test = test_df[features].copy()
    categorical_cols = [c for c in ["gender", "race"] if c in x_train.columns]

    for col in categorical_cols:
        normalizer = normalize_gender if col == "gender" else normalize_race
        x_train[col] = x_train[col].apply(normalizer)
        x_test[col] = x_test[col].apply(normalizer)

    if model_type == "catboost":
        for col in categorical_cols:
            x_train[col] = x_train[col].astype(str).fillna("NA")
            x_test[col] = x_test[col].astype(str).fillna("NA")
        categorical_info = [x_train.columns.get_loc(c) for c in categorical_cols]
    elif model_type == "lightgbm":
        for col in categorical_cols:
            cats = pd.Index(
                sorted(
                    set(x_train[col].astype(str).dropna())
                    | set(x_test[col].astype(str).dropna())
                )
            )
            x_train[col] = pd.Categorical(x_train[col].astype(str), categories=cats)
            x_test[col] = pd.Categorical(x_test[col].astype(str), categories=cats)
        categorical_info = categorical_cols
    else:
        for col in categorical_cols:
            encoder = LabelEncoder()
            both = pd.concat([x_train[col], x_test[col]], ignore_index=True).astype(str)
            encoder.fit(both)
            x_train[col] = encoder.transform(x_train[col].astype(str))
            x_test[col] = encoder.transform(x_test[col].astype(str))
        categorical_info = []

    numeric_cols = [c for c in x_train.columns if c not in categorical_cols]
    if numeric_cols:
        means = x_train[numeric_cols].apply(pd.to_numeric, errors="coerce").mean()
        x_train[numeric_cols] = (
            x_train[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(means)
        )
        x_test[numeric_cols] = (
            x_test[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(means)
        )
    return x_train, x_test, categorical_info


def build_model(model_type: str, random_state: int):
    if model_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            **SELECTED_PARAMS["xgboost"],
        )
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            verbosity=-1,
            force_row_wise=True,
            deterministic=True,
            random_state=random_state,
            n_jobs=-1,
            **SELECTED_PARAMS["lightgbm"],
        )
    if model_type == "catboost":
        return cb.CatBoostClassifier(
            random_state=random_state,
            verbose=False,
            thread_count=-1,
            **SELECTED_PARAMS["catboost"],
        )
    raise ValueError(f"Unsupported model_type={model_type}")


def fit_model(model, model_type: str, x_train, y_train, categorical_info) -> None:
    if model_type == "lightgbm":
        model.fit(
            x_train,
            y_train,
            categorical_feature=categorical_info if categorical_info else None,
        )
    elif model_type == "catboost":
        model.fit(
            x_train,
            y_train,
            cat_features=categorical_info if categorical_info else None,
        )
    else:
        model.fit(x_train, y_train)


def auprc_trapz(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(y_prob, edges[1:-1], right=True)
    out = 0.0
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        out += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.sum() / len(y_true)
    return float(out)


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
    values = []
    n = len(y_true)
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

    def structural(pos_scores, neg_scores):
        comp = (
            (pos_scores[:, None] > neg_scores[None, :]).astype(float)
            + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).astype(float)
        )
        return comp.mean(axis=1), comp.mean(axis=0), comp.mean()

    v10_a, v01_a, auc_a = structural(a_pos, a_neg)
    v10_b, v01_b, auc_b = structural(b_pos, b_neg)
    sx = np.cov(np.vstack([v10_a, v10_b]), ddof=1) / m
    sy = np.cov(np.vstack([v01_a, v01_b]), ddof=1) / n
    cov = sx + sy
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return (np.nan, np.nan)
    z = (auc_a - auc_b) / math.sqrt(var)
    return (float(z), float(2 * stats.norm.sf(abs(z))))


def paired_auprc_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float, float]:
    observed = auprc_trapz(y_true, pred_a) - auprc_trapz(y_true, pred_b)
    if n_bootstrap <= 0:
        return (observed, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    diffs = []
    n = len(y_true)
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


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(y_prob, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        lo, hi = wilson_interval(int(y_true[mask].sum()), int(mask.sum()))
        rows.append(
            {
                "bin": b + 1,
                "n": int(mask.sum()),
                "predicted_mean": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "observed_ci_low": lo,
                "observed_ci_high": hi,
                "predicted_min": float(y_prob[mask].min()),
                "predicted_max": float(y_prob[mask].max()),
            }
        )
    return pd.DataFrame(rows)


def dca_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y_true = y_true.astype(int)
    rows = []
    n = len(y_true)
    prevalence = float(y_true.mean())
    for threshold in thresholds:
        pred = y_prob >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        odds = threshold / (1 - threshold)
        nb = tp / n - fp / n * odds
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(nb),
                "treat_all": float(prevalence - (1 - prevalence) * odds),
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def format_ci(x: float, lo: float, hi: float) -> str:
    return f"{x:.3f} ({lo:.3f}-{hi:.3f})"


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def run_validation(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(args.train_file)
    eval_sets = {
        "internal": pd.read_csv(args.internal_test_file),
        "external": pd.read_csv(args.external_test_file),
    }
    feature_sets = build_feature_sets(train_df)
    y_train = train_df["label"].astype(int).to_numpy()

    metrics_rows = []
    delong_rows = []
    auprc_rows = []
    all_prediction_frames = {}
    all_calibration_rows = []
    all_dca_rows = []

    for model_type in MODEL_ORDER:
        for dataset_name, test_df in eval_sets.items():
            y_test = test_df["label"].astype(int).to_numpy()
            pred_frame = pd.DataFrame({"stay_id": test_df["stay_id"], "y_true": y_test})
            probs_by_version = {}

            for version in VERSION_ORDER:
                x_train, x_test, categorical_info = prepare_features(
                    train_df, test_df, feature_sets[version], model_type
                )
                model = build_model(model_type, args.random_state)
                fit_model(model, model_type, x_train, y_train, categorical_info)
                y_prob = model.predict_proba(x_test)[:, 1]
                probs_by_version[version] = y_prob
                pred_frame[f"v{version}_prob"] = y_prob

                auroc = float(roc_auc_score(y_test, y_prob))
                auprc = auprc_trapz(y_test, y_prob)
                auroc_ci = bootstrap_ci(
                    y_test, y_prob, roc_auc_score, args.bootstrap, args.random_state + version
                )
                auprc_ci = bootstrap_ci(
                    y_test, y_prob, auprc_trapz, args.bootstrap, args.random_state + 100 + version
                )
                metrics_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_type,
                        "version": VERSION_LABELS[version],
                        "n_train": len(train_df),
                        "n_eval": len(test_df),
                        "n_eval_AKI": int(y_test.sum()),
                        "n_eval_non_AKI": int((y_test == 0).sum()),
                        "n_features": len(feature_sets[version]),
                        "AUROC": auroc,
                        "AUROC_ci_low": auroc_ci[0],
                        "AUROC_ci_high": auroc_ci[1],
                        "AUPRC": auprc,
                        "AUPRC_ci_low": auprc_ci[0],
                        "AUPRC_ci_high": auprc_ci[1],
                        "average_precision": float(average_precision_score(y_test, y_prob)),
                        "Brier": float(brier_score_loss(y_test, y_prob)),
                        "ECE10": ece_score(y_test, y_prob, args.calibration_bins),
                    }
                )

                cal = calibration_bins(y_test, y_prob, args.calibration_bins)
                cal.insert(0, "version", VERSION_LABELS[version])
                cal.insert(0, "model", model_type)
                cal.insert(0, "dataset", dataset_name)
                all_calibration_rows.append(cal)

                dca = dca_curve(y_test, y_prob, np.linspace(0.01, 0.99, 99))
                dca.insert(0, "version", VERSION_LABELS[version])
                dca.insert(0, "model", model_type)
                dca.insert(0, "dataset", dataset_name)
                all_dca_rows.append(dca)

            pred_path = args.out_dir / f"{dataset_name}_{model_type}_predictions.csv"
            pred_frame.to_csv(pred_path, index=False)
            all_prediction_frames[(dataset_name, model_type)] = pred_frame

            for label, a, b in [("V2_vs_V1", 2, 1), ("V3_vs_V1", 3, 1), ("V3_vs_V2", 3, 2)]:
                z, p = delong_test(y_test, probs_by_version[a], probs_by_version[b])
                delong_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_type,
                        "comparison": label,
                        "AUROC_a": float(roc_auc_score(y_test, probs_by_version[a])),
                        "AUROC_b": float(roc_auc_score(y_test, probs_by_version[b])),
                        "AUROC_diff": float(
                            roc_auc_score(y_test, probs_by_version[a])
                            - roc_auc_score(y_test, probs_by_version[b])
                        ),
                        "z": z,
                        "p_value": p,
                    }
                )
                diff, p_prc, lo, hi = paired_auprc_bootstrap(
                    y_test,
                    probs_by_version[a],
                    probs_by_version[b],
                    args.bootstrap,
                    args.random_state + a * 100 + b,
                )
                auprc_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_type,
                        "comparison": label,
                        "AUPRC_diff": diff,
                        "p_value": p_prc,
                        "diff_ci_low": lo,
                        "diff_ci_high": hi,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    delong_df = pd.DataFrame(delong_rows)
    auprc_df = pd.DataFrame(auprc_rows)
    calibration_df = pd.concat(all_calibration_rows, ignore_index=True)
    dca_df = pd.concat(all_dca_rows, ignore_index=True)

    metrics_df.to_csv(args.out_dir / "metrics_internal_external.csv", index=False)
    delong_df.to_csv(args.out_dir / "delong_internal_external.csv", index=False)
    auprc_df.to_csv(args.out_dir / "auprc_bootstrap_internal_external.csv", index=False)
    calibration_df.to_csv(args.out_dir / "calibration_bins_internal_external.csv", index=False)
    dca_df.to_csv(args.out_dir / "dca_curves_internal_external.csv", index=False)
    (args.out_dir / "selected_params.json").write_text(
        json.dumps(SELECTED_PARAMS, indent=2), encoding="utf-8"
    )
    run_config = {
        "train_file": str(args.train_file),
        "internal_test_file": str(args.internal_test_file),
        "external_test_file": str(args.external_test_file),
        "bootstrap": args.bootstrap,
        "random_state": args.random_state,
        "calibration_bins": args.calibration_bins,
        "dca_focus_max": args.dca_focus_max,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    draw_all_figures(args, metrics_df, delong_df, calibration_df, dca_df, all_prediction_frames)
    write_summary(args.out_dir, metrics_df, delong_df, auprc_df)
    return metrics_df, delong_df, auprc_df


def draw_all_figures(
    args: argparse.Namespace,
    metrics_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    dca_df: pd.DataFrame,
    prediction_frames: dict,
) -> None:
    for dataset_name in ["internal", "external"]:
        draw_dataset_figure(
            args,
            dataset_name,
            metrics_df,
            delong_df,
            calibration_df,
            dca_df,
            prediction_frames,
            dca_xmax=args.dca_focus_max,
            suffix="focused",
        )
        draw_dataset_figure(
            args,
            dataset_name,
            metrics_df,
            delong_df,
            calibration_df,
            dca_df,
            prediction_frames,
            dca_xmax=0.99,
            suffix="fullrange",
        )


def draw_dataset_figure(
    args: argparse.Namespace,
    dataset_name: str,
    metrics_df: pd.DataFrame,
    delong_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    dca_df: pd.DataFrame,
    prediction_frames: dict,
    dca_xmax: float,
    suffix: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(15.2, 10.4), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.11, top=0.91, wspace=0.24, hspace=0.34)
    panel_letters = [f"{chr(ord('A') + c)}{r + 1}" for r in range(3) for c in range(4)]
    legend_handles = [
        Line2D([0], [0], color=COLORS[1], lw=2.2, label="V1"),
        Line2D([0], [0], color=COLORS[2], lw=2.2, label="V2"),
        Line2D([0], [0], color=COLORS[3], lw=2.2, label="V3"),
        Line2D([0], [0], color="#BDBDBD", lw=1.1, ls="--", label="Prevalence"),
        Line2D([0], [0], color=COLORS["all"], lw=1.2, ls="--", label="Treat all"),
        Line2D([0], [0], color=COLORS["none"], lw=1.2, ls=":", label="Treat none"),
    ]
    dataset_cal_max = float(calibration_df.query("dataset == @dataset_name")["predicted_max"].max())

    for r, model_type in enumerate(MODEL_ORDER):
        pred = prediction_frames[(dataset_name, model_type)]
        y_true = pred["y_true"].to_numpy().astype(int)
        prevalence = float(y_true.mean())
        v3_metrics = metrics_df.query(
            "dataset == @dataset_name and model == @model_type and version == 'V3'"
        ).iloc[0]

        ax = axes[r, 0]
        for version in VERSION_ORDER:
            version_label = VERSION_LABELS[version]
            y_prob = pred[f"v{version}_prob"].to_numpy()
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            m = metrics_df.query(
                "dataset == @dataset_name and model == @model_type and version == @version_label"
            ).iloc[0]
            ax.plot(fpr, tpr, color=COLORS[version], lw=1.9)
        ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=0.9, ls="--")
        p_v3v2 = delong_df.query(
            "dataset == @dataset_name and model == @model_type and comparison == 'V3_vs_V2'"
        )["p_value"].iloc[0]
        ax.set_title(f"{MODEL_LABELS[model_type]} ROC", fontsize=8.3, pad=8)
        ax.text(
            0.985,
            0.055,
            f"V3 AUROC {v3_metrics.AUROC:.3f} ({v3_metrics.AUROC_ci_low:.3f}-{v3_metrics.AUROC_ci_high:.3f})\n"
            f"V3 vs V2 p={p_v3v2:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
        )
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        ax = axes[r, 1]
        for version in VERSION_ORDER:
            version_label = VERSION_LABELS[version]
            y_prob = pred[f"v{version}_prob"].to_numpy()
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            m = metrics_df.query(
                "dataset == @dataset_name and model == @model_type and version == @version_label"
            ).iloc[0]
            ax.plot(recall, precision, color=COLORS[version], lw=1.9)
        ax.axhline(prevalence, color="#BDBDBD", lw=0.9, ls="--")
        ax.set_title(f"{MODEL_LABELS[model_type]} PRC", fontsize=8.3, pad=8)
        ax.text(
            0.985,
            0.985,
            f"V3 AUPRC {v3_metrics.AUPRC:.3f} ({v3_metrics.AUPRC_ci_low:.3f}-{v3_metrics.AUPRC_ci_high:.3f})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.02)

        ax = axes[r, 2]
        ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=0.9, ls="--")
        for version in VERSION_ORDER:
            version_label = VERSION_LABELS[version]
            cal = calibration_df.query(
                "dataset == @dataset_name and model == @model_type and version == @version_label"
            )
            ax.errorbar(
                cal["predicted_mean"],
                cal["observed_rate"],
                yerr=[
                    cal["observed_rate"] - cal["observed_ci_low"],
                    cal["observed_ci_high"] - cal["observed_rate"],
                ],
                fmt="-o",
                color=COLORS[version],
                lw=1.6,
                ms=3.2,
                elinewidth=0.8,
                capsize=1.5,
                alpha=0.95,
                zorder=3,
            )
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed AKI rate")
        if suffix == "focused":
            x_right = min(0.70, max(0.62, dataset_cal_max + 0.05))
            ax.set_xlim(0, x_right)
            ax.set_ylim(0, x_right)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_title(f"{MODEL_LABELS[model_type]} calibration", fontsize=8.3, pad=8)

        ax = axes[r, 3]
        model_dca = dca_df.query("dataset == @dataset_name and model == @model_type")
        ref = model_dca.query("version == 'V1'").copy()
        ref = ref[ref["threshold"] <= dca_xmax]
        ax.plot(ref["threshold"], ref["treat_none"], color=COLORS["none"], lw=1.0, ls=":")
        ax.plot(ref["threshold"], ref["treat_all"], color=COLORS["all"], lw=1.0, ls="--")
        y_values = []
        for version in VERSION_ORDER:
            version_label = VERSION_LABELS[version]
            d = model_dca.query("version == @version_label").copy()
            d = d[d["threshold"] <= dca_xmax]
            y_values.extend(d["net_benefit"].tolist())
            ax.plot(d["threshold"], d["net_benefit"], color=COLORS[version], lw=1.9)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_xlim(0, dca_xmax)
        if y_values:
            ymax = max(
                0.02,
                min(0.25, max(np.nanpercentile(y_values, 98), float(ref["treat_all"].max())) + 0.02),
            )
            ymin = max(
                -0.10,
                min(
                    -0.01,
                    min(np.nanpercentile(y_values, 2), float(ref["treat_all"].min()), 0.0) - 0.02,
                ),
            )
            ax.set_ylim(ymin, ymax)
        ax.set_title(f"{MODEL_LABELS[model_type]} DCA", fontsize=8.3, pad=8)

    for i, ax in enumerate(axes.flat):
        ax.text(
            -0.20,
            1.10,
            panel_letters[i],
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        fontsize=7.5,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.4,
    )
    fig.suptitle(
        f"{dataset_name.capitalize()} validation: ROC, PRC, calibration, and DCA",
        fontsize=11.5,
        fontweight="bold",
        y=0.975,
    )
    out_base = args.out_dir / f"figure_{dataset_name}_roc_prc_calibration_dca_3x4_{suffix}"
    fig.savefig(out_base.with_suffix(".png"), dpi=600)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def write_summary(out_dir: Path, metrics_df: pd.DataFrame, delong_df: pd.DataFrame, auprc_df: pd.DataFrame) -> None:
    compact = metrics_df[
        [
            "dataset",
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
            "ECE10",
        ]
    ].copy()
    compact["AUROC_95CI"] = compact.apply(
        lambda x: format_ci(x.AUROC, x.AUROC_ci_low, x.AUROC_ci_high), axis=1
    )
    compact["AUPRC_95CI"] = compact.apply(
        lambda x: format_ci(x.AUPRC, x.AUPRC_ci_low, x.AUPRC_ci_high), axis=1
    )
    v3v2 = delong_df.query("comparison == 'V3_vs_V2'").copy()
    lines = [
        "# MIMIC-IV Modeling Cohort: Internal and External Validation",
        "",
        "## Protocol Note",
        "",
        (
            "Models were trained on the locked MIMIC-IV modeling-cohort training split only. "
            "The eICU cohort was used only for final external evaluation and was not used for "
            "hyperparameter selection, early stopping, probability calibration, threshold selection, "
            "or feature selection. Main results use conventional tree-boosting parameters without "
            "`scale_pos_weight`; class weighting can be explored as a sensitivity analysis but should "
            "not be used to tune the external validation result."
        ),
        "",
        "## Metrics",
        "",
        compact[
            ["dataset", "model", "version", "n_features", "AUROC_95CI", "AUPRC_95CI", "Brier", "ECE10"]
        ].to_markdown(index=False),
        "",
        "## AUROC DeLong Tests: V3 vs V2",
        "",
        v3v2.to_markdown(index=False),
        "",
        "## AUPRC Paired Bootstrap Tests",
        "",
        auprc_df.query("comparison == 'V3_vs_V2'").to_markdown(index=False),
        "",
        "## Selected Parameters",
        "",
        "Parameters are rounded, conventional tree-boosting settings and are saved in `selected_params.json`.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_df, delong_df, _ = run_validation(args)
    print(args.out_dir)
    print(metrics_df[["dataset", "model", "version", "AUROC", "AUPRC", "Brier"]].to_string(index=False))
    print(delong_df.query("comparison == 'V3_vs_V2'").to_string(index=False))


if __name__ == "__main__":
    main()
