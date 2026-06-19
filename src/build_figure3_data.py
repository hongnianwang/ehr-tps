#!/usr/bin/env python3
"""Build aggregate result tables for Figure 3.

ROC and PRC use raw model probabilities. Calibration and DCA use Platt
logistic calibration fitted only on MIMIC-IV training-set out-of-fold
predictions, then applied unchanged to the MIMIC-IV holdout and eICU
external predictions.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold

from run_mimic_internal_external_validation import (
    VERSION_LABELS,
    VERSION_ORDER,
    build_feature_sets,
    build_model,
    ece_score,
    fit_model,
    prepare_features,
)


@dataclass
class Calibrator:
    model: LogisticRegression

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        x = logit_clip(p_raw).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=root / "data" / "mimiciv" / "processed_modeling_cohort" / "processed_train_with_shapelets.csv",
    )
    parser.add_argument(
        "--internal-pred-file",
        type=Path,
        default=root / "results" / "mimic_internal_external_validation_selected" / "internal_xgboost_predictions.csv",
    )
    parser.add_argument(
        "--external-pred-file",
        type=Path,
        default=root / "results" / "mimic_internal_external_validation_selected" / "external_xgboost_predictions.csv",
    )
    parser.add_argument("--model", default="xgboost", choices=["xgboost"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--curve-grid-n", type=int, default=201)
    parser.add_argument("--dca-grid-n", type=int, default=901)
    parser.add_argument("--dca-min", type=float, default=0.05)
    parser.add_argument("--dca-max", type=float, default=0.65)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "results" / "model_performance",
    )
    return parser.parse_args()


def logit_clip(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_platt(y: np.ndarray, p_raw: np.ndarray) -> Calibrator:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(logit_clip(p_raw).reshape(-1, 1), y.astype(int))
    return Calibrator(model=model)


def make_oof_predictions(
    train_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    folds: int,
    seed: int,
) -> np.ndarray:
    y = train_df["label"].astype(int).to_numpy()
    oof = np.full(len(train_df), np.nan)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), start=1):
        fold_train = train_df.iloc[tr_idx].copy()
        fold_valid = train_df.iloc[va_idx].copy()
        x_train, x_valid, categorical_info = prepare_features(
            fold_train,
            fold_valid,
            features,
            model_type,
        )
        model = build_model(model_type, seed + fold)
        fit_model(model, model_type, x_train, y[tr_idx], categorical_info)
        oof[va_idx] = model.predict_proba(x_valid)[:, 1]

    if np.isnan(oof).any():
        raise RuntimeError("OOF prediction contains missing values.")
    return oof


def load_prediction_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"stay_id", "y_true", "v1_prob", "v2_prob", "v3_prob"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def roc_xy(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr


def pr_xy(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(y, p)
    x = recall[::-1]
    yv = precision[::-1]
    df = pd.DataFrame({"x": x, "y": yv}).groupby("x", as_index=False)["y"].max().sort_values("x")
    return df["x"].to_numpy(), df["y"].to_numpy()


def interp_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    uniq = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].max().sort_values("x")
    if len(uniq) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, uniq["x"].to_numpy(), uniq["y"].to_numpy())


def bootstrap_curve(
    y: np.ndarray,
    p: np.ndarray,
    curve: str,
    grid: np.ndarray,
    bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    mat = []
    metric_values = []
    n = len(y)
    for _ in range(bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        if curve == "roc":
            x, yy = roc_xy(y[idx], p[idx])
        else:
            x, yy = pr_xy(y[idx], p[idx])
        mat.append(interp_curve(x, yy, grid))
        metric_values.append(float(auc(x, yy)))
    if not mat:
        nan = np.full_like(grid, np.nan, dtype=float)
        return nan, nan, nan, (np.nan, np.nan)
    arr = np.vstack(mat)
    lower = np.nanpercentile(arr, 2.5, axis=0)
    upper = np.nanpercentile(arr, 97.5, axis=0)
    median = np.nanpercentile(arr, 50, axis=0)
    metric_ci = (float(np.percentile(metric_values, 2.5)), float(np.percentile(metric_values, 97.5)))
    return median, lower, upper, metric_ci


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        lo, hi = wilson_interval(int(y[mask].sum()), int(mask.sum()))
        rows.append(
            {
                "bin": b + 1,
                "n": int(mask.sum()),
                "predicted_mean": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
                "observed_ci_low": lo,
                "observed_ci_high": hi,
                "predicted_min": float(p[mask].min()),
                "predicted_max": float(p[mask].max()),
            }
        )
    return pd.DataFrame(rows)


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y = y.astype(int)
    n = len(y)
    prevalence = float(y.mean())
    rows = []
    for threshold in thresholds:
        pred = p >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(tp / n - fp / n * odds),
                "treat_all": float(prevalence - (1 - prevalence) * odds),
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_file)
    feature_sets = build_feature_sets(train_df)
    y_train = train_df["label"].astype(int).to_numpy()

    calibrators: dict[int, Calibrator] = {}
    oof_rows = []
    for version in VERSION_ORDER:
        print(f"Building OOF predictions for {args.model} {VERSION_LABELS[version]}...")
        oof = make_oof_predictions(
            train_df=train_df,
            features=feature_sets[version],
            model_type=args.model,
            folds=args.folds,
            seed=args.random_state + 100 * version,
        )
        calibrators[version] = fit_platt(y_train, oof)
        oof_rows.append(
            pd.DataFrame(
                {
                    "stay_id": train_df["stay_id"].to_numpy(),
                    "y_true": y_train,
                    "version": VERSION_LABELS[version],
                    "oof_prob_raw": oof,
                    "oof_prob_calibrated": calibrators[version].predict(oof),
                }
            )
        )
    pd.concat(oof_rows, ignore_index=True).to_csv(
        args.out_dir / f"{args.model}_internal_oof_platt_predictions.csv",
        index=False,
    )

    eval_sets = {
        "internal": load_prediction_file(args.internal_pred_file),
        "external": load_prediction_file(args.external_pred_file),
    }
    grid = np.linspace(0, 1, args.curve_grid_n)
    dca_thresholds = np.linspace(args.dca_min, args.dca_max, args.dca_grid_n)

    curve_rows = []
    calibration_rows = []
    dca_rows = []
    metric_rows = []

    for dataset_name, df in eval_sets.items():
        y = df["y_true"].astype(int).to_numpy()
        for version in VERSION_ORDER:
            version_label = VERSION_LABELS[version]
            raw_col = f"v{version}_prob"
            p_raw = df[raw_col].astype(float).to_numpy()
            p_cal = calibrators[version].predict(p_raw)
            for curve_name in ["roc", "prc"]:
                x0, y0 = roc_xy(y, p_raw) if curve_name == "roc" else pr_xy(y, p_raw)
                metric = float(auc(x0, y0))
                mid, lo, hi, metric_ci = bootstrap_curve(
                    y=y,
                    p=p_raw,
                    curve="roc" if curve_name == "roc" else "pr",
                    grid=grid,
                    bootstrap=args.bootstrap,
                    seed=args.random_state + (10000 if dataset_name == "external" else 0) + 100 * version + (7 if curve_name == "prc" else 0),
                )
                curve_rows.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "model": args.model,
                            "version": version_label,
                            "curve": curve_name,
                            "x": grid,
                            "y": interp_curve(x0, y0, grid),
                            "y_boot_median": mid,
                            "y_ci_low": lo,
                            "y_ci_high": hi,
                        }
                    )
                )
                metric_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": args.model,
                        "version": version_label,
                        "metric": "AUROC" if curve_name == "roc" else "AUPRC",
                        "value": metric,
                        "ci_low": metric_ci[0],
                        "ci_high": metric_ci[1],
                        "probability": "raw",
                    }
                )

            cal = calibration_bins(y, p_cal, args.calibration_bins)
            cal.insert(0, "probability", "internal_oof_platt")
            cal.insert(0, "version", version_label)
            cal.insert(0, "model", args.model)
            cal.insert(0, "dataset", dataset_name)
            calibration_rows.append(cal)

            dca = decision_curve(y, p_cal, dca_thresholds)
            dca.insert(0, "probability", "internal_oof_platt")
            dca.insert(0, "version", version_label)
            dca.insert(0, "model", args.model)
            dca.insert(0, "dataset", dataset_name)
            dca_rows.append(dca)

            metric_rows.extend(
                [
                    {
                        "dataset": dataset_name,
                        "model": args.model,
                        "version": version_label,
                        "metric": "Brier",
                        "value": float(brier_score_loss(y, p_cal)),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "probability": "internal_oof_platt",
                    },
                    {
                        "dataset": dataset_name,
                        "model": args.model,
                        "version": version_label,
                        "metric": "ECE10",
                        "value": float(ece_score(y, p_cal, args.calibration_bins)),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "probability": "internal_oof_platt",
                    },
                ]
            )


    pd.concat(curve_rows, ignore_index=True).to_csv(
        args.out_dir / f"{args.model}_roc_prc_curves_bootstrap.csv",
        index=False,
    )
    pd.concat(calibration_rows, ignore_index=True).to_csv(
        args.out_dir / f"{args.model}_calibration_bins_internal_oof_platt.csv",
        index=False,
    )
    pd.concat(dca_rows, ignore_index=True).to_csv(
        args.out_dir / f"{args.model}_dca_curves_internal_oof_platt.csv",
        index=False,
    )
    pd.DataFrame(metric_rows).to_csv(
        args.out_dir / f"{args.model}_figure3_metrics.csv",
        index=False,
    )

    print(f"Wrote model-performance aggregate results to {args.out_dir}")


if __name__ == "__main__":
    main()
