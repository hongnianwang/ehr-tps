#!/usr/bin/env python3
"""Generate internal/external performance tables with 95% CI.

Outputs:
- Table2_internal_performance.*
- Table3_external_performance.* (includes PPV@0.5)
- Table2_3_internal_external_performance_combined.* (merged table)
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

MODELS = ["lightgbm", "xgboost", "catboost"]
MODEL_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--internal_pred_dir",
        default=os.path.join(root, "results", "mimic_internal_external_validation_selected"),
    )
    p.add_argument(
        "--external_pred_dir",
        default=os.path.join(root, "results", "mimic_internal_external_validation_selected"),
    )
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--outdir", default=os.path.join(root, "results", "paper_tables"))
    return p.parse_args()


def delong_roc_test(y_true, y_pred_1, y_pred_2) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred_1 = np.asarray(y_pred_1, dtype=float)
    y_pred_2 = np.asarray(y_pred_2, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    a_pos, a_neg = y_pred_1[pos], y_pred_1[neg]
    b_pos, b_neg = y_pred_2[pos], y_pred_2[neg]
    m, n = len(a_pos), len(a_neg)
    if m < 2 or n < 2:
        return np.nan, np.nan

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
        return np.nan, np.nan
    z = (auc_a - auc_b) / math.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def bootstrap_auprc_test(y_true, y_pred_1, y_pred_2, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)
    n = len(y_true)

    p1, r1, _ = precision_recall_curve(y_true, y_pred_1)
    p2, r2, _ = precision_recall_curve(y_true, y_pred_2)
    original_diff = auc(r1, p1) - auc(r2, p2)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_true[idx]
        p1b, r1b, _ = precision_recall_curve(yb, y_pred_1[idx])
        p2b, r2b, _ = precision_recall_curve(yb, y_pred_2[idx])
        diffs.append(auc(r1b, p1b) - auc(r2b, p2b))
    diffs = np.array(diffs)
    p = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return p, original_diff


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def load_preds(pred_dir: str, cohort_prefix: str | None = None) -> Dict[str, Dict[int, pd.DataFrame]]:
    out: Dict[str, Dict[int, pd.DataFrame]] = {}
    for model in MODELS:
        out[model] = {}
        wide_path = (
            os.path.join(pred_dir, f"{cohort_prefix}_{model}_predictions.csv")
            if cohort_prefix is not None
            else None
        )
        if wide_path is not None and os.path.exists(wide_path):
            wide = pd.read_csv(wide_path)
            for v in [1, 2, 3]:
                out[model][v] = pd.DataFrame(
                    {
                        "y_true": wide["y_true"],
                        "y_prob": wide[f"v{v}_prob"],
                    }
                )
        else:
            for v in [1, 2, 3]:
                path = os.path.join(pred_dir, f"{model}_v{v}_predictions.csv")
                out[model][v] = pd.read_csv(path)
    return out


def ppv_point(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_hat = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_hat == 1))
    fp = np.sum((y_true == 0) & (y_hat == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan


def metric_bundle(y_true, y_prob, n_boot=1000, seed=42, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else np.nan
    p, r, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(r, p)
    ppv = ppv_point(y_true, y_prob, threshold=threshold)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    auroc_bs, auprc_bs, ppv_bs = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_true[idx]
        sb = y_prob[idx]

        if len(np.unique(yb)) >= 2:
            auroc_bs.append(roc_auc_score(yb, sb))

        pb, rb, _ = precision_recall_curve(yb, sb)
        auprc_bs.append(auc(rb, pb))

        ppv_b = ppv_point(yb, sb, threshold=threshold)
        if np.isfinite(ppv_b):
            ppv_bs.append(ppv_b)

    def ci(vals):
        if len(vals) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {
        "auroc": auroc,
        "auroc_ci": ci(auroc_bs),
        "auprc": auprc,
        "auprc_ci": ci(auprc_bs),
        "ppv": ppv,
        "ppv_ci": ci(ppv_bs),
    }


def evaluate_models(preds: Dict[str, Dict[int, pd.DataFrame]], n_boot=1000, seed=42, threshold=0.5):
    res = {}
    for model in MODELS:
        res[model] = {}
        for v in [1, 2, 3]:
            df = preds[model][v]
            y = df["y_true"].values
            s = df["y_prob"].values
            mb = metric_bundle(y, s, n_boot=n_boot, seed=seed + 100 * v + len(model), threshold=threshold)
            mb["y"] = y
            mb["s"] = s
            res[model][v] = mb

        # Significance for Configuration 3 vs Configuration 2.
        y = res[model][2]["y"]
        s2 = res[model][2]["s"]
        s3 = res[model][3]["s"]
        _, p_roc = delong_roc_test(y, s3, s2)
        p_pr, _ = bootstrap_auprc_test(y, s3, s2, n_bootstrap=n_boot, seed=seed)
        res[model]["p_roc_v3v2"] = p_roc
        res[model]["p_pr_v3v2"] = p_pr

    return res


def format_cell(point: float, ci: Tuple[float, float], add_stars: str = "") -> str:
    return f"{point:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]{add_stars}"


def build_table_from_results(res, include_ppv=False, threshold=0.5) -> pd.DataFrame:
    rows = []

    rows.append(["AUROC [95% CI]", "", "", ""])
    for v in [1, 2, 3]:
        vals = []
        for m in MODELS:
            st = stars(res[m]["p_roc_v3v2"]) if v == 3 else ""
            vals.append(format_cell(res[m][v]["auroc"], res[m][v]["auroc_ci"], st))
        rows.append([f"Configuration {v}", *vals])

    rows.append(["AUPRC [95% CI]", "", "", ""])
    for v in [1, 2, 3]:
        vals = []
        for m in MODELS:
            st = stars(res[m]["p_pr_v3v2"]) if v == 3 else ""
            vals.append(format_cell(res[m][v]["auprc"], res[m][v]["auprc_ci"], st))
        rows.append([f"Configuration {v}", *vals])

    if include_ppv:
        rows.append([f"PPV@{threshold:.2f} [95% CI]", "", "", ""])
        for v in [1, 2, 3]:
            vals = []
            for m in MODELS:
                vals.append(format_cell(res[m][v]["ppv"], res[m][v]["ppv_ci"], ""))
            rows.append([f"Configuration {v}", *vals])

    return pd.DataFrame(rows, columns=["Configuration", "LightGBM", "XGBoost", "CatBoost"])


def build_combined_table(internal_res, external_res, threshold=0.5) -> pd.DataFrame:
    rows = []

    # Internal cohort
    for metric_key, metric_label, pkey in [
        ("auroc", "AUROC [95% CI]", "p_roc_v3v2"),
        ("auprc", "AUPRC [95% CI]", "p_pr_v3v2"),
    ]:
        for v in [1, 2, 3]:
            vals = []
            for m in MODELS:
                st = stars(internal_res[m][pkey]) if v == 3 else ""
                vals.append(format_cell(internal_res[m][v][metric_key], internal_res[m][v][f"{metric_key}_ci"], st))
            rows.append(["Internal (MIMIC holdout)", metric_label, f"Configuration {v}", *vals])

    # External cohort
    for metric_key, metric_label, pkey in [
        ("auroc", "AUROC [95% CI]", "p_roc_v3v2"),
        ("auprc", "AUPRC [95% CI]", "p_pr_v3v2"),
    ]:
        for v in [1, 2, 3]:
            vals = []
            for m in MODELS:
                st = stars(external_res[m][pkey]) if v == 3 else ""
                vals.append(format_cell(external_res[m][v][metric_key], external_res[m][v][f"{metric_key}_ci"], st))
            rows.append(["External (eICU)", metric_label, f"Configuration {v}", *vals])

    for v in [1, 2, 3]:
        vals = []
        for m in MODELS:
            vals.append(format_cell(external_res[m][v]["ppv"], external_res[m][v]["ppv_ci"], ""))
        rows.append(["External (eICU)", f"PPV@{threshold:.2f} [95% CI]", f"Configuration {v}", *vals])

    return pd.DataFrame(
        rows,
        columns=["Cohort", "Metric", "Configuration", "LightGBM", "XGBoost", "CatBoost"],
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    internal_preds = load_preds(args.internal_pred_dir, cohort_prefix="internal")
    external_preds = load_preds(args.external_pred_dir, cohort_prefix="external")

    internal_res = evaluate_models(internal_preds, n_boot=args.bootstrap, seed=args.seed, threshold=args.threshold)
    external_res = evaluate_models(
        external_preds, n_boot=args.bootstrap, seed=args.seed + 99, threshold=args.threshold
    )

    t2 = build_table_from_results(internal_res, include_ppv=False, threshold=args.threshold)
    t3 = build_table_from_results(external_res, include_ppv=True, threshold=args.threshold)
    t23 = build_combined_table(internal_res, external_res, threshold=args.threshold)

    t2_csv = os.path.join(args.outdir, "Table2_internal_performance.csv")
    t3_csv = os.path.join(args.outdir, "Table3_external_performance.csv")
    t23_csv = os.path.join(args.outdir, "Table2_3_internal_external_performance_combined.csv")

    t2_md = os.path.join(args.outdir, "Table2_internal_performance.md")
    t3_md = os.path.join(args.outdir, "Table3_external_performance.md")
    t23_md = os.path.join(args.outdir, "Table2_3_internal_external_performance_combined.md")

    t2.to_csv(t2_csv, index=False)
    t3.to_csv(t3_csv, index=False)
    t23.to_csv(t23_csv, index=False)

    with open(t2_md, "w", encoding="utf-8") as f:
        f.write("# Table 2. Internal Performance of Gradient Boosting Models\n\n")
        f.write(t2.to_markdown(index=False))
        f.write("\n\n")
        f.write(
            "Notes: values are point estimates with bootstrap 95% CIs. "
            "Asterisks in Configuration 3 indicate Configuration 3 vs Configuration 2 significance (* p<0.05; ** p<0.01; *** p<0.001).\n"
        )

    with open(t3_md, "w", encoding="utf-8") as f:
        f.write("# Table 3. External Validation Performance in eICU\n\n")
        f.write(t3.to_markdown(index=False))
        f.write("\n\n")
        f.write(
            f"Notes: values are point estimates with bootstrap 95% CIs; PPV is computed at threshold={args.threshold:.2f}. "
            "Asterisks in Configuration 3 indicate Configuration 3 vs Configuration 2 significance for AUROC/AUPRC.\n"
        )

    with open(t23_md, "w", encoding="utf-8") as f:
        f.write("# Table 2-3 Combined. Internal and External Validation Performance\n\n")
        f.write(t23.to_markdown(index=False))
        f.write("\n\n")
        f.write(
            f"Notes: values are point estimates with bootstrap 95% CIs; PPV is computed at threshold={args.threshold:.2f}. "
            "Asterisks in Configuration 3 indicate Configuration 3 vs Configuration 2 significance for AUROC/AUPRC within each cohort.\n"
        )

    print("Saved:")
    print(t2_csv)
    print(t2_md)
    print(t3_csv)
    print(t3_md)
    print(t23_csv)
    print(t23_md)


if __name__ == "__main__":
    main()
