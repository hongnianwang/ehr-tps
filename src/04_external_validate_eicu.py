"""Train models on MIMIC and externally validate on eICU.

This script uses the aligned shapelet-enhanced datasets:
- MIMIC train: data/mimiciv/processed/processed_train_with_shapelets.csv
- eICU test:  data/eicu/processed/processed_test_with_shapelets.csv

Outputs:
- Per-model/version predictions CSV
- Per-model/version serialized model
- metrics_summary.csv / metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    parser = argparse.ArgumentParser(
        description="Train on MIMIC and externally validate on eICU."
    )
    parser.add_argument(
        "--train_file",
        default=os.path.join(
            project_root, "data", "mimiciv", "processed", "processed_train_with_shapelets.csv"
        ),
    )
    parser.add_argument(
        "--test_file",
        default=os.path.join(
            project_root, "data", "eicu", "processed", "processed_test_with_shapelets.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(project_root, "results", "external_validation_eicu"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm", "catboost"],
        choices=["xgboost", "lightgbm", "catboost"],
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        type=int,
        default=[3],
        choices=[1, 2, 3],
        help="1=demo+last, 2=+min/max, 3=+shapelet",
    )
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def normalize_gender(v) -> str:
    if pd.isna(v):
        return "Unknown"
    s = str(v).strip().lower()
    if s in {"0", "0.0", "female", "f"}:
        return "Female"
    if s in {"1", "1.0", "male", "m"}:
        return "Male"
    if s in {"2", "2.0", "other"}:
        return "Other"
    if s in {"unknown", "unk", "nan", ""}:
        return "Unknown"
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
    if "hisp" in s:
        return "Hispanic"
    if "white" in s:
        return "White"
    if s in {"other", "native", "american indian", "pacific islander"}:
        return "Other"
    if s in {"unknown", "unk", "nan", ""}:
        return "Unknown"
    return "Other"


def harmonize_demographics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if "gender" in train_df.columns and "gender" in test_df.columns:
        train_df["gender"] = train_df["gender"].apply(normalize_gender)
        test_df["gender"] = test_df["gender"].apply(normalize_gender)
    if "race" in train_df.columns and "race" in test_df.columns:
        train_df["race"] = train_df["race"].apply(normalize_race)
        test_df["race"] = test_df["race"].apply(normalize_race)


def build_feature_versions(df: pd.DataFrame) -> Dict[int, List[str]]:
    demographic_cols = [c for c in ["gender", "age", "race"] if c in df.columns]
    last_cols = [col for col in df.columns if "_last" in col]
    min_cols = [col for col in df.columns if "_min" in col]
    max_cols = [col for col in df.columns if "_max" in col]

    shapelet_cols: List[str] = []
    for col in df.columns:
        if (
            col not in demographic_cols
            and "_last" not in col
            and "_min" not in col
            and "_max" not in col
            and col not in {"stay_id", "label"}
            and "_" in col
        ):
            shapelet_cols.append(col)

    return {
        1: demographic_cols + last_cols,
        2: demographic_cols + last_cols + min_cols + max_cols,
        3: demographic_cols + last_cols + min_cols + max_cols + shapelet_cols,
    }


def preprocess_for_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
    model_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[int]]:
    x_train = train_df[list(features)].copy()
    x_test = test_df[list(features)].copy()
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    categorical_cols = [c for c in ["gender", "race"] if c in x_train.columns]
    cat_indices = [x_train.columns.get_loc(c) for c in categorical_cols]

    numeric_cols = [col for col in x_train.columns if col not in categorical_cols]
    for col in numeric_cols:
        mean_val = x_train[col].mean()
        x_train[col] = x_train[col].fillna(mean_val)
        x_test[col] = x_test[col].fillna(mean_val)

    if model_type == "lightgbm":
        for col in categorical_cols:
            categories = pd.Index(
                sorted(set(x_train[col].astype(str).dropna()) | set(x_test[col].astype(str).dropna()))
            )
            x_train[col] = pd.Categorical(x_train[col].astype(str), categories=categories)
            x_test[col] = pd.Categorical(x_test[col].astype(str), categories=categories)
    elif model_type == "catboost":
        for col in categorical_cols:
            x_train[col] = x_train[col].astype(str).fillna("NA")
            x_test[col] = x_test[col].astype(str).fillna("NA")
    else:
        for col in categorical_cols:
            le = LabelEncoder()
            both = pd.concat([x_train[col], x_test[col]], axis=0).astype(str)
            le.fit(both)
            x_train[col] = le.transform(x_train[col].astype(str))
            x_test[col] = le.transform(x_test[col].astype(str))

    return x_train, x_test, y_train, y_test, cat_indices


def create_model(model_type: str, random_state: int):
    if model_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=800,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=0.1,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=4,
        )
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            verbosity=-1,
            random_state=random_state,
            force_row_wise=True,
            deterministic=True,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=5,
            n_estimators=800,
            min_child_samples=10,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=0.8,
            n_jobs=4,
        )
    if model_type == "catboost":
        return cb.CatBoostClassifier(
            iterations=800,
            learning_rate=0.08,
            depth=7,
            l2_leaf_reg=9,
            random_state=random_state,
            verbose=False,
            thread_count=4,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    pred_pos_rate = float((tp + fp) / (tp + tn + fp + fn))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(auc(recall_curve, precision_curve)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ppv": ppv,
        "npv": npv,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "pred_pos_rate": pred_pos_rate,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, run_id)
    pred_dir = os.path.join(outdir, "predictions")
    model_dir = os.path.join(outdir, "models")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    harmonize_demographics(train_df, test_df)

    feature_versions = build_feature_versions(train_df)
    stay_id_test = test_df["stay_id"].to_numpy()

    results: List[Dict[str, float]] = []

    for model_type in args.models:
        for version in args.versions:
            features = feature_versions[version]
            x_train, x_test, y_train, y_test, cat_indices = preprocess_for_model(
                train_df, test_df, features, model_type
            )

            model = create_model(model_type, args.random_state)
            if model_type == "lightgbm":
                model.fit(
                    x_train,
                    y_train,
                    categorical_feature=[c for c in ["gender", "race"] if c in x_train.columns],
                )
            elif model_type == "catboost":
                model.fit(x_train, y_train, cat_features=cat_indices if cat_indices else None)
            else:
                model.fit(x_train, y_train)

            y_prob = model.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metric = evaluate(y_test, y_pred, y_prob)
            metric.update(
                {
                    "model": model_type,
                    "version": version,
                    "n_features": len(features),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "n_test_pos": int(y_test.sum()),
                    "n_test_neg": int((1 - y_test).sum()),
                }
            )
            results.append(metric)

            pred_path = os.path.join(pred_dir, f"{model_type}_v{version}_predictions.csv")
            pd.DataFrame(
                {
                    "stay_id": stay_id_test,
                    "y_true": y_test,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                }
            ).to_csv(pred_path, index=False)

            model_path = os.path.join(model_dir, f"{model_type}_v{version}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            print(
                f"[DONE] {model_type} v{version}: "
                f"AUROC={metric['auroc']:.4f}, AUPRC={metric['auprc']:.4f}, "
                f"F1={metric['f1']:.4f}"
            )

    summary_df = pd.DataFrame(results).sort_values(["model", "version"]).reset_index(drop=True)
    summary_csv = os.path.join(outdir, "metrics_summary.csv")
    summary_json = os.path.join(outdir, "metrics_summary.json")
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    run_cfg = {
        "run_id": run_id,
        "train_file": os.path.abspath(args.train_file),
        "test_file": os.path.abspath(args.test_file),
        "models": args.models,
        "versions": args.versions,
        "random_state": args.random_state,
        "output_dir": outdir,
    }
    with open(os.path.join(outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print("\n=== External validation finished ===")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Predictions : {pred_dir}")
    print(f"Models      : {model_dir}")


if __name__ == "__main__":
    main()
