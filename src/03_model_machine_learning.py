# Model Performance Comparison and Statistical Testing
# ================================================================

import logging
import os
from datetime import datetime

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration
base_dir = os.getcwd()
DATA_DIR = f"{base_dir}/../data/mimiciv/processed"
OUTPUT_DIR = f"{base_dir}/../results/model_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
log_file = os.path.join(
    OUTPUT_DIR, f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)


def delong_roc_test(y_true, y_pred_1, y_pred_2):
    """DeLong test for comparing two ROC curves."""
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)

    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return np.nan, np.nan

    def auc_contribution(preds):
        preds_pos = preds[pos_indices][:, np.newaxis]
        preds_neg = preds[neg_indices][np.newaxis, :]
        comparisons = preds_pos > preds_neg
        ties = preds_pos == preds_neg
        scores = comparisons.astype(float) + 0.5 * ties.astype(float)
        return scores.mean(axis=1)

    v1 = auc_contribution(y_pred_1)
    v2 = auc_contribution(y_pred_2)
    diff = v1 - v2
    mean_diff = np.mean(diff)

    if len(diff) < 2:
        return np.nan, np.nan

    std_diff = np.std(diff, ddof=1) / np.sqrt(len(pos_indices))

    if std_diff == 0:
        return np.nan, np.nan

    z = mean_diff / std_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p


def bootstrap_auprc_test(y_true, y_pred_1, y_pred_2, n_bootstrap=1000):
    """Bootstrap test for comparing AUPRC values."""
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)

    n_samples = len(y_true)

    precision_1, recall_1, _ = precision_recall_curve(y_true, y_pred_1)
    precision_2, recall_2, _ = precision_recall_curve(y_true, y_pred_2)
    auprc_1 = auc(recall_1, precision_1)
    auprc_2 = auc(recall_2, precision_2)
    original_diff = auprc_1 - auprc_2

    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[indices]
        pred_1_boot = y_pred_1[indices]
        pred_2_boot = y_pred_2[indices]

        try:
            precision_1_boot, recall_1_boot, _ = precision_recall_curve(
                y_boot, pred_1_boot
            )
            precision_2_boot, recall_2_boot, _ = precision_recall_curve(
                y_boot, pred_2_boot
            )
            auprc_1_boot = auc(recall_1_boot, precision_1_boot)
            auprc_2_boot = auc(recall_2_boot, precision_2_boot)
            bootstrap_diffs.append(auprc_1_boot - auprc_2_boot)
        except:
            continue

    bootstrap_diffs = np.array(bootstrap_diffs)
    p_value = 2 * min(np.mean(bootstrap_diffs >= 0), np.mean(bootstrap_diffs <= 0))
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return p_value, (ci_lower, ci_upper), original_diff


def load_datasets(
    train_path, test_path=None, random_state=42, preprocessing_for="xgboost_catboost"
):
    """Load and preprocess datasets for three feature versions."""

    train_df = pd.read_csv(train_path)
    if test_path:
        test_df = pd.read_csv(test_path)
        has_test = True
    else:
        has_test = False

    demographic_cols = ["gender", "age", "race"]
    last_cols = [col for col in train_df.columns if "_last" in col]
    min_cols = [col for col in train_df.columns if "_min" in col]
    max_cols = [col for col in train_df.columns if "_max" in col]

    shapelet_cols = []
    for col in train_df.columns:
        if (
            col not in demographic_cols
            and "_last" not in col
            and "_min" not in col
            and "_max" not in col
            and col != "stay_id"
            and col != "label"
            and "_" in col
        ):
            shapelet_cols.append(col)

    v1_features = demographic_cols + last_cols
    v2_features = demographic_cols + last_cols + min_cols + max_cols
    v3_features = demographic_cols + last_cols + min_cols + max_cols + shapelet_cols

    datasets = {}

    for version, features in zip([1, 2, 3], [v1_features, v2_features, v3_features]):
        X_train = train_df[features].copy()
        y_train = train_df["label"].copy()

        categorical_cols = [col for col in ["gender", "race"] if col in X_train.columns]

        if preprocessing_for == "lightgbm":
            for col in categorical_cols:
                X_train[col] = X_train[col].astype("category")
            categorical_info = categorical_cols
        else:
            categorical_info = []
            for i, col in enumerate(categorical_cols):
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                categorical_info.append(i)

        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
        if numeric_cols:
            X_train[numeric_cols] = X_train[numeric_cols].fillna(
                X_train[numeric_cols].mean()
            )

        if has_test:
            X_test = test_df[features].copy()
            y_test = test_df["label"].copy()

            if preprocessing_for == "lightgbm":
                for col in categorical_cols:
                    X_test[col] = X_test[col].astype("category")
            else:
                for col in categorical_cols:
                    le = LabelEncoder()
                    le.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))

            if numeric_cols:
                for col in numeric_cols:
                    if X_test[col].isna().any():
                        X_test[col] = X_test[col].fillna(X_train[col].mean())
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state
            )

        datasets[version] = (X_train, X_test, y_train, y_test, categorical_info)

    return datasets


def evaluate_model(y_true, y_pred, y_prob=None):
    """Evaluate model performance."""
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        results["ROC AUC"] = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        results["AUPRC"] = auc(recall, precision)

    return results


def build_model_single_version(dataset_version, model_type="xgboost"):
    """Build and evaluate model for single dataset version."""

    if model_type.lower() == "lightgbm":
        datasets = load_datasets(
            train_path=os.path.join(DATA_DIR, "processed_train_with_shapelets.csv"),
            test_path=os.path.join(DATA_DIR, "processed_test_with_shapelets.csv"),
            preprocessing_for="lightgbm",
        )
    else:
        datasets = load_datasets(
            train_path=os.path.join(DATA_DIR, "processed_train_with_shapelets.csv"),
            test_path=os.path.join(DATA_DIR, "processed_test_with_shapelets.csv"),
        )

    X_train, X_test, y_train, y_test, categorical_info = datasets[dataset_version]

    logging.info(f"Building {model_type.upper()} - Version {dataset_version}")
    logging.info(
        f"Features: {X_train.shape[1]}, Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}"
    )

    if model_type.lower() == "xgboost":
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=800,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=0.1,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
    elif model_type.lower() == "lightgbm":
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            verbosity=-1,
            random_state=42,
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
        )
    elif model_type.lower() == "catboost":
        model = cb.CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            random_state=42,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_type.lower() in ["lightgbm"]:
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_info if categorical_info else None,
        )
    elif model_type.lower() in ["catboost"]:
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_info if categorical_info else None,
        )
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = evaluate_model(y_test, y_pred, y_prob)

    logging.info(f"{model_type.upper()} Version {dataset_version} Results:")
    for metric, value in results.items():
        logging.info(f"  {metric}: {value:.4f}")

    return {
        "model": model,
        "predictions": y_pred,
        "probabilities": y_prob,
        "metrics": results,
        "y_test": y_test,
        "feature_names": X_train.columns.tolist(),
    }


def perform_statistical_tests(version_results):
    """Perform and log statistical significance tests."""

    logging.info("\n" + "=" * 60)
    logging.info("Statistical Significance Tests")
    logging.info("=" * 60)

    comparisons = [
        ("Version 2", "Version 1", 2, 1),
        ("Version 3", "Version 1", 3, 1),
        ("Version 3", "Version 2", 3, 2),
    ]

    for comp_name1, comp_name2, v1, v2 in comparisons:
        if v1 in version_results and v2 in version_results:
            y_test = version_results[v1]["y_test"]
            y_prob_v1 = version_results[v1]["probabilities"]
            y_prob_v2 = version_results[v2]["probabilities"]

            roc_auc_v1 = version_results[v1]["metrics"]["ROC AUC"]
            roc_auc_v2 = version_results[v2]["metrics"]["ROC AUC"]
            auprc_v1 = version_results[v1]["metrics"]["AUPRC"]
            auprc_v2 = version_results[v2]["metrics"]["AUPRC"]

            logging.info(f"\n{comp_name1} vs {comp_name2}:")
            logging.info(
                f"  ROC AUC: {roc_auc_v1:.4f} vs {roc_auc_v2:.4f} (diff: {roc_auc_v1 - roc_auc_v2:.4f})"
            )
            logging.info(
                f"  AUPRC: {auprc_v1:.4f} vs {auprc_v2:.4f} (diff: {auprc_v1 - auprc_v2:.4f})"
            )

            z_roc, p_roc = delong_roc_test(y_test, y_prob_v1, y_prob_v2)
            logging.info(
                f"  DeLong test: Z={z_roc:.4f}, p={p_roc:.4f} {'(significant)' if p_roc < 0.05 else '(not significant)'}"
            )

            p_prc, ci_prc, _ = bootstrap_auprc_test(y_test, y_prob_v1, y_prob_v2)
            logging.info(
                f"  Bootstrap test: p={p_prc:.4f}, CI=[{ci_prc[0]:.4f}, {ci_prc[1]:.4f}] {'(significant)' if p_prc < 0.05 else '(not significant)'}"
            )


def run_all_models():
    """Run analysis for all three model types."""

    models_to_run = ["xgboost", "lightgbm", "catboost"]

    for model_type in models_to_run:
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting {model_type.upper()} Analysis")
        logging.info(f"{'='*60}")

        version_results = {}

        # Build models for all versions
        for version in [1, 2, 3]:
            version_results[version] = build_model_single_version(version, model_type)

        # Perform statistical tests
        perform_statistical_tests(version_results)

        # Save predictions only
        predictions_df = pd.DataFrame(
            {
                "y_true": version_results[1]["y_test"],
                "v1_prob": version_results[1]["probabilities"],
                "v2_prob": version_results[2]["probabilities"],
                "v3_prob": version_results[3]["probabilities"],
            }
        )
        predictions_file = os.path.join(OUTPUT_DIR, f"{model_type}_predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)
        logging.info(f"\nPredictions saved to: {predictions_file}")

        # Log summary metrics
        logging.info("\n" + "-" * 40)
        logging.info(f"{model_type.upper()} Summary:")
        for version in [1, 2, 3]:
            metrics = version_results[version]["metrics"]
            logging.info(
                f"Version {version}: ROC AUC={metrics['ROC AUC']:.4f}, AUPRC={metrics['AUPRC']:.4f}"
            )


if __name__ == "__main__":
    run_all_models()
    logging.info("\n" + "=" * 60)
    logging.info("All analyses completed successfully!")
    logging.info(f"Log saved to: {log_file}")
