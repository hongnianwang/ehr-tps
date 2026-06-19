"""
SHAP analysis with:
1) unit-aware feature display names (var_display_map + units_map),
2) colored SHAP summary plot (feature-value color encoding),
3) per-patient V2 vs V3 error-correction waterfall plots.

"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
MAX_DISPLAY = 20

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DATA_DIR = Path(
    os.environ.get(
        "EMRTPS_MIMIC_MODELING_DIR",
        REPO_DIR / "data" / "mimiciv" / "processed_modeling_cohort",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "EMRTPS_SHAP_OUTPUT_DIR",
        REPO_DIR / "results" / "shap_analysis",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "processed_train_with_shapelets.csv"
TEST_PATH = DATA_DIR / "processed_test_with_shapelets.csv"
EICU_TEST_PATH = Path(
    os.environ.get(
        "EMRTPS_EICU_TEST_PATH",
        REPO_DIR / "data" / "eicu" / "processed" / "processed_test_with_shapelets.csv",
    )
)


def normalize_gender(v):
    if pd.isna(v):
        return "Unknown"
    s = str(v).strip().lower()
    if s in {"0", "0.0", "female", "f"}:
        return "Female"
    if s in {"1", "1.0", "male", "m"}:
        return "Male"
    return "Unknown"


def normalize_race(v):
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
    return "Unknown"


def format_feature_name(feature_name: str) -> str:
    var_display_map = {
        "heart": "Heart Rate",
        "heart_rate": "Heart Rate",
        "sbp": "SBP",
        "dbp": "DBP",
        "spo2": "SpO2",
        "o2_saturation": "O2 Saturation",
        "bun": "BUN",
        "creatinine": "Creatinine",
        "potassium": "Potassium",
        "glucose": "Glucose",
        "sodium": "Sodium",
        "chloride": "Chloride",
        "bicarbonate": "Bicarbonate",
        "calcium": "Calcium",
        "magnesium": "Magnesium",
        "hemoglobin": "Hemoglobin",
        "hematocrit": "Hematocrit",
        "platelet": "Platelet",
        "wbc": "WBC",
        "rbc": "RBC",
        "lactate": "Lactate",
        "respiratory_rate": "Respiratory Rate",
        "temperature": "Temperature",
        "gender": "Gender",
        "age": "Age",
        "race": "Race",
    }

    units_map = {
        "heart": "bpm",
        "heart_rate": "bpm",
        "sbp": "mmHg",
        "dbp": "mmHg",
        "spo2": "%",
        "o2_saturation": "%",
        "bun": "mg/dL",
        "creatinine": "mg/dL",
        "potassium": "mEq/L",
        "glucose": "mg/dL",
        "sodium": "mEq/L",
        "chloride": "mEq/L",
        "bicarbonate": "mEq/L",
        "calcium": "mg/dL",
        "magnesium": "mg/dL",
        "hemoglobin": "g/dL",
        "hematocrit": "%",
        "platelet": "x10^3/uL",
        "wbc": "x10^3/uL",
        "rbc": "x10^6/uL",
        "lactate": "mmol/L",
        "respiratory_rate": "breaths/min",
        "temperature": "C",
    }

    # Handle standard aggregate suffix first (supports variables with underscores)
    for suffix in ["_last", "_min", "_max"]:
        if feature_name.endswith(suffix):
            var_name = feature_name[: -len(suffix)]
            disp = var_display_map.get(var_name, var_name.upper())
            suffix_name = suffix.replace("_", "").capitalize()
            unit = units_map.get(var_name, "")
            return f"{disp} {suffix_name} ({unit})" if unit else f"{disp} {suffix_name}"

    # Handle shapelet-like names: var_<numeric-sequence>
    if "_" in feature_name:
        parts = feature_name.split("_", 1)
        var_name = parts[0]
        tail = parts[1]
        if any(ch.isdigit() for ch in tail):
            disp = var_display_map.get(var_name, var_name.upper())
            unit = units_map.get(var_name, "")
            values = tail.split("-")
            cleaned_values = []
            for v in values:
                try:
                    num = float(v)
                    cleaned_values.append(str(int(num)) if num == int(num) else v)
                except Exception:
                    cleaned_values.append(v)
            values_clean = "-".join(cleaned_values)
            return f"{disp} ({values_clean} {unit})" if unit else f"{disp} ({values_clean})"

    return var_display_map.get(feature_name, feature_name)


def get_feature_type(feature_name: str) -> str:
    if feature_name in {"gender", "age", "race"}:
        return "demographic"
    if feature_name.endswith("_last"):
        return "last"
    if feature_name.endswith("_min"):
        return "min"
    if feature_name.endswith("_max"):
        return "max"
    if "_" in feature_name and any(ch.isdigit() for ch in feature_name.split("_", 1)[1]):
        return "shapelet"
    return "other"


@dataclass
class DatasetPack:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    categorical_info: List


class SHAPAnalyzer:
    def __init__(self, train_path: Path, test_path: Path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.feature_sets = self._build_feature_sets(self.train_df)
        self.datasets: Dict[int, DatasetPack] = {}
        self.models: Dict[str, dict] = {}
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self.shap_values: Dict[str, dict] = {}

    @staticmethod
    def _build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
        demographic_cols = ["gender", "age", "race"]
        last_cols = [c for c in df.columns if c.endswith("_last")]
        min_cols = [c for c in df.columns if c.endswith("_min")]
        max_cols = [c for c in df.columns if c.endswith("_max")]

        shapelet_cols = []
        for c in df.columns:
            if (
                c not in demographic_cols
                and not c.endswith("_last")
                and not c.endswith("_min")
                and not c.endswith("_max")
                and c not in {"stay_id", "label"}
                and "_" in c
            ):
                shapelet_cols.append(c)

        return {
            "v1": demographic_cols + last_cols,
            "v2": demographic_cols + last_cols + min_cols + max_cols,
            "v3": demographic_cols + last_cols + min_cols + max_cols + shapelet_cols,
        }

    def load_datasets(self, preprocessing_for: str):
        for version in [1, 2, 3]:
            feats = self.feature_sets[f"v{version}"]
            x_train = self.train_df[feats].copy()
            x_test = self.test_df[feats].copy()
            y_train = self.train_df["label"].copy()
            y_test = self.test_df["label"].copy()

            # Preserve stay_id index for per-patient analysis
            x_train.index = self.train_df["stay_id"].values
            x_test.index = self.test_df["stay_id"].values
            y_train.index = x_train.index
            y_test.index = x_test.index

            cat_cols = [c for c in ["gender", "race"] if c in x_train.columns]
            categorical_info = []

            if preprocessing_for == "lightgbm":
                for c in cat_cols:
                    x_train[c] = x_train[c].astype("category")
                    x_test[c] = x_test[c].astype("category")
                categorical_info = cat_cols

            elif preprocessing_for == "catboost":
                if "gender" in x_train.columns:
                    x_train["gender"] = x_train["gender"].apply(normalize_gender)
                    x_test["gender"] = x_test["gender"].apply(normalize_gender)
                if "race" in x_train.columns:
                    x_train["race"] = x_train["race"].apply(normalize_race)
                    x_test["race"] = x_test["race"].apply(normalize_race)
                for c in cat_cols:
                    x_train[c] = x_train[c].astype(str).fillna("NA")
                    x_test[c] = x_test[c].astype(str).fillna("NA")
                    categorical_info.append(x_train.columns.get_loc(c))

            else:
                for c in cat_cols:
                    le = LabelEncoder()
                    merged = pd.concat([x_train[c], x_test[c]], axis=0).astype(str)
                    le.fit(merged)
                    x_train[c] = le.transform(x_train[c].astype(str))
                    x_test[c] = le.transform(x_test[c].astype(str))
                    categorical_info.append(x_train.columns.get_loc(c))

            numeric_cols = [c for c in x_train.columns if c not in cat_cols]
            if numeric_cols:
                means = x_train[numeric_cols].mean()
                x_train[numeric_cols] = x_train[numeric_cols].fillna(means)
                x_test[numeric_cols] = x_test[numeric_cols].fillna(means)

            self.datasets[version] = DatasetPack(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                categorical_info=categorical_info,
            )

    @staticmethod
    def _build_model(model_type: str):
        if model_type == "xgboost":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=4,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.10,
                reg_lambda=1.00,
                gamma=0.02,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            )
        if model_type == "lightgbm":
            return lgb.LGBMClassifier(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                verbosity=-1,
                random_state=RANDOM_STATE,
                force_row_wise=True,
                deterministic=True,
                learning_rate=0.04,
                num_leaves=80,
                max_depth=6,
                n_estimators=1500,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=0.50,
                reg_alpha=0.01,
                min_split_gain=0.02,
            )
        if model_type == "catboost":
            return cb.CatBoostClassifier(
                iterations=900,
                learning_rate=0.12,
                depth=6,
                l2_leaf_reg=6,
                random_strength=1.0,
                bagging_temperature=0.50,
                random_state=RANDOM_STATE,
                verbose=False,
            )
        raise ValueError(f"Unsupported model_type: {model_type}")

    def train_model(self, model_type: str, version: int):
        data = self.datasets[version]
        model = self._build_model(model_type)

        if model_type == "lightgbm":
            model.fit(
                data.x_train,
                data.y_train,
                categorical_feature=data.categorical_info if data.categorical_info else None,
            )
        elif model_type == "catboost":
            model.fit(
                data.x_train,
                data.y_train,
                cat_features=data.categorical_info if data.categorical_info else None,
            )
        else:
            model.fit(data.x_train, data.y_train)

        prob = model.predict_proba(data.x_test)[:, 1]
        auc = roc_auc_score(data.y_test, prob)
        pred = model.predict(data.x_test)

        key = f"{model_type}_v{version}"
        self.models[key] = {
            "model": model,
            "auc": float(auc),
            "pred": pred,
            "prob": prob,
            "y_test": data.y_test,
        }
        print(f"[{key}] AUROC = {auc:.4f}")
        return key

    def create_shap_explainer(self, model_key: str, sample_size: Optional[int] = None):
        version = int(model_key.split("_v")[-1])
        model = self.models[model_key]["model"]
        x_test = self.datasets[version].x_test

        if sample_size is not None and len(x_test) > sample_size:
            rng = np.random.RandomState(RANDOM_STATE)
            keep_ids = rng.choice(x_test.index.values, sample_size, replace=False)
            x_sample = x_test.loc[keep_ids].copy()
        else:
            x_sample = x_test.copy()

        explainer = shap.TreeExplainer(model)
        try:
            sv = explainer.shap_values(x_sample, check_additivity=False)
        except TypeError:
            sv = explainer.shap_values(x_sample)

        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.asarray(expected_value).ravel()[-1])
        else:
            expected_value = float(expected_value)

        self.explainers[model_key] = explainer
        self.shap_values[model_key] = {
            "values": sv,
            "data": x_sample,
            "expected_value": expected_value,
        }
        print(f"[{model_key}] SHAP values ready: {sv.shape}")

    def plot_shap_summary(self, model_key: str, max_display: int = 20):
        shap_data = self.shap_values[model_key]
        sv = shap_data["values"]
        x = shap_data["data"]
        formatted_names = [format_feature_name(c) for c in x.columns]

        plt.figure(figsize=(8.2, 9.2), facecolor="white")
        shap.summary_plot(
            sv,
            x,
            feature_names=formatted_names,
            max_display=max_display,
            show=False,
            color_bar=True,
        )
        plt.xlabel("SHAP value")
        plt.tight_layout()
        out = OUTPUT_DIR / f"{model_key}_summary_plot_colored.png"
        plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[Saved] {out}")

        plt.figure(figsize=(6.8, 7.5), facecolor="white")
        shap.summary_plot(
            sv,
            x,
            feature_names=formatted_names,
            max_display=max_display,
            plot_type="bar",
            show=False,
        )
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        out_bar = OUTPUT_DIR / f"{model_key}_summary_bar_colored.png"
        plt.savefig(out_bar, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[Saved] {out_bar}")

    def _prepare_external_v3_features(self, external_df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        feats = self.feature_sets["v3"]
        x_ext = external_df[feats].copy()
        train_pack = self.datasets[3]
        x_train_ref = train_pack.x_train

        cat_cols = [c for c in ["gender", "race"] if c in x_ext.columns]

        if model_type == "catboost":
            if "gender" in x_ext.columns:
                x_ext["gender"] = x_ext["gender"].apply(normalize_gender)
            if "race" in x_ext.columns:
                x_ext["race"] = x_ext["race"].apply(normalize_race)
            for c in cat_cols:
                x_ext[c] = x_ext[c].astype(str).fillna("NA")

        elif model_type == "lightgbm":
            # keep categorical dtype consistent with training columns
            for c in cat_cols:
                x_ext[c] = x_ext[c].astype("category")
                if c in x_train_ref.columns:
                    cats = x_train_ref[c].cat.categories
                    x_ext[c] = x_ext[c].cat.set_categories(cats)

        else:
            # mimic internal preprocessing: label-encode using train+test mimic domain
            for c in cat_cols:
                le = LabelEncoder()
                merged = pd.concat([self.train_df[c], self.test_df[c]], axis=0).astype(str)
                le.fit(merged)
                ext_vals = x_ext[c].astype(str)
                unknown_mask = ~ext_vals.isin(set(le.classes_))
                if unknown_mask.any():
                    ext_vals = ext_vals.where(~unknown_mask, le.classes_[0])
                x_ext[c] = le.transform(ext_vals)

        numeric_cols = [c for c in x_ext.columns if c not in cat_cols]
        if numeric_cols:
            means = x_train_ref[numeric_cols].mean()
            x_ext[numeric_cols] = x_ext[numeric_cols].fillna(means)

        return x_ext

    def plot_external_summary(self, model_key: str, external_test_path: Path, tag: str = "eicu", max_display: int = 20):
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not trained yet.")
        if not external_test_path.exists():
            print(f"[Skip] External file not found: {external_test_path}")
            return

        model_type = model_key.split("_v")[0]
        model = self.models[model_key]["model"]
        ext_df = pd.read_csv(external_test_path)
        x_ext = self._prepare_external_v3_features(ext_df, model_type=model_type)

        explainer = shap.TreeExplainer(model)
        try:
            sv = explainer.shap_values(x_ext, check_additivity=False)
        except TypeError:
            sv = explainer.shap_values(x_ext)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]

        formatted_names = [format_feature_name(c) for c in x_ext.columns]

        plt.figure(figsize=(8.2, 9.2), facecolor="white")
        shap.summary_plot(
            sv,
            x_ext,
            feature_names=formatted_names,
            max_display=max_display,
            show=False,
            color_bar=True,
        )
        plt.xlabel("SHAP value")
        plt.tight_layout()
        out = OUTPUT_DIR / f"{model_key}_{tag}_summary_plot_colored.png"
        plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[Saved] {out}")

    def analyze_feature_groups(self, model_key: str):
        shap_data = self.shap_values[model_key]
        sv = shap_data["values"]
        x = shap_data["data"]
        feature_names = list(x.columns)

        groups = {
            "Demographics": [c for c in feature_names if c in {"gender", "age", "race"}],
            "Last": [c for c in feature_names if c.endswith("_last")],
            "Min": [c for c in feature_names if c.endswith("_min")],
            "Max": [c for c in feature_names if c.endswith("_max")],
            "Shapelets": [
                c
                for c in feature_names
                if c not in {"gender", "age", "race"}
                and not c.endswith("_last")
                and not c.endswith("_min")
                and not c.endswith("_max")
                and "_" in c
            ],
        }

        rows = []
        for g, feats in groups.items():
            idx = [feature_names.index(f) for f in feats if f in feature_names]
            if not idx:
                continue
            val = float(np.abs(sv[:, idx]).sum(axis=1).mean())
            rows.append({"group": g, "mean_abs_shap": val})

        gdf = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
        group_order = ["Demographics", "Last", "Min", "Max", "Shapelets"]
        gdf["group"] = pd.Categorical(gdf["group"], categories=group_order, ordered=True)
        gdf = gdf.sort_values("group")
        gdf["percent"] = 100 * gdf["mean_abs_shap"] / gdf["mean_abs_shap"].sum()

        feat_abs = np.abs(sv).mean(axis=0)
        fdf = pd.DataFrame(
            {
                "feature_raw_name": feature_names,
                "feature_display_name": [format_feature_name(c) for c in feature_names],
                "mean_abs_shap": feat_abs,
                "feature_type": [get_feature_type(c) for c in feature_names],
            }
        ).sort_values("mean_abs_shap", ascending=False)

        gcsv = OUTPUT_DIR / f"{model_key}_group_contribution.csv"
        fcsv = OUTPUT_DIR / f"{model_key}_feature_importance.csv"
        gdf.to_csv(gcsv, index=False)
        fdf.to_csv(fcsv, index=False)

        plt.figure(figsize=(4.8, 3.8), facecolor="white")
        base_color = "#4C78A8"
        shapelet_color = "#D6604D"
        x_pos = np.arange(len(gdf))
        plt.bar(
            x_pos,
            gdf["percent"],
            color=[
                shapelet_color if g == "Shapelets" else base_color
                for g in gdf["group"].astype(str)
            ],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.7,
        )
        for i, v in enumerate(gdf["percent"].values):
            plt.text(i, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        plt.xticks(x_pos, gdf["group"], rotation=20, ha="right")
        plt.ylabel("Contribution (%)")
        plt.grid(axis="y", alpha=0.25)
        plt.ylim(0, max(45, float(gdf["percent"].max()) * 1.15))
        plt.tight_layout()
        out = OUTPUT_DIR / f"{model_key}_feature_groups_percent.png"
        plt.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[Saved] {out}")
        return gdf, fdf

    def plot_error_correction_waterfall(
        self, model_key_v2: str, model_key_v3: str, num_samples: int = 30
    ):
        out_dir = OUTPUT_DIR / f"error_correction_{model_key_v3.split('_v')[0]}"
        out_dir.mkdir(parents=True, exist_ok=True)

        v2 = self.models[model_key_v2]
        v3 = self.models[model_key_v3]
        y_true = v2["y_test"]

        common_ids = y_true.index.intersection(v3["y_test"].index)
        y = y_true.loc[common_ids]
        p2 = pd.Series(v2["pred"], index=v2["y_test"].index).loc[common_ids]
        p3 = pd.Series(v3["pred"], index=v3["y_test"].index).loc[common_ids]

        corrected = common_ids[(p2 != y) & (p3 == y) & (y == 1)]
        if len(corrected) == 0:
            print(f"[{model_key_v3}] No corrected positive samples for waterfall.")
            return

        rng = np.random.RandomState(RANDOM_STATE)
        if len(corrected) > num_samples:
            chosen = rng.choice(corrected.values, num_samples, replace=False)
        else:
            chosen = corrected.values

        shap_v2 = self.shap_values[model_key_v2]
        shap_v3 = self.shap_values[model_key_v3]
        x2 = shap_v2["data"]
        x3 = shap_v3["data"]
        sv2 = shap_v2["values"]
        sv3 = shap_v3["values"]
        ev2 = shap_v2["expected_value"]
        ev3 = shap_v3["expected_value"]

        records = []
        summary_lines = []

        for sid in chosen:
            if sid not in x2.index or sid not in x3.index:
                continue

            i2 = list(x2.index).index(sid)
            i3 = list(x3.index).index(sid)

            v2_vals = sv2[i2]
            v3_vals = sv3[i3]
            d2 = x2.loc[sid]
            d3 = x3.loc[sid]

            fn2 = [format_feature_name(c) for c in x2.columns]
            fn3 = [format_feature_name(c) for c in x3.columns]

            # record per-feature data for later auditing
            for j, col in enumerate(x2.columns):
                records.append(
                    {
                        "sample_id": sid,
                        "version": "V2",
                        "feature_raw_name": col,
                        "feature_display_name": fn2[j],
                        "feature_value": d2.iloc[j],
                        "shap_value": float(v2_vals[j]),
                        "abs_shap_value": float(abs(v2_vals[j])),
                        "feature_type": get_feature_type(col),
                    }
                )
            for j, col in enumerate(x3.columns):
                records.append(
                    {
                        "sample_id": sid,
                        "version": "V3",
                        "feature_raw_name": col,
                        "feature_display_name": fn3[j],
                        "feature_value": d3.iloc[j],
                        "shap_value": float(v3_vals[j]),
                        "abs_shap_value": float(abs(v3_vals[j])),
                        "feature_type": get_feature_type(col),
                    }
                )

            # detailed text
            det = out_dir / f"sample_{sid}_details.txt"
            with open(det, "w", encoding="utf-8") as f:
                f.write("=" * 88 + "\n")
                f.write(f"Sample ID: {sid}\n")
                f.write("Error correction: V2 wrong -> V3 correct (true AKI=1)\n")
                f.write("=" * 88 + "\n\n")
                f.write(f"V2 output: {ev2 + np.sum(v2_vals):.6f}\n")
                f.write(f"V3 output: {ev3 + np.sum(v3_vals):.6f}\n\n")
                f.write("Top 12 |SHAP| in V2:\n")
                rank2 = np.argsort(np.abs(v2_vals))[::-1][:12]
                for k in rank2:
                    val = d2.iloc[k]
                    sval = v2_vals[k]
                    if isinstance(val, (int, float, np.number)):
                        vtxt = f"{float(val):.4f}"
                    else:
                        vtxt = str(val)
                    f.write(f"  - {fn2[k]} | value={vtxt} | SHAP={sval:.6f}\n")
                f.write("\nTop 12 |SHAP| in V3:\n")
                rank3 = np.argsort(np.abs(v3_vals))[::-1][:12]
                for k in rank3:
                    val = d3.iloc[k]
                    sval = v3_vals[k]
                    if isinstance(val, (int, float, np.number)):
                        vtxt = f"{float(val):.4f}"
                    else:
                        vtxt = str(val)
                    f.write(f"  - {fn3[k]} | value={vtxt} | SHAP={sval:.6f}\n")

            summary_lines.append(
                f"sample={sid} | v2={ev2 + np.sum(v2_vals):.6f} | v3={ev3 + np.sum(v3_vals):.6f}"
            )

            # waterfall figure
            fig, axes = plt.subplots(2, 1, figsize=(3.2, 10.2), dpi=300)
            plt.subplots_adjust(left=0.60, right=0.98, hspace=0.30, top=0.97, bottom=0.04)

            plt.sca(axes[0])
            exp2 = shap.Explanation(
                values=v2_vals,
                base_values=ev2,
                data=d2.values,
                feature_names=fn2,
            )
            shap.plots.waterfall(exp2, max_display=12, show=False)
            axes[0].tick_params(axis="x", labelsize=9)
            axes[0].tick_params(axis="y", labelsize=9)

            plt.sca(axes[1])
            exp3 = shap.Explanation(
                values=v3_vals,
                base_values=ev3,
                data=d3.values,
                feature_names=fn3,
            )
            shap.plots.waterfall(exp3, max_display=12, show=False)
            axes[1].tick_params(axis="x", labelsize=9)
            axes[1].tick_params(axis="y", labelsize=9)

            out_img = out_dir / f"sample_{sid}_v2_vs_v3_comparison.png"
            plt.savefig(out_img, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        if records:
            rec_df = pd.DataFrame(records)
            rec_df.to_csv(out_dir / "feature_value_shap_analysis.csv", index=False)

            with open(out_dir / "shap_analysis_summary.txt", "w", encoding="utf-8") as f:
                f.write("SHAP Error-Correction Summary (V2 -> V3)\n")
                f.write("=" * 60 + "\n")
                for line in summary_lines:
                    f.write(line + "\n")

            stat_df = (
                rec_df.groupby(["version", "feature_type"], as_index=False)["abs_shap_value"]
                .agg(["mean", "std", "max", "count"])
                .reset_index()
            )
            stat_df.to_csv(out_dir / "feature_value_shap_statistics.csv", index=False)
            print(f"[Saved] waterfall and audit files -> {out_dir}")


def compare_feature_groups_all_models(output_dir: Path = OUTPUT_DIR):
    """
    Reproduce the original concentric-donut style from legacy
    figure4_shap_analysis.py::compare_feature_groups_all_models(),
    using current retrained model data.
    """
    model_order = ["xgboost", "lightgbm", "catboost"]
    group_order = ["Demographics", "Last", "Min", "Max", "Shapelets"]

    # Load current model group contributions (raw mean_abs_shap, not percentages)
    results = {}
    rows = []
    for model in model_order:
        csv_path = output_dir / f"{model}_v3_group_contribution.csv"
        if not csv_path.exists():
            continue
        gdf = pd.read_csv(csv_path)
        model_dict = {}
        for g in group_order:
            sub = gdf[gdf["group"] == g]
            val = float(sub["mean_abs_shap"].iloc[0]) if len(sub) > 0 else 0.0
            model_dict[g] = val
            rows.append({"model": model, "group": g, "mean_abs_shap": val})
        results[model] = model_dict

    if "xgboost" not in results:
        print("[compare_feature_groups_all_models] Missing xgboost group data.")
        return None

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "all_models_group_contribution_raw.csv", index=False)

    # ===== Original style starts here =====
    fig, ax = plt.subplots(figsize=(6, 5))

    groups = list(results["xgboost"].keys())

    colors = {
        "Demographics": "#4393C3",
        "Last": "#4393C3",
        "Min": "#4393C3",
        "Max": "#4393C3",
        "Shapelets": "#D6604D",
    }
    group_colors = [colors.get(g, "#4393C3") for g in groups]

    ring_width = 0.2
    rings = [
        {"model": "catboost", "inner": 0.3, "outer": 0.5},
        {"model": "lightgbm", "inner": 0.5, "outer": 0.7},
        {"model": "xgboost", "inner": 0.7, "outer": 0.9},
    ]

    for ring in rings:
        sizes = list(results[ring["model"]].values())
        ax.pie(
            sizes,
            labels=None,
            colors=group_colors,
            startangle=90,
            radius=ring["outer"],
            wedgeprops={
                "width": ring_width,
                "edgecolor": "white",
                "linewidth": 2,
                "alpha": 0.7,
            },
            pctdistance=(ring["inner"] + ring["outer"]) / 2 / ring["outer"],
            textprops={"fontsize": 12, "weight": "normal"},
        )

    ax.text(
        0,
        0,
        "Feature Group\nContributions",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )

    models_info = [
        {"model": "CatBoost", "radius": 0.4, "angle": -90},
        {"model": "LightGBM", "radius": 0.6, "angle": -90},
        {"model": "XGBoost", "radius": 0.8, "angle": -90},
    ]
    for info in models_info:
        angle_rad = np.radians(info["angle"])
        x = info["radius"] * np.cos(angle_rad)
        y = info["radius"] * np.sin(angle_rad)
        ax.text(
            x,
            y,
            info["model"],
            ha="center",
            va="center",
            fontsize=14,
            color="#303030",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    groups_ordered = ["Shapelets", "Max", "Min", "Last", "Demographics"]
    angle_offset = 90
    total = sum(results["xgboost"].values())
    cumulative = 0

    for group in groups_ordered:
        value = results["xgboost"][group]
        angle = angle_offset - (cumulative + value / 2) / total * 360
        cumulative += value
        angle_rad = np.radians(angle)

        slice_percentage = value / total if total > 0 else 0
        if slice_percentage < 0.05:
            x_arrow = 0.91 * np.cos(angle_rad)
            y_arrow = 0.91 * np.sin(angle_rad)
            x_text = 1.0 * np.cos(angle_rad)
            y_text = 1.0 * np.sin(angle_rad)
            ax.annotate(
                group,
                xy=(x_arrow, y_arrow),
                xytext=(x_text, y_text),
                ha="center",
                va="bottom",
                fontsize=18,
                color="black",
                weight="normal",
                arrowprops=dict(arrowstyle="-", color="gray", lw=1.2),
            )
        else:
            radius = 1.0
            x = radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            if x > 0.1:
                ha = "left"
            elif x < -0.1:
                ha = "right"
            else:
                ha = "center"
            ax.text(
                x,
                y,
                group,
                ha=ha,
                va="center",
                fontsize=18,
                color="black",
                weight="normal",
            )

    ax.axis("equal")
    plt.tight_layout()

    out_path = output_dir / "all_models_feature_groups_concentric_clean.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Saved] {out_path}")
    return out_df


def replot_feature_group_bars_legacy(output_dir: Path = OUTPUT_DIR):
    """
    Replot per-model feature-group bars in legacy style from the original script:
    - Contribution percentage (percent), while keeping legacy visual style
    - same colors and typography logic as legacy analyze_feature_groups()
    """
    model_order = ["xgboost", "lightgbm", "catboost"]
    model_display = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}
    group_order = ["Demographics", "Last", "Min", "Max", "Shapelets"]

    color_map = {
        "Demographics": "#4393C3",
        "Last": "#4393C3",
        "Min": "#4393C3",
        "Max": "#4393C3",
        "Shapelets": "#D6604D",
    }

    for model in model_order:
        csv_path = output_dir / f"{model}_v3_group_contribution.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
        df = df.sort_values("group")

        groups = df["group"].tolist()
        total = float(df["mean_abs_shap"].sum())
        contributions = (
            (df["mean_abs_shap"] / total * 100.0).tolist() if total > 0 else [0.0] * len(df)
        )
        bar_colors = [color_map.get(g, "#4393C3") for g in groups]

        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        x_pos = np.arange(len(groups))
        bars = ax.bar(
            x_pos,
            contributions,
            color=bar_colors,
            alpha=0.7,
            edgecolor="gray",
            linewidth=1.0,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, fontsize=12, rotation=45, ha="right")
        ax.set_ylabel("Contribution (%)", fontsize=13)
        ax.set_title(f"Feature group contributions ({model_display.get(model, model)})", fontsize=14)
        if len(contributions) > 0:
            ax.set_ylim(0, max(contributions) * 1.15)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, contributions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        plt.tight_layout()
        out_path = output_dir / f"{model}_v3_feature_groups_legacy_percent.png"
        plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[Saved] {out_path}")


def main():
    np.random.seed(RANDOM_STATE)
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.family"] = "sans-serif"

    print("=" * 70)
    print("SHAP retrained v3 script")
    print(f"Train: {TRAIN_PATH}")
    print(f"Test : {TEST_PATH}")
    print(f"Out  : {OUTPUT_DIR}")
    print("=" * 70)

    manifest = {
        "train": str(TRAIN_PATH),
        "test": str(TEST_PATH),
        "output": str(OUTPUT_DIR),
        "models": {},
    }

    # V3 colored summary + group analysis for all models
    for model_type in ["xgboost", "lightgbm", "catboost"]:
        analyzer = SHAPAnalyzer(TRAIN_PATH, TEST_PATH)
        if model_type == "lightgbm":
            analyzer.load_datasets(preprocessing_for="lightgbm")
        elif model_type == "catboost":
            analyzer.load_datasets(preprocessing_for="catboost")
        else:
            analyzer.load_datasets(preprocessing_for="xgboost")

        k3 = analyzer.train_model(model_type, 3)
        analyzer.create_shap_explainer(k3, sample_size=None)
        analyzer.plot_shap_summary(k3, max_display=MAX_DISPLAY)
        analyzer.plot_external_summary(k3, EICU_TEST_PATH, tag="eicu", max_display=MAX_DISPLAY)
        analyzer.analyze_feature_groups(k3)

        manifest["models"][model_type] = {"v3_auroc": analyzer.models[k3]["auc"]}

    # Error-correction waterfall (V2 vs V3) for XGBoost by default
    analyzer_wf = SHAPAnalyzer(TRAIN_PATH, TEST_PATH)
    analyzer_wf.load_datasets(preprocessing_for="xgboost")
    k2 = analyzer_wf.train_model("xgboost", 2)
    k3 = analyzer_wf.train_model("xgboost", 3)
    analyzer_wf.create_shap_explainer(k2, sample_size=None)
    analyzer_wf.create_shap_explainer(k3, sample_size=None)
    analyzer_wf.plot_error_correction_waterfall(k2, k3, num_samples=30)

    # Restore the cross-model group comparison utility
    compare_feature_groups_all_models(OUTPUT_DIR)
    replot_feature_group_bars_legacy(OUTPUT_DIR)

    manifest["waterfall_model"] = "xgboost"
    manifest["waterfall_versions"] = ["v2", "v3"]
    with open(OUTPUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
