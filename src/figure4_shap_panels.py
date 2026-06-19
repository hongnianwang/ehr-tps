#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from figure4_shap_analysis import (
    EICU_TEST_PATH,
    MAX_DISPLAY,
    OUTPUT_DIR as SHAP_ANALYSIS_DIR,
    RANDOM_STATE,
    TEST_PATH,
    TRAIN_PATH,
    SHAPAnalyzer,
    format_feature_name,
    get_feature_type,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUT_DIR = REPO_DIR / "results" / "figure4_shap_panels"
FIGURE_DIR = REPO_DIR / "results" / "figures" / "figure4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#4393C3"
RED = "#D6604D"
TEAL = "#79B8B0"
GRAY = "#4D4D4D"
LIGHT_GRID = "#E8E8E8"
SHAP_CMAP = LinearSegmentedColormap.from_list("paper_shap_blue_red", [BLUE, "#F7F7F7", RED])

MODEL_ORDER = ["xgboost", "lightgbm", "catboost"]
MODEL_LABEL = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}
GROUP_ORDER = ["Demographics", "Last", "Min", "Max", "Shapelets"]


def patch_shap_colors() -> None:
    try:
        import shap.plots.colors as shap_colors

        shap_colors.red_blue = SHAP_CMAP
        shap_colors.red_blue_no_bounds = SHAP_CMAP
        shap_colors.blue_rgb = mpl.colors.to_rgb(BLUE)
        shap_colors.red_rgb = mpl.colors.to_rgb(RED)
    except Exception:
        pass


def set_style() -> None:
    patch_shap_colors()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_group_contributions() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        csv_path = SHAP_ANALYSIS_DIR / f"{model}_v3_group_contribution.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}. Run figure4_shap_analysis.py first."
            )
        df = pd.read_csv(csv_path)
        if "percent" not in df.columns:
            total = df["mean_abs_shap"].sum()
            df["percent"] = 100 * df["mean_abs_shap"] / total
        df["model"] = model
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
    out["model_label"] = out["model"].map(MODEL_LABEL)
    out = out.sort_values(["model", "group"])
    out.to_csv(OUT_DIR / "figure4a_group_contributions.csv", index=False)
    return out


def train_xgboost_for_shap() -> tuple[SHAPAnalyzer, str, str]:
    analyzer = SHAPAnalyzer(TRAIN_PATH, TEST_PATH)
    analyzer.load_datasets(preprocessing_for="xgboost")
    k2 = analyzer.train_model("xgboost", 2)
    k3 = analyzer.train_model("xgboost", 3)
    analyzer.create_shap_explainer(k2, sample_size=None)
    analyzer.create_shap_explainer(k3, sample_size=None)
    return analyzer, k2, k3


def plot_group_panel(ax: plt.Axes, df: pd.DataFrame, model: str, panel_label: str) -> None:
    sub = df[df["model"] == model].sort_values("group")
    x = np.arange(len(GROUP_ORDER))
    vals = sub.set_index("group").reindex(GROUP_ORDER)["percent"].fillna(0).values
    colors = [RED if group == "Shapelets" else BLUE for group in GROUP_ORDER]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.7, alpha=0.78)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, rotation=35, ha="right", fontsize=6.8)
    ax.set_ylim(0, max(42, vals.max() * 1.18))
    ax.set_title(f"Feature group contributions ({MODEL_LABEL[model]})", fontsize=8, pad=5)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        -0.22,
        1.13,
        panel_label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _normalize_feature_values(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    if np.all(~np.isfinite(values)):
        return np.full(values.shape, 0.5)
    finite = values[np.isfinite(values)]
    lo, hi = np.nanpercentile(finite, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(finite), np.nanmax(finite)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(values.shape, 0.5)
    normed = (values - lo) / (hi - lo)
    return np.clip(normed, 0, 1)


def plot_beeswarm(
    ax: plt.Axes,
    analyzer: SHAPAnalyzer,
    model_key: str,
    max_display: int = 18,
    panel_label: str | None = "B",
    model_label: str = "XGBoost",
    write_top_csv: bool = True,
) -> pd.DataFrame:
    shap_data = analyzer.shap_values[model_key]
    sv = shap_data["values"]
    x = shap_data["data"]
    feature_names = list(x.columns)
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:max_display]
    top_idx = top_idx[np.argsort(mean_abs[top_idx])]

    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for pos, feat_idx in enumerate(top_idx):
        shap_vals = sv[:, feat_idx]
        feat_vals = pd.to_numeric(x.iloc[:, feat_idx], errors="coerce").to_numpy(dtype=float)
        color_vals = _normalize_feature_values(feat_vals)
        jitter = rng.normal(0, 0.055, size=len(shap_vals))
        ax.scatter(
            shap_vals,
            np.full_like(shap_vals, pos, dtype=float) + jitter,
            c=color_vals,
            cmap=SHAP_CMAP,
            s=9,
            alpha=0.72,
            edgecolors="none",
            rasterized=False,
        )
        rows.append(
            {
                "feature_raw_name": feature_names[feat_idx],
                "feature_display_name": format_feature_name(feature_names[feat_idx]),
                "feature_type": get_feature_type(feature_names[feat_idx]),
                "mean_abs_shap": float(mean_abs[feat_idx]),
            }
        )

    ax.axvline(0, color="#9E9E9E", lw=1.0, zorder=0)
    ax.set_yticks(np.arange(len(top_idx)))
    ax.set_yticklabels([format_feature_name(feature_names[i]) for i in top_idx], fontsize=6.8)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=8)
    ax.set_title(model_label, fontsize=8.5, pad=5)
    ax.grid(axis="y", color=LIGHT_GRID, linestyle=(0, (1.5, 3.0)), linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)
    if panel_label:
        ax.text(
            -0.13,
            1.06,
            panel_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
        )

    x_key = 1.035
    y0, y1 = 0.12, 0.48
    for j, value in enumerate(np.linspace(0, 1, 40)[:-1]):
        ya = y0 + (y1 - y0) * j / 39
        yb = y0 + (y1 - y0) * (j + 1) / 39
        ax.plot(
            [x_key, x_key],
            [ya, yb],
            transform=ax.transAxes,
            color=SHAP_CMAP(value),
            lw=5,
            solid_capstyle="butt",
            clip_on=False,
        )
    ax.text(x_key + 0.03, y1, "High", transform=ax.transAxes, fontsize=6.5, va="center", ha="left")
    ax.text(x_key + 0.03, y0, "Low", transform=ax.transAxes, fontsize=6.5, va="center", ha="left")

    top_df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    if write_top_csv:
        top_df.to_csv(OUT_DIR / "figure4b_xgboost_v3_top_shap_features.csv", index=False)
    return top_df


def train_model_for_shap(model_type: str) -> tuple[SHAPAnalyzer, str]:
    analyzer = SHAPAnalyzer(TRAIN_PATH, TEST_PATH)
    if model_type == "lightgbm":
        analyzer.load_datasets(preprocessing_for="lightgbm")
    elif model_type == "catboost":
        analyzer.load_datasets(preprocessing_for="catboost")
    else:
        analyzer.load_datasets(preprocessing_for="xgboost")
    key = analyzer.train_model(model_type, 3)
    analyzer.create_shap_explainer(key, sample_size=None)
    return analyzer, key


def build_standalone_shap_figures(max_display: int = 20) -> dict:
    set_style()
    outputs = {}
    figure_names = {
        "xgboost": "Fig4_shap_xgboost_v3_retrained",
        "lightgbm": "FigS3_shap_lightgbm_v3_retrained",
        "catboost": "FigS4_shap_catboost_v3_retrained",
    }
    for model_type in MODEL_ORDER:
        analyzer, key = train_model_for_shap(model_type)
        fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=300)
        top_df = plot_beeswarm(
            ax,
            analyzer,
            key,
            max_display=max_display,
            panel_label=None,
            model_label=MODEL_LABEL[model_type],
            write_top_csv=False,
        )
        top_df.to_csv(OUT_DIR / f"supplement_{model_type}_v3_top_shap_features.csv", index=False)
        png_out = OUT_DIR / f"supplement_{model_type}_shap_beeswarm.png"
        pdf_out = OUT_DIR / f"supplement_{model_type}_shap_beeswarm.pdf"
        figure_png = FIGURE_DIR / f"{figure_names[model_type]}.png"
        figure_pdf = FIGURE_DIR / f"{figure_names[model_type]}.pdf"
        fig.savefig(png_out, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_out, bbox_inches="tight")
        fig.savefig(figure_png, dpi=600, bbox_inches="tight")
        fig.savefig(figure_pdf, bbox_inches="tight")
        plt.close(fig)
        outputs[model_type] = {
            "png": str(png_out),
            "pdf": str(pdf_out),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
            "top_features_csv": str(OUT_DIR / f"supplement_{model_type}_v3_top_shap_features.csv"),
        }
    with open(OUT_DIR / "figure4_shap_panels_supplement_manifest.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    return outputs


def select_waterfall_sample(analyzer: SHAPAnalyzer, k2: str, k3: str, max_terms: int = 9) -> int:
    y = analyzer.models[k2]["y_test"]
    ids = y.index.intersection(analyzer.models[k3]["y_test"].index)
    y = y.loc[ids]
    p2_prob = pd.Series(analyzer.models[k2]["prob"], index=analyzer.models[k2]["y_test"].index).loc[ids]
    p3_prob = pd.Series(analyzer.models[k3]["prob"], index=analyzer.models[k3]["y_test"].index).loc[ids]
    p2 = (p2_prob >= 0.5).astype(int)
    p3 = (p3_prob >= 0.5).astype(int)

    corrected = ids[(y == 1) & (p2 != y) & (p3 == y)]
    if len(corrected) == 0:
        corrected = ids[y == 1]
    sv3 = analyzer.shap_values[k3]["values"]
    x3 = analyzer.shap_values[k3]["data"]
    shap_score_rows = []
    for sid in corrected:
        row_idx = list(x3.index).index(sid)
        vals = sv3[row_idx]
        order = np.argsort(np.abs(vals))[::-1]
        total_abs = float(np.abs(vals).sum())
        top_abs = float(np.abs(vals[order[:max_terms]]).sum())
        rest_sum_abs = float(abs(np.sum(vals[order[max_terms:]])))
        coverage = top_abs / total_abs if total_abs > 0 else 0.0
        improvement = float(p3_prob.loc[sid] - p2_prob.loc[sid])
        composite = improvement + 0.35 * coverage - 0.12 * rest_sum_abs
        shap_score_rows.append(
            {
                "stay_id": int(sid),
                "label": int(y.loc[sid]),
                "v2_prob": float(p2_prob.loc[sid]),
                "v3_prob": float(p3_prob.loc[sid]),
                "prob_improvement": improvement,
                "top_shap_abs_coverage": coverage,
                "rest_shap_sum_abs": rest_sum_abs,
                "selection_score": composite,
            }
        )
    score_df = pd.DataFrame(shap_score_rows).sort_values("selection_score", ascending=False)
    chosen = int(score_df["stay_id"].iloc[0])
    pd.DataFrame(
        {
            "stay_id": score_df["stay_id"].astype(int).values,
            "label": score_df["label"].astype(int).values,
            "v2_prob": score_df["v2_prob"].values,
            "v3_prob": score_df["v3_prob"].values,
            "prob_improvement": score_df["prob_improvement"].values,
            "top_shap_abs_coverage": score_df["top_shap_abs_coverage"].values,
            "rest_shap_sum_abs": score_df["rest_shap_sum_abs"].values,
            "selection_score": score_df["selection_score"].values,
        }
    ).to_csv(OUT_DIR / "figure4c_candidate_corrected_samples.csv", index=False)
    return chosen


def _format_value(value: object) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        val = float(value)
        if abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        return f"{val:.2g}"
    return str(value)


def _show_feature_value(raw_name: str, value: object) -> bool:
    if raw_name == "other_features":
        return False
    if get_feature_type(raw_name) == "shapelet":
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(value))


def plot_single_waterfall(
    ax: plt.Axes,
    shap_values: np.ndarray,
    values: pd.Series,
    raw_feature_names: list[str],
    feature_names: list[str],
    expected_value: float,
    title: str,
    max_terms: int = 9,
) -> list[dict]:
    order = np.argsort(np.abs(shap_values))[::-1]
    top = list(order[:max_terms])
    rest = [i for i in order[max_terms:]]

    terms = []
    for i in top:
        terms.append(
            {
                "name": feature_names[i],
                "raw_name": raw_feature_names[i],
                "value": values.iloc[i],
                "shap": float(shap_values[i]),
            }
        )
    if rest:
        terms.append(
            {
                "name": f"{len(rest)} other features",
                "raw_name": "other_features",
                "value": "",
                "shap": float(np.sum(shap_values[rest])),
            }
        )

    terms = terms[::-1]
    y_pos = np.arange(len(terms))
    vals = np.array([t["shap"] for t in terms])
    colors = [RED if v >= 0 else BLUE for v in vals]
    left = np.minimum(0, vals)
    width = np.abs(vals)

    ax.barh(y_pos, width, left=left, height=0.62, color=colors, alpha=0.76, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#BDBDBD", lw=0.8)
    for yv, val in zip(y_pos, vals):
        ha = "left" if val >= 0 else "right"
        x_txt = val + (0.035 if val >= 0 else -0.035)
        ax.text(x_txt, yv, f"{val:+.2f}", ha=ha, va="center", fontsize=6.3, color=colors[yv])

    labels = []
    for term in terms:
        label = term["name"]
        if _show_feature_value(term["raw_name"], term["value"]):
            label = f"{_format_value(term['value'])} = {label}"
        labels.append(label)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_title(
        f"{title}   f(x)={expected_value + float(np.sum(shap_values)):.2f}, E[f(X)]={expected_value:.2f}",
        fontsize=7.2,
        loc="left",
        pad=2,
    )
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("SHAP contribution", fontsize=7)
    lim = max(0.5, float(np.max(np.abs(vals))) * 1.35)
    ax.set_xlim(-lim, lim)
    return terms


def plot_waterfall_panel(
    axes: list[plt.Axes],
    analyzer: SHAPAnalyzer,
    k2: str,
    k3: str,
    sample_id: int,
) -> pd.DataFrame:
    records = []
    for ax, key, title in zip(axes, [k2, k3], ["Configuration 2", "Configuration 3"]):
        shap_data = analyzer.shap_values[key]
        x = shap_data["data"]
        sv = shap_data["values"]
        expected_value = shap_data["expected_value"]
        row_idx = list(x.index).index(sample_id)
        values = x.loc[sample_id]
        raw_names = list(x.columns)
        display_names = [format_feature_name(c) for c in raw_names]
        terms = plot_single_waterfall(
            ax,
            sv[row_idx],
            values,
            raw_names,
            display_names,
            expected_value,
            title,
            max_terms=9,
        )
        for term in terms:
            records.append(
                {
                    "stay_id": sample_id,
                    "configuration": title,
                    "feature_display_name": term["name"],
                    "feature_value": term["value"],
                    "shap_value": term["shap"],
                }
            )

    axes[0].text(
        -0.15,
        1.13,
        "C",
        transform=axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )
    out = pd.DataFrame(records)
    out.to_csv(OUT_DIR / "figure4c_waterfall_selected_sample.csv", index=False)
    return out


def build_figure() -> dict:
    set_style()
    group_df = load_group_contributions()
    analyzer, k2, k3 = train_xgboost_for_shap()
    sample_id = select_waterfall_sample(analyzer, k2, k3, max_terms=9)

    fig = plt.figure(figsize=(8.9, 7.55), dpi=300)
    outer = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 2.8],
        width_ratios=[1.04, 1.0],
        hspace=0.28,
        wspace=0.56,
    )

    top = outer[0, :].subgridspec(1, 3, wspace=0.34)
    for ax, model, label in zip([fig.add_subplot(top[0, i]) for i in range(3)], MODEL_ORDER, ["A1", "A2", "A3"]):
        plot_group_panel(ax, group_df, model, label)
    fig.axes[0].set_ylabel("Contribution (%)", fontsize=8)

    ax_b = fig.add_subplot(outer[1, 0])
    plot_beeswarm(ax_b, analyzer, k3, max_display=min(MAX_DISPLAY, 16))

    right = outer[1, 1].subgridspec(2, 1, hspace=0.34)
    ax_c1 = fig.add_subplot(right[0, 0])
    ax_c2 = fig.add_subplot(right[1, 0])
    plot_waterfall_panel([ax_c1, ax_c2], analyzer, k2, k3, sample_id)

    png_out = OUT_DIR / "figure4_shap_panels.png"
    pdf_out = OUT_DIR / "figure4_shap_panels.pdf"
    figure_png = FIGURE_DIR / "figure4_shap_panels.png"
    figure_pdf = FIGURE_DIR / "figure4_shap_panels.pdf"
    fig.savefig(png_out, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(figure_png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "train": str(TRAIN_PATH),
        "test": str(TEST_PATH),
        "external_test_reference": str(EICU_TEST_PATH),
        "shap_analysis_source_dir": str(SHAP_ANALYSIS_DIR),
        "output_dir": str(OUT_DIR),
        "figure_dir": str(FIGURE_DIR),
        "outputs": {
            "png": str(png_out),
            "pdf": str(pdf_out),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
        },
        "palette": {
            "low_or_non_shapelet": BLUE,
            "high_or_shapelet": RED,
            "middle": "#F7F7F7",
        },
        "layout_decision": "A4 omitted because A1-A3 already provide model-specific group contributions; the concentric ring is redundant and less quantitatively readable.",
        "waterfall_sample_id": sample_id,
        "waterfall_model": "xgboost",
        "waterfall_versions": ["v2", "v3"],
    }
    with open(OUT_DIR / "figure4_shap_panels_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    manifest = build_figure()
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
