#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data" / "mimiciv"
OUT_DIR = Path(
    os.environ.get(
        "EMRTPS_FIGURE4D_OUTPUT_DIR",
        BASE_DIR / "results" / "figure4d_case_examples",
    )
)
FIGURE_DIR = Path(
    os.environ.get(
        "EMRTPS_FIGURE_EXPORT_DIR",
        BASE_DIR / "results" / "figures" / "figure4",
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RAW_AKI_TS = Path(os.environ.get("EMRTPS_MIMIC_AKI_TS_PATH", DATA_DIR / "data-aki_ts.csv"))
PROCESSED_TEST = Path(
    os.environ.get(
        "EMRTPS_MIMIC_PROCESSED_TEST_PATH",
        DATA_DIR / "processed" / "processed_test_with_shapelets.csv",
    )
)

BLUE = "#4393C3"
RED = "#D6604D"
ORANGE = "#F4A261"
GRAY = "#4D4D4D"
LIGHT_GRID = "#E8E8E8"
PATIENT_BLUE = "#2C7FB8"

VARIABLE_AGGREGATION = {
    "potassium": 4,
    "creatinine": 4,
    "bun": 4,
    "glucose": 4,
    "spo2": 1,
    "heart_rate": 1,
    "sbp": 1,
    "dbp": 1,
}


@dataclass(frozen=True)
class CaseSpec:
    panel: str
    stay_id: int | None
    variable: str
    variable_label: str
    unit: str
    pattern_label: str
    shapelet_values: tuple[float, ...]
    color: str


CASE_SPECS = [
    CaseSpec(
        panel="D1",
        stay_id=None,
        variable="spo2",
        variable_label="SpO2",
        unit="%",
        pattern_label="oxygenation instability",
        shapelet_values=(98, 96, 96, 98, 95),
        color=PATIENT_BLUE,
    ),
    CaseSpec(
        panel="D2",
        stay_id=None,
        variable="dbp",
        variable_label="DBP",
        unit="mmHg",
        pattern_label="diastolic hypotensive dip",
        shapelet_values=(68, 34, 60.5, 61),
        color=PATIENT_BLUE,
    ),
    CaseSpec(
        panel="D3",
        stay_id=None,
        variable="heart_rate",
        variable_label="Heart rate",
        unit="bpm",
        pattern_label="heart-rate decline",
        shapelet_values=(84, 87, 78, 73, 77),
        color=PATIENT_BLUE,
    ),
]


def resolve_case_specs(candidate_table: pd.DataFrame) -> list[CaseSpec]:
    """Resolve representative cases from local data without hard-coding stay IDs."""
    resolved: list[CaseSpec] = []
    for spec in CASE_SPECS:
        if spec.stay_id is not None:
            resolved.append(spec)
            continue
        sub = candidate_table[candidate_table["panel_template"] == spec.panel].sort_values("selection_score")
        if sub.empty:
            raise ValueError(f"No candidate case found for {spec.panel}")
        resolved.append(replace(spec, stay_id=int(sub.iloc[0]["stay_id"])))
    return resolved


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#333333",
            "axes.labelsize": 8,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def z_normalized_distance(values: np.ndarray, pattern: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    pattern = np.asarray(pattern, dtype=float)
    if np.std(values) > 1e-6 and np.std(pattern) > 1e-6:
        values = (values - values.mean()) / values.std()
        pattern = (pattern - pattern.mean()) / pattern.std()
    return float(np.sqrt(np.sum((values - pattern) ** 2)))


def raw_scaled_distance(values: np.ndarray, pattern: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    pattern = np.asarray(pattern, dtype=float)
    scale = max(float(np.nanmax(values) - np.nanmin(values)), float(np.nanmax(pattern) - np.nanmin(pattern)), 1.0)
    return float(np.sqrt(np.mean(((values - pattern) / scale) ** 2)))


def aggregate_patient_series(ts: pd.DataFrame, stay_id: int, variable: str) -> pd.DataFrame:
    sub = ts[(ts["stay_id"] == stay_id) & ts[variable].notna()].copy()
    if sub.empty:
        raise ValueError(f"No {variable} measurements found for stay_id={stay_id}")

    sub = sub.sort_values("charttime")
    icu_time = sub["icu_intime"].iloc[0]
    aki_time = sub["aki_time"].iloc[0]
    bin_hours = VARIABLE_AGGREGATION[variable]

    rows = []
    # Use the last measurement in each bin, matching the processed S3M matrix
    # construction and avoiding overplotting raw bedside measurements.
    for _, row in sub.iterrows():
        hours_from_icu = (row["charttime"] - icu_time).total_seconds() / 3600.0
        if bin_hours == 4:
            bin_idx = int((hours_from_icu + 8) / 4)
            bin_center_time = icu_time + pd.Timedelta(hours=(-8 + bin_idx * 4 + 2))
        else:
            bin_idx = int(hours_from_icu)
            bin_center_time = icu_time + pd.Timedelta(hours=bin_idx)

        rows.append(
            {
                "stay_id": stay_id,
                "variable": variable,
                "bin_idx": bin_idx,
                "bin_center_time": bin_center_time,
                "charttime": row["charttime"],
                "value": float(row[variable]),
                "icu_intime": icu_time,
                "aki_time": aki_time,
            }
        )

    df = pd.DataFrame(rows).sort_values("charttime")
    df = df.groupby("bin_idx", as_index=False).tail(1).sort_values("bin_idx").reset_index(drop=True)
    df["hours_to_aki"] = (df["bin_center_time"] - aki_time).dt.total_seconds() / 3600.0
    return df


def find_best_match(series_df: pd.DataFrame, pattern: tuple[float, ...]) -> dict:
    pattern_arr = np.asarray(pattern, dtype=float)
    window = len(pattern_arr)
    candidates = []

    for start in range(0, len(series_df) - window + 1):
        win = series_df.iloc[start : start + window].copy()
        if win["bin_idx"].diff().dropna().max() > 1:
            continue
        if win["bin_center_time"].max() > win["aki_time"].iloc[0]:
            continue

        values = win["value"].to_numpy(dtype=float)
        zdist = z_normalized_distance(values, pattern_arr)
        rdist = raw_scaled_distance(values, pattern_arr)
        corr = 0.0
        if np.std(values) > 1e-6 and np.std(pattern_arr) > 1e-6:
            corr = float(np.corrcoef(values, pattern_arr)[0, 1])

        h_end = float(win["hours_to_aki"].max())
        h_start = float(win["hours_to_aki"].min())
        # Prefer matches before the prediction window and close to AKI onset.
        horizon_penalty = abs(h_end + 16.0) / 24.0
        score = zdist + 1.3 * rdist - 0.65 * corr + 0.25 * horizon_penalty
        candidates.append(
            {
                "score": score,
                "z_distance": zdist,
                "raw_scaled_distance": rdist,
                "correlation": corr,
                "h_start": h_start,
                "h_end": h_end,
                "values": values,
                "indices": list(win.index),
                "window_df": win,
            }
        )

    if not candidates:
        raise ValueError("No valid pre-AKI match found")
    return sorted(candidates, key=lambda x: x["score"])[0]


def format_sequence(values: np.ndarray | tuple[float, ...]) -> str:
    out = []
    for value in values:
        val = float(value)
        out.append(f"{val:.0f}" if abs(val - round(val)) < 1e-6 else f"{val:.1f}")
    return "-".join(out)


def plot_case_panel(ax: plt.Axes, spec: CaseSpec, series_df: pd.DataFrame, match: dict) -> dict:
    display_df = series_df[(series_df["hours_to_aki"] >= -36) & (series_df["hours_to_aki"] <= 1)].copy()
    match_df = match["window_df"].copy()
    pattern = np.asarray(spec.shapelet_values, dtype=float)

    ax.axvspan(-12, 0, color=ORANGE, alpha=0.13, lw=0, zorder=0)
    ax.axvline(-12, color=ORANGE, lw=0.8, linestyle="--", alpha=0.85, zorder=1)
    ax.axvline(0, color=RED, lw=1.3, linestyle="--", alpha=0.95, zorder=1)

    ax.plot(
        display_df["hours_to_aki"],
        display_df["value"],
        color=spec.color,
        lw=1.4,
        marker="o",
        markersize=2.8,
        alpha=0.72,
        label="Patient trajectory",
        zorder=2,
    )
    ax.plot(
        match_df["hours_to_aki"],
        match_df["value"],
        color=RED,
        lw=3.0,
        marker="o",
        markersize=4.0,
        alpha=0.95,
        label="Matched subsequence",
        zorder=4,
    )
    ax.plot(
        match_df["hours_to_aki"],
        pattern,
        color=GRAY,
        lw=1.8,
        linestyle=(0, (2, 2)),
        marker="s",
        markersize=3.8,
        alpha=0.9,
        label="Reference shapelet",
        zorder=3,
    )

    observed_seq = format_sequence(match_df["value"].to_numpy())
    shapelet_seq = format_sequence(pattern)
    ax.set_title(
        spec.panel,
        loc="left",
        fontsize=8.4,
        fontweight="bold",
        pad=6,
    )

    ymin = min(display_df["value"].min(), pattern.min())
    ymax = max(display_df["value"].max(), pattern.max())
    pad = max((ymax - ymin) * 0.18, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-36, 1)
    ax.set_ylabel(f"{spec.variable_label} ({spec.unit})")
    ax.grid(True, color=LIGHT_GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_top = ymax + pad * 0.45
    ax.text(
        -6,
        y_top,
        "12-h\nwindow",
        color="#B05A00",
        fontsize=5.8,
        ha="center",
        va="top",
        linespacing=0.85,
    )
    ax.text(
        0.35,
        y_top,
        "AKI",
        color=RED,
        fontsize=5.8,
        ha="left",
        va="top",
    )

    return {
        "panel": spec.panel,
        "stay_id": spec.stay_id,
        "variable": spec.variable,
        "variable_label": spec.variable_label,
        "pattern_label": spec.pattern_label,
        "unit": spec.unit,
        "shapelet_sequence": shapelet_seq,
        "observed_sequence": observed_seq,
        "z_distance": match["z_distance"],
        "raw_scaled_distance": match["raw_scaled_distance"],
        "correlation": match["correlation"],
        "hours_to_aki_start": match["h_start"],
        "hours_to_aki_end": match["h_end"],
    }


def build_candidate_table(ts: pd.DataFrame, processed_test: pd.DataFrame) -> pd.DataFrame:
    aki_ids = set(processed_test.loc[processed_test["label"] == 1, "stay_id"].astype(int))
    rows = []
    for spec in CASE_SPECS:
        for stay_id in sorted(aki_ids):
            try:
                series_df = aggregate_patient_series(ts, stay_id, spec.variable)
                match = find_best_match(series_df, spec.shapelet_values)
            except Exception:
                continue
            rows.append(
                {
                    "panel_template": spec.panel,
                    "stay_id": stay_id,
                    "variable": spec.variable,
                    "pattern_label": spec.pattern_label,
                    "shapelet_sequence": format_sequence(spec.shapelet_values),
                    "observed_sequence": format_sequence(match["values"]),
                    "selection_score": match["score"],
                    "z_distance": match["z_distance"],
                    "raw_scaled_distance": match["raw_scaled_distance"],
                    "correlation": match["correlation"],
                    "hours_to_aki_start": match["h_start"],
                    "hours_to_aki_end": match["h_end"],
                }
            )
    out = pd.DataFrame(rows).sort_values(["panel_template", "selection_score"])
    out.to_csv(OUT_DIR / "figure4_shapelet_examples_candidates.csv", index=False)
    return out


def main() -> None:
    set_style()
    ts = pd.read_csv(RAW_AKI_TS, parse_dates=["charttime", "icu_intime", "aki_time"])
    processed_test = pd.read_csv(PROCESSED_TEST)
    candidate_table = build_candidate_table(ts, processed_test)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.85), dpi=300, sharex=True)
    selected_rows = []
    trajectory_rows = []

    resolved_specs = resolve_case_specs(candidate_table)

    for ax, spec in zip(axes, resolved_specs):
        series_df = aggregate_patient_series(ts, spec.stay_id, spec.variable)
        match = find_best_match(series_df, spec.shapelet_values)
        selected_rows.append(plot_case_panel(ax, spec, series_df, match))

        display_df = series_df[(series_df["hours_to_aki"] >= -36) & (series_df["hours_to_aki"] <= 1)].copy()
        display_df["panel"] = spec.panel
        display_df["is_matched_bin"] = display_df.index.isin(match["indices"])
        trajectory_rows.append(display_df)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=6.6,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.7,
    )
    for ax in axes:
        ax.set_xlabel("")
        ax.set_xticks([-36, -24, -12, 0])

    fig.supxlabel("Hours before AKI onset", fontsize=8.0, y=0.14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.30, wspace=0.36)

    png_out = OUT_DIR / "figure4_shapelet_examples.png"
    pdf_out = OUT_DIR / "figure4_shapelet_examples.pdf"
    figure_png = FIGURE_DIR / "figure4_shapelet_examples.png"
    figure_pdf = FIGURE_DIR / "figure4_shapelet_examples.pdf"
    fig.savefig(png_out, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(figure_png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(OUT_DIR / "figure4_shapelet_examples_selected.csv", index=False)
    pd.concat(trajectory_rows, ignore_index=True).to_csv(OUT_DIR / "figure4_shapelet_examples_trajectories.csv", index=False)

    manifest = {
        "task": "Figure 4D representative shapelet case examples",
        "raw_ts": str(RAW_AKI_TS),
        "processed_test": str(PROCESSED_TEST),
        "output_dir": str(OUT_DIR),
        "figure_dir": str(FIGURE_DIR),
        "selection_rule": (
            "Internal holdout AKI cases were screened for close pre-AKI matches "
            "to selected shapelet patterns using z-normalized distance, raw-scaled "
            "distance, and correlation; final examples prioritize visual clarity."
        ),
        "outputs": {
            "png": str(png_out),
            "pdf": str(pdf_out),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
            "selected_examples_csv": str(OUT_DIR / "figure4_shapelet_examples_selected.csv"),
            "candidate_rankings_csv": str(OUT_DIR / "figure4_shapelet_examples_candidates.csv"),
            "panel_trajectories_csv": str(OUT_DIR / "figure4_shapelet_examples_trajectories.csv"),
        },
        "selected_examples": selected_rows,
        "candidate_rows": int(len(candidate_table)),
    }
    with open(OUT_DIR / "figure4_shapelet_examples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
