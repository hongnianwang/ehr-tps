#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure4_shap_panels import (
    MAX_DISPLAY,
    MODEL_ORDER,
    FIGURE_DIR,
    load_group_contributions,
    plot_beeswarm,
    plot_group_panel,
    plot_waterfall_panel,
    select_waterfall_sample,
    set_style,
    train_xgboost_for_shap,
)
from figure4_shapelet_examples import (
    OUT_DIR as FIG4D_OUT_DIR,
    PROCESSED_TEST,
    RAW_AKI_TS,
    aggregate_patient_series,
    build_candidate_table,
    find_best_match,
    plot_case_panel,
    resolve_case_specs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUT_DIR = REPO_DIR / "results" / "figures" / "figure4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def build_figure() -> dict:
    set_style()

    group_df = load_group_contributions()
    analyzer, k2, k3 = train_xgboost_for_shap()
    sample_id = select_waterfall_sample(analyzer, k2, k3, max_terms=9)

    ts = pd.read_csv(RAW_AKI_TS, parse_dates=["charttime", "icu_intime", "aki_time"])
    processed_test = pd.read_csv(PROCESSED_TEST)
    candidate_table = build_candidate_table(ts, processed_test)
    resolved_case_specs = resolve_case_specs(candidate_table)

    fig = plt.figure(figsize=(8.9, 10.4), dpi=300)
    outer = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.05, 3.55, 1.72],
        hspace=0.40,
    )

    # A1-A3: feature group contributions.
    top = outer[0].subgridspec(1, 3, wspace=0.34)
    top_axes = [fig.add_subplot(top[0, i]) for i in range(3)]
    for ax, model, label in zip(top_axes, MODEL_ORDER, ["A1", "A2", "A3"]):
        plot_group_panel(ax, group_df, model, label)
    top_axes[0].set_ylabel("Contribution (%)", fontsize=8)

    # B-C: global and local SHAP explanations.
    middle = outer[1].subgridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.56)
    ax_b = fig.add_subplot(middle[0, 0])
    plot_beeswarm(ax_b, analyzer, k3, max_display=min(MAX_DISPLAY, 16))

    c_grid = middle[0, 1].subgridspec(2, 1, hspace=0.34)
    ax_c1 = fig.add_subplot(c_grid[0, 0])
    ax_c2 = fig.add_subplot(c_grid[1, 0])
    plot_waterfall_panel([ax_c1, ax_c2], analyzer, k2, k3, sample_id)

    # D1-D3: representative real-patient shapelet matches.
    bottom = outer[2].subgridspec(1, 3, wspace=0.36)
    bottom_axes = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    for ax, spec in zip(bottom_axes, resolved_case_specs):
        series_df = aggregate_patient_series(ts, spec.stay_id, spec.variable)
        match = find_best_match(series_df, spec.shapelet_values)
        plot_case_panel(ax, spec, series_df, match)

    for ax in bottom_axes:
        ax.set_xlabel("")
        ax.set_xticks([-36, -24, -12, 0])
    bottom_axes[1].set_xlabel("Hours before AKI onset", fontsize=8)

    handles, labels = bottom_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        fontsize=6.6,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.7,
    )

    fig.subplots_adjust(left=0.08, right=0.985, top=0.985, bottom=0.075)

    png_out = OUT_DIR / "figure4_interpretability.png"
    pdf_out = OUT_DIR / "figure4_interpretability.pdf"
    figure_png = FIGURE_DIR / "figure4_interpretability.png"
    figure_pdf = FIGURE_DIR / "figure4_interpretability.pdf"
    fig.savefig(png_out, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(figure_png, dpi=600, bbox_inches="tight")
    fig.savefig(figure_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_out}")
    print(f"Saved {pdf_out}")
    return {"png": png_out, "pdf": pdf_out}


def main() -> None:
    build_figure()


if __name__ == "__main__":
    main()
