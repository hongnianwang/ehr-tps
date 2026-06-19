import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
base_dir = os.getcwd()
DATA_DIR = f"{base_dir}/../data/mimiciv/ts_vital"
SHAPELET_DIR = f"{base_dir}/../results/s3m_out"
OUTPUT_DIR = f"{base_dir}/../results/shapelet_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Plotting style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.linewidth"] = 1.0  # 修改：从0.5增加到1.0
plt.rcParams["axes.edgecolor"] = "#333333"  # 新增：设置轴线颜色为深灰色
plt.rcParams["xtick.color"] = "#333333"  # 新增：设置x轴刻度颜色
plt.rcParams["ytick.color"] = "#333333"  # 新增：设置y轴刻度颜色
plt.rcParams["axes.labelcolor"] = "#333333"  # 新增：设置轴标签颜色


COLORS = {
    "Non-AKI": "#4393C3",  # 蓝色
    "AKI": "#D6604D",  # 红色
    "shapelet": "#2A9D8F",  # 绿色
}


def load_ts_data(feature_name):
    """Load time series data and labels."""
    file_path = os.path.join(DATA_DIR, f"Train_{feature_name}.csv")
    df = pd.read_csv(file_path, header=None)
    labels = df.iloc[:, 0]
    timeseries = df.iloc[:, 1:]
    return timeseries, labels


def load_shapelet_data(feature_name, file_suffix):
    """Load shapelet JSON data."""
    file_path = os.path.join(SHAPELET_DIR, f"{feature_name}_{file_suffix}.json")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def parse_shapelet_string(shapelet_str):
    """Extract numeric values from shapelet string."""
    parts = shapelet_str.split("_")
    if len(parts) == 2:
        values_str = parts[1]
        return [float(val) for val in values_str.split("-")]
    return None


def find_matching_shapelet(shapelet_data, target_values):
    """Find matching shapelet in JSON data."""
    for shapelet in shapelet_data["shapelets"]:
        if len(shapelet["shapelet"]) == len(target_values):
            if all(
                abs(a - b) < 0.001 for a, b in zip(shapelet["shapelet"], target_values)
            ):
                return shapelet
    return None


def find_similar_sequences(
    timeseries,
    labels,
    shapelet_values,
    n_positive=12,
    n_negative=12,
):
    """Find sequences most similar to shapelet using Euclidean distance."""
    all_matches = []

    for idx, row in timeseries.iterrows():
        sequence = row.dropna().values
        label = labels.iloc[idx]

        if len(sequence) < len(shapelet_values):
            continue

        min_dist = float("inf")
        best_start = 0

        for i in range(len(sequence) - len(shapelet_values) + 1):
            subseq = sequence[i : i + len(shapelet_values)]
            dist = np.sqrt(np.sum((subseq - shapelet_values) ** 2))

            if dist < min_dist:
                min_dist = dist
                best_start = i

        all_matches.append(
            {
                "index": idx,
                "distance": min_dist,
                "start": best_start,
                "sequence": sequence,
                "label": label,
            }
        )

    all_matches.sort(key=lambda x: x["distance"])

    positive_matches = [m for m in all_matches if m["label"] == 1][:n_positive]
    negative_matches = [m for m in all_matches if m["label"] == 0][:n_negative]

    # 输出被选中的样本索引
    print(f"\nSelected Non-AKI indices: {[m['index'] for m in negative_matches]}")
    print(f"Selected AKI indices: {[m['index'] for m in positive_matches]}")

    return positive_matches, negative_matches


def create_comparison_subplot(
    feature_name, shapelet_str, file_suffix
):
    """Create comparison visualization showing both AKI and Non-AKI matches."""

    shapelet_values = parse_shapelet_string(shapelet_str)
    if not shapelet_values:
        print(f"Cannot parse shapelet string: {shapelet_str}")
        return

    # Load data
    timeseries, labels = load_ts_data(feature_name)
    shapelet_data = load_shapelet_data(feature_name, file_suffix)

    shapelet_info = find_matching_shapelet(shapelet_data, shapelet_values)
    if not shapelet_info:
        print(f"Shapelet not found in JSON file")
        return

    # Find similar sequences for both groups
    positive_matches, negative_matches = find_similar_sequences(
        timeseries,
        labels,
        shapelet_values,
        n_positive=10,
        n_negative=10,
    )

    # Calculate average distances
    avg_aki_dist = np.mean([m["distance"] for m in positive_matches])
    avg_nonaki_dist = np.mean([m["distance"] for m in negative_matches])

    print(f"\n{feature_name} - {shapelet_str}")
    print(f"Number of Non-AKI matches: {len(negative_matches)}")
    print(f"Number of AKI matches: {len(positive_matches)}")
    print(f"Average distance (Non-AKI): {avg_nonaki_dist:.2f}")
    print(f"Average distance (AKI): {avg_aki_dist:.2f}")
    print(f"Difference: {avg_aki_dist - avg_nonaki_dist:.2f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=300)

    # Set labels based on feature
    if feature_name == "spo2":
        clean_values = "100-96-98-97-94"
        y_label = f"SpO₂ ({clean_values}%)"
    elif feature_name == "heart":
        clean_values = "88-98-82-86.3"
        y_label = f"Heart Rate ({clean_values} bpm)"
    elif feature_name == "bun":
        y_label = "BUN (mg/dL)"
        clean_values = shapelet_str.split("_")[1]
        title_suffix = f"BUN ({clean_values})"
    elif feature_name == "creatinine":
        y_label = "Creatinine (mg/dL)"
        clean_values = shapelet_str.split("_")[1]
        title_suffix = f"Creatinine ({clean_values})"
    elif feature_name == "potassium":
        y_label = "Potassium (mmol/L)"
        clean_values = shapelet_str.split("_")[1]
        title_suffix = f"Potassium ({clean_values})"
    elif feature_name == "sbp":
        y_label = "SBP (mmHg)"
        clean_values = shapelet_str.split("_")[1]
        title_suffix = f"SBP ({clean_values})"
    elif feature_name == "dbp":
        y_label = "DBP (mmHg)"
        clean_values = shapelet_str.split("_")[1]
        title_suffix = f"DBP ({clean_values})"
    else:
        y_label = feature_name
        clean_values = shapelet_str.split("_")[1]
        title_suffix = feature_name

    # Collect all values for y-axis range
    all_values = []
    for match in positive_matches + negative_matches:
        sequence = match["sequence"]
        start = match["start"]
        display_start = max(0, start - 6)
        display_end = min(len(sequence), start + len(shapelet_values) + 6)
        all_values.extend(sequence[display_start:display_end])
    all_values.extend(shapelet_values)

    y_min = min(all_values) - 2
    y_max = max(all_values) + 2

    # Plot Non-AKI sequences (蓝色)
    for match in negative_matches:
        sequence = match["sequence"]
        start = match["start"]

        display_start = max(0, start - 6)
        display_end = min(len(sequence), start + len(shapelet_values) + 6)

        x_offset = -start
        time_points = np.arange(display_start, display_end) + x_offset

        ax.plot(
            time_points,
            sequence[display_start:display_end],
            color=COLORS["Non-AKI"],
            alpha=0.35,
            linewidth=0.9,
        )

    # Plot AKI sequences (红色)
    for match in positive_matches:
        sequence = match["sequence"]
        start = match["start"]

        display_start = max(0, start - 6)
        display_end = min(len(sequence), start + len(shapelet_values) + 6)

        x_offset = -start
        time_points = np.arange(display_start, display_end) + x_offset

        ax.plot(
            time_points,
            sequence[display_start:display_end],
            color=COLORS["AKI"],
            alpha=0.35,
            linewidth=0.9,
        )

    # Plot shapelet pattern (绿色加粗)
    shapelet_time = np.arange(len(shapelet_values))
    ax.plot(
        shapelet_time,
        shapelet_values,
        "o-",
        color=COLORS["shapelet"],
        linewidth=2.8,
        markersize=7,
        label=f"Shapelet Pattern",
        zorder=10,
    )

    # Set x-axis
    x_min = -6
    x_max = len(shapelet_values) + 5
    major_ticks = list(range(-6, len(shapelet_values) + 6, 2))
    ax.set_xticks(major_ticks)

    tick_labels = []
    for tick in major_ticks:
        if tick < 0:
            tick_labels.append(f"{tick}h")
        elif tick == 0:
            tick_labels.append("Start")
        elif tick == len(shapelet_values) - 1:
            tick_labels.append("End")
        elif tick < len(shapelet_values):
            tick_labels.append("")
        else:
            tick_labels.append(f"+{tick-len(shapelet_values)+1}h")

    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Time relative to shapelet (hours)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Styling
    # Styling - ggplot2风格
    ax.grid(
        True, linestyle="-", alpha=0.15, color="#E5E5E5", linewidth=0.5
    )  # 修改网格线
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)  # 加粗
    ax.spines["bottom"].set_linewidth(1.3)  # 加粗
    ax.spines["left"].set_color("#333333")  # 新增：深灰色
    ax.spines["bottom"].set_color("#333333")  # 新增：深灰色

    # 设置刻度线
    ax.tick_params(
        axis="both", which="major", length=4, width=1, colors="#333333"
    )  # 新增

    # Legend with average distances
    from matplotlib.lines import Line2D

    n_samples = len(negative_matches)

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=COLORS["Non-AKI"],
            lw=2,
            alpha=0.7,
            label=f"Non-AKI (top {n_samples} closest matches)",  # 修改这里
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["AKI"],
            lw=2,
            alpha=0.7,
            label=f"AKI (top {n_samples} closest matches)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["shapelet"],
            lw=2.8,
            marker="o",
            label="Shapelet Pattern",
        ),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9, frameon=False)

    # Save
    output_path = os.path.join(
        OUTPUT_DIR, f"{feature_name}_{clean_values}_comparison.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=600, facecolor="white")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    """Generate comparison visualizations for all shapelets."""
    shapelet_configs = [
        {
            "feature": "spo2",
            "shapelet_str": "spo2_100.0-96.0-98.0-97.0-94.0",
            "file_suffix": "3_8",
        },
        {
            "feature": "heart",
            "shapelet_str": "heart_88.0-98.0-82.0-86.3",
            "file_suffix": "3_8",
        },
    ]

    for config in shapelet_configs:
        create_comparison_subplot(
            feature_name=config["feature"],
            shapelet_str=config["shapelet_str"],
            file_suffix=config["file_suffix"],
        )


if __name__ == "__main__":
    main()
