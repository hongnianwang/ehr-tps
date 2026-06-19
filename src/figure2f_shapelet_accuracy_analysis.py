# Shapelet Classification Performance Comparison
# ================================================================

import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
base_dir = os.getcwd()
DATA_DIR = f"{base_dir}/../results/csv_results"
OUTPUT_DIR = f"{base_dir}/../results/shapelet_performance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 0.5

# Shapelet patterns
shapelet_patterns = [
    "bun_17-17-9",
    "sbp_107-125-131-104-134-133",
    "heart_88-98-82-86.3",
    "heart_84-87-78-73-77",
    "creatinine_0.8-0.8-0.9-0.7-0.7-0.7",
    "spo2_100-96-98-97-94",
    "spo2_98-96-96-98-95",
    "dbp_68-34-60.5-61",
    "dbp_42-69.7-59-53-82-60",
    "potassium_4.2-3.4-3.4-3.4-3.4-3.4",
]

# File mapping
file_mapping = {
    "bun": ["bun_2_4_metrics.csv", "bun_5_6_metrics.csv"],
    "sbp": ["sbp_2_6_metrics.csv"],
    "heart": ["heart_2_6_metrics.csv"],
    "creatinine": ["creatinine_2_4_metrics.csv", "creatinine_5_6_metrics.csv"],
    "spo2": ["spo2_2_6_metrics.csv"],
    "dbp": ["dbp_2_6_metrics.csv"],
    "potassium": ["potassium_2_4_metrics.csv", "potassium_5_6_metrics.csv"],
}


def parse_shapelet_string(shapelet_str):
    """Extract variable name and values from shapelet string."""
    parts = shapelet_str.split("_")
    if len(parts) < 2:
        return None, None

    variable = parts[0]
    values = parts[1].split("-")
    try:
        values = [float(v) for v in values]
        return variable, values
    except:
        return None, None


def compare_shapelets(shapelet1, shapelet2):
    """Compare two shapelets for equality."""
    if len(shapelet1) != len(shapelet2):
        return False
    return np.allclose(shapelet1, shapelet2, rtol=1e-5, atol=1e-8)


def find_acc_for_shapelet(csv_file, target_shapelet):
    """Find accuracy for specific shapelet in CSV file."""
    try:
        df = pd.read_csv(csv_file)

        if "acc" not in df.columns or "shapelet" not in df.columns:
            return None

        for idx, row in df.iterrows():
            shapelet = row["shapelet"]

            if isinstance(shapelet, str):
                try:
                    shapelet = ast.literal_eval(shapelet)
                except:
                    continue

            if isinstance(shapelet, list):
                if compare_shapelets(shapelet, target_shapelet):
                    return row["acc"]

        return None

    except Exception as e:
        return None


def extract_accuracies(prefix="Test"):
    """Extract accuracies for specified prefix (Train or Test)."""
    results = {}

    for pattern in shapelet_patterns:
        variable, target_values = parse_shapelet_string(pattern)

        if variable is None or target_values is None:
            continue

        if variable not in file_mapping:
            continue

        for csv_filename in file_mapping[variable]:
            csv_filename_with_prefix = f"{prefix}_{csv_filename}"
            csv_path = os.path.join(DATA_DIR, csv_filename_with_prefix)

            if not os.path.exists(csv_path):
                continue

            acc = find_acc_for_shapelet(csv_path, target_values)

            if acc is not None:
                results[pattern] = acc
                break

    return results


def format_shapelet_name(name):
    """Format shapelet name with full values and units."""
    units_map = {
        "bun": "mg/dL",
        "sbp": "mmHg",
        "dbp": "mmHg",
        "heart": "bpm",
        "creatinine": "mg/dL",
        "spo2": "%",
        "potassium": "mEq/L",
    }

    var_display_map = {
        "heart": "Heart Rate",
        "sbp": "SBP",
        "dbp": "DBP",
        "bun": "BUN",
        "creatinine": "Creatinine",
        "spo2": "SpO₂",
        "potassium": "Potassium",
    }

    parts = name.split("_")
    var_name = parts[0]
    values = parts[1] if len(parts) > 1 else ""

    unit = units_map.get(var_name, "")
    var_display = var_display_map.get(var_name, var_name.upper())

    return f"{var_display} ({values} {unit})"


def create_comparison_barplot(labels, train_acc, test_acc):
    """Create horizontal bar plot with training on top, test on bottom."""
    fig, ax = plt.subplots(figsize=(9, 8.2), dpi=300)

    y_pos = np.arange(len(labels))
    bar_height = 0.37

    # Blue color scheme
    bars1 = ax.barh(
        y_pos + bar_height / 2,
        train_acc,
        bar_height,
        label="Training Set",
        color="#A8C8DC",
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax.barh(
        y_pos - bar_height / 2,
        test_acc,
        bar_height,
        label="Test Set",
        color="#4393C3",
        edgecolor="black",
        linewidth=0.8,
    )

    # Add value labels
    for bars, acc_list in [(bars1, train_acc), (bars2, test_acc)]:
        for bar, val in zip(bars, acc_list):
            ax.text(
                val + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=15,
            )

    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=17)
    ax.set_xlabel("Accuracy", fontsize=15)

    # Set x-axis range
    min_acc = min(min(train_acc), min(test_acc))
    max_acc = max(max(train_acc), max(test_acc))
    ax.set_xlim(min_acc - 0.05, max_acc + 0.08)

    # Legend
    ax.legend(loc="lower right", fontsize=13.5, frameon=False, fancybox=False)

    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Baseline (gray)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    return fig


# Extract accuracies
train_results = extract_accuracies(prefix="Train")
test_results = extract_accuracies(prefix="Test")

# Create DataFrame
train_df = pd.DataFrame(
    list(train_results.items()), columns=["shapelet_pattern", "train_acc"]
)
test_df = pd.DataFrame(
    list(test_results.items()), columns=["shapelet_pattern", "test_acc"]
)

# Merge and filter
results_df = pd.merge(train_df, test_df, on="shapelet_pattern")
results_df = results_df[
    (results_df["train_acc"].notna()) & (results_df["test_acc"].notna())
]
results_df = results_df.sort_values("test_acc", ascending=True)

# Save results
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "shapelet_train_test_accuracies.csv"), index=False
)

# Generate plot
if len(results_df) > 0:
    labels = [format_shapelet_name(s) for s in results_df["shapelet_pattern"]]
    train_acc = results_df["train_acc"].tolist()
    test_acc = results_df["test_acc"].tolist()

    fig = create_comparison_barplot(labels, train_acc, test_acc)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "shapelet_accuracy_comparison.png"),
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )
    plt.close()
