# Decision Tree and Confusion Matrix Visualizations for Shapelet Classification
# ================================================================

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import chi2_contingency

# Configuration
base_dir = os.getcwd()
OUTPUT_DIR = f"{base_dir}/../results/shapelet_decision_trees"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 0.5


def create_decision_tree_flow(ax, data_dict, shapelet_name):
    """Create horizontal decision tree flow visualization."""

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    threshold = data_dict["threshold"]
    total_n = data_dict["total_n"]
    cm = data_dict["cm"]
    mode = data_dict["mode"]

    # Color scheme
    color_non_aki = "#E8F4F8"
    color_aki = "#FFF4E8"

    # Title above root node
    ax.text(2.2, 3.15, shapelet_name, ha="center", va="bottom", fontsize=10)

    # Root node
    root_box = FancyBboxPatch(
        (1.6, 2.1),
        1.2,
        0.8,
        boxstyle="round,pad=0.03",
        edgecolor="black",
        facecolor="white",
        linewidth=1.2,
    )
    ax.add_patch(root_box)
    ax.text(
        2.2, 2.5, f"All Patients\n(n={total_n})", ha="center", va="center", fontsize=11
    )

    # Decision criterion
    ax.text(
        3.45,
        2.5,
        f'Distance {"<" if mode == "greater" else ">"} {threshold}?',
        ha="center",
        va="center",
        fontsize=10,
    )

    # Branch arrows
    arrow_yes = FancyArrowPatch(
        (2.8, 2.75),
        (4.3, 3.4),
        connectionstyle="arc3,rad=0.1",
        arrowstyle="->",
        mutation_scale=20,
        linewidth=1.2,
        color="black",
    )
    ax.add_patch(arrow_yes)
    ax.text(3.5, 3.15, "Yes", ha="center", fontsize=10)

    arrow_no = FancyArrowPatch(
        (2.8, 2.25),
        (4.3, 1.6),
        connectionstyle="arc3,rad=-0.1",
        arrowstyle="->",
        mutation_scale=20,
        linewidth=1.2,
        color="black",
    )
    ax.add_patch(arrow_no)
    ax.text(3.5, 1.85, "No", ha="center", fontsize=10)

    # Leaf nodes (specific to each shapelet type)
    if "SpO" in shapelet_name:
        # Upper node: Non-AKI
        pred0_box = FancyBboxPatch(
            (4.3, 3.15),
            1.3,
            0.7,
            boxstyle="round,pad=0.04",
            facecolor=color_non_aki,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(pred0_box)
        ax.text(4.95, 3.65, "Non-AKI", ha="center", va="center", fontsize=11)
        ax.text(4.95, 3.45, "[677, 530]", ha="center", va="center", fontsize=10)  # 修改
        ax.text(
            4.95, 3.25, "samples=72%", ha="center", va="center", fontsize=10
        )  # 1207/1678

        # Lower node: AKI
        pred1_box = FancyBboxPatch(
            (4.3, 1.35),
            1.3,
            0.7,
            boxstyle="round,pad=0.04",
            facecolor=color_aki,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(pred1_box)
        ax.text(4.95, 1.85, "AKI", ha="center", va="center", fontsize=11)
        ax.text(4.95, 1.65, "[162, 309]", ha="center", va="center", fontsize=10)  # 修改
        ax.text(
            4.95, 1.45, "samples=28%", ha="center", va="center", fontsize=10
        )  # 471/1678

    else:
        # Upper node: Non-AKI
        pred0_box = FancyBboxPatch(
            (4.3, 3.15),
            1.3,
            0.7,
            boxstyle="round,pad=0.04",
            facecolor=color_non_aki,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(pred0_box)
        ax.text(4.95, 3.65, "Non-AKI", ha="center", va="center", fontsize=11)
        ax.text(4.95, 3.45, "[315, 170]", ha="center", va="center", fontsize=10)  # 修改
        ax.text(
            4.95, 3.25, "samples=29%", ha="center", va="center", fontsize=10
        )  # 485/1678

        # Lower node: AKI
        pred1_box = FancyBboxPatch(
            (4.3, 1.35),
            1.3,
            0.7,
            boxstyle="round,pad=0.04",
            facecolor=color_aki,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(pred1_box)
        ax.text(4.95, 1.85, "AKI", ha="center", va="center", fontsize=11)
        ax.text(4.95, 1.65, "[524, 669]", ha="center", va="center", fontsize=10)  # 修改
        ax.text(
            4.95, 1.45, "samples=71%", ha="center", va="center", fontsize=10
        )  # 1193/1678


def create_confusion_matrix_table(ax_table, data_dict):
    """Create confusion matrix table with chi-square test."""

    cm = data_dict["cm"]

    ax_table.axis("tight")
    ax_table.axis("off")

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(cm)
    p_text = "P < 0.001" if p_value < 0.001 else f"P = {p_value:.3f}"

    # Table data
    table_data = [
        ["", "Predicted\nNon-AKI", "Predicted\nAKI"],
        ["Actual\nNon-AKI", str(cm[0, 0]), str(cm[0, 1])],
        ["Actual\nAKI", str(cm[1, 0]), str(cm[1, 1])],
        ["Chi-square\ntest", p_text, ""],
    ]

    # Create table
    table = ax_table.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.33, 0.33, 0.33],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Header styling
    for col in range(3):
        table[(0, col)].set_facecolor("white")

    # Row label styling
    table[(1, 0)].set_facecolor("white")
    table[(2, 0)].set_facecolor("white")

    # Confusion matrix cell coloring
    max_val = cm.max()
    min_val = cm.min()

    for i in range(1, 3):
        for j in range(1, 3):
            val = cm[i - 1, j - 1]
            intensity = (
                (val - min_val) / (max_val - min_val) if max_val > min_val else 0
            )
            color_intensity = 0.98 - intensity * 0.08
            table[(i, j)].set_facecolor((color_intensity, 1, color_intensity))

    # Chi-square row styling
    table[(3, 0)].set_facecolor("white")
    table[(3, 1)].set_facecolor("white")
    table[(3, 2)].set_facecolor("white")
    table[(3, 0)].set_text_props(style="italic", fontsize=11)
    table[(3, 1)].set_text_props(fontsize=11)


# # Shapelet classification data
# hr_data = {
#     "threshold": 65.29,
#     "total_n": 1343,
#     "cm": np.array([[219, 452], [107, 564]]),
#     "accuracy": 0.58345753,
#     "mode": "greater",
# }

# spo2_data = {
#     "threshold": 16,
#     "total_n": 1343,
#     "cm": np.array([[556, 115], [434, 238]]),
#     "accuracy": 0.591654247,
#     "mode": "greater",
# }
# Shapelet classification data
hr_data = {
    "threshold": 81.49,
    "total_n": 1678,  # 315+524+170+669
    "cm": np.array([[315, 524], [170, 669]]),
    "accuracy": (315 + 669) / 1678,  # 0.587
    "mode": "greater",
}


spo2_data = {
    "threshold": 16,
    "total_n": 1678,  # 677+162+530+309
    "cm": np.array([[677, 162], [530, 309]]),
    "accuracy": (677 + 309) / 1678,  # 0.588
    "mode": "greater",
}
# Generate visualizations
# Heart Rate decision tree
fig1, ax1 = plt.subplots(figsize=(7, 3.8), dpi=600)
create_decision_tree_flow(ax1, hr_data, "Heart Rate Shapelet\n(84-87-78-73-77 bpm)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "hr_decision_tree.png"),
    bbox_inches="tight",
    dpi=600,
    facecolor="white",
)
plt.close()

# Heart Rate confusion matrix
fig2, ax2 = plt.subplots(figsize=(3.5, 3), dpi=300)
create_confusion_matrix_table(ax2, hr_data)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "hr_performance_table.png"),
    bbox_inches="tight",
    dpi=600,
    facecolor="white",
)
plt.close()

# SpO2 decision tree
fig3, ax3 = plt.subplots(figsize=(7, 3.8), dpi=600)
create_decision_tree_flow(ax3, spo2_data, "SpO2 Shapelet\n(100-96-98-97-94 %)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "spo2_decision_tree.png"),
    bbox_inches="tight",
    dpi=600,
    facecolor="white",
)
plt.close()

# SpO2 confusion matrix
fig4, ax4 = plt.subplots(figsize=(3.5, 3), dpi=600)
create_confusion_matrix_table(ax4, spo2_data)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "spo2_performance_table.png"),
    bbox_inches="tight",
    dpi=600,
    facecolor="white",
)
plt.close()
