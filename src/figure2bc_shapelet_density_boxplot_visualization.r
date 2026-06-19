# Density and Boxplot Visualizations for Shapelet Analysis
# ================================================================

library(ggplot2)
library(dplyr)
library(ggsignif)

args <- commandArgs(trailingOnly = TRUE)
full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- full_args[grepl("^--file=", full_args)]
default_root <- if (length(file_arg) >= 1) {
    normalizePath(file.path(dirname(sub("^--file=", "", file_arg[[1]])), ".."))
} else {
    normalizePath(getwd())
}
root_dir <- if (length(args) >= 1) normalizePath(args[[1]]) else default_root
setwd(root_dir)

# Load data
df <- read.csv("data/mimiciv/temp/for_R.csv", check.names = FALSE, stringsAsFactors = FALSE)

# Function: Create density and boxplot comparisons
plot_comparison <- function(data, col_name, x_lim = NULL, pct_cut = NULL) {
    # Prepare data
    plot_data <- data.frame(
        value = data[[col_name]],
        group = factor(data$label, levels = c(0, 1), labels = c("Non-AKI", "AKI"))
    )
    plot_data <- plot_data[!is.na(plot_data$value), ]

    # Calculate p-value
    p_val <- wilcox.test(value ~ group, data = plot_data)$p.value

    # Determine significance label
    if (p_val < 0.001) {
        p_label <- "***"
    } else if (p_val < 0.01) {
        p_label <- "**"
    } else if (p_val < 0.05) {
        p_label <- "*"
    } else {
        p_label <- "ns"
    }

    v_just <- ifelse(p_label == "ns", 0, 0.5)

    # Apply percentile truncation if specified
    plot_data_density <- plot_data
    if (!is.null(pct_cut)) {
        q_low <- quantile(plot_data$value, pct_cut, na.rm = TRUE)
        q_high <- quantile(plot_data$value, 1 - pct_cut, na.rm = TRUE)
        plot_data_density <- plot_data[plot_data$value >= q_low & plot_data$value <= q_high, ]
    }

    # ✅ 扩展的变量名格式化
    if (grepl("spo2_100\\.0-96\\.0-98\\.0-97\\.0-94\\.0", col_name)) {
        clean_name <- "SpO₂ (100-96-98-97-94 %)"
    } else if (grepl("spo2_97\\.0-98\\.0-95\\.0-95\\.0-97\\.0-97\\.0", col_name)) {
        clean_name <- "SpO₂ (97-98-95-95-97-97 %)" # ✅ 使用实际的模式
    } else if (grepl("heart_88\\.0-98\\.0-82\\.0-86\\.3", col_name)) {
        clean_name <- "Heart Rate (88-98-82-86.3 bpm)" # ✅ 新增
    } else if (grepl("heart_8", col_name)) {
        clean_name <- "Heart Rate (84-87-78-73-77 bpm)"
    } else if (col_name == "o2_saturation_last") {
        clean_name <- "SpO₂ Last Value (%)"
    } else if (col_name == "heart_rate_last") {
        clean_name <- "Heart Rate Last Value (bpm)"
    } else {
        clean_name <- col_name
    }

    # Density plot
    p1 <- ggplot(plot_data_density, aes(x = value, fill = group)) +
        geom_density(alpha = 0.5, color = "black", linewidth = 0.5) +
        scale_fill_manual(values = c("Non-AKI" = "#4393C3", "AKI" = "#D6604D")) +
        theme_classic() +
        labs(x = clean_name, y = "Density") +
        theme(
            legend.position = "none",
            axis.title = element_text(size = 11),
            axis.text = element_text(size = 10)
        )

    if (!is.null(x_lim)) {
        p1 <- p1 + xlim(x_lim)
    }

    # Boxplot
    y_max <- max(plot_data_density$value, na.rm = TRUE)
    y_min <- min(plot_data_density$value, na.rm = TRUE)
    y_range <- y_max - y_min

    p2 <- ggplot(plot_data_density, aes(x = group, y = value, fill = group)) +
        geom_boxplot(alpha = 0.5, outlier.shape = 21, outlier.size = 1.5) +
        geom_signif(
            comparisons = list(c("Non-AKI", "AKI")),
            annotations = p_label,
            map_signif_level = FALSE,
            textsize = 6,
            vjust = v_just,
            y_position = y_max + y_range * 0.08
        ) +
        scale_fill_manual(values = c("#4393C3", "#D6604D")) +
        scale_y_continuous(limits = c(y_min - y_range * 0.05, y_max + y_range * 0.15)) +
        theme_classic() +
        theme(
            legend.position = "none",
            axis.title.x = element_blank(),
            axis.title.y = element_text(size = 11),
            axis.text = element_text(size = 10)
        ) +
        labs(y = clean_name)

    return(list(density = p1, boxplot = p2, p_value = p_val))
}

# ================================================================
# Generate plots
# ================================================================

# SpO₂ Last Value
p1 <- plot_comparison(df, "o2_saturation_last")

# SpO₂ Shapelet Pattern 1 (Continuous Decline: 100-96-98-97-94)
p2 <- plot_comparison(df, "spo2_100.0-96.0-98.0-97.0-94.0", x_lim = c(0, 80), pct_cut = 0.025)

# ✅ SpO₂ Shapelet Pattern 2 (Fluctuation: 97-98-95-95-97-97)
p5 <- plot_comparison(df, "spo2_97.0-98.0-95.0-95.0-97.0-97.0", x_lim = c(0, 80), pct_cut = 0.025)

# Heart Rate Last Value
p3 <- plot_comparison(df, "heart_rate_last", x_lim = c(0, 400))

# # Heart Rate Shapelet (84-87-78-73-77)
# p4 <- plot_comparison(df, "heart_84.0-87.0-78.0-73.0-77.0", pct_cut = 0.025)

# Heart Rate Shapelet (84-87-78-73-77)"heart_88-98-82-86.3",

p4 <- plot_comparison(df, "heart_88.0-98.0-82.0-86.3", pct_cut = 0.025)

# ================================================================
# Save plots
# ================================================================

# SpO₂ Last Value
ggsave("results/shapelet_density_and_box/density_o2_last.png", p1$density,
    width = 4.5, height = 3.2, dpi = 500
)
ggsave("results/shapelet_density_and_box/box_o2_last.png", p1$boxplot,
    width = 1.9, height = 2.8, dpi = 500
)

# SpO₂ Shapelet Pattern 1 (Continuous Decline)
ggsave("results/shapelet_density_and_box/density_spo2_shapelet_decline.png", p2$density,
    width = 4.5, height = 3.2, dpi = 500
)
ggsave("results/shapelet_density_and_box/box_spo2_shapelet_decline.png", p2$boxplot,
    width = 1.9, height = 2.8, dpi = 500
)

# ✅ SpO₂ Shapelet Pattern 2 (Fluctuation)
ggsave("results/shapelet_density_and_box/density_spo2_shapelet_fluctuation.png", p5$density,
    width = 4.5, height = 3.2, dpi = 500
)
ggsave("results/shapelet_density_and_box/box_spo2_shapelet_fluctuation.png", p5$boxplot,
    width = 1.9, height = 2.8, dpi = 500
)

# Heart Rate Last Value
ggsave("results/shapelet_density_and_box/density_hr_last.png", p3$density,
    width = 4.5, height = 3.2, dpi = 500
)
ggsave("results/shapelet_density_and_box/box_hr_last.png", p3$boxplot,
    width = 1.9, height = 2.8, dpi = 500
)

# Heart Rate Shapelet
ggsave("results/shapelet_density_and_box/density_hr_shapelet.png", p4$density,
    width = 4.5, height = 3.2, dpi = 500
)
ggsave("results/shapelet_density_and_box/box_hr_shapelet.png", p4$boxplot,
    width = 1.9, height = 2.8, dpi = 500
)

# ================================================================
# Print p-values for reporting
# ================================================================
cat("\n=== P-values Summary ===\n")
cat(sprintf("SpO₂ Last Value: p = %.4f\n", p1$p_value))
cat(sprintf("SpO₂ Shapelet Pattern 1 (Decline): p = %.4f\n", p2$p_value))
cat(sprintf("SpO₂ Shapelet Pattern 2 (Fluctuation): p = %.4f\n", p5$p_value))
cat(sprintf("Heart Rate Last Value: p = %.4f\n", p3$p_value))
cat(sprintf("Heart Rate Shapelet: p = %.4f\n", p4$p_value))
cat("========================\n")
