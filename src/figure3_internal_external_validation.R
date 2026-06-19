#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})

parse_args <- function(args) {
  out <- list(
    data_dir = "results/model_performance",
    outdir = "results/figures/figure3",
    model = "xgboost",
    prefix = "figure3_internal_external_validation",
    dca_min = 0.05,
    dca_max = 0.65,
    width = 8.6,
    height = 12.0,
    dpi = 400
  )

  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    val <- if (i + 1 <= length(args)) args[[i + 1]] else NULL
    if (!startsWith(key, "--") || is.null(val) || startsWith(val, "--")) {
      i <- i + 1
      next
    }
    key <- sub("^--", "", key)
    if (key %in% c("data_dir", "outdir", "model", "prefix")) out[[key]] <- val
    if (key %in% c("dca_min", "dca_max", "width", "height")) out[[key]] <- as.numeric(val)
    if (key == "dpi") out[[key]] <- as.integer(val)
    i <- i + 2
  }
  out
}

version_style <- tibble(
  version = c("V1", "V2", "V3"),
  label = c("Baseline", "+ Min/Max", "+ Shapelets"),
  color = c("#5AAE61", "#D6604D", "#4393C3"),
  fill = c("#5AAE61", "#D6604D", "#4393C3"),
  linewidth = c(0.82, 0.82, 1.05)
)

theme_pub <- function() {
  theme_bw(base_size = 9.4) +
    theme(
      panel.grid.major = element_line(linewidth = 0.22, color = "#ECE7DF"),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "#B8B8B8", linewidth = 0.35),
      plot.title = element_text(face = "bold", size = 9.8, color = "#1F2937", hjust = 0),
      axis.title = element_text(size = 8.8, color = "#111827"),
      axis.text = element_text(size = 8.0, color = "#374151"),
      legend.text = element_text(size = 8.6),
      plot.margin = margin(6, 7, 6, 7)
    )
}

add_tag <- function(p, tag) {
  p +
    labs(tag = tag) +
    theme(
      plot.tag = element_text(face = "bold", size = 10.8, color = "#111827"),
      plot.tag.position = c(0.012, 1.015)
    )
}

metric_text <- function(metrics, dataset_key, metric_key) {
  d <- metrics %>%
    filter(.data$dataset == .env$dataset_key, .data$metric == .env$metric_key) %>%
    left_join(version_style, by = "version") %>%
    arrange(match(.data$version, version_style$version))
  if (nrow(d) == 0) return("")
  paste(
    sprintf("%s %.3f [%.3f-%.3f]", d$label, d$value, d$ci_low, d$ci_high),
    collapse = "\n"
  )
}

scalar_metric_text <- function(metrics, dataset_key, metric_key) {
  d <- metrics %>%
    filter(.data$dataset == .env$dataset_key, .data$metric == .env$metric_key) %>%
    left_join(version_style, by = "version") %>%
    arrange(match(.data$version, version_style$version))
  if (nrow(d) == 0) return("")
  paste(sprintf("%s %.3f", d$label, d$value), collapse = "\n")
}

plot_roc <- function(curves, metrics, dataset_key, title_text, show_legend = FALSE) {
  d <- curves %>%
    filter(.data$dataset == .env$dataset_key, .data$curve == "roc") %>%
    left_join(version_style, by = "version")
  p <- ggplot(d, aes(x = x, y = y, color = label, fill = label, linewidth = label)) +
    geom_ribbon(aes(ymin = y_ci_low, ymax = y_ci_high), alpha = 0.13, color = NA, show.legend = FALSE) +
    geom_abline(intercept = 0, slope = 1, color = "#9CA3AF", linetype = "dotted", linewidth = 0.55) +
    geom_line(alpha = 0.98) +
    scale_color_manual(values = setNames(version_style$color, version_style$label), guide = "none") +
    scale_fill_manual(values = setNames(version_style$fill, version_style$label), guide = "none") +
    scale_linewidth_manual(values = setNames(version_style$linewidth, version_style$label), guide = "none") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = title_text, x = "1 - Specificity", y = "Sensitivity", color = NULL, fill = NULL) +
    theme_pub() +
    theme(legend.position = "none")
}

plot_prc <- function(curves, metrics, dataset_key, title_text) {
  d <- curves %>%
    filter(.data$dataset == .env$dataset_key, .data$curve == "prc") %>%
    left_join(version_style, by = "version")
  baseline <- ifelse(dataset_key == "internal", 168 / 1354, 644 / 4228)
  ggplot(d, aes(x = x, y = y, color = label, fill = label, linewidth = label)) +
    geom_ribbon(aes(ymin = y_ci_low, ymax = y_ci_high), alpha = 0.13, color = NA, show.legend = FALSE) +
    geom_hline(yintercept = baseline, color = "#9CA3AF", linetype = "dotted", linewidth = 0.55) +
    geom_line(alpha = 0.98) +
    scale_color_manual(values = setNames(version_style$color, version_style$label), guide = "none") +
    scale_fill_manual(values = setNames(version_style$fill, version_style$label), guide = "none") +
    scale_linewidth_manual(values = setNames(version_style$linewidth, version_style$label), guide = "none") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(title = title_text, x = "Recall", y = "Precision", color = NULL, fill = NULL) +
    theme_pub() +
    theme(legend.position = "none")
}

plot_calibration <- function(calibration, metrics, dataset_key, title_text) {
  d <- calibration %>%
    filter(.data$dataset == .env$dataset_key) %>%
    left_join(version_style, by = "version")
  axis_breaks <- seq(0, 1, by = 0.25)
  axis_labels <- sprintf("%.2f", axis_breaks)
  ggplot(d, aes(x = predicted_mean, y = observed_rate, color = label)) +
    geom_abline(intercept = 0, slope = 1, color = "#9CA3AF", linetype = "dotted", linewidth = 0.55) +
    geom_errorbar(aes(ymin = observed_ci_low, ymax = observed_ci_high), width = 0, alpha = 0.48, linewidth = 0.45) +
    geom_point(size = 1.25, alpha = 0.95) +
    geom_line(linewidth = 0.68, alpha = 0.82) +
    scale_color_manual(values = setNames(version_style$color, version_style$label), guide = "none") +
    scale_x_continuous(limits = c(0, 1), breaks = axis_breaks, labels = axis_labels, expand = expansion(mult = 0, add = 0)) +
    scale_y_continuous(limits = c(0, 1), breaks = axis_breaks, labels = axis_labels, expand = expansion(mult = 0, add = 0)) +
    coord_fixed(ratio = 1, clip = "on") +
    labs(title = title_text, x = "Predicted risk", y = "Observed AKI rate", color = NULL) +
    theme_pub() +
    theme(legend.position = "none") +
    annotate("text", x = 0.55, y = 0.18, label = scalar_metric_text(metrics, dataset_key, "Brier"), hjust = 0, size = 2.35, lineheight = 1.03, color = "#1F2937")
}

plot_dca <- function(dca, dataset_key, title_text, dca_min, dca_max) {
  d <- dca %>%
    filter(.data$dataset == .env$dataset_key, threshold >= .env$dca_min, threshold <= .env$dca_max) %>%
    left_join(version_style, by = "version")
  ref <- d %>% filter(version == "V1")
  y_max <- max(d$net_benefit, na.rm = TRUE)
  ylim <- c(-0.005, max(0.045, y_max + 0.01))

  ggplot(d, aes(x = threshold, y = net_benefit, color = label, linewidth = label)) +
    geom_line(alpha = 0.98, lineend = "round") +
    geom_line(data = ref, aes(x = threshold, y = treat_all), inherit.aes = FALSE, color = "#6B7280", linetype = "dashed", linewidth = 0.58) +
    geom_hline(yintercept = 0, color = "#111827", linetype = "dotted", linewidth = 0.50) +
    scale_color_manual(values = setNames(version_style$color, version_style$label), guide = "none") +
    scale_linewidth_manual(values = setNames(version_style$linewidth, version_style$label), guide = "none") +
    coord_cartesian(xlim = c(dca_min, dca_max), ylim = ylim, expand = FALSE) +
    labs(title = title_text, x = "Threshold probability", y = "Net benefit", color = NULL) +
    theme_pub() +
    theme(legend.position = "none")
}

legend_panel <- function() {
  d <- version_style %>%
    mutate(
      x0 = c(0.25, 0.48, 0.70),
      x1 = x0 + 0.040,
      xt = x1 + 0.012,
      y = 0.5
    )
  ggplot(d) +
    geom_segment(aes(x = x0, xend = x1, y = y, yend = y, color = label), linewidth = 0.68, lineend = "round") +
    geom_text(aes(x = xt, y = y, label = label), hjust = 0, size = 2.45, color = "#111827") +
    scale_color_manual(values = setNames(version_style$color, version_style$label), guide = "none") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
    theme_void() +
    theme(plot.margin = margin(0, 4, 0, 4))
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- normalizePath(".")
  data_dir <- file.path(root, args$data_dir)
  outdir <- file.path(root, args$outdir)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  curves <- read.csv(file.path(data_dir, sprintf("%s_roc_prc_curves_bootstrap.csv", args$model)), stringsAsFactors = FALSE)
  calibration <- read.csv(file.path(data_dir, sprintf("%s_calibration_bins_internal_oof_platt.csv", args$model)), stringsAsFactors = FALSE)
  dca <- read.csv(file.path(data_dir, sprintf("%s_dca_curves_internal_oof_platt.csv", args$model)), stringsAsFactors = FALSE)
  metrics <- read.csv(file.path(data_dir, sprintf("%s_figure3_metrics.csv", args$model)), stringsAsFactors = FALSE)

  p_a <- add_tag(plot_roc(curves, metrics, "internal", "Internal validation: ROC", show_legend = TRUE), "A")
  p_b <- add_tag(plot_roc(curves, metrics, "external", "External validation: ROC", show_legend = FALSE), "B")
  p_c <- add_tag(plot_prc(curves, metrics, "internal", "Internal validation: PRC"), "C")
  p_d <- add_tag(plot_prc(curves, metrics, "external", "External validation: PRC"), "D")
  p_e <- add_tag(plot_calibration(calibration, metrics, "internal", "Internal validation: calibration"), "E")
  p_f <- add_tag(plot_calibration(calibration, metrics, "external", "External validation: calibration"), "F")
  p_g <- add_tag(plot_dca(dca, "internal", "Internal validation: DCA", args$dca_min, args$dca_max), "G")
  p_h <- add_tag(plot_dca(dca, "external", "External validation: DCA", args$dca_min, args$dca_max), "H")

  body <- wrap_plots(
    p_a, p_b, p_c, p_d, p_e, p_f, p_g, p_h,
    ncol = 2,
    byrow = TRUE
  )

  fig <- wrap_plots(body, legend_panel(), ncol = 1, heights = c(1, 0.055))

  png_path <- file.path(outdir, paste0(args$prefix, ".png"))
  pdf_path <- file.path(outdir, paste0(args$prefix, ".pdf"))
  ggsave(png_path, fig, width = args$width, height = args$height, dpi = args$dpi, bg = "white")
  ggsave(pdf_path, fig, width = args$width, height = args$height, bg = "white")

  note <- c(
    "Figure 3. Internal and external validation of the XGBoost AKI prediction model.",
    "Panels A-B show receiver operating characteristic curves; panels C-D show precision-recall curves; panels E-F show calibration curves; panels G-H show decision curve analysis.",
    "Curves compare Baseline, + Min/Max, and + Shapelets feature sets.",
    "ROC and PRC panels use raw predicted probabilities with pointwise bootstrap 95% confidence bands.",
    "Calibration and decision-curve panels use Platt-calibrated probabilities fitted only on MIMIC-IV training-set out-of-fold predictions and then applied unchanged to the MIMIC-IV holdout and eICU external-validation cohorts.",
    sprintf("The decision-curve threshold range is %.2f-%.2f.", args$dca_min, args$dca_max)
  )
  writeLines(note, file.path(outdir, paste0(args$prefix, "_README.md")))

  cat(png_path, "\n")
  cat(pdf_path, "\n")
}

main()
