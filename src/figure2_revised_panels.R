#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(patchwork)
  library(scales)
})

parse_args <- function(args) {
  out <- list(
    project_root = NULL,
    results_root = NULL,
    output_dir = NULL,
    example1_shapelet = "spo2_100-96-98-97-94",
    example2_shapelet = "spo2_98-96-96-98-95",
    context_window = 4,
    dpi = 450
  )

  if (length(args) == 0) {
    return(out)
  }

  for (arg in args) {
    if (!startsWith(arg, "--")) {
      next
    }
    parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    key <- gsub("-", "_", parts[[1]])
    value <- if (length(parts) >= 2) parts[[2]] else TRUE

    if (key %in% c("project_root", "results_root", "output_dir", "example1_shapelet", "example2_shapelet")) {
      out[[key]] <- value
    } else if (key %in% c("context_window", "dpi")) {
      out[[key]] <- as.integer(value)
    }
  }

  out
}

get_repo_root <- function() {
  full_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- full_args[grepl("^--file=", full_args)]
  if (length(file_arg) >= 1) {
    script_path <- sub("^--file=", "", file_arg[[1]])
    return(normalizePath(file.path(dirname(script_path), "..")))
  }
  normalizePath(file.path(getwd(), ".."))
}

resolve_existing_root <- function(candidates, required_rel) {
  for (candidate in candidates) {
    if (is.null(candidate)) {
      next
    }
    candidate <- path.expand(candidate)
    required_path <- file.path(candidate, required_rel)
    if (file.exists(required_path)) {
      return(normalizePath(candidate))
    }
  }

  stop(
    paste0(
      "Could not locate a root containing '", required_rel, "'. Checked:\n",
      paste(vapply(candidates, function(x) if (is.null(x)) "<NULL>" else x, character(1)), collapse = "\n")
    )
  )
}

resolve_paths <- function(cli_args) {
  repo_root <- get_repo_root()
  anchor_root <- if (basename(dirname(repo_root)) == "release_repo") dirname(dirname(repo_root)) else dirname(repo_root)

  project_candidates <- list(
    cli_args$project_root,
    repo_root,
    file.path(anchor_root, "ehr-tps"),
    file.path(anchor_root, "EHR-TPS")
  )
  results_candidates <- list(
    cli_args$results_root,
    file.path(repo_root, "results"),
    file.path(anchor_root, "results"),
    file.path(anchor_root, "ehr-tps", "results"),
    file.path(anchor_root, "EHR-TPS", "results")
  )

  project_root <- resolve_existing_root(project_candidates, file.path("data", "mimiciv", "temp", "for_R.csv"))
  results_root <- resolve_existing_root(results_candidates, "csv_results")

  output_dir <- if (!is.null(cli_args$output_dir)) {
    normalizePath(path.expand(cli_args$output_dir), mustWork = FALSE)
  } else {
    normalizePath(file.path(repo_root, "results", "figures_internal", "figure2_revised_panels"), mustWork = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  list(
    repo_root = repo_root,
    project_root = project_root,
    results_root = results_root,
    output_dir = output_dir
  )
}

min_subsequence_distance <- function(sequence, shapelet) {
  best_distance <- Inf
  max_start <- length(sequence) - length(shapelet) + 1L
  for (start in seq_len(max_start)) {
    subseq <- sequence[start:(start + length(shapelet) - 1L)]
    distance <- sqrt(sum((subseq - shapelet)^2))
    if (distance < best_distance) {
      best_distance <- distance
    }
  }
  best_distance
}

COLORS <- list(
  non_aki = "#5B9BD5",
  aki = "#D97767",
  shapelet = "#2A9D8F",
  matched = "#7B8794",
  train = "#B4D1E9",
  test = "#F5C4BC",
  distance_bg = "#F6FBF8",
  grid = "#E3E9EE",
  chance = "#8A8F98",
  text = "#2F3640"
)

CURATED_EXAMPLES <- list(
  "spo2_98-96-96-98-95" = list(split = "train", row = 995)
)

PANEL_LAYOUT <- list(
  # top_row_widths = c(1.44, 0.56),
  top_row_widths = c(1.32, 0.68),
  b_internal_widths = c(1, 1, 1),
  # main_row_heights = c(1.12, 0.88),
  main_row_heights = c(1.25, 0.75),
  standalone_a_width = 4.8,
  standalone_a_height = 6.8,
  standalone_c_width = 7.2,
  standalone_c_height = 6.8,
  standalone_b_width = 12.0,
  standalone_b_height = 4.9
)

METADATA <- list(
  heart = list(
    display_name = "Heart Rate",
    unit = "bpm",
    ts_filename = "Train_heart.csv",
    last_col = "heart_rate_last"
  ),
  spo2 = list(
    display_name = "SpO2",
    unit = "%",
    ts_filename = "Train_spo2.csv",
    last_col = "o2_saturation_last"
  )
)

SUMMARY_FILE_MAPPING <- list(
  bun = c("bun_2_4_metrics.csv", "bun_5_6_metrics.csv"),
  sbp = c("sbp_2_6_metrics.csv"),
  heart = c("heart_2_6_metrics.csv"),
  creatinine = c("creatinine_2_4_metrics.csv", "creatinine_5_6_metrics.csv"),
  spo2 = c("spo2_2_6_metrics.csv"),
  dbp = c("dbp_2_6_metrics.csv"),
  potassium = c("potassium_2_4_metrics.csv", "potassium_5_6_metrics.csv")
)

SUMMARY_PATTERNS <- c(
  "bun_17-17-9",
  "sbp_107-125-131-104-134-133",
  "heart_88-98-82-86.3",
  "heart_84-87-78-73-77",
  "creatinine_0.8-0.8-0.9-0.7-0.7-0.7",
  "spo2_100-96-98-97-94",
  "spo2_98-96-96-98-95",
  "dbp_68-34-60.5-61",
  "dbp_42-69.7-59-53-82-60",
  "potassium_4.2-3.4-3.4-3.4-3.4-3.4"
)

format_value <- function(x) {
  if (abs(x - round(x)) < 1e-8) {
    as.character(as.integer(round(x)))
  } else {
    sub("\\.?0+$", "", sprintf("%.1f", x))
  }
}

format_pattern <- function(values) {
  paste(vapply(values, format_value, character(1)), collapse = "-")
}

format_shapelet_subtitle <- function(config) {
  paste0(config$display_name, " (", format_pattern(config$values), ")")
}

format_last_subtitle <- function(config) {
  paste0(config$display_name, " (latest measurement)")
}

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs)) rhs else lhs
}

parse_shapelet_string <- function(shapelet_str) {
  parts <- strsplit(shapelet_str, "_", fixed = TRUE)[[1]]
  values <- as.numeric(strsplit(parts[[2]], "-", fixed = TRUE)[[1]])
  list(feature_key = parts[[1]], values = values)
}

compare_shapelets <- function(x, y, tol = 1e-8) {
  length(x) == length(y) && all(abs(x - y) < tol)
}

theme_paper <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      text = element_text(color = COLORS$text),
      axis.title = element_text(face = "plain"),
      plot.title = element_text(face = "bold", hjust = 0, size = base_size + 0.5),
      plot.subtitle = element_text(hjust = 0, size = base_size - 1.8, color = COLORS$text),
      legend.position = "top",
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 1),
      panel.grid.major.y = element_line(color = COLORS$grid, linewidth = 0.35),
      panel.grid.minor = element_blank(),
      plot.margin = margin(8, 10, 8, 8)
    )
}

save_plot_pair <- function(plot_obj, stem, output_dir, width, height, dpi = 450) {
  png_path <- file.path(output_dir, paste0(stem, ".png"))
  pdf_path <- file.path(output_dir, paste0(stem, ".pdf"))
  ggsave(png_path, plot_obj, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(pdf_path, plot_obj, width = width, height = height, device = "pdf", bg = "white")
  cat("Saved:", png_path, "\n")
  cat("Saved:", pdf_path, "\n")
}

save_combined_plot <- function(plot_obj, stem, output_dir, width_mm, height_mm, dpi = 450) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  save_plot_pair(plot_obj, stem, output_dir, width = width_in, height = height_in, dpi = dpi)
}

load_for_r_data <- function(project_root) {
  read_csv(
    file.path(project_root, "data", "mimiciv", "temp", "for_R.csv"),
    show_col_types = FALSE,
    progress = FALSE
  )
}

load_timeseries <- function(project_root, ts_filename) {
  ts_path <- file.path(project_root, "data", "mimiciv", "ts_vital", ts_filename)
  df <- read_csv(ts_path, col_names = FALSE, show_col_types = FALSE, progress = FALSE)
  labels <- as.integer(df[[1]])
  values <- as.matrix(df[, -1, drop = FALSE])
  storage.mode(values) <- "double"
  list(values = values, labels = labels)
}

load_single_sequence <- function(project_root, ts_filename, split, row_1based) {
  file_name <- if (split == "test") sub("^Train_", "Test_", ts_filename) else ts_filename
  ts_obj <- load_timeseries(project_root, file_name)
  sequence <- as.numeric(ts_obj$values[row_1based, ])
  sequence <- sequence[!is.na(sequence)]
  list(sequence = sequence, label = ts_obj$labels[[row_1based]])
}

compute_best_match <- function(sequence, shapelet) {
  best_distance <- Inf
  best_start <- 1L
  max_start <- length(sequence) - length(shapelet) + 1L

  for (start in seq_len(max_start)) {
    subseq <- sequence[start:(start + length(shapelet) - 1L)]
    distance <- sqrt(sum((subseq - shapelet)^2))
    if (distance < best_distance) {
      best_distance <- distance
      best_start <- start
    }
  }

  list(distance = best_distance, start = best_start)
}

find_representative_match <- function(values, labels, shapelet) {
  matches <- vector("list", nrow(values))
  keep_idx <- 0L

  for (row_idx in seq_len(nrow(values))) {
    sequence <- as.numeric(values[row_idx, ])
    sequence <- sequence[!is.na(sequence)]
    if (length(sequence) < length(shapelet)) {
      next
    }

    match <- compute_best_match(sequence, shapelet)
    keep_idx <- keep_idx + 1L
    matches[[keep_idx]] <- list(
      row_idx = row_idx,
      label = labels[[row_idx]],
      distance = match$distance,
      start = match$start,
      sequence = sequence
    )
  }

  matches <- matches[seq_len(keep_idx)]
  if (length(matches) == 0) {
    stop("No representative match could be found.")
  }

  distances <- vapply(matches, function(x) x$distance, numeric(1))
  matches <- matches[order(distances)]
  matches[[1]]
}

match_distance_column <- function(columns, feature_key, values) {
  prefix <- paste0(feature_key, "_")
  for (column in columns) {
    if (!startsWith(column, prefix)) {
      next
    }
    parsed <- tryCatch(suppressWarnings(parse_shapelet_string(column)), error = function(e) NULL)
    if (is.null(parsed)) {
      next
    }
    if (any(is.na(parsed$values))) {
      next
    }
    if (compare_shapelets(parsed$values, values)) {
      return(column)
    }
  }
  stop("Could not find a shapelet-distance column for ", feature_key, "_", format_pattern(values))
}

build_rep_config <- function(shapelet_str, available_columns) {
  parsed <- parse_shapelet_string(shapelet_str)
  metadata <- METADATA[[parsed$feature_key]]
  if (is.null(metadata)) {
    stop("Unsupported representative feature: ", parsed$feature_key)
  }

  matched_col <- tryCatch(
    match_distance_column(available_columns, parsed$feature_key, parsed$values),
    error = function(e) NULL
  )

  list(
    feature_key = parsed$feature_key,
    values = parsed$values,
    display_name = metadata$display_name,
    unit = metadata$unit,
    ts_filename = metadata$ts_filename,
    last_col = metadata$last_col,
    distance_col = matched_col
  )
}

build_distance_vector <- function(project_root, config) {
  shapelet <- config$values

  compute_from_matrix <- function(ts_path) {
    ts_obj <- load_timeseries(project_root, basename(ts_path))
    apply(
      ts_obj$values,
      1,
      function(row) {
        sequence <- as.numeric(row)
        sequence <- sequence[!is.na(sequence)]
        if (length(sequence) < length(shapelet)) {
          return(NA_real_)
        }
        min_subsequence_distance(sequence, shapelet)
      }
    )
  }

  train_path <- file.path(project_root, "data", "mimiciv", "ts_vital", config$ts_filename)
  test_path <- file.path(
    project_root, "data", "mimiciv", "ts_vital",
    sub("^Train_", "Test_", config$ts_filename)
  )

  train_dist <- compute_from_matrix(train_path)
  test_dist <- if (file.exists(test_path)) compute_from_matrix(test_path) else numeric(0)

  c(as.numeric(train_dist), as.numeric(test_dist))
}

build_example_plot <- function(project_root, config, context_window = 4L, compact = FALSE, panel_label = NULL, title_override = NULL) {
  shapelet_key <- paste0(config$feature_key, "_", format_pattern(config$values))
  curated <- CURATED_EXAMPLES[[shapelet_key]]

  if (!is.null(curated)) {
    seq_obj <- load_single_sequence(project_root, config$ts_filename, curated$split, curated$row)
    best_match <- compute_best_match(seq_obj$sequence, config$values)
    representative <- list(
      sequence = seq_obj$sequence,
      start = best_match$start,
      distance = best_match$distance,
      label = seq_obj$label
    )
  } else {
    ts_obj <- load_timeseries(project_root, config$ts_filename)
    representative <- find_representative_match(ts_obj$values, ts_obj$labels, config$values)
  }

  sequence <- representative$sequence
  start <- representative$start
  shapelet_len <- length(config$values)

  display_start <- max(1L, start - context_window)
  display_end <- min(length(sequence), start + shapelet_len - 1L + context_window)
  display_values <- sequence[display_start:display_end]
  x_values <- seq(display_start, display_end) - start

  matched_df <- tibble(
    x = x_values,
    value = display_values
  )
  shapelet_df <- tibble(
    x = seq_len(shapelet_len) - 1L,
    value = config$values
  )

  xticks <- unique(c(min(x_values), -2L, 0L, shapelet_len - 1L, max(x_values)))
  xticks <- xticks[order(xticks)]
  xlabels <- vapply(
    xticks,
    function(tick) {
      if (tick == 0) {
        "Start"
      } else if (tick == shapelet_len - 1L) {
        "End"
      } else if (tick < 0) {
        paste0(tick, "h")
      } else {
        paste0("+", tick - shapelet_len + 1L, "h")
      }
    },
    character(1)
  )

  base_size <- if (compact) 7.6 else 11
  matched_line_width <- if (compact) 0.62 else 0.9
  matched_point_size <- if (compact) 1.35 else 1.8
  shapelet_line_width <- if (compact) 0.92 else 1.25
  shapelet_point_size <- if (compact) 1.95 else 2.4
  x_axis_title <- NULL
  x_padding <- if (compact) expansion(mult = c(0.12, 0.16)) else expansion(mult = c(0.03, 0.07))
  y_padding <- if (compact) expansion(mult = c(0.3, 0.3)) else expansion(mult = c(0.15, 0.15))


  panel_title <- if (!is.null(title_override)) {
    title_override
  } else if (!is.null(panel_label)) {
    panel_label
  } else {
    NULL
  }
  panel_subtitle <- format_shapelet_subtitle(config)

  ggplot() +
    annotate(
      "rect",
      xmin = -0.1, xmax = shapelet_len - 0.9,
      ymin = -Inf, ymax = Inf,
      fill = COLORS$shapelet, alpha = 0.08
    ) +
    geom_line(
      data = matched_df,
      aes(x = x, y = value),
      linewidth = matched_line_width,
      color = COLORS$matched
    ) +
    geom_point(
      data = matched_df,
      aes(x = x, y = value),
      size = matched_point_size,
      color = COLORS$matched
    ) +
    geom_line(
      data = shapelet_df,
      aes(x = x, y = value),
      linewidth = shapelet_line_width,
      color = COLORS$shapelet
    ) +
    geom_point(
      data = shapelet_df,
      aes(x = x, y = value),
      size = shapelet_point_size,
      color = COLORS$shapelet
    ) +
    scale_x_continuous(breaks = xticks, labels = xlabels, expand = x_padding) +
    scale_y_continuous(expand = y_padding) +
    coord_cartesian(clip = "off") +
    labs(
      title = if (!is.null(panel_title) && nzchar(panel_title)) panel_title else NULL,
      subtitle = panel_subtitle,
      x = x_axis_title,
      y = paste0(config$display_name, " (", config$unit, ")")
    ) +
    theme_paper(base_size = base_size) +
    theme(
      legend.position = "none",
      plot.title = element_text(size = if (compact) 8.8 else 10.8),
      plot.subtitle = element_text(size = if (compact) 6.3 else 7.6, color = COLORS$text),
      axis.title = element_text(size = if (compact) 6.7 else 11),
      axis.title.x = element_text(lineheight = 0.92),
      axis.text = element_text(size = if (compact) 6.2 else 10),
      plot.margin = if (compact) margin(5, 8, 2, 8) else margin(8, 10, 8, 8)
    )
}

format_p_label <- function(p_value) {
  if (is.na(p_value)) {
    return("NA")
  }
  if (p_value < 0.001) {
    return("***")
  }
  if (p_value < 0.01) {
    return("**")
  }
  if (p_value < 0.05) {
    return("*")
  }
  "ns"
}

build_single_distribution_plot <- function(
    data, column, title, subtitle = NULL, y_label, emphasize = FALSE,
    title_color = NULL,
    compact = FALSE) {
  plot_df <- tibble(
    value = data[[column]],
    Group = factor(ifelse(data$label == 1, "AKI", "Non-AKI"), levels = c("Non-AKI", "AKI"))
  ) |>
    filter(!is.na(value))

  p_value <- wilcox.test(value ~ Group, data = plot_df)$p.value

  display_df <- plot_df
  if (emphasize) {
    upper_limit <- quantile(display_df$value, 0.99, na.rm = TRUE)
    display_df <- filter(display_df, value <= upper_limit)
  }

  y_min <- min(display_df$value, na.rm = TRUE)
  y_max <- max(display_df$value, na.rm = TRUE)
  y_range <- max(y_max - y_min, 1e-8)
  top_padding <- 0.30
  x_expand_left <- if (compact) {
    0.22
  } else {
    0.08
  }
  x_expand_right <- if (compact) 0.14 else 0.05

  # ==============================================
  # 核心修复：C1 和 C2/C3 分开设置标注高度！
  # emphasize = FALSE → C1图（最后值）单独高度
  # emphasize = TRUE  → C2/C3图（距离）统一高度
  # ==============================================
  if (emphasize) {
    # C2、C3 图：距离图（绿色背景）→ 固定低位
    bracket_y <- y_max + 0.15 * y_range
    text_y <- bracket_y + 0.06 * y_range
  } else {
    # C1 图：最后值图（白色背景）→ 固定高位，和C2/C3对齐
    bracket_y <- y_max + 0.15 * y_range
    text_y <- bracket_y + 0.11 * y_range
  }

  base_size <- if (compact) 8.6 else 10.5
  # 两套字号：显著性符号大，ns单独缩小
  sig_size <- if (compact) 3.5 else 4.0
  ns_size <- if (compact) 2.2 else 2.8
  corner_size <- if (compact) 2.35 else 2.9

  p <- ggplot(display_df, aes(x = Group, y = value, fill = Group)) +
    geom_violin(width = 0.88, alpha = 0.88, color = NA, trim = TRUE) +
    geom_boxplot(
      width = 0.2,
      outlier.shape = NA,
      fill = "white",
      color = COLORS$text,
      linewidth = 0.45
    ) +
    annotate("segment", x = 1, xend = 2, y = bracket_y, yend = bracket_y, linewidth = 0.35, color = COLORS$text) +
    annotate("segment", x = 1, xend = 1, y = bracket_y - 0.02 * y_range, yend = bracket_y, linewidth = 0.35, color = COLORS$text) +
    annotate("segment", x = 2, xend = 2, y = bracket_y - 0.02 * y_range, yend = bracket_y, linewidth = 0.35, color = COLORS$text) +
    annotate(
      "text",
      x = 1.5,
      y = text_y,
      label = format_p_label(p_value),
      size = ifelse(format_p_label(p_value) == "ns", ns_size, sig_size),
      fontface = "bold",
      color = COLORS$text
    ) +
    scale_fill_manual(values = c("Non-AKI" = COLORS$non_aki, "AKI" = COLORS$aki)) +
    scale_x_discrete(expand = expansion(add = c(x_expand_left, x_expand_right))) +
    coord_cartesian(ylim = c(y_min - 0.07 * y_range, y_max + top_padding * y_range), clip = "off") +
    labs(
      title = title,
      subtitle = subtitle,
      x = NULL,
      y = y_label
    ) +
    theme_paper(base_size = base_size) +
    theme(
      legend.position = "none",
      panel.background = if (emphasize) {
        element_rect(fill = COLORS$distance_bg, color = "#D9EDE4", linewidth = 0.5)
      } else {
        element_rect(fill = "white", color = NA)
      },
      plot.title = element_text(
        color = title_color %||% if (emphasize) COLORS$shapelet else COLORS$text,
        size = if (compact) 8.8 else 10.5
      ),
      plot.subtitle = element_text(size = if (compact) 6.2 else 7.4, color = COLORS$text),
      axis.title = element_text(size = if (compact) 7.4 else 10.5),
      axis.text = element_text(size = if (compact) 7.0 else 9.5),
      plot.margin = if (compact) margin(2, 8, 3, 8) else margin(8, 10, 8, 8)
    )

  p
}

build_comparison_plot <- function(data, project_root, config, compact = FALSE, panel_label = NULL, title_override = NULL) {
  last_title <- if (compact) "Last value" else "Last-recorded value"
  distance_title <- if (compact) "Distance" else "Shapelet distance"

  plot_data <- data
  distance_col <- config$distance_col
  if (is.null(distance_col) || !(distance_col %in% names(plot_data))) {
    distance_col <- paste0(config$feature_key, "_derived_", format_pattern(config$values))
    plot_data[[distance_col]] <- build_distance_vector(project_root, config)
  }

  p_last <- build_single_distribution_plot(
    data = plot_data,
    column = config$last_col,
    title = last_title,
    subtitle = NULL,
    y_label = paste0(config$display_name, " (", config$unit, ")"),
    emphasize = FALSE,
    compact = compact
  )

  p_distance <- build_single_distribution_plot(
    data = plot_data,
    column = distance_col,
    title = distance_title,
    subtitle = format_shapelet_subtitle(config),
    y_label = "Distance",
    emphasize = TRUE,
    compact = compact
  )

  panel_title <- if (!is.null(title_override)) {
    title_override
  } else if (is.null(panel_label)) {
    config$display_name
  } else {
    idx <- if (panel_label %in% c("A", "C")) "1" else "2"
    paste0(panel_label, "  ", config$display_name, " pattern ", idx)
  }

  combined <- p_last + p_distance
  if (!is.null(panel_title) && nzchar(panel_title)) {
    combined <- combined + plot_annotation(title = panel_title)
  }

  combined &
    theme(
      plot.title = element_text(
        face = "bold",
        hjust = 0,
        size = if (compact) 10.4 else 12,
        color = COLORS$text
      )
    )
}

build_examples_panel <- function(example1_plot, example2_plot, vertical = FALSE) {
  if (vertical) {
    top_plot <- example1_plot +
      labs(x = NULL) +
      theme(
        axis.title.x = element_blank(),
        plot.margin = margin(5, 8, 0, 8)
      )
    bottom_plot <- example2_plot +
      theme(
        plot.margin = margin(0, 8, 2, 8)
      )
    wrap_plots(list(top_plot, bottom_plot), ncol = 1)
  } else {
    wrap_plots(list(example1_plot, example2_plot), nrow = 1)
  }
}

build_combined_spo2_comparison_plot <- function(
    data, project_root, config1, config2, compact = FALSE, panel_label = NULL) {
  plot_data <- data

  distance_col_1 <- config1$distance_col
  if (is.null(distance_col_1) || !(distance_col_1 %in% names(plot_data))) {
    distance_col_1 <- paste0(config1$feature_key, "_derived_", format_pattern(config1$values))
    plot_data[[distance_col_1]] <- build_distance_vector(project_root, config1)
  }

  distance_col_2 <- config2$distance_col
  if (is.null(distance_col_2) || !(distance_col_2 %in% names(plot_data))) {
    distance_col_2 <- paste0(config2$feature_key, "_derived_", format_pattern(config2$values))
    plot_data[[distance_col_2]] <- build_distance_vector(project_root, config2)
  }

  last_title <- "C1"
  dist1_title <- "C2"
  dist2_title <- "C3"

  p_last <- build_single_distribution_plot(
    data = plot_data,
    column = config1$last_col,
    title = last_title,
    subtitle = format_last_subtitle(config1),
    y_label = paste0(config1$display_name, " (", config1$unit, ")"),
    emphasize = FALSE,
    compact = compact
  )

  p_dist1 <- build_single_distribution_plot(
    data = plot_data,
    column = distance_col_1,
    title = dist1_title,
    subtitle = format_shapelet_subtitle(config1),
    y_label = "Distance",
    emphasize = TRUE,
    title_color = COLORS$text,
    compact = compact
  )

  p_dist2 <- build_single_distribution_plot(
    data = plot_data,
    column = distance_col_2,
    title = dist2_title,
    subtitle = format_shapelet_subtitle(config2),
    y_label = "Distance",
    emphasize = TRUE,
    title_color = COLORS$text,
    compact = compact
  )

  wrap_plots(list(p_last, p_dist1, p_dist2), nrow = 1, widths = PANEL_LAYOUT$b_internal_widths)
}

parse_metric_shapelet <- function(shapelet_str) {
  clean <- gsub("^\\[|\\]$", "", shapelet_str)
  values <- as.numeric(strsplit(clean, ",\\s*")[[1]])
  values[!is.na(values)]
}

find_accuracy_for_shapelet <- function(csv_path, target_shapelet) {
  if (!file.exists(csv_path) || file.info(csv_path)$size == 0) {
    return(NA_real_)
  }

  df <- tryCatch(
    read_csv(csv_path, show_col_types = FALSE, progress = FALSE),
    error = function(e) NULL
  )
  if (is.null(df) || !all(c("shapelet", "acc") %in% names(df))) {
    return(NA_real_)
  }

  for (row_idx in seq_len(nrow(df))) {
    candidate <- parse_metric_shapelet(df$shapelet[[row_idx]])
    if (compare_shapelets(candidate, target_shapelet)) {
      return(as.numeric(df$acc[[row_idx]]))
    }
  }

  NA_real_
}

format_summary_label <- function(pattern) {
  parsed <- parse_shapelet_string(pattern)
  feature_key <- parsed$feature_key
  values <- parsed$values

  display_map <- c(
    heart = "Heart Rate",
    spo2 = "SpO2",
    sbp = "SBP",
    dbp = "DBP",
    bun = "BUN",
    creatinine = "Creatinine",
    potassium = "Potassium"
  )
  unit_map <- c(
    heart = "bpm",
    spo2 = "%",
    sbp = "mmHg",
    dbp = "mmHg",
    bun = "mg/dL",
    creatinine = "mg/dL",
    potassium = "mEq/L"
  )

  paste0(
    display_map[[feature_key]], "\n(",
    format_pattern(values), " ", unit_map[[feature_key]], ")"
  )
}

collect_summary_results <- function(results_root) {
  rows <- list()
  keep_idx <- 0L

  for (pattern in SUMMARY_PATTERNS) {
    parsed <- parse_shapelet_string(pattern)
    files <- SUMMARY_FILE_MAPPING[[parsed$feature_key]]
    if (is.null(files)) {
      next
    }

    for (filename in files) {
      train_path <- file.path(results_root, "csv_results", paste0("Train_", filename))
      test_path <- file.path(results_root, "csv_results", paste0("Test_", filename))
      train_acc <- find_accuracy_for_shapelet(train_path, parsed$values)
      test_acc <- find_accuracy_for_shapelet(test_path, parsed$values)

      if (!is.na(train_acc) && !is.na(test_acc)) {
        keep_idx <- keep_idx + 1L
        rows[[keep_idx]] <- tibble(
          pattern = pattern,
          label = format_summary_label(pattern),
          train_acc = train_acc,
          test_acc = test_acc
        )
        break
      }
    }
  }

  bind_rows(rows) |>
    arrange(desc(test_acc))
}

build_summary_plot <- function(results_root, compact = FALSE, panel_label = NULL, title_override = NULL) {
  summary_df <- collect_summary_results(results_root)
  summary_df <- summary_df |>
    mutate(
      y = rev(seq_len(n()))
    )

  long_df <- bind_rows(
    transmute(summary_df, label, y = y + 0.16, dataset = "Training set", accuracy = train_acc),
    transmute(summary_df, label, y = y - 0.16, dataset = "Test set", accuracy = test_acc)
  ) |>
    mutate(
      ymin = y - 0.11,
      ymax = y + 0.11,
      xmin = 0.50,
      xmax = accuracy,
      x_label = accuracy + 0.0025
    )

  x_max <- max(long_df$accuracy) + 0.016
  x_min <- 0.498
  panel_x_max <- x_max + if (compact) 0.003 else 0

  base_size <- if (compact) 7.3 else 10.5
  value_label_size <- if (compact) 2.25 else 2.9

  panel_title <- if (!is.null(title_override)) {
    title_override
  } else if (!is.null(panel_label)) {
    panel_label
  } else {
    NULL
  }

  ggplot(long_df) +
    geom_vline(xintercept = 0.50, linetype = "dashed", linewidth = 0.4, color = COLORS$chance) +
    geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = dataset),
      color = "#5A6570",
      linewidth = 0.32
    ) +
    geom_text(
      aes(x = x_label, y = y, label = sprintf("%.3f", accuracy)),
      hjust = 0,
      size = value_label_size,
      color = COLORS$text
    ) +
    scale_fill_manual(
      values = c("Training set" = COLORS$train, "Test set" = COLORS$test),
      breaks = c("Training set", "Test set")
    ) +
    scale_x_continuous(
      breaks = pretty_breaks(6)
    ) +
    scale_y_continuous(
      breaks = summary_df$y,
      labels = summary_df$label,
      expand = expansion(mult = c(0.05, if (compact) 0.01 else 0.015))
    ) +
    coord_cartesian(xlim = c(x_min, panel_x_max), clip = "off") +
    labs(
      title = if (!is.null(panel_title) && nzchar(panel_title)) panel_title else NULL,
      x = "Accuracy",
      y = NULL
    ) +
    theme_paper(base_size = base_size) +
    theme(
      legend.position = "inside",
      legend.position.inside = c(0.97, 0.03),
      legend.justification = c(1, 0),
      legend.direction = "vertical",
      legend.background = element_rect(fill = alpha("white", 0.9), color = NA),
      panel.grid.major.x = element_line(color = COLORS$grid, linewidth = 0.35),
      panel.grid.major.y = element_blank(),
      legend.text = element_text(size = if (compact) 6.8 else 9.5),
      legend.key.size = unit(if (compact) 7 else 12, "pt"),
      axis.text.y = element_text(size = if (compact) 6.1 else 9.5),
      axis.ticks.y = element_blank(),
      axis.text.x = element_text(size = if (compact) 6.7 else 9.5),
      axis.title.x = element_text(size = if (compact) 7.2 else 10.5),
      plot.title = element_text(
        size = if (compact) 8.8 else 10.8,
        face = "bold",
        hjust = 0,
        margin = margin(
          t = if (compact) 19.2 else 17.2, # 往下移
          b = if (compact) 8 else 12,
          l = if (compact) 15 else 17 # 往右移
        )
      ),
      plot.title.position = "plot",
      plot.margin = if (compact) margin(8, 8, 1, 8) else margin(8, 10, 8, 8)
    )
}

build_main_figure2 <- function(a_plot, b_plot, c_plot) {
  top_row <- wrap_plots(
    list(
      wrap_elements(
        full = a_plot + theme(
          legend.position = "inside",
          legend.position.inside = c(0.97, 0.03),
          legend.justification = c(1, 0),
          legend.direction = "vertical",
          legend.background = element_rect(fill = alpha("white", 0.9), color = NA),
          legend.key.size = unit(7, "pt"),
          legend.margin = margin(1, 1, 1, 1),
          legend.text = element_text(size = 7.2),
          plot.margin = margin(-11, 6, 2, 6)
        )
      ),
      wrap_elements(full = b_plot & theme(plot.margin = margin(4, 6, 2, 6)))
    ),
    nrow = 1,
    widths = PANEL_LAYOUT$top_row_widths
  )

  wrap_plots(
    list(
      top_row,
      wrap_elements(full = c_plot & theme(plot.margin = margin(4, 6, 3, 6)))
    ),
    ncol = 1,
    heights = PANEL_LAYOUT$main_row_heights
  )
}

main <- function() {
  cli_args <- parse_args(commandArgs(trailingOnly = TRUE))
  paths <- resolve_paths(cli_args)

  for_r_df <- load_for_r_data(paths$project_root)

  example1_config <- build_rep_config(cli_args$example1_shapelet, names(for_r_df))
  example2_config <- build_rep_config(cli_args$example2_shapelet, names(for_r_df))

  example1_example <- build_example_plot(
    paths$project_root, example1_config, cli_args$context_window,
    title_override = "B1"
  )
  example2_example <- build_example_plot(
    paths$project_root, example2_config, cli_args$context_window,
    title_override = "B2"
  )
  examples_panel <- build_examples_panel(example1_example, example2_example, vertical = TRUE)
  distributions_panel <- build_combined_spo2_comparison_plot(
    for_r_df, paths$project_root, example1_config, example2_config
  )
  summary_plot <- build_summary_plot(paths$results_root, title_override = "A")

  example1_example_compact <- build_example_plot(
    paths$project_root, example1_config, cli_args$context_window,
    compact = TRUE, title_override = "B1"
  )
  example2_example_compact <- build_example_plot(
    paths$project_root, example2_config, cli_args$context_window,
    compact = TRUE, title_override = "B2"
  )
  examples_panel_compact <- build_examples_panel(example1_example_compact, example2_example_compact, vertical = TRUE)
  distributions_panel_compact <- build_combined_spo2_comparison_plot(
    for_r_df, paths$project_root, example1_config, example2_config,
    compact = TRUE
  )
  summary_plot_compact <- build_summary_plot(paths$results_root, compact = TRUE, title_override = "A")
  main_figure2 <- build_main_figure2(
    a_plot = summary_plot_compact,
    b_plot = examples_panel_compact,
    c_plot = distributions_panel_compact
  )

  save_plot_pair(summary_plot, "figure2A_shapelet_summary", paths$output_dir, width = PANEL_LAYOUT$standalone_c_width, height = PANEL_LAYOUT$standalone_c_height, dpi = cli_args$dpi)
  save_plot_pair(examples_panel, "figure2B_spo2_examples", paths$output_dir, width = PANEL_LAYOUT$standalone_a_width, height = PANEL_LAYOUT$standalone_a_height, dpi = cli_args$dpi)
  save_plot_pair(distributions_panel, "figure2C_spo2_distributions", paths$output_dir, width = PANEL_LAYOUT$standalone_b_width, height = PANEL_LAYOUT$standalone_b_height, dpi = cli_args$dpi)
  save_combined_plot(main_figure2, "figure2_main_text_layout", paths$output_dir, width_mm = 146, height_mm = 118, dpi = cli_args$dpi)

  cat("Project root:", paths$project_root, "\n")
  cat("Results root:", paths$results_root, "\n")
  cat("Output dir:", paths$output_dir, "\n")
}

main()
