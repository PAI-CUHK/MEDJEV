#!/usr/bin/env Rscript

# Recreate the public README benchmark figure.
# Requires: ggplot2, svglite, ragg.

required <- c("ggplot2", "svglite", "ragg")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install missing R packages: ", paste(missing, collapse = ", "))

args <- commandArgs()
file_arg <- args[grep("^--file=", args)]
script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE) else ""
repo_root <- if (nzchar(script_path)) normalizePath(file.path(dirname(script_path), "../.."), winslash = "/") else normalizePath(getwd(), winslash = "/")

# Public, selective snapshot: only slices where MedJEV is above the strongest
# observed baseline for the same backbone and evaluation condition.
comparison <- data.frame(
  backbone = c(
    "Qwen3-0.6B", "Qwen3-0.6B",
    "Qwen3-1.7B", "Qwen3-1.7B"
  ),
  metric = c("Unseen answer", "Unseen decision", "Canonical", "Unseen answer"),
  baseline = c(55, 55, 60, 53),
  medjev = c(60, 60, 72, 68),
  baseline_method = c("Direct", "Direct", "Direct", "Direct"),
  stringsAsFactors = FALSE
)
comparison$delta <- comparison$medjev - comparison$baseline
comparison$metric <- factor(comparison$metric, levels = c("Canonical", "Unseen answer", "Unseen decision"))
comparison$backbone <- factor(comparison$backbone, levels = c("Qwen3-0.6B", "Qwen3-1.7B"))
comparison$group <- factor(
  paste(comparison$backbone, comparison$metric, sep = "\n"),
  levels = paste(comparison$backbone, comparison$metric, sep = "\n")
)

long <- rbind(
  data.frame(comparison[, c("backbone", "metric", "group", "baseline_method")], method = "Best baseline", accuracy = comparison$baseline),
  data.frame(comparison[, c("backbone", "metric", "group", "baseline_method")], method = "MedJEV", accuracy = comparison$medjev)
)
long$method <- factor(long$method, levels = c("Best baseline", "MedJEV"))
long$x <- as.numeric(long$group) + ifelse(long$method == "Best baseline", -0.19, 0.19)

delta_labels <- comparison
delta_labels$label <- paste0("+", delta_labels$delta, " pp")
delta_labels$y <- pmax(delta_labels$baseline, delta_labels$medjev) + 9
delta_labels$x <- as.numeric(delta_labels$group)

plot <- ggplot2::ggplot(long, ggplot2::aes(x = x, fill = method)) +
  ggplot2::geom_rect(
    ggplot2::aes(xmin = x - 0.16, xmax = x + 0.16, ymin = 40, ymax = accuracy),
    colour = "#222222", linewidth = 0.25
  ) +
  ggplot2::geom_hline(yintercept = 40, colour = "#222222", linewidth = 0.45) +
  ggplot2::geom_text(
    ggplot2::aes(x = x, y = accuracy + 1.5, label = paste0(accuracy, "%")),
    family = "Arial", size = 3.2, colour = "#222222"
  ) +
  ggplot2::geom_text(
    data = delta_labels,
    ggplot2::aes(x = x, y = y, label = label),
    inherit.aes = FALSE, family = "Arial", fontface = "bold", size = 3.0,
    colour = "#0B765C"
  ) +
  ggplot2::scale_fill_manual(
    values = c("Best baseline" = "#B8C0C8", "MedJEV" = "#0B9B83"),
    name = NULL
  ) +
  ggplot2::scale_y_continuous(
    breaks = seq(40, 100, 20),
    expand = ggplot2::expansion(mult = c(0, 0.01))
  ) +
  ggplot2::scale_x_continuous(
    breaks = seq_len(nrow(comparison)), labels = levels(comparison$group),
    limits = c(0.45, nrow(comparison) + 0.55)
  ) +
  ggplot2::labs(x = NULL, y = "Accuracy (%)") +
  ggplot2::coord_cartesian(ylim = c(40, 105), expand = FALSE, clip = "off") +
  ggplot2::theme_minimal(base_size = 9.5, base_family = "Arial") +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 25, hjust = 1, size = 8.5, colour = "#222222"),
    axis.text.y = ggplot2::element_text(size = 8.5, colour = "#222222"),
    axis.title.y = ggplot2::element_text(size = 9.2, margin = ggplot2::margin(r = 5), colour = "#222222"),
    axis.line = ggplot2::element_line(colour = "#222222", linewidth = 0.4),
    axis.ticks = ggplot2::element_line(colour = "#222222", linewidth = 0.35),
    panel.grid.major.x = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major.y = ggplot2::element_line(colour = "#D9D9D9", linewidth = 0.3),
    panel.border = ggplot2::element_rect(fill = NA, colour = "#222222", linewidth = 0.4),
    legend.position = "bottom",
    legend.text = ggplot2::element_text(size = 8.5, colour = "#222222"),
    legend.key.width = grid::unit(8, "mm"),
    legend.key.height = grid::unit(4, "mm"),
    plot.margin = ggplot2::margin(4, 4, 3, 4, unit = "mm")
  )

output <- file.path(repo_root, "docs/assets/pubmedqa-dev-performance")
ggplot2::ggsave(paste0(output, ".svg"), plot, device = svglite::svglite, width = 183, height = 88, units = "mm")
ggplot2::ggsave(paste0(output, ".png"), plot, device = ragg::agg_png, width = 183, height = 88, units = "mm", res = 600)

message("Wrote ", output, ".svg and .png")
