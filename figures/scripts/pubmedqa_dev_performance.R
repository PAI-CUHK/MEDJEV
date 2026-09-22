options(warn = 2)

repo_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source("C:/Users/JIAN/.codex/skills/nature-r-figure-studio/scripts/nature_r_helpers.R", local = TRUE)
source("C:/Users/JIAN/.codex/skills/nature-r-figure-studio/scripts/nature_r_audit.R", local = TRUE)

required_packages <- c("ggplot2", "dplyr", "tidyr", "scales", "svglite", "ragg")
missing <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))

source_data <- file.path(repo_root, "figures/source_data/pubmedqa_dev_performance.csv")
data <- utils::read.csv(source_data, stringsAsFactors = FALSE, check.names = FALSE)
expected_columns <- c(
  "backbone", "method", "canonical", "unseen_answer", "unseen_decision",
  "permutation_behavior"
)
if (!identical(names(data), expected_columns)) {
  stop("Unexpected source columns: ", paste(names(data), collapse = ", "))
}
if (nrow(data) != 12L) stop("Expected 12 benchmark rows, found ", nrow(data))
score_columns <- c("canonical", "unseen_answer", "unseen_decision")
if (any(!vapply(data[score_columns], is.numeric, logical(1)))) stop("Scores must be numeric.")
if (any(unlist(data[score_columns]) < 0 | unlist(data[score_columns]) > 100)) {
  stop("Scores must be percentages in the range 0-100.")
}

profile <- profile_figure_data(data, group_cols = c("backbone", "method"))
profile_dir <- file.path(repo_root, "figures/audit/data_profile")
write_figure_profile(profile, profile_dir)

data$backbone <- factor(data$backbone, levels = c("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3.5-4B"))
data$method <- factor(data$method, levels = c("Direct", "Direct SFT", "Fixed", "MedJEV"))
long <- tidyr::pivot_longer(
  data,
  cols = dplyr::all_of(score_columns),
  names_to = "metric",
  values_to = "accuracy"
)
long$metric <- factor(
  long$metric,
  levels = score_columns,
  labels = c("Canonical", "Unseen answer", "Unseen decision")
)

method_colours <- c(
  "Direct" = "#2C6EAB",
  "Direct SFT" = "#D9772B",
  "Fixed" = "#7B5A8E",
  "MedJEV" = "#1B8A5A"
)
method_shapes <- c("Direct" = 16, "Direct SFT" = 17, "Fixed" = 15, "MedJEV" = 18)
method_linetypes <- c("Direct" = "solid", "Direct SFT" = "dashed", "Fixed" = "dotted", "MedJEV" = "solid")

plot <- ggplot2::ggplot(
  long,
  ggplot2::aes(
    x = metric, y = accuracy, group = method,
    colour = method, linetype = method, shape = method
  )
) +
  ggplot2::geom_line(
    ggplot2::aes(linewidth = method == "MedJEV"),
    lineend = "round"
  ) +
  ggplot2::geom_point(size = 2.3, stroke = 0.25) +
  ggplot2::facet_wrap(~backbone, nrow = 1) +
  ggplot2::scale_colour_manual(values = method_colours, name = "Method") +
  ggplot2::scale_shape_manual(values = method_shapes, name = "Method") +
  ggplot2::scale_linetype_manual(values = method_linetypes, name = "Method") +
  ggplot2::scale_linewidth_manual(values = c(`FALSE` = 0.65, `TRUE` = 1.05), guide = "none") +
  ggplot2::scale_y_continuous(
    limits = c(0, 100), breaks = seq(0, 100, 20),
    expand = ggplot2::expansion(mult = c(0.01, 0.03))
  ) +
  ggplot2::labs(
    x = NULL, y = "Accuracy (%)",
    caption = "PubMedQA development only; 50 examples; identity-restored accuracy averaged across four candidate permutations."
  ) +
  theme_nature_compact(base_size = 7.5, base_family = "Arial") +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 25, hjust = 1, vjust = 1, size = 6.5),
    axis.text.y = ggplot2::element_text(size = 6.5),
    axis.title.y = ggplot2::element_text(size = 7.2, margin = ggplot2::margin(r = 5)),
    strip.background = ggplot2::element_blank(),
    strip.text = ggplot2::element_text(size = 7.5, face = "bold", colour = "#222222"),
    panel.grid.major = ggplot2::element_line(colour = "#D9D9D9", linewidth = 0.3),
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(fill = NA, colour = "#222222", linewidth = 0.45),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = ggplot2::element_text(size = 6.5),
    legend.title = ggplot2::element_text(size = 6.8, face = "bold"),
    legend.key.width = grid::unit(9, "mm"),
    legend.key.height = grid::unit(4, "mm"),
    plot.caption = ggplot2::element_text(size = 5.7, colour = "#666666", hjust = 0),
    panel.spacing = grid::unit(3, "mm")
  ) +
  ggplot2::guides(
    colour = ggplot2::guide_legend(order = 1, override.aes = list(linewidth = 0.8)),
    shape = ggplot2::guide_legend(order = 1),
    linetype = ggplot2::guide_legend(order = 1)
  )

output_base <- file.path(repo_root, "docs/assets/pubmedqa-dev-performance")
dir.create(dirname(output_base), recursive = TRUE, showWarnings = FALSE)
save_nature_bundle(
  plot, output_base, width_mm = 183, height_mm = 74, dpi = 600,
  profile = "nature-main", stage = "submission", data_status = "verified", strict = TRUE
)

manifest <- data.frame(
  panel = letters[1:3],
  evidential_role = c("absolute benchmark", "wording transfer", "decision semantics"),
  unique_question = c(
    "How accurate is each method on the canonical schema?",
    "How accurate is each method on unseen answer wording?",
    "How accurate is each method on unseen decision wording?"
  ),
  medium = rep("quantitative", 3),
  source = rep("figures/source_data/pubmedqa_dev_performance.csv", 3),
  data_status = rep("verified", 3),
  stringsAsFactors = FALSE
)
write_panel_manifest(
  manifest,
  file.path(repo_root, "figures/audit/pubmedqa_dev_panel_manifest.csv"),
  stage = "submission"
)

audit <- audit_export_bundle(
  output_base, width_mm = 183, height_mm = 74, dpi = 600,
  profile = "nature-main", stage = "submission", data_status = "verified"
)
write_figure_audit(audit, file.path(repo_root, "figures/audit/pubmedqa_dev_export_audit.csv"))
utils::write.csv(
  audit_palette_accessibility(unname(method_colours)),
  file.path(repo_root, "figures/audit/pubmedqa_dev_palette_audit.csv"), row.names = FALSE
)

cat("Rendered latest PubMedQA development figure to ", output_base, ".{svg,pdf,tiff,png}\n", sep = "")
