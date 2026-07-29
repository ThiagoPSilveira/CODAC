#' Execute the CODAC_Compare Analysis Pipeline
#'
#' Wrapper for the CODAC_Compare Python engine. Unlike Single/Flex (which analyze
#' one target series at a time), CODAC_Compare works with MULTIPLE GROUPS: it fits
#' rhythmicity per group and performs pairwise differential-rhythm comparisons
#' between groups (e.g. WT vs KO).
#'
#' The data columns must be ordered by group: for each group, all timepoints,
#' each with its replicates. So the number of value columns must equal
#' length(groups) * length(timepoints) * n_observations.
#'
#' @param data A dataframe: target names in column 1, then the value columns
#'   ordered by group (see above).
#' @param timepoints Numeric/character vector of collection times (e.g., c(2,6,10,14,18,22)).
#' @param groups Character vector naming the groups, IN THE SAME ORDER as the
#'   value-column blocks in `data` (e.g., c("WT_chow","WT_cdHFD","KO_chow","KO_cdHFD")).
#' @param n_observations Integer. Number of replicates per timepoint.
#' @param comparisons Optional list of group pairs to compare, each a length-2
#'   character vector, e.g. list(c("WT_chow","KO_chow"), c("WT_cdHFD","WT_chow")).
#'   If NULL, all pairwise combinations are compared.
#' @param targets_to_plot Character vector. Specific targets to plot.
#' @param r2_threshold Minimum R-squared threshold (default: 0.4).
#' @param p_threshold Significance level for p-value (default: 0.05).
#' @param p_value_option Multiple-testing strategy for the PER-GROUP rhythmicity
#'   test: 'FDR' or 'RAW' (default: 'FDR').
#' @param p_value_comparison Which p-values drive the PAIRWISE COMPARISON decisions
#'   (Mesor_Change, the Cat 2/3 loss/gain confidence, and the Cat 4-7
#'   amplitude/phase split): 'RAW' (raw pairwise p-values) or 'FDR'
#'   (Benjamini-Hochberg adjusted). This is INDEPENDENT of `p_value_option`.
#'   Both raw and FDR columns are always exported; only the decisions switch
#'   (default: 'RAW', which reproduces the previously validated behavior).
#' @param min_rhythmicity Minimum rhythmicity tier (default: 'ARRHYTHMIC').
#' @param rhythmicity_cutoff Rhythmicity level at which a group counts as rhythmic
#'   in the classification: 'ARRHYTHMIC','LOW','MEDIUM','HIGH','EXTREMELY HIGH' (default: 'HIGH').
#' @param interval_var Interval variance flexibility: 1, 2, or 3 (default: 1).
#' @param plot_flag Character. "Y" to generate individual plots (default: "N").
#' @param plot_all Character. "Y" to plot all, "N" to plot only targets_to_plot (default: "N").
#' @param time_label X-axis label in plots (default: 'ZT').
#' @param missing_data_action How to handle missing values: 'KEEP', 'IMPUTE', or 'REMOVE' (default: 'KEEP').
#' @param amp_stringency Amplitude-filter strictness from 0 to 1: 0 = off, 0.5 = default (validated), 1 = strictest (default: 0.5).
#' @param exclude_medium Logical. If TRUE, drop targets classified as MEDIUM in any group before the comparisons (default: TRUE).

#' @return A dataframe with the per-target fits and pairwise comparison results.
#'   All p-value, delta and metric columns are returned as numeric (not text),
#'   so they can be filtered directly (e.g. `dplyr::filter(p_diff_amplitude < 0.05)`)
#'   without any `as.numeric()` conversion.
#' @export
codac_compare <- function(data,
                             timepoints,
                             groups,
                             n_observations = 1,
                             comparisons = NULL,
                             targets_to_plot = NULL,
                             r2_threshold = 0.4,
                             p_threshold = 0.05,
                             p_value_option = 'FDR',
                             p_value_comparison = 'RAW',
                             min_rhythmicity = 'HIGH',
                             rhythmicity_cutoff = 'HIGH',
                             amp_stringency = 0.5,
                             interval_var = 1,
                             missing_data_action = 'KEEP',
                             plot_flag = "N",
                             plot_all = "N",
                             exclude_medium = TRUE,
                             time_label = 'ZT') {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Please install it using install.packages('reticulate').")
  }

  message("[INFO] Checking Python dependencies...")
  reticulate::py_require(c("pandas", "numpy", "scipy", "matplotlib", "statsmodels", "scikit-learn", "tqdm", "openpyxl", "xlsxwriter"))

  reticulate::py_run_string("pass")
  py <- reticulate::import_main()

  py$IS_R_INJECTED   <- TRUE

  # Data and design
  py$df_input          <- data
  py$r_timepoints      <- timepoints
  py$r_n_obs           <- n_observations
  py$r_groups          <- groups
  py$r_comparisons     <- comparisons
  py$r_targets_to_plot <- targets_to_plot

  # Statistical parameters
  py$r2_threshold        <- r2_threshold
  py$p_threshold         <- p_threshold
  py$p_value_option      <- p_value_option
  py$p_value_comparison  <- p_value_comparison
  py$min_rhythmicity     <- min_rhythmicity
  py$rhythmicity_cutoff  <- rhythmicity_cutoff
  py$amp_stringency      <- amp_stringency
  py$interval_var        <- interval_var
  py$missing_data_action <- missing_data_action
  py$exclude_medium      <- exclude_medium

  py$plot_flag          <- plot_flag
  py$plot_all           <- plot_all
  py$time_label         <- time_label

  script_path <- system.file("python", "CODAC_Compare_Py.py", package = "CODAC")

  if (script_path == "") {
    stop("[ERROR] Python script not found. Ensure 'CODAC_Compare_Py.py' is in the 'inst/python' folder.")
  }

  message("\n[INFO] Starting the CODAC_Compare Pipeline via Python engine...")

  reticulate::py_run_file(script_path)

  # The engine returns a single per-target "master table" (df_master), the same
  # one exported to Excel: each target's per-group fits followed by its pairwise
  # comparisons, laid out together. It comes back as a native R data.frame.
  results <- as.data.frame(reticulate::py_to_r(py$df_master))

  message("[SUCCESS] Compare Pipeline finished successfully!")

  return(results)
}
