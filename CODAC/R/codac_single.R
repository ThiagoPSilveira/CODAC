#' Execute the CODAC_Single Analysis Pipeline
#'
#' This function acts as a wrapper for the CODA Python engine, performing
#' non-linear Cosinor optimization and multi-criteria rhythmicity scoring.
#'
#' @param data A dataframe containing the expression data.
#' @param timepoints Numeric vector of collection times (e.g., c(2, 6, 10, 14, 18, 22)).
#' @param n_observations Integer. Number of replicates per timepoint.
#' @param targets_to_plot Character vector. Specific targets to plot (e.g., c("Ak4", "Per2")).
#' @param r2_threshold Minimum R-squared threshold for goodness of fit (default: 0.4).
#' @param p_threshold Significance level for p-value (default: 0.05).
#' @param p_value_option Multiple testing correction strategy: 'FDR' or 'RAW' (default: 'FDR').
#' @param min_rhythmicity Minimum rhythmicity tier: 'ARRHYTHMIC', 'LOW', 'MEDIUM', 'HIGH', 'EXTREMELY HIGH' (default: 'HIGH').
#' @param missing_data_action How to handle NaNs: 'KEEP', 'IMPUTE', or 'REMOVE' (default: 'KEEP').
#' @param interval_var Interval variance flexibility: 1, 2, or 3 (default: 1).
#' @param amp_stringency Amplitude-filter strictness from 0 to 1: 0 = off, 0.5 = default, 1 = strictest (default: 0.5).
#' @param period_mode Period fitting mode: 'fixed' or 'variable' (default: 'fixed').
#' @param fixed_period Fixed period value in hours (default: 24.0).
#' @param period_lower Lower limit for variable period in hours (default: 20.0).
#' @param period_upper Upper limit for variable period in hours (default: 28.0).
#' @param plot_flag Character. "Y" to generate individual plots for each gene (default: "N").
#' @param plot_all Character. "Y" to plot all. "N" to plot only those in the input list (default: "Y").
#' @param time_label X-axis label in plots: 'ZT', 'CT', or 'Clock' (default: 'ZT').
#'
#' @return A dataframe containing the rhythmicity results.
#' @export
codac_single <- function(data,
                            timepoints,
                            n_observations = 1,
                            targets_to_plot = NULL,
                            r2_threshold = 0.4,
                            p_threshold = 0.05,
                            p_value_option = 'FDR',
                            min_rhythmicity = 'HIGH',
                            missing_data_action = 'KEEP',
                            interval_var = 1,
                            amp_stringency = 0.5,
                            period_mode = 'fixed',
                            fixed_period = 24.0,
                            period_lower = 20.0,
                            period_upper = 28.0,
                            plot_flag = "Y",
                            plot_all = "N",
                            time_label = 'ZT') {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Please install it using install.packages('reticulate').")
  }

  message("[INFO] Checking Python dependencies...")
  reticulate::py_require(c("pandas", "numpy", "scipy", "matplotlib", "statsmodels", "scikit-learn", "tqdm", "openpyxl", "xlsxwriter"))

  reticulate::py_run_string("pass")

  py <- reticulate::import_main()

  py$IS_R_INJECTED   <- TRUE


  py$df_input          <- data
  py$r_timepoints      <- timepoints
  py$r_n_obs           <- n_observations
  py$r_targets_to_plot <- targets_to_plot


  py$r2_threshold        <- r2_threshold
  py$p_threshold         <- p_threshold
  py$p_value_option      <- p_value_option
  py$min_rhythmicity     <- min_rhythmicity
  py$missing_data_action <- missing_data_action
  py$interval_var        <- interval_var
  py$amp_stringency <- amp_stringency

  py$period_mode         <- period_mode
  py$fixed_period        <- fixed_period
  py$period_lower        <- period_lower
  py$period_upper        <- period_upper

  py$plot_flag           <- plot_flag
  py$plot_all            <- plot_all
  py$time_label          <- time_label

  script_path <- system.file("python", "CODAC_Single_Py.py", package = "CODAC")

  if (script_path == "") {
    stop("[ERROR] Python script not found. Ensure 'CODAC_Single_Py.py' is in the 'inst/python' folder.")
  }

  message("\n[INFO] Starting the CODAC_Single Pipeline via Python engine...")

  reticulate::py_run_file(script_path)

  results <- reticulate::py_to_r(py$df_results)
  results <- as.data.frame(results)

  message("[SUCCESS] Single Pipeline finished successfully!")

  return(results)
}
