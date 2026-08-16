# ==============================================================================
# CODAC SINGLE - ANALYSIS SCRIPT
# ==============================================================================
# You do NOT need to install Python or any Python libraries by hand. The first
# time you run the analysis, R sets up everything automatically (this needs an
# internet connection and may take a minute).
#
# Results (the Excel table and the plots) are saved automatically in a folder
# called "CODA_Results", created right next to your data file.
# ==============================================================================


# ------------------------------------------------------------------------------
# 0. SETUP  (installs the package and its dependencies)
# ------------------------------------------------------------------------------
install.packages("reticulate")   # connects R to Python (use a recent version)
install.packages("remotes")      # used to install the package from GitHub
# NOTE: after the install below, R will RESTART automatically. That is normal.    <--------- Important
# When it finishes, run the rest of the script (from Section 1 onwards).          <--------- Important

# This line RE-DOWNLOADS the latest version of the package from GitHub every
# time you run the script. Keeping it here guarantees you always test the most
# up-to-date code, without having to check the repository yourself.
#
# >>> You MAY comment out the line below to run faster, BUT only do so if you
#     are sure you already have the latest version of the code installed. <<<
remotes::install_github("ThiagoPSilveira/CODAC", subdir = "CODAC")

#
# After installing (or updating) the package, RESTART R (Session > Restart R)
# before running the analysis.


# ------------------------------------------------------------------------------
# 1. POINT TO YOUR DATA FILE
# ------------------------------------------------------------------------------
# Tab-delimited file: genes in rows, samples in columns, gene name in column 1.
data_path <- "C:/path/to/your/single_gene_data.txt"   # <-- EDIT THIS

# Work in the folder that contains your data file. The engine automatically
# creates a "CODA_Results" folder here (same behavior as the PyCharm run),
# so we do NOT create it in R -- otherwise you would get CODA_Results/CODA_Results.
setwd(dirname(data_path))
results_path <- file.path(dirname(data_path), "CODA_Results")


# ------------------------------------------------------------------------------
# 2. LOAD THE PACKAGE AND THE DATA
# ------------------------------------------------------------------------------
library(CODAC)

# ///////////////////////////////////////////////////////////////////////////
# IMPORTANT - DECIMAL SEPARATOR (the 'dec' argument below)
# ---------------------------------------------------------------------------
# Numbers in your file use EITHER a period (e.g. 12.34) OR a comma (e.g. 12,34)
# as the decimal mark. You MUST tell R which one your file uses:
#     dec = "."   ->  for period decimals  (e.g. 12.34)   [default here]
#     dec = ","   ->  for comma decimals   (e.g. 12,34)
#
# Open your data file and check. If it is wrong, the numbers are read as text
# and EVERY gene is skipped as "insufficient valid data" - so if you see that,
# switch the value of dec below and run again.
# ///////////////////////////////////////////////////////////////////////////
expression_data <- read.table(data_path, sep = "\t", header = TRUE,
                              check.names = FALSE, dec = ",")

# Safety net: drop fully-empty columns (e.g. from trailing tabs in the header)
expression_data <- expression_data[, colSums(is.na(expression_data)) < nrow(expression_data)]

# Quick check: this should equal length(timepoints) * n_observations.
cat("Value columns detected:", ncol(expression_data) - 1, "\n")


# ------------------------------------------------------------------------------
# 3. RUN THE ANALYSIS
# ------------------------------------------------------------------------------
# The very first run sets up Python automatically. Please be patient.
analysis_results <- codac_single(

  # --- Data & experimental design --------------------------------------------
  data            = expression_data,            # The expression table loaded above
  timepoints      = c(2, 6, 10, 14, 18, 22),    # Collection times (hours)
  n_observations  = 4,                          # Replicates per timepoint
  targets_to_plot = c("Per2"),                  # Target(s) to plot, e.g. c("Per2", "Arntl")

  # --- Statistical parameters ------------------------------------------------
  r2_threshold        = 0.4,        # Minimum R-squared (goodness of fit)
  p_threshold         = 0.05,       # Significance level (p-value)
  p_value_option      = 'FDR',      # 'FDR' (Benjamini-Hochberg) or 'RAW'
  min_rhythmicity     = 'HIGH',     # 'ARRHYTHMIC','LOW','MEDIUM','HIGH','EXTREMELY HIGH'
  missing_data_action = 'KEEP',     # 'KEEP', 'IMPUTE', or 'REMOVE'
  interval_var        = 1,          # Interval variance flexibility (1, 2, or 3)
  amp_stringency      = 0.5,        # Amplitude-filter strictness: 0 = off, 0.5 = default, 1 = strictest

  # --- Period parameters -----------------------------------------------------
  period_mode   = 'fixed',          # 'fixed' or 'variable'
  fixed_period  = 24.0,             # Fixed period in hours (used when 'fixed')
  period_lower  = 20.0,             # Lower limit in hours (used when 'variable')
  period_upper  = 28.0,             # Upper limit in hours (used when 'variable')

  # --- Visual & export parameters --------------------------------------------
  plot_flag   = "Y",                # "Y" generates an individual plot per gene
  plot_all    = "N",                # "N" plots only targets_to_plot (safe for big datasets)
  time_label  = 'ZT'                # X-axis label: 'ZT', 'CT', or 'Clock'
)


# ------------------------------------------------------------------------------
# 4. VIEW / EXPORT THE RESULTS
# ------------------------------------------------------------------------------
# Safety net: make sure the result is a native R data.frame, so View() and
# write.csv() work even if the package returns a Python (pandas) object.
if (!is.data.frame(analysis_results)) {
  analysis_results <- as.data.frame(reticulate::py_to_r(analysis_results))
}

View(analysis_results)
cat("\n[SUCCESS] Done! Results were saved in:\n", results_path, "\n")
