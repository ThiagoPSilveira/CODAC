# ==============================================================================
# CODAC_Multi - ANALYSIS SCRIPT
# ==============================================================================
# You do NOT need to install Python or any Python libraries by hand. The first
# time you run the analysis, R sets up everything automatically (this needs an
# internet connection and may take a minute).
#
# CODAC_Multi works with MULTIPLE GROUPS: it fits rhythmicity per group and
# performs pairwise differential-rhythm comparisons between groups. The returned
# table is the per-gene "master table": each gene's per-group fits and pairwise
# comparisons laid out together (same as the Excel report).
#
# Results (the Excel table and the plots) are saved automatically in a folder
# called "CODA_Results", created right next to your data file.
#
# NOTE: after the install in Section 0, R will RESTART automatically. That is
# normal. When it finishes, run the rest of the script (from Section 1 onwards).
# ==============================================================================


# ------------------------------------------------------------------------------
# 0. SETUP  (installs the package and its dependencies)
# ------------------------------------------------------------------------------
install.packages("reticulate")   # connects R to Python (use a recent version)
install.packages("remotes")      # used to install the package from GitHub

# Normal install (fast; skips the download if you already have the latest version):
remotes::install_github("ThiagoPSilveira/CODAC", subdir = "CODAC")

# >>> If you were told there is a NEW version, and the line above prints
#     "Skipping install ... SHA has not changed", run THIS line instead ONCE to
#     force the update, then restart R:
# remotes::install_github("ThiagoPSilveira/CODAC", subdir = "CODAC", force = TRUE)

# If the repository is PRIVATE, the install may fail with a 404 error. In that
# case, create a GitHub token once, then re-run the install line:
#   usethis::create_github_token()
#   gitcreds::gitcreds_set()
#
# After installing (or updating) the package, RESTART R (Session > Restart R)
# before running the analysis.


# ------------------------------------------------------------------------------
# 1. POINT TO YOUR DATA FILE
# ------------------------------------------------------------------------------
# Tab-delimited file: target names in column 1, then the value columns ordered
# BY GROUP: for each group, all timepoints, each with its replicates. So the
# number of value columns must equal length(groups) * length(timepoints) * n_observations.
data_path <- "C:/path/to/your/multi_gene_data.txt"   # <-- EDIT THIS

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
#     dec = "."   ->  for period decimals  (e.g. 12.34)
#     dec = ","   ->  for comma decimals   (e.g. 12,34)
#
# Open your data file and check. If it is wrong, the numbers are read as text
# and the column count may even "explode" (far more columns than expected) - so
# if the check below is not what you expect, switch dec and run again.
# ///////////////////////////////////////////////////////////////////////////
expression_data <- read.table(data_path, sep = "\t", header = TRUE,
                              check.names = FALSE, dec = ".")

# Safety net: drop fully-empty columns (e.g. from trailing tabs in the header)
expression_data <- expression_data[, colSums(is.na(expression_data)) < nrow(expression_data)]

# Quick check: this must equal length(groups) * length(timepoints) * n_observations.
cat("Value columns detected:", ncol(expression_data) - 1, "\n")


# ------------------------------------------------------------------------------
# 3. RUN THE ANALYSIS
# ------------------------------------------------------------------------------
# The very first run sets up Python automatically. Please be patient.
analysis_results <- codac_multi(

  # --- Data & experimental design --------------------------------------------
  data            = expression_data,
  timepoints      = c(2, 6, 10, 14, 18, 22),        # Collection times (hours)
  n_observations  = 4,                              # Replicates per timepoint

  # groups: names IN THE SAME ORDER as the value-column blocks in the data file
  groups          = c("group1", "group2", "group3", "group4"),

  # comparisons: list of group pairs to compare. Set to NULL to compare all pairs.
  comparisons     = list(c("group1", "group2"),
                         c("group3", "group4"),
                         c("group1", "group3"),
                         c("group2", "group4")),

  targets_to_plot = c("TARGET_ID_HERE"),              # Target(s) to plot

  # --- Statistical parameters ------------------------------------------------
  r2_threshold        = 0.4,          # Minimum R-squared (goodness of fit)
  p_threshold         = 0.05,         # Significance level (p-value)
  missing_data_action = 'KEEP',       # How to handle missing values: 'KEEP', 'IMPUTE', or 'REMOVE'
  p_value_option      = 'FDR',        # PER-GROUP rhythmicity: 'FDR' or 'RAW'. Own alpha via c('FDR', 0.1)
  p_value_comparison  = 'RAW',        # PAIRWISE decisions: 'RAW' or 'FDR'. Own alpha via c('RAW', 0.05)
  selection_criterion = 'BIC',        # MULTI-GROUP grouping criterion: 'BIC' (conservative, default) or 'AICc'
  p_value_global      = 'FDR',        # GLOBAL gates: 'FDR' (default) or 'RAW'. Own alpha via c('FDR', 0.1)
  rhythm_diff_correction = 'all_targets', # p_rhythm_diff FDR: 'all_targets' (genome-wide) or 'screened_pooled' (two-stage, recovers power)
  permute_B           = 0,             # diagnostic: permutations to calibrate the gate's empirical FDR (0 = off)
  min_rhythmicity     = 'HIGH',       # 'ARRHYTHMIC','LOW','MEDIUM','HIGH','EXTREMELY HIGH'
  rhythmicity_cutoff  = 'HIGH',       # Rhythmicity level to count a group as rhythmic (excludes MEDIUM by default)
  amp_stringency      = 0.5,          # Amplitude-filter strictness: 0 = off, 0.5 = default, 1 = strictest
  interval_var        = 1,            # Interval variance flexibility (1, 2, or 3)
  exclude_medium      = TRUE,         # TRUE = drop targets that are MEDIUM in any group

  # --- Visual & export parameters --------------------------------------------
  plot_flag   = "Y",                  # "Y" generates an individual plot per target
  plot_all    = "N",                  # "N" plots only targets_to_plot (safe for big datasets)
  time_label  = 'ZT'                  # X-axis label: 'ZT', 'CT', or 'Clock'
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

# ------------------------------------------------------------------------------
# 5. POST-PROCESSING  (category selection, per-pair UpSet, amplitude histogram)
# ------------------------------------------------------------------------------
# Everything here runs on `analysis_results` (the master table). It does NOT
# need the package to be reinstalled -- edit and re-run freely.
#
# The master table has, per Target, several rows: one per group (the individual
# fits) PLUS the pairwise comparison rows. Biological_Category, Mesor_Change and
# LossGain_Confidence are filled ONLY on the comparison rows, and each comparison
# row belongs to a specific Pair. So all selection/plotting below is done
# PER PAIR, and keeps the WHOLE Target block (group_by(Target) + any(...)).
#
# NOTE 1: Biological_Category, Mesor_Change and LossGain_Confidence already
# reflect the `p_value_comparison` choice ('RAW' or 'FDR') made in Section 3 --
# nothing extra is needed here to switch between them.
# NOTE 2: all p-value / delta / metric columns come back as NUMERIC, so you can
# filter them directly, e.g.:
#   results_clean %>% filter(p_diff_amplitude < 0.05)
#   results_clean %>% filter(p_diff_phase_FDR < 0.05)
# (no as.numeric() needed).

# Install the post-processing helper packages if they are missing (first run on a
# new machine). They are NOT part of CODAC_Multi -- only this Section 5 uses them.
.pp_pkgs <- c("dplyr", "UpSetR", "ggplot2")
.pp_missing <- .pp_pkgs[!vapply(.pp_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(.pp_missing) > 0) {
  message("[INFO] Installing post-processing packages: ", paste(.pp_missing, collapse = ", "))
  install.packages(.pp_missing)
}

library(dplyr)
library(UpSetR)
library(ggplot2)

# The 7 categories, in order.
category_labels <- c(
  "Cat 1: Arrhythmic",
  "Cat 2: rhythmic_group_1_only",
  "Cat 3: rhythmic_group_2_only",
  "Cat 4: rhythmic_both_unchanged",
  "Cat 5: rhythmic_with_changes_only_amp",
  "Cat 6: rhythmic_with_changes_only_phase",
  "Cat 7: rhythmic_with_changes_amp_phase"
)

# Drop the blank spacer rows once (rows with no category).
results_clean <- analysis_results %>%
  filter(!is.na(Biological_Category) & Biological_Category != "")

# All comparison pairs actually present in the results.
all_pairs <- results_clean %>%
  filter(!is.na(Pair) & Pair != "") %>%
  pull(Pair) %>%
  unique()

cat("Comparison pairs found:", paste(all_pairs, collapse = " | "), "\n")


# ------------------------------------------------------------------------------
# 5a. CATEGORY SELECTION  (objects in the session, per pair x category)
# ------------------------------------------------------------------------------
# Builds a nested list: category_selection[[pair]][[category]] is the full
# Target blocks for that pair+category. Also split by Mesor_Change
# (Conserved / Different), under $..._mesor_conserved / $..._mesor_diff.
# Nothing is written to disk -- these are just objects to inspect / View().

category_selection <- list()

for (pr in all_pairs) {
  df_pair <- results_clean %>% filter(Pair == pr)
  targets_in_pair <- unique(df_pair$Target)

  # Restrict the master table to the Targets of THIS pair, so a Target's full
  # block travels with it (the group rows have Pair == "" and would be lost by a
  # plain Pair filter, so we re-select the whole block by Target).
  df_pair_full <- analysis_results %>% filter(Target %in% targets_in_pair)

  per_cat <- list()
  for (cat in category_labels) {
    # Targets that are in this category FOR THIS PAIR.
    tg <- df_pair %>%
      filter(Biological_Category == cat) %>%
      pull(Target) %>% unique()
    if (length(tg) == 0) next

    block <- df_pair_full %>% filter(Target %in% tg)

    # Mesor split (Mesor_Change lives on the comparison rows of this pair).
    tg_cons <- df_pair %>%
      filter(Biological_Category == cat, Mesor_Change == "Conserved") %>%
      pull(Target) %>% unique()
    tg_diff <- df_pair %>%
      filter(Biological_Category == cat, Mesor_Change == "Different") %>%
      pull(Target) %>% unique()

    per_cat[[cat]] <- list(
      all             = block,
      mesor_conserved = df_pair_full %>% filter(Target %in% tg_cons),
      mesor_diff      = df_pair_full %>% filter(Target %in% tg_diff)
    )
  }
  category_selection[[pr]] <- per_cat
}

# Example of how to reach one selection (uncomment to use):
# View(category_selection[[ all_pairs[1] ]][["Cat 5: rhythmic_with_changes_only_amp"]]$all)
# View(category_selection[[ all_pairs[1] ]][["Cat 2: rhythmic_group_1_only"]]$mesor_diff)


# ------------------------------------------------------------------------------
# 5b. UPSET PLOT  (ONE per comparison pair)
# ------------------------------------------------------------------------------
# For each pair, build category -> Target sets and draw an UpSet showing how the
# categories distribute/intersect. One JPEG per pair, saved in results_path.

for (pr in all_pairs) {
  df_pair <- results_clean %>% filter(Pair == pr)

  target_sets <- lapply(category_labels, function(cat) {
    df_pair %>% filter(Biological_Category == cat) %>% pull(Target) %>% unique()
  })
  names(target_sets) <- category_labels

  # UpSetR errors on empty sets -> drop categories with zero targets.
  target_sets <- target_sets[sapply(target_sets, length) > 0]

  if (length(target_sets) < 1) {
    cat("[skip] no categories to plot for pair:", pr, "\n")
    next
  }

  cat("\nUpSet set sizes for pair", pr, ":\n")
  print(sapply(target_sets, length))

  upset_data <- fromList(target_sets)

  # Safe file name from the pair label.
  safe_pair <- gsub("[^A-Za-z0-9]+", "_", pr)
  out_file  <- file.path(results_path, paste0("CODA_UpSet_", safe_pair, ".jpeg"))

  jpeg(filename = out_file, width = 12, height = 7, units = "in", res = 300)
  # UpSet must be printed inside the open device.
  print(
    upset(
      upset_data,
      nsets          = length(target_sets),
      nintersects    = NA,
      order.by       = "freq",
      main.bar.color = "steelblue",
      sets.bar.color = "grey40",
      text.scale     = 1.3,
      mb.ratio       = c(0.6, 0.4)
    )
  )
  dev.off()
  cat("  -> saved:", out_file, "\n")
}


# ------------------------------------------------------------------------------
# 5c. AMPLITUDE HISTOGRAM WITH QUARTILES  (kept as-is, over all comparison rows)
# ------------------------------------------------------------------------------
amp_df <- results_clean %>%
  mutate(Amplitude = as.numeric(Amplitude)) %>%
  filter(!is.na(Amplitude))

q <- quantile(amp_df$Amplitude, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)

p_amp <- ggplot(amp_df, aes(x = Amplitude)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  geom_vline(xintercept = q, linetype = "dashed", color = "red", linewidth = 0.7) +
  annotate("text", x = q, y = Inf,
           label = paste0(c("Q1", "Median", "Q3"), "\n", round(q, 3)),
           vjust = 1.3, hjust = -0.1, color = "red", size = 3.5) +
  xlim(0, 1) +
  labs(title = "Distribution of Amplitude",
       subtitle = paste0("Q1 = ", round(q[1], 3),
                         "   Median = ", round(q[2], 3),
                         "   Q3 = ", round(q[3], 3)),
       x = "Amplitude", y = "Count") +
  theme_minimal(base_size = 13)

print(p_amp)

ggsave(filename = file.path(results_path, "Amplitude_Histogram_Quartiles.jpeg"),
       plot = p_amp, width = 8, height = 6, units = "in", dpi = 300)

cat("\n[SUCCESS] Post-processing done. UpSet plots and amplitude histogram saved in:\n",
    results_path, "\n")
