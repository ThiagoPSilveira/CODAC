#' Check, diagnose and repair the value-column layout of a CODAC data frame
#'
#' The CODAC engines read the value columns by POSITION, expecting exactly
#' \code{groups x timepoints x n_observations} columns in group-major order. When
#' samples are physically missing from a file, the column count no longer matches
#' and the analysis stops with an "expected N, found M" error.
#'
#' \code{codac_check_columns()} takes a data frame whose value columns are named
#' \code{Group_ZT<time>_<rep>} (e.g. \code{CON_ZT6_1}), diagnoses the common
#' problems (duplicate names, stray whitespace, unparseable names, group/timepoint
#' typos), then rebuilds the full expected set of columns -- adding any missing
#' sample as \code{NaN} (which the engines mask with \code{missing_data_action =
#' 'KEEP'}) and putting every column back in the exact order the engine expects.
#' It is a convenience wrapper to run once, before \code{codac_compare()} /
#' \code{codac_multi()} (or the single-group tools), on data with missing samples.
#'
#' @param data A data frame: first column = target/gene names, remaining columns =
#'   values named \code{Group_ZT<time>_<rep>}.
#' @param groups Character vector of group names, in the intended column order.
#' @param timepoints Numeric vector of collection times (hours).
#' @param n_observations Number of replicates per timepoint.
#' @param verbose If \code{TRUE} (default), print the diagnostic report.
#'
#' @return The data frame with exactly \code{groups x timepoints x n_observations}
#'   value columns, missing samples filled with \code{NaN}, ordered as the engine
#'   expects. Errors out (without modifying anything) if it finds duplicate,
#'   unparseable, or mislabeled columns, so real-but-misnamed data is never
#'   silently replaced by fake \code{NaN} columns.
#'
#' @export
codac_check_columns <- function(data, groups, timepoints, n_observations,
                                 verbose = TRUE) {
  say <- function(...) if (isTRUE(verbose)) cat(...)

  if (!is.data.frame(data) || ncol(data) < 2)
    stop("`data` must be a data frame with an ID column plus value columns.")

  id_col   <- colnames(data)[1]
  all_cols <- colnames(data)

  # ---- 1. Duplicate column names -------------------------------------------
  dup_names <- unique(all_cols[duplicated(all_cols)])
  if (length(dup_names) > 0)
    say("!!! DUPLICATE COLUMN NAMES (can cause silent column loss):\n  -> ",
        paste(dup_names, collapse = ", "), "\n\n")
  else
    say("OK: no duplicate column names.\n\n")

  # ---- 2. Whitespace in names ----------------------------------------------
  ws_cols <- all_cols[trimws(all_cols) != all_cols]
  if (length(ws_cols) > 0)
    say("!!! COLUMN NAMES WITH LEADING/TRAILING WHITESPACE:\n  -> ",
        paste(ws_cols, collapse = ", "), "\n\n")
  else
    say("OK: no whitespace issues in column names.\n\n")

  # ---- 3. Parse existing value columns -------------------------------------
  # Accept zero-padding in the timepoint and any case of "zt".
  existing_cols <- setdiff(all_cols, id_col)
  parse_col <- function(cn) {
    m <- regmatches(cn, regexec("^(.*)_[Zz][Tt]0*([0-9]+)_([0-9]+)$", cn))[[1]]
    if (length(m) == 4)
      list(group = m[2], timepoint = as.numeric(m[3]), rep = as.numeric(m[4]))
    else
      NULL
  }
  parsed <- lapply(existing_cols, parse_col)
  names(parsed) <- existing_cols
  ok <- !vapply(parsed, is.null, logical(1))

  unparsed_cols <- existing_cols[!ok]
  if (length(unparsed_cols) > 0)
    say("!!! COLUMNS THAT DON'T MATCH Group_ZT<time>_<rep>:\n  -> ",
        paste(unparsed_cols, collapse = ", "),
        "\n  (typo in group name, missing underscore, extra suffix, ...)\n\n")
  else
    say("OK: every value column parses as Group_ZT<time>_<rep>.\n\n")

  parsed_groups <- unique(vapply(parsed[ok], function(x) x$group, character(1)))
  unknown_groups <- setdiff(parsed_groups, groups)
  if (length(unknown_groups) > 0)
    say("!!! COLUMNS USE A GROUP NAME NOT IN `groups`:\n  -> ",
        paste(unknown_groups, collapse = ", "),
        "\n  (typo, or `groups` needs updating to match the file)\n\n")

  parsed_tps <- unique(vapply(parsed[ok], function(x) x$timepoint, numeric(1)))
  unknown_tps <- setdiff(parsed_tps, timepoints)
  if (length(unknown_tps) > 0)
    say("!!! COLUMNS USE A TIMEPOINT NOT IN `timepoints`:\n  -> ",
        paste(unknown_tps, collapse = ", "), "\n\n")

  # ---- 4. Stop on fatal issues (before touching the data) ------------------
  if (length(dup_names) > 0 || length(unparsed_cols) > 0 ||
      length(unknown_groups) > 0 || length(unknown_tps) > 0) {
    stop("Fix the issues flagged above (duplicates / unparseable columns / ",
         "unknown group or timepoint names) BEFORE filling -- otherwise the ",
         "fill would add fake NaN columns for real data that is just misnamed.")
  }

  # ---- 5. Normalize names to the canonical form ----------------------------
  # This makes zero-padded / lowercase-zt columns match the expected names.
  canonical <- vapply(existing_cols, function(cn) {
    p <- parsed[[cn]]
    paste0(p$group, "_ZT", p$timepoint, "_", p$rep)
  }, character(1))
  colnames(data)[match(existing_cols, colnames(data))] <- canonical
  existing_cols <- canonical

  # ---- 6. Expected set, missing -> NaN, reorder ----------------------------
  expected_cols <- unlist(lapply(groups, function(g)
    unlist(lapply(timepoints, function(t)
      paste0(g, "_ZT", t, "_", seq_len(n_observations))))))

  missing_cols <- setdiff(expected_cols, existing_cols)
  extra_cols   <- setdiff(existing_cols, expected_cols)

  say("==================================================================\n")
  say("Expected value columns :", length(expected_cols), "\n")
  say("Existing value columns :", length(existing_cols), "\n")
  say("Missing -> filled NaN   :", length(missing_cols), "\n")
  if (length(missing_cols) > 0) say("  -> ", paste(missing_cols, collapse = ", "), "\n")
  if (length(extra_cols) > 0)
    say("Extra (NOT expected, EXCLUDED from output):", length(extra_cols),
        "\n  -> ", paste(extra_cols, collapse = ", "),
        "\n  (check `n_observations` / group / timepoint settings)\n")
  say("==================================================================\n")

  if (length(missing_cols) > 0) data[missing_cols] <- NaN

  # Output exactly ID + expected columns, in engine order (extras dropped so the
  # positional column count matches what the engine requires).
  data <- data[, c(id_col, expected_cols), drop = FALSE]

  say("[DONE] ", ncol(data) - 1, " value columns (",
      length(expected_cols), " expected).\n")
  data
}
