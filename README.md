# CODAC

**Circadian Oscillation Detection, Analysis and Comparison**

CODAC is a single R package that bundles four tools for circadian time-series
analysis, all sharing one validated Python analysis engine (run from R, no manual
Python setup needed):

| Function | Tool | Use it when… |
|---|---|---|
| `codac_single()` | **Single** | You have **one group** and want per-target rhythmicity (fixed 24 h period). |
| `codac_flex()` | **Flex** | You want the method to **choose the waveform** (standard, linear trend, damped, rapidly damped) — good for bioluminescence and multi-cycle data. |
| `codac_compare()` | **Compare** | You have **two or more groups** and want **pairwise** differential rhythmicity (group 1 vs group 2, etc.). |
| `codac_multi()` | **Multi** | You have **several groups** and want the single best **grouping** of them (which groups share a rhythm / a baseline), à la dryR but on the CODA engine. |

> This is a Python analysis engine wrapped as an R package. **You do not need to
> install Python or any Python library by hand** — the package sets that up
> automatically on first use.

---

## 1. Requirements

- **R** (version 4.0 or newer) and, recommended, **RStudio**.
- An **internet connection** on the first run (to set up the Python engine).
- The R packages `reticulate` and `remotes` (the analysis scripts install them for
  you). The post-processing sections also use `dplyr`, `UpSetR`, and `ggplot2`.

You do **not** need to install Python manually.

---

## 2. Installation

This repository is **private**, so R needs a GitHub access token to download it.

**Step 1 — set up a GitHub token (only once per machine):**
```r
install.packages("usethis")
install.packages("gitcreds")
usethis::create_github_token()   # opens the browser; create and copy the token
gitcreds::gitcreds_set()         # paste the token when prompted
```

**Step 2 — install the package:**
```r
install.packages("reticulate")
install.packages("remotes")
remotes::install_github("ThiagoPSilveira/CODAC", subdir = "CODAC")
```

If told there is a **new version** but the install is skipped ("SHA has not
changed"), force it once:
```r
remotes::install_github("ThiagoPSilveira/CODAC", subdir = "CODAC", force = TRUE)
```

**Step 3 — restart R** (*Session > Restart R*) before running an analysis.

Confirm the install:
```r
library(CODAC)
ls("package:CODAC")            # codac_single, codac_flex, codac_compare, codac_multi
packageVersion("CODAC")
?codac_multi                   # opens the help page
```

> A **404 error** on install almost always means the token is missing or wrong —
> redo Step 1.

---

## 3. Preparing your data

The data file is **tab-delimited** (`.txt`):

- **Column 1:** the target (gene) names.
- **Remaining columns:** the measured values.

For **Single** and **Flex** (one group): the value columns are
`timepoints × replicates`, in order.

For **Compare** and **Multi** (several groups): the value columns are
`groups × timepoints × replicates`, ordered **by group** — for each group, all
timepoints, each with its replicates.

### ⚠️ Group order (Compare / Multi)

The `groups` argument order **must match** the order of the column blocks in the
file, otherwise values are assigned to the wrong group. Group names in `groups`
and `comparisons` must also match **exactly** (watch for stray underscores).

### ⚠️ Decimal separator — the most common mistake

Numbers use **either** a period (`12.34`) **or** a comma (`12,34`); tell R which
via the `dec` argument in the run script. If wrong, numbers are read as text and
every target is skipped (or the column count "explodes"). If results look empty
or the column count is off, switch `dec` between `"."` and `","` and rerun.

### Missing samples — `codac_check_columns()`

The engines read the value columns **by position** and expect exactly
`groups × timepoints × replicates` of them. If some samples are physically absent
from the file, the count no longer matches and the analysis stops with an
"expected N, found M" error.

If your value columns are named `Group_ZT<time>_<rep>` (e.g. `CON_ZT6_1`), run
`codac_check_columns()` once before the analysis. It diagnoses the usual culprits
(duplicate names, stray whitespace, group/timepoint typos), then rebuilds the full
expected set of columns — filling any missing sample with `NaN` (masked by
`missing_data_action = 'KEEP'`) and restoring the exact order the engine needs:

```r
expression_data <- codac_check_columns(
  expression_data,
  groups         = c("CON", "MMI", "T3"),
  timepoints     = c(2, 6, 10, 14, 18, 22),
  n_observations = 4
)
analysis_results <- codac_multi(data = expression_data, ...)
```

It stops (without changing anything) if it finds mislabeled columns, so real data
is never silently replaced by fake `NaN` columns.

---

## 4. Running an analysis

Use the `run_analysis_CODAC_*.R` script for the tool you want. Edit the data path,
the decimal separator, and the parameters, then run. The first run sets up the
Python engine automatically.

### Parameters shared by all tools

| Parameter | Meaning |
|---|---|
| `timepoints` | Collection times in hours, e.g. `c(2, 6, 10, 14, 18, 22)` |
| `n_observations` | Number of replicates per timepoint |
| `r2_threshold` | Minimum R² for a fit to count as good (default `0.4`) |
| `p_threshold` | Significance level for the rhythmicity p-value (default `0.05`) |
| `p_value_option` | Multiple-testing for the **rhythmicity** p-value: `'FDR'` (Benjamini-Hochberg) or `'RAW'` (default `'FDR'`) |
| `amp_stringency` | Amplitude-filter strictness, `0`–`1` (default `0.5`) — see §5 |
| `min_rhythmicity` | Minimum rhythmicity tier a target must reach to be kept (default `'HIGH'`) |
| `missing_data_action` | `'KEEP'` (mask NaNs, default), `'IMPUTE'` (fill from surviving replicates) or `'REMOVE'` |
| `plot_flag` / `plot_all` / `targets_to_plot` | Whether to draw per-target plots, and for which targets |
| `time_label` | X-axis label: `'ZT'`, `'CT'`, or `'Clock'` |

### Tool-specific parameters

**Flex:** `period_mode` (`'fixed'` or `'variable'`), `fixed_period` (default `24`),
`period_lower`/`period_upper` (variable-mode bounds, default `20`/`28`),
`interval_var` (`1`/`2`/`3`).

**Compare / Multi:** `groups`, `comparisons` (list of pairs, or `NULL` for all
pairs), `rhythmicity_cutoff` (tier at which a group counts as rhythmic, default
`'HIGH'`), `exclude_medium` (drop MEDIUM targets before comparing, default `TRUE`),
`p_value_comparison` (source of the **pairwise** decision p-values: `'RAW'` or
`'FDR'`, default `'RAW'`).

**Multi only:** `selection_criterion` — the information criterion for the grouping
selection: `'BIC'` (default, conservative) or `'AICc'` (more sensitive). See §6.5.
`p_value_global` — which p-value gates the grouping: `'FDR'` (default,
Benjamini-Hochberg across all targets) or `'RAW'`. See §6.5.

---

## 5. The amplitude filter (`amp_stringency`)

A rhythm is only trusted if its amplitude stands out from noise — a target can be
statistically significant yet oscillate too little to matter biologically. CODAC
applies an **adaptive** per-target threshold (reported as `Amp. Minimum`), based
on the target's expression level and variability, with an absolute noise floor.
One dial controls how demanding it is:

| `amp_stringency` | Effect |
|---|---|
| `0.0` | Filter off — amplitude never rejects |
| `0.5` | Default, validated |
| `1.0` | Strictest — requires twice the default amplitude |

A target passes the amplitude criterion when `Amplitude ≥ Amp. Minimum`.

---

## 6. Understanding the output

Results are written to a **`CODA_Results`** folder next to your data (an Excel
table + a `plots/` subfolder), and the same table is returned in R as
`analysis_results`.

### 6.1 Columns produced by every tool (the per-target fit)

These describe the fitted rhythm of each target (per group, for Compare/Multi):

| Column | Meaning |
|---|---|
| `Target` | The gene/target name (case preserved from your file). |
| `Mesor` | The rhythm-adjusted mean — the baseline the oscillation sits on. Not the same as a simple average when data are uneven. |
| `Amplitude` | The size of the oscillation. For Single/Compare/Multi it is the **peak-to-mesor distance** of the fitted curve (equals the cosine amplitude for a pure rhythm); for Flex it is the fitted amplitude parameter of the winning model (see §6.3). |
| `Amp. Minimum` | The adaptive amplitude threshold used for this target (see §5). A target passes when `Amplitude ≥ Amp. Minimum`. |
| `Phase` | The **acrophase in decimal hours** — the time of the fitted peak (e.g. `13.75`). This is the value used in all downstream math. |
| `Phase (h:min)` | The same acrophase in **clock format**, where digits after the dot are minutes (`13.45` = 13 h 45 min). Display only — never do arithmetic on it. |
| `Period` | The period in hours. Fixed at 24 for Single/Compare/Multi; fitted by Flex in variable-period mode. |
| `R2` | Coefficient of determination — how well the fitted curve tracks the data (0–1). |
| `P-value` | Rhythmicity significance from a nested F-test (fitted cosinor vs a flat line). Raw value. |
| `P-value (FDR)` | The Benjamini-Hochberg–adjusted rhythmicity p-value (across all fits). Which of `P-value`/`P-value (FDR)` drives the classification is set by `p_value_option`. |
| `Interval` | `In` / `Out` flag of the waveform-prominence test: whether the fitted curve sweeps **beyond** the inter-percentile band of the observed data (`Out` = prominent oscillation). |
| `Probability` | The rhythmicity tier, from a 0–4 count of criteria met (significance, R², amplitude, prominence): `EXTREMELY HIGH` (4), `HIGH` (3), `MEDIUM` (2), `LOW` (1), `ARRHYTHMIC` (0). This multi-criteria tier — not the p-value alone — is CODAC's rhythmicity call. |

> **How rhythmicity is decided.** Rather than trusting the p-value alone (which
> over-calls rhythms in large datasets), CODAC scores four independent criteria —
> significance, effect size (R²), a biologically meaningful amplitude, and
> waveform prominence — and reports how many were met as the `Probability` tier.

### 6.2 CODAC_Single

Single produces exactly the columns in §6.1, one row per target. Nothing extra.

### 6.3 CODAC_Flex (extra columns)

Flex fits four models per target and picks the best by AICc, adding:

| Column | Meaning |
|---|---|
| `Winning_Model` | Which model best described the target: `standard` (constant amplitude), `linear` (with a baseline trend), `dampened` (amplitude decaying exponentially), or `dampened_fast` (decaying faster than exponential). |
| `AICc` | The corrected Akaike Information Criterion of the winning model. AICc balances fit against the number of parameters, so a more complex model wins only if it earns it. It is **relative** — meaningful only comparing the four models of the *same* target, not across targets, and not an absolute goodness measure (that is R²/`Probability`). |
| `Flex_Parameter` | The extra coefficient of the winning model: the slope for `linear`, the decay rate for the damped models (0 for `standard`). |
| `Half_Life` | For the damped models only: the time for the amplitude to fall to half its initial value. Empty for `standard`/`linear`. |

Because the linear and damped models change the waveform, Flex reports the fitted
amplitude parameter **A** directly (rather than a peak-to-mesor distance), so the
baseline trend or the decay does not inflate the reported amplitude.

### 6.4 CODAC_Compare (pairwise columns)

For each target, Compare gives the per-group fits (§6.1) plus one row per
**pairwise comparison**, carrying:

| Column | Meaning |
|---|---|
| `Pair` | The two groups compared, as `Group 1 vs Group 2`. |
| `Rhythm_Status` | Which groups are rhythmic: `Both rhythmic`, `Group 1 only`, `Group 2 only`, `Neither rhythmic`. |
| `Biological_Category` | The rhythm-change category (see the table below). |
| `Mesor_Change` | The **baseline** comparison, reported separately from rhythm: `Different`, `Conserved`, or `Undetermined`. |
| `Delta_Mesor`, `Delta_Amplitude`, `Delta_Phase` | The change in each component, computed as **Group 1 − Group 2** (matching the `Group 1 vs Group 2` label). A positive value means the first group is higher. |
| `p_diff_mesor`, `p_diff_amplitude`, `p_diff_phase` | The raw p-value for a difference in each component (nested NLS F-test). |
| `p_diff_mesor_FDR`, `p_diff_amplitude_FDR`, `p_diff_phase_FDR` | The Benjamini-Hochberg–adjusted version of each, applied **per component within each pair**. `p_value_comparison` chooses which set (raw or FDR) drives the decisions; both are always exported so you can compare. |
| `LossGain_Confidence` | For `Cat 2`/`Cat 3` only (rhythmic in one group): `High confidence` when the amplitude also differs significantly, `Weak evidence` otherwise — a guard against over-calling a rhythm gain/loss (cf. Pelikan et al., 2022). |

Amplitude and phase are compared **only when both groups are rhythmic**; the mesor
is handled separately (`Mesor_Change`), so it does not enter these categories:

| Category | Meaning |
|---|---|
| `Cat 1: Arrhythmic` | Neither group is rhythmic |
| `Cat 2: rhythmic_group_1_only` | Only the first group is rhythmic |
| `Cat 3: rhythmic_group_2_only` | Only the second group is rhythmic |
| `Cat 4: rhythmic_both_unchanged` | Both rhythmic; amplitude and phase unchanged |
| `Cat 5: rhythmic_with_changes_only_amp` | Both rhythmic; amplitude differs |
| `Cat 6: rhythmic_with_changes_only_phase` | Both rhythmic; phase differs |
| `Cat 7: rhythmic_with_changes_amp_phase` | Both rhythmic; amplitude and phase differ |

### 6.5 CODAC_Multi (global tests + grouping)

Multi keeps everything Compare produces (the per-group fits and the pairwise
comparisons) and adds a **target-level** view: three global tests and the best
**grouping** of the groups on two independent axes.

**The three global tests** (one value per target):

| Column | Question it answers |
|---|---|
| `p_global_rhythm` | Does *any* group show a rhythm? (small = at least one group is rhythmic) |
| `p_rhythm_diff` | Do the rhythms *differ between groups*? (small = the rhythm is not the same everywhere) |
| `p_mesor_diff` | Do the *baselines* differ between groups? (small = at least one group has a different mesor) |

Each of these is also reported Benjamini-Hochberg–corrected across all targets, in
a companion `_FDR` column (`p_global_rhythm_FDR`, `p_rhythm_diff_FDR`,
`p_mesor_diff_FDR`). Because these tests run once per target over thousands of
targets, the corrected value is the right one for a genome-wide screen. The
`p_value_global` parameter chooses which drives the grouping **gate** —
`'FDR'` (default) or `'RAW'` — while both columns are always exported. With the
default, a target whose rhythm difference is significant raw but not after
correction is folded back into "shared rhythm" rather than split.

**The grouping** — the single best configuration of which groups share the same
rhythm, and, separately, which share the same baseline:

| Column | Meaning |
|---|---|
| `Grouping` | The winning **rhythm** grouping. Which groups are **rhythmic** is decided first, by CODA's own per-group multi-criteria tier: a group counts as rhythmic only if its `Probability` reaches `rhythmicity_cutoff` (so a `LOW` group is treated as arrhythmic, the same bar `codac_compare()` uses). Then, among the rhythmic groups, model selection decides **how they share** the rhythm: `{G1,G2} != {G3}` (G1, G2 share it, G3 differs), `{G1,G2,G3} (all equal)` / `All groups rhythmic (shared rhythm)`, `{G1} ; arrhythmic: G2,G3` (only G1 rhythmic), or `All groups arrhythmic` (none reach the cutoff). If two or more groups are rhythmic but the omnibus `p_rhythm_diff` could not be computed, it reads `Undetermined (insufficient data)`. |
| `Grouping_Model` | A short, stable code for the winning rhythm model (`M01`–`M15` for three groups), for easy filtering in R. See the legend below; empty when the grouping is `Undetermined`. |
| `Grouping_Confidence` | The strength of evidence for that grouping, in `[0, 1]` (the criterion weights of all candidate models sum to 1). Near 1 = decisive; low = the top models were close (an uncertain call). `NA` when no grouping was searched. |
| `Grouping_IC_Gap` | The information-criterion margin to the runner-up model. A small gap is another sign of a close call. |
| `Grouping_Mesor`, `Grouping_Mesor_Confidence`, `Grouping_Mesor_IC_Gap` | Exactly the same three, but for the **baseline (mesor)** axis, gated by `p_mesor_diff`. Its "no difference" label is `All groups equal (same baseline)`, and `Undetermined (insufficient data)` when the mesor omnibus could not be computed. |
| `Grouping_Mesor_Model` | The stable code for the winning mesor model (`MM1`–`MM5` for three groups); see the legend below. |

The grouping is chosen by **model selection**: CODAC fits every possible
configuration of rhythm-sharing across the groups and keeps the one with the best
information criterion, set by `selection_criterion`:

- **`BIC`** (default) — penalizes complexity more strongly, so it avoids splitting
  groups that are actually the same. More conservative; recommended.
- **`AICc`** — more sensitive to subtle differences, at a higher false-positive
  risk.

Because the criterion is a *relative* comparison of models, `Grouping_Confidence`
(not a p-value) is its measure of certainty; the `p_*_diff` gates above are the
companion hypothesis tests. The two axes are independent, so a target can have an
identical rhythm across groups while its baseline splits (or vice versa).

> **Reading a Multi row, in three steps:** `p_rhythm_diff` tells you *whether*
> there is a rhythm difference; `Grouping` tells you *what* the structure is; and
> `Grouping_Confidence` tells you *how much to trust* that structure. A significant
> `p_rhythm_diff` with a low confidence means "there is a difference, but the exact
> grouping is uncertain" — worth checking the pairwise rows for that target.

#### Model legend (three groups)

The `Grouping_Model` / `Grouping_Mesor_Model` codes are stable for a **fixed
number of groups**. For three groups the rhythm axis has 15 models and the mesor
axis 5 (`G1`, `G2`, `G3` in the order given in `groups`; `!=` separates
different-rhythm blocks; `arr` = arrhythmic in that model):

| Rhythm | Meaning | | Rhythm | Meaning |
|---|---|---|---|---|
| `M01` | arrhythmic in all | | `M09` | {G2,G3} rhythmic; G1 arr |
| `M02` | {G1} rhythmic; G2,G3 arr | | `M10` | {G2} != {G3}; G1 arr |
| `M03` | {G2} rhythmic; G1,G3 arr | | `M11` | all rhythmic, one shared rhythm |
| `M04` | {G3} rhythmic; G1,G2 arr | | `M12` | {G1} != {G2,G3} |
| `M05` | {G1,G2} rhythmic; G3 arr | | `M13` | {G1,G2} != {G3} |
| `M06` | {G1} != {G2}; G3 arr | | `M14` | {G1,G3} != {G2} |
| `M07` | {G1,G3} rhythmic; G2 arr | | `M15` | {G1} != {G2} != {G3} |
| `M08` | {G1} != {G3}; G2 arr | | | |

| Mesor | Meaning |
|---|---|
| `MM1` | all groups share one baseline |
| `MM2` | {G1} != {G2,G3} |
| `MM3` | {G1,G2} != {G3} |
| `MM4` | {G1,G3} != {G2} |
| `MM5` | {G1} != {G2} != {G3} |

Filter in R by code, e.g. keep the "one group loses the rhythm" cases:
`analysis_results %>% filter(Grouping_Model %in% c("M05","M07","M09"))`.

> **How the grouping relates to the per-group tiers and the global tests.** The
> `Grouping` deliberately follows CODA's per-group `Probability` tier for the
> "who is rhythmic" question: a group is placed in a rhythmic block only if it
> reaches `rhythmicity_cutoff`, so a weakly-oscillating `LOW` group is treated as
> arrhythmic even if a bare significance test would flag it. This means
> `p_global_rhythm` (a pure "is there any rhythm?" p-value) can be significant
> while the `Grouping` still reads `All groups arrhythmic` — that is expected, not
> a contradiction: `p_global_rhythm` only asks whether *some* oscillation is
> statistically detectable, whereas the grouping requires a rhythm strong enough
> to pass CODA's multi-criteria bar. The model-selection step (and `p_rhythm_diff`)
> then only decides how the groups that *are* rhythmic share their rhythm.

---

## 7. Plots

In the `CODA_Results/plots/` folder:

- **Per-target plots** — the fitted curve(s) over the data (all groups overlaid
  for Compare/Multi; annotated with the winning model for Flex).
- **Heatmaps** — rhythmic targets ordered by acrophase; for Compare/Multi, also
  per-category heatmaps with two panels (one per group, same targets/order,
  z-scored so only amplitude/phase show). Per-gene labels are drawn only when a
  panel has at most 50 targets, to keep large heatmaps readable.
- **Summary figures** — R² and amplitude distributions, R²-vs-p-value scatter, and
  a phase rose (polar) plot.
- **Flex** adds a pie chart of how often each model won.

---

## 8. Post-processing (Compare / Multi)

The lower part of the `run_analysis_CODAC_Compare.R` / `_Multi.R` scripts runs
extra analyses on `analysis_results` (plain R — edit and rerun freely; the helper
packages `dplyr`, `UpSetR`, `ggplot2` install themselves on first use):

- **Category selection** — builds nested lists of the target blocks per pair and
  per category (further split by `Mesor_Change`), for inspection with `View()`.
- **UpSet plot, one per pair** — how targets distribute across the 7 categories.
- **Amplitude histogram with quartiles** — the basis of the amplitude filter.

### Filtering the results

Each target spans several rows (one per group, plus the comparison rows), and
`Biological_Category`/`Mesor_Change` appear only on the comparison rows. Filter
while keeping the whole target by grouping on `Target`:

```r
library(dplyr)
analysis_results %>%
  group_by(Target) %>%
  filter(any(Biological_Category == "Cat 4: rhythmic_both_unchanged")) %>%
  ungroup()
```

Use the exact category names from §6.4. When there are several pairs, filter by
`Pair` as well.

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| **404 error** on install | Private repo + missing token → redo the token setup (§2, Step 1). |
| **Every target "insufficient valid data"** or column count off | Wrong decimal separator → switch `dec` between `"."` and `","`. |
| **Results assigned to the wrong group** | `groups` order does not match the column order in the file → fix the order. |
| **A comparison/heatmap is missing** | A group name in `comparisons` does not exactly match the data → align the names. |
| **`could not find function "codac_multi"`** | Package not loaded → `library(CODAC)`. |
| **Changes not showing up** | Old version still loaded → restart R after installing (use `force = TRUE`), confirm with `packageVersion("CODAC")`. |
| **Rtools warning on install** | Harmless — CODAC has no C/Fortran to compile; the install still completes (`* DONE (CODAC)`). |

---

*CODAC is under active validation. Please report issues or unexpected results to:
thiago.silveira@gp.ita.br*
