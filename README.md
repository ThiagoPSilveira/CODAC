# CODAC

**Circadian Oscillation Detection, Analysis and Comparison**

CODAC is a single R package that bundles four tools for circadian time-series
analysis, all sharing one validated Python analysis engine (run from R, no manual
Python setup needed):

| Function | Tool | Use it when… |
|---|---|---|
| `codac_single()` | **Single** | You have **one group** and want per-target rhythmicity (fixed 24 h period). |
| `codac_flex()` | **Flex** | You want the method to **choose the waveform** (standard, linear trend, damped, rapidly damped) |
| `codac_compare()` | **Compare** | You have **two or more groups** and want **pairwise** differential rhythmicity (group 1 vs group 2, etc.). |
| `codac_multi()` | **Multi** | You have **several groups** and want the single best **grouping** of them (which groups share a rhythm / a baseline) |

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

### Group order (Compare / Multi)

The `groups` argument order **must match** the order of the column blocks in the
file, otherwise values are assigned to the wrong group. Group names in `groups`
and `comparisons` must also match **exactly** (watch for stray underscores).

### Decimal separator — the most common mistake

Numbers use **either** a period (`12.34`) **or** a comma (`12,34`); tell R which
via the `dec` argument in the run script. If wrong, numbers are read as text and
every target is skipped (or the column count "explodes"). If results look empty
or the column count is off, switch `dec` between `"."` and `","` and rerun.

### Missing samples — `codac_check_columns()`

This function reads the value columns **by position** and expect exactly
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
| `p_value_option` | Multiple-testing for the **rhythmicity** p-value: `'FDR'` (Benjamini-Hochberg) or `'RAW'` (default `'FDR'`). Each `p_value_*` option can also carry its **own alpha** as a `c(method, alpha)` pair — e.g. `p_value_option = c('FDR', 0.1)` — instead of sharing the single `p_threshold` |
| `amp_stringency` | Amplitude-filter strictness, `0`–`1` (default `0.5`) — see §5 |
| `min_rhythmicity` | Minimum rhythmicity tier a target must reach to be kept (default `'HIGH'`) |
| `missing_data_action` | `'KEEP'` (mask NaNs, default), `'IMPUTE'` (fill from surviving replicates) or `'REMOVE'` |
| `plot_flag` / `plot_all` / `targets_to_plot` | Whether to draw per-target plots, and for which targets |
| `time_label` | X-axis label: `'ZT'`, `'CT'`, or `'Clock'` |
| `interval_var` | (`1`/`2`/`3`). Sets the width of the noise band used by the waveform prominence criterion. 

### Tool-specific parameters

**Flex:** `period_mode` (`'fixed'` or `'variable'`), `fixed_period` (default `24`),
`period_lower`/`period_upper` (variable-mode bounds, default `20`/`28`). 

**Compare / Multi:** `groups`, `comparisons` (list of pairs, or `NULL` for all
pairs), `rhythmicity_cutoff` (tier at which a group counts as rhythmic, default
`'HIGH'`), `exclude_medium` (drop MEDIUM targets before comparing, default `TRUE`),
`p_value_comparison` (source of the **pairwise** decision p-values: `'RAW'` or
`'FDR'`, default `'RAW'`).

**Multi only:** `selection_criterion` — the information criterion for the grouping
selection: `'BIC'` (default, conservative) or `'AICc'` (more sensitive). See §6.5.
`p_value_global` — which p-value gates the grouping: `'FDR'` (default,
Benjamini-Hochberg across all targets) or `'RAW'`. See §6.5.
`rhythm_diff_correction` — how the `p_rhythm_diff` gate is FDR-corrected:
`'all_targets'` (default, genome-wide) or `'screened_pooled'` (two-stage — BH only
among targets passing an orthogonal pooled shared-rhythm screen, which recovers the
power a genome-wide correction loses when most targets are arrhythmic). See §6.5.
`permute_B` — diagnostic only (default `0` = off): number of label permutations
used to estimate the rhythm-difference gate's **empirical FDR** inside CODAC's own
engine. Writes `rhythm_diff_calibration.csv`. Slow, so use a modest `B` (e.g. 100).

---

## 5. The amplitude filter (`amp_stringency`)

CODAC applies an **adaptive** per-target threshold (reported as `Amp. Minimum`), based
on the target's expression level and variability, with an absolute noise floor, as shown below: 

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
| `Target` | The target name (case preserved from your file). |
| `Mesor` | The rhythm-adjusted mean. |
| `Amplitude` | The size of the oscillation. For Single/Compare/Multi it is the **peak-to-mesor distance** of the fitted curve; for Flex it is the fitted amplitude parameter of the winning model (see §6.3). |
| `Amp. Minimum` | The adaptive amplitude threshold used for this target (see §5). A target passes when `Amplitude ≥ Amp. Minimum`. |
| `Phase` | The **acrophase in decimal hours** — the time of the fitted peak (e.g. `13.75`). |
| `Phase (h:min)` | Acrophase in **clock format** (`13.45` = 13 h 45 min). |
| `Period` | The period in hours. Fixed at 24 for Single/Compare/Multi; fitted by Flex in variable-period mode. |
| `R2` | Goodness of fit (0–1). |
| `P-value` | Rhythmicity significance from a nested F-test (fitted cosinor vs a flat line). Raw value. |
| `P-value (FDR)` | The Benjamini-Hochberg–adjusted rhythmicity p-value (across all fits). |
| `Interval` | `In` / `Out` flag of the waveform-prominence test: whether the fitted curve sweeps **beyond** the inter-percentile band of the observed data (`Out` = prominent oscillation). |
| `Probability` | The rhythmicity tier, from a 0–4 count of criteria met: `EXTREMELY HIGH` (4), `HIGH` (3), `MEDIUM` (2), `LOW` (1), `ARRHYTHMIC` (0). |

> **How rhythmicity is decided.** Rather than trusting the p-value alone, CODAC scores four independent criteria: 
> significance, effect size (R²), a meaningful amplitude, and
> waveform prominence, and reports how many were met as the `Probability` tier.

### 6.2 CODAC_Single

Single produces exactly the columns in §6.1, one row per target. Nothing extra.

### 6.3 CODAC_Flex (extra columns)

Flex fits four models per target and picks the best by AICc, adding:

| Column | Meaning |
|---|---|
| `Winning_Model` | Which model best described the target: `standard` (constant amplitude), `linear` (with a baseline trend), `damped` (amplitude decaying exponentially), or `damped_fast` (decaying faster than exponential). |
| `AICc` | The corrected Akaike Information Criterion of the winning model. 
| `Flex_Parameter` | The extra coefficient of the winning model: the slope for `linear`, the decay rate for the damped models (0 for `standard`). |
| `Half_Life` | For the damped models only: the time for the amplitude to fall to half its initial value. Empty for `standard`/`linear`. |

Because the linear and damped models change the waveform, Flex reports the fitted
amplitude parameter **A** directly (rather than a peak-to-mesor distance).

### 6.4 CODAC_Compare (pairwise columns)

For each target, Compare gives the per-group fits (§6.1) plus one row per
**pairwise comparison**, carrying:

| Column | Meaning |
|---|---|
| `Pair` | The two groups compared, as `Group 1 vs Group 2`. |
| `Rhythm_Status` | Which groups are rhythmic: `Both rhythmic`, `Group 1 only`, `Group 2 only`, `Neither rhythmic`. |
| `Biological_Category` | The rhythm-change category (see the table below). |
| `Mesor_Change` | The **baseline** comparison, reported separately from rhythm: `Different`, `Conserved`, or `Undetermined`. |
| `Delta_Mesor`, `Delta_Amplitude`, `Delta_Phase` | The change in each component, computed as **Group 1 − Group 2** (matching the `Group 1 vs Group 2` label). |
| `p_diff_mesor`, `p_diff_amplitude`, `p_diff_phase` | The raw p-value for a difference in each component (nested NLS F-test). |
| `p_diff_mesor_FDR`, `p_diff_amplitude_FDR`, `p_diff_phase_FDR` | The Benjamini-Hochberg–adjusted version of each, applied **per component within each pair**. `p_value_comparison` chooses which set (raw or FDR) drives the decisions. |
| `LossGain_Confidence` | For `Cat 2`/`Cat 3` only (rhythmic in one group): `High confidence` when the amplitude also differs significantly, `Weak evidence` otherwise. 

Phase is compared **only when both groups are rhythmic**. The mesor
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

### 6.5 CODAC_Multi (global tests and grouping)

Multi keeps everything Compare produces and adds a **target-level** view: four
global tests, and the best **grouping** of the experimental groups on two axes,
rhythm and mesor. 

#### Global tests

Each test returns one value per target.

| Column | Question it answers |
|---|---|
| `p_global_rhythm` | Does *any* group show a rhythm? Small = at least one group is rhythmic. |
| `p_pooled_rhythm` | Does a rhythm *common to the groups* exist? A shared-rhythm test on 2 df, used to screen the correction family (see below). |
| `p_rhythm_diff` | Do the rhythms *differ between groups*? Small = the rhythm is not the same everywhere. |
| `p_mesor_diff` | Do the *baselines* differ between groups? Small = at least one group has a different mesor. |

Each test is also reported Benjamini–Hochberg–corrected across all targets, in a
companion `_FDR` column. Both columns are always exported. The `p_value_global`
parameter only decides which of the two drives the grouping gate, `'FDR'`
(default) or `'RAW'`.

#### Recovering power with a screened correction (`rhythm_diff_correction`)

Correcting `p_rhythm_diff` for all targets may be over-conservative, especially if
the majority of the targets are arrhythmic. The `'screened_pooled'` mode addresses
this with a two-stage design. Targets first have to pass a BH-corrected screen on
the pooled shared-rhythm test (`p_pooled_rhythm`, 2 df). Those that pass form the
family within which `p_rhythm_diff` is then BH-corrected, reported as
`p_rhythm_diff_FDR_screened`. Screening in this manner does not bias the child test,
since the pooled test is orthogonal to the group-by-rhythm interaction measured by
`p_rhythm_diff`. The genome-wide `p_rhythm_diff_FDR` is
still reported, and is the value that drives the gate when
`rhythm_diff_correction = 'all_targets'` (the default). Companion columns are
`rhythm_screen_pass`, `rhythm_diff_family_size`, `p_rhythm_diff_FDR_screened`, and
`rhythm_diff_gate_used`.

The `permute_B` diagnostic permutes group labels within each timepoint — a null in
which the groups share the rhythm — to estimate the empirical FDR of the gate in
CODAC's own engine, written to `rhythm_diff_calibration.csv`.

#### The grouping

The grouping is the single best configuration of which groups share the same rhythm
and, separately, which share the same mesor.

Which groups count as rhythmic is decided first, by CODA's own per-group
multi-criteria tier. A group enters a rhythmic block only if its `Probability`
reaches `rhythmicity_cutoff`, the same bar `codac_compare()` uses, so a `LOW` group
is treated as arrhythmic. Model selection then decides only *how* the rhythmic groups share the rhythm.

> Importantly, `p_global_rhythm` can be significant while `Grouping` still reads
> `All groups arrhythmic`. That is expected, as the p-value asks only whether *some* oscillation is statistically detectable, whereas
> the grouping requires a rhythm strong enough to pass CODA's multi-criteria bar.

| Column | Meaning |
|---|---|
| `Grouping` | The winning rhythm grouping, written as blocks of groups that share a rhythm, separated by `!=`, with any arrhythmic groups listed after a semicolon: `{G1,G2} != {G3}`, `{G1,G2} != {G3,G4} ; arrhythmic: G5`, `{G1} ; arrhythmic: G2,G3`. The two extremes have their own labels, `All groups rhythmic (shared rhythm)` and `All groups arrhythmic`. Reads `Undetermined (insufficient data)` when two or more groups are rhythmic but the omnibus `p_rhythm_diff` could not be computed. |
| `Grouping_Model` | A code for the winning rhythm model. See the legend below. Empty when the grouping is `Undetermined`. |
| `Grouping_Confidence` | Strength of evidence for that grouping, in `[0, 1]`, since the criterion weights of all candidate models sum to 1. Near 1 = decisive; low = the top models were close. `NA` when no grouping was searched. |
| `Grouping_IC_Gap` | The information-criterion margin to the runner-up model. A small gap is another sign of a close call. |
| `Grouping_Mesor`, `Grouping_Mesor_Confidence`, `Grouping_Mesor_IC_Gap` | The same three columns for the **baseline (mesor)** axis, gated by `p_mesor_diff`. Its no-difference label is `All groups equal (same baseline)`, and `Undetermined (insufficient data)` when the mesor omnibus could not be computed. |
| `Grouping_Mesor_Model` | A code for the winning mesor model; see the legend below. |

#### How the grouping is chosen (`selection_criterion`)

CODAC fits every possible configuration of rhythm-sharing across the groups and
keeps the one with the best information criterion.

- **`BIC`** (default) penalizes complexity more strongly, so it avoids splitting
  groups that are actually the same. Use it for genome-wide screens, and whenever
  avoiding false "groups differ" calls matters more than catching every subtle one.
- **`AICc`** is more sensitive to small amplitude and phase differences, at a higher
  false-positive rate. Use it for targeted analyses of a handful of candidate
  targets.

Where the two criteria would disagree on a target, its `Grouping_Confidence` and
`Grouping_IC_Gap` tend to be low anyway, which is a built-in signal that the call is
borderline. Because the criterion compares models relative to each other,
`Grouping_Confidence` rather than a p-value is its measure of certainty; the
`p_*_diff` gates above are the companion hypothesis tests.

The model space grows quickly with the number of groups, so the search is best kept
to the groups a question actually needs. With `k` groups the rhythm axis has
`Bell(k+1)` models and the mesor axis `Bell(k)`:

| Groups | Rhythm models | Mesor models |
|---|---|---|
| 2 | 5 | 2 |
| 3 | 15 | 5 |
| 4 | 52 | 15 |
| 5 | 203 | 52 |
| 6 | 877 | 203 |

> **Reading a Multi row.** Read the row in three steps. `p_rhythm_diff` tells you
> *whether* there is a rhythm difference, `Grouping` tells you *what* the structure
> is, and `Grouping_Confidence` tells you *how much to trust* that structure. A
> significant `p_rhythm_diff` with a low confidence means there is a difference but
> the exact grouping is uncertain, which is worth checking against the pairwise rows
> for that target.

#### `Biological_Category` follows the grouping

In Compare the category is decided pair by pair. In Multi the grouping decides it,
so that a target is never called *shared* by its grouping and *changed* by one of
its pairs. For any pair in which both groups are rhythmic:

- both in the **same** rhythm block → **Cat 4** (unchanged)
- in **different** rhythm blocks → **Cat 5/6/7** (changed), with the component taken
  from the pairwise amplitude and phase p-values

Every pair of a model is categorized on this rule, so one target can carry both. In
a model such as `{G1,G2} != {G3}`, the pair G1–G2 is Cat 4 while G1–G3 and G2–G3 are
Cat 5/6/7. A model in which every rhythmic group sits in one block is Cat 4 on all of
its pairs, and a model that puts each rhythmic group in its own block is Cat 5/6/7 on
all of them. Pairs involving an arrhythmic group keep their loss/gain category and
are untouched by the grouping.

The grouping can therefore override an isolated pairwise test: a pair sitting in the
same block is Cat 4 even when its raw amplitude or phase p-value was significant.
Nothing is discarded — `Delta_Amplitude`, `Delta_Phase`, `p_diff_amplitude`, and
`p_diff_phase` are written unchanged, so the pairwise result stays visible next to
the reconciled category.

#### When `Grouping_Confidence` is a number

`Grouping_Confidence` is the weight of the model-selection search, which requires two or more rhythmic
groups whose rhythms differ (`p_rhythm_diff` past the gate). By model:

- **Always `NA`** — the model with none or one rhythmic group.
- With none or one rhythmic group there is no shared-versus-split
  question to weigh, so there is nothing to score. 
- **Always a number** — every model that places two or more rhythmic groups in
  different blocks. 
- **Either** — every model in which all the rhythmic groups sit in a single shared
  block. A number when the search ran and chose shared; `NA` when the gate was closed
  and sharing was assumed without a search.

An `NA` confidence does not mean missing information. It means the structure was
decided without a model-selection contest. The per-component numbers always live in
the pairwise columns, whatever the grouping says: `Biological_Category`,
`p_diff_amplitude` / `p_diff_amplitude_FDR`, `p_diff_phase` / `p_diff_phase_FDR`,
`Delta_Amplitude`, and `Delta_Phase` for whether and by how much amplitude and phase
differ between each pair; and `LossGain_Confidence`, for the models with exactly one
rhythmic group, for the confidence that a rhythm was gained or lost.

Use `Grouping` and `Grouping_Model` for the structure, and the pairwise columns for
the per-component numbers, including when the grouping confidence is `NA`.

#### Model legend

A rhythm model is one assignment of the groups to rhythmic blocks, plus the set left
arrhythmic. A mesor model is one partition of all the groups into baseline blocks,
with no arrhythmic option, since every group has a baseline. The codes are `M01`,
`M02`, … on the rhythm axis and `MM1`, `MM2`, … on the mesor axis, numbered in a
fixed order:

1. by the number of rhythmic groups, from none to all (rhythm axis only);
2. then by the groups in that rhythmic set, in the order given in `groups`;
3. then from the coarsest partition of that set (one shared block) to the finest
   (every group on its own).

The codes are stable for a fixed number of groups, so they can be compared across
targets and across runs of the same design. They are **not** stable across designs:
`M07` in a three-group run and `M07` in a four-group run are different models, so any
filter written on codes has to be rewritten when the number of groups changes.

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

Because the codes shift with the number of groups, filter on the grouping
*structure* rather than on codes wherever the design might change. To keep the "one
group loses the rhythm" cases — that is, every model in which the rhythmic groups
share one block and at least one group is arrhythmic:

```r
# structural filter, works for any number of groups
analysis_results %>%
  filter(grepl("arrhythmic", Grouping), !grepl("!=", Grouping))

# equivalent code filter, three groups only
analysis_results %>%
  filter(Grouping_Model %in% c("M05", "M07", "M09"))
```
## 7. Plots

In the `CODA_Results/plots/` folder:

- **Per-target plots** — the fitted curve(s) over the data (all groups overlaid
  for Compare/Multi; annotated with the winning model for Flex).
- **Heatmaps** — rhythmic targets ordered by acrophase, z-scored row-wise so only
  amplitude and phase show (the mesor is removed by construction). Per-gene labels
  are drawn only when a panel has at most 50 targets.
  - **Compare** draws per-category heatmaps (two panels, one per group).
  - **Multi** draws **model-based** heatmaps instead: one per grouping model
    (`heatmap_model_M02.png` …), with all groups side by side and genes ordered by
    the acrophase of the model's rhythmic group (`M01` is ordered by the mesor
    difference). A **consolidated** heatmap (`heatmap_consolidated.png`) stacks
    models M02–M15 vertically with the groups as columns, and everything is also
    bundled into a single **`CODAC_Multi_heatmaps.pdf`**.
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
