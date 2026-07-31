import os
import csv
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import re
import statsmodels.formula.api as smf
import sys
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.svm import SVC
from tqdm import tqdm
from statsmodels.stats.anova import anova_lm
from itertools import combinations
from scipy.optimize import OptimizeWarning

# ================================================================
# SPYDER VARIABLE EXPLORER — accessible objects after running
# ================================================================
df_results     = None
df_comparisons = None
df_expression  = None
fig_r2_high    = None
fig_r2_low     = None
fig_r2_all     = None
fig_r2_pvalue  = None
fig_amplitude  = None
fig_polar      = None
fig_heatmap    = None

# ==================================================================
# AMPLITUDE FILTER — single stringency dial (0.0 to 1.0)
# ==================================================================
# A rhythm is only trusted if its amplitude is large enough to stand out from
# noise. This one parameter sets how demanding that test is:
#
#   amp_stringency = 0.0  -> filter OFF (amplitude is never a reason to reject;
#                            even tiny/noise-level rhythms pass)
#   amp_stringency = 0.5  -> DEFAULT, validated behavior (recommended)
#   amp_stringency = 1.0  -> most stringent (requires twice the default amplitude)
#
# Values in between scale linearly. The threshold is ADAPTIVE: it adjusts to
# each target's expression level and variability; amp_stringency just scales it.
amp_stringency = 0.5

# --- Internal shape of the adaptive threshold (NOT user-facing) ---
# These define HOW the threshold adapts per target. Tune the filter via
# amp_stringency above; only change these if you want to reshape the filter.
_amp_floor      = 0.15   # absolute noise floor (amplitude below this = noise)
_amp_mean_ratio = 0.10   # >= 10% of mean expression (minimum fold-change ~1.22)
_amp_std_ratio  = 0.50   # >= 50% of the data's std dev (meaningful share of variance)

#-------------------------------------------------------------------
# Circular (cosinor) function to fit: k + a * cos((x/T)*2pi - f)
#-------------------------------------------------------------------
def circular_function(x, k, a, f, T):
    r = (x / T) * (2 * np.pi)
    return k + a * np.cos(r - f)

#-------------------------------------------------------------------
# Read observed-data file using metadata from Input.txt
# Expected format:
# gene_name   02  02  02  02  02  02  02  02  06  06  ...
# The first column must be gene_name.
# The remaining columns contain only observed values.
#-------------------------------------------------------------------
def read_data_file(data_file, input_config, df_raw=None):
    n_timepoints = input_config['n_timepoints']
    n_observations = input_config['n_observations']
    timepoints = input_config['timepoints']
    groups = input_config['groups']
    expected_data_columns = len(groups) * len(timepoints) * n_observations
    def normalize_time_label(value):
        value_str = str(value).strip()
        if value_str.endswith('.0'):
            value_str = value_str[:-2]
        elif '.' in value_str:
            left_part, right_part = value_str.rsplit('.', 1)
            if right_part.isdigit():
                value_str = left_part
        value_str = value_str.strip()
        if value_str.isdigit():
            value_str = value_str.zfill(2)
        return value_str

    def parse_numeric_value(raw_value):
        if pd.isna(raw_value):
            return np.nan
        raw_str = str(raw_value).strip()
        if raw_str == '' or raw_str.lower() in ['na', 'none', 'null']:
            return np.nan
        raw_str = raw_str.replace(',', '.')
        return pd.to_numeric(raw_str, errors='coerce')

    # File reading (or use the DataFrame injected by R, if provided)
    injected = df_raw is not None
    if not injected:
        df_raw = pd.read_csv(data_file, sep='\t', dtype=str)
    else:
        df_raw = df_raw.astype(str)
    df_raw = df_raw.dropna(axis=1, how='all')
    cleaned_columns = [str(col).strip() if col is not None else '' for col in df_raw.columns]
    df_raw.columns = cleaned_columns
    valid_columns = [col for col in df_raw.columns if str(col).strip() != '']
    df_raw = df_raw[valid_columns]
    found_data_columns = len(df_raw.columns) - 1
    if found_data_columns != expected_data_columns:
        raise ValueError(f'Number of data columns mismatch. Expected: {expected_data_columns}, found: {found_data_columns}')

    # Header verification (only in file mode; when injected from R the column
    # order is controlled by the caller, so we validate by count, not by header)
    if not injected:
        observed_headers = [normalize_time_label(col) for col in df_raw.columns[1:]]
        expected_headers = []
        for group_name in groups:
            for time_value in timepoints:
                for rep_idx in range(n_observations):
                    expected_headers.append(normalize_time_label(time_value))
        if observed_headers != expected_headers:
            raise ValueError('The schedule header is not in the expected order defined by Input.txt.')

    # CREATE THE TABLE (DataFrame)
    all_records = []

    for _, row in df_raw.iterrows():
        gene_name = str(row.iloc[0]).strip()
        if gene_name == '' or gene_name.lower() in ['none', 'null']:
            continue
        col_idx = 1
        for group_name in groups:
            for time_value in timepoints:
                # We store the time as a float for Cossinor calculations.
                t_val = float(time_value)
                for rep_idx in range(n_observations):
                    val = parse_numeric_value(row.iloc[col_idx])
                    all_records.append({'Gene': gene_name,'Group': group_name,'Time': t_val,'Value': val})
                    col_idx += 1

    # We return a single table. This makes it easier to filter N groups later.
    df_long = pd.DataFrame(all_records)

    return df_long, df_raw

#-------------------------------------------------------------------
# Read the Metadata
#-------------------------------------------------------------------
def read_input_file(input_file):
    config = {}
    comparisons = []
    genes_to_plot = []
    current_section = 'config'

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comments.
            if not line or line.startswith('#'):
                continue

            # Detects section change
            if line == '[Comparisons]':
                current_section = 'comparisons'
                continue
            elif line == '[GenesToPlot]':
                current_section = 'genes_to_plot'
                continue

            # Processing according to the section we are in.
            if current_section == 'config':
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                config[key] = value

            elif current_section == 'comparisons':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 2:
                    comparisons.append((parts[0], parts[1]))

            elif current_section == 'genes_to_plot':
                # Save the gene already in uppercase to ensure compatibility.
                genes_to_plot.append(line.upper())

    # Old mapping to accommodate common spelling variations
    try:
        n_timepoints = int(config['number of timepoints'])
        n_observations = int(config['number of observations'])

        # Take the string of schedules and groups and turn it into a list.
        timepoints = [x.strip() for x in config['timepoints'].split(',')]
        groups = [x.strip() for x in config['groups'].split(',')]
    except KeyError as e:
        raise KeyError(f"Required key not found in Input.txt: {e}")

    # Consistency validations
    if len(timepoints) != n_timepoints:
        raise ValueError(
            f'The number of listed times ({len(timepoints)}) does not match the defined "Number of Timepoints" ({n_timepoints}).')
    if len(groups) < 2:
        raise ValueError('For CODAC_Multi, you must define at least 2 groups in Input.txt.')

    # The return now includes comparisons and genes for plotting.
    return {
        'n_timepoints': n_timepoints,
        'n_observations': n_observations,
        'timepoints': timepoints,
        'groups': groups,
        'comparisons': comparisons,  # Nova lista de tuplas
        'genes_to_plot': genes_to_plot  # Nova lista de strings
    }
#-------------------------------------------------------------------
# Calculate R-squared
#-------------------------------------------------------------------
def calc_r2(y_obs, y_hat):
    # We created a mask to only consider data where we have valid data.
    mask = ~np.isnan(y_obs) & ~np.isnan(y_hat)
    y_o = y_obs[mask]
    y_h = y_hat[mask]
    # If there is insufficient data after filtering, it returns 0.0.
    if len(y_o) < 2:
        return 0.0
    ss_res = np.sum((y_o - y_h) ** 2)
    ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

#-------------------------------------------------------------------
# Calculate metrics
#-------------------------------------------------------------------
def calculate_full_metrics(group_list):
    # Calculate the average of each ZT (ignoring NaNs)
    means = [np.nanmean(g) for g in group_list if len(g) > 0]
    if not means:
        return [], 0, 0
    mean_of_means = np.mean(means)
    std_dev = np.std(means)
    return means, mean_of_means, std_dev

#-------------------------------------------------------------------
# Fit linear cosinor model — used to compute p-value vs null model
#-------------------------------------------------------------------
def fit_linear_cosinor(X, y, period=24.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # NaNs filter
    mask = ~np.isnan(y)
    n_samples = np.sum(mask)

    # If there is insufficient data (minimum 3 points for 3 parameters)
    if n_samples < 3:
        return {"M": np.nan, "B": np.nan, "C": np.nan,"amplitude": 0, "phase": 0,"y_pred": np.full_like(y, np.nan),"r2": 0, "ss_res": np.inf, "n": n_samples}
    X_valid = X[mask]
    y_valid = y[mask]
    omega = 2.0 * np.pi / period

    # Matrix Design: [1, cos(wt), sin(wt)]
    X_design = np.column_stack((np.ones_like(X_valid),np.cos(omega * X_valid),np.sin(omega * X_valid)))

    # Solving by Ordinary Least Squares (OLS)
    try:
        coef, _, _, _ = np.linalg.lstsq(X_design, y_valid, rcond=None)
        M, B, C = coef

        # Prediction for the complete original vector (X)
        X_design_full = np.column_stack((np.ones_like(X), np.cos(omega * X), np.sin(omega * X)))
        y_pred = X_design_full.dot(coef)
        amplitude = np.sqrt(B ** 2 + C ** 2)
        phase = np.arctan2(-C, B)

        # Sums of Squares for posterior F-statistic
        ss_res = np.sum((y_valid - y_pred[mask]) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"M": M, "B": B, "C": C,"amplitude": amplitude, "phase": phase,"y_pred": y_pred, "r2": r2, "ss_res": ss_res, "n": n_samples}
    except:
        # If a single numerical error occurs
        return {"M": np.nan, "B": np.nan, "C": np.nan,"amplitude": 0, "phase": 0,"y_pred": np.full_like(y, np.nan),"r2": 0, "ss_res": np.inf, "n": n_samples}

#-------------------------------------------------------------------
# Rhythmicity test: cosinor vs. null (constant) model via F-test
#-------------------------------------------------------------------
def test_rhythmicity_cosinor(X, y, period=24.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # NaNs filter
    mask = ~np.isnan(y)
    X_valid = X[mask]
    y_valid = y[mask]
    n = len(y_valid)

    # We need at least 4 points to test a 3-parameter model.
    if n < 4:
        return np.nan, 1.0, {"k": np.mean(y_valid) if n > 0 else np.nan, "ss_res": np.inf}, None

    # Null Model: y = média
    k = np.mean(y_valid)
    ss_res_null = np.sum((y_valid - k) ** 2)

    # Cossinor Model
    cosinor = fit_linear_cosinor(X_valid, y_valid, period=period)
    ss_res_cos = cosinor["ss_res"]

    # Degrees of Freedom. Cosmin has 3 parameters (M, B, C), Null has 1 (k). Difference = 2.
    df1 = 2
    df2 = n - 3

    # If the fit is worse than the mean or there are insufficient degrees of freedom.
    if df2 <= 0 or ss_res_cos >= ss_res_null:
        return 0.0, 1.0, {"k": k, "ss_res": ss_res_null}, cosinor

    # F-statistic
    F = ((ss_res_null - ss_res_cos) / df1) / (ss_res_cos / df2)

    # P-value using the survival function (sf), which is more accurate than 1 - cdf
    p_value = stats.f.sf(F, df1, df2)

    return F, p_value, {"k": k, "ss_res": ss_res_null}, cosinor

#-------------------------------------------------------------------
# Nonlinear curve fitting (refined version)
#-------------------------------------------------------------------
def perform_curve_fit(X, y, fixed_period=False, bounds_period=(18.0, 24.0), p0_linear=None):
    # Ensure that X and y are arrays and without NaNs for the curve_fit.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    X_v, y_v = X[mask], y[mask]

    if len(y_v) < 4:
        return None, None, None

    # 1. Definition of initial kicks (p0)
    if p0_linear and not np.isnan(p0_linear['M']):
        k0, a0, f0 = p0_linear['M'], p0_linear['amplitude'], p0_linear['phase']
    else:
        k0 = np.mean(y_v)
        a0 = (np.max(y_v) - np.min(y_v)) / 2

        idx_max = np.argmax(y_v)
        t_peak = X_v.values[idx_max] if isinstance(X_v, pd.Series) else X_v[idx_max]
        f0 = (t_peak / 24.0) * (2 * np.pi)

    # 2. Adjustment with FIXED Period (24h)
    if fixed_period:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, _ = curve_fit(lambda x, k, a, f: circular_function(x, k, a, f, 24.0), X_v, y_v, p0=[k0, a0, f0],maxfev=10000)
            T_final = 24.0
            p_final = (*params, T_final)
            y_pred = circular_function(X_v, *p_final)
            return p_final, calc_r2(y_v, y_pred), y_pred
        except (RuntimeError, ValueError):
            return None, None, None

    # 3. Adjustment with FREE Period
    try:
        p0_free = [k0, a0, f0, 24.0]
        lower_bounds = [-np.inf, 0, -2 * np.pi, bounds_period[0]]
        upper_bounds = [np.inf, np.inf, 2 * np.pi, bounds_period[1]]

        params, _ = curve_fit(circular_function, X_v, y_v,p0=p0_free,bounds=(lower_bounds, upper_bounds),maxfev=20000)
        y_pred = circular_function(X_v, *params)
        return params, calc_r2(y_v, y_pred), y_pred
    except (RuntimeError, ValueError):
        return None, None, None

# -------------------------------------------------------------------
# Outlier detection: IQR within each timepoint (across replicates)
# -------------------------------------------------------------------
def check_outliers(data):
    # data is a list of per-timepoint replicate arrays. The IQR is computed
    # WITHIN each timepoint, so a bad replicate can be flagged without ever
    # touching genuine rhythm peaks/troughs (which live across timepoints).
    outliers = []
    for group in data:
        g_array = np.asarray(group, dtype=float)
        g_valid = g_array[~np.isnan(g_array)]

        # The IQR needs at least a little dispersion to make sense.
        if len(g_valid) < 2:
            continue

        Q1 = np.percentile(g_valid, 25)
        Q3 = np.percentile(g_valid, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        group_outliers = g_valid[(g_valid < lower_bound) | (g_valid > upper_bound)]
        outliers.extend(group_outliers)

    return len(outliers) > 0, outliers

# -------------------------------------------------------------------
# Handle missing data: drop, impute, or keep as is (Ignoring NaNs)
# -------------------------------------------------------------------
def handle_missing_data_df(df_long, missing_data_action='KEEP'):
    # 1. Searching for missing data (NaN)
    mask_missing = df_long['Value'].isna()

    if not mask_missing.any():
        print("No missing data detected.")
        return df_long

    # 2. Shows the gene and group with missing data
    df_missing = df_long[mask_missing]
    missing_info = df_missing[['Gene', 'Group']].drop_duplicates()
    genes_with_nan = missing_info['Gene'].unique()

    # 3. Writes the warning
    print(f" ⚠️ Warning: Missing data detected in {len(genes_with_nan)} gene(s)!")

    for gene in genes_with_nan:
        group_missing  = missing_info[missing_info['Gene'] == gene]['Group'].tolist()
        print(f"  -> Gene: {gene:<10} | Group(s): {', '.join(group_missing)}")

    # 4. Define the decision automatically for R integration
    # Accept text values (KEEP / IMPUTE / REMOVE), case-insensitive.
    # Default KEEP. Legacy numeric codes ('1','2','3') still work.
    decision = str(missing_data_action).strip().upper()

    # 5. Apply the user's decision
    if decision in ('IMPUTE', '1'):
        print("\nImputing missing values with group/timepoint mean...")
        means = df_long.groupby(['Gene', 'Group', 'Time'])['Value'].transform('mean')
        df_long['Value'] = df_long['Value'].fillna(means)
    elif decision in ('REMOVE', '2'):
        print("\nRemoving genes with missing data from the analysis...")
        df_long = df_long[~df_long['Gene'].isin(genes_with_nan)]
    else:
        print("\nKeeping existing data. NaNs will be ignored by fitting functions.")

    print("Missing data handling complete!")
    return df_long

# -------------------------------------------------------------------
# Computing the difference between phase (Circular Difference)
# -------------------------------------------------------------------
def circular_phase_diff(p1, p2, period=24.0):
    # If p1 or p2 are NaNs, return NaN
    if np.isnan(p1) or np.isnan(p2):
        return np.nan
    diff = (p1 - p2) % period
    if diff > period / 2:
        diff -= period
    return diff

# -------------------------------------------------------------------
# Auxiliary function to compute harmonic terms for Linear Modeling
# -------------------------------------------------------------------
def add_harmonic_terms(df, time_col='Time', period=24.0):
    # Adds sine and cosine columns to the DataFrame to allow for linear Cossinor adjustments.
    df = df.copy()

    # We guarantee that there are no NaNs in the time for the calculation.
    omega = 2.0 * np.pi / period

    # We created the linearized columns.
    df['cos_t'] = np.cos(omega * df[time_col])
    df['sin_t'] = np.sin(omega * df[time_col])

    return df

#-------------------------------------------------------------------
# Fit Multigroup Models using Likelihood Ratio logic
#-------------------------------------------------------------------
def fit_multigroup_models(df_gene, expr_col='Value', group_col='Group'):
    # Adjusts three levels of models for statistical comparison.
    # df_gene must contain the cos_t and sin_t columns (generated by add_harmonic_terms).

    formula0 = f"Q('{expr_col}') ~ C(Q('{group_col}'))"
    formula1 = f"Q('{expr_col}') ~ C(Q('{group_col}')) + cos_t + sin_t"
    formula2 = f"Q('{expr_col}') ~ C(Q('{group_col}')) + cos_t + sin_t + C(Q('{group_col}')):cos_t + C(Q('{group_col}')):sin_t"

    try:
        model0 = smf.ols(formula0, data=df_gene).fit()

        # Model 1 (Reduced Model): A single rhythmic curve for all groups.
        model1 = smf.ols(formula1, data=df_gene).fit()

        # Model 2 (Complete Model): Independent rhythmic curves for each group
        model2 = smf.ols(formula2, data=df_gene).fit()
        return model0, model1, model2
    except Exception as e:
        # If the data is insufficient or constant, it returns None to avoid blocking the loop.
        return None, None, None

# -------------------------------------------------------------------
# Compute Differential Statistics (The "CODAC_Multi" Heart)
# -------------------------------------------------------------------
def get_global_tests(model0, model1, model2, group_col='Group'):
    # Performs model comparison tests (Extra Sum-of-Squares F-test).
    out = {'p_global_rhythm': np.nan,
           'p_rhythm_diff': np.nan}

    if model0 is None or model1 is None or model2 is None:
        return out

    # 1. General Rhythmicity Test (Does any group have rhythm?)
    try:
        anova_a = anova_lm(model0, model2)
        out['p_global_rhythm'] = float(anova_a['Pr(>F)'].iloc[1])
    except:
        pass

    # 2. Differential Rhythm Test (Does the phase/amplitude change between groups?)
    try:
        anova_b = anova_lm(model1, model2)
        out['p_rhythm_diff'] = float(anova_b['Pr(>F)'].iloc[1])
    except:
        pass

    return out

# ===================================================================
# MULTI-GROUP GROUPING SELECTION (CODAC_Multi)
# -------------------------------------------------------------------
# Instead of only reading pairwise contrasts (which do not compose when
# they are non-transitive: G1~G2, G2~G3, but G1!=G3), CODAC_Multi finds the
# single best GROUPING of the groups by model selection -- the same spirit
# as dryR, but on the CODA cosinor/GLM engine.
#
# The RHYTHM axis enumerates, for each subset of rhythmic groups, every way
# the rhythmic groups can be partitioned into "same-rhythm" blocks. For N
# groups this is Bell(N+1) models (N=2 -> 5, N=3 -> 15, N=4 -> 52). The mesor
# is a free-per-group nuisance here (it has its own axis; see the mesor
# selection). Each model is fitted as a linear GLM (shared cos/sin within a
# block, none for arrhythmic groups) and scored by an information criterion.
# The winner is the lowest-IC model; its criterion weight is reported as a
# 0-1 confidence.
# ===================================================================
def set_partitions(collection):
    # Yields every partition of `collection` into non-empty blocks.
    collection = list(collection)
    if len(collection) == 0:
        yield []
        return
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in set_partitions(collection[1:]):
        for i, block in enumerate(smaller):
            yield smaller[:i] + [[first] + block] + smaller[i + 1:]
        yield [[first]] + smaller

#-------------------------------------------------------------------
# Models
#-------------------------------------------------------------------
def enumerate_rhythm_models(groups):
    # Each model = (frozenset of rhythmic groups, tuple of same-rhythm blocks).
    # Groups outside the rhythmic set are arrhythmic in that model.
    groups = list(groups)
    models = []
    for k in range(0, len(groups) + 1):
        for rhythmic_set in combinations(groups, k):
            if k == 0:
                models.append((frozenset(), tuple()))
            else:
                for part in set_partitions(rhythmic_set):
                    blocks = tuple(sorted(tuple(sorted(b)) for b in part))
                    models.append((frozenset(rhythmic_set), blocks))
    return models

#-------------------------------------------------------------------
# Information criterion
#-------------------------------------------------------------------
def _ic_from_design(X, y, criterion='BIC'):
    # Fits y ~ X by OLS and returns the information criterion.
    # k = number of design columns (mean-structure params), matching the AICc
    # convention used by CODAflex (variance is not counted). BIC uses k*ln(n).
    n = len(y)
    k = X.shape[1]
    if k == 0 or n <= k + 1:
        return np.inf
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    if rss <= 0:
        rss = 1e-10
    base = n * np.log(rss / n)
    if str(criterion).strip().upper() in ('AICC', 'AIC'):
        aic = base + 2 * k
        return aic + (2 * k * (k + 1)) / (n - k - 1) if (n - k - 1) > 0 else aic
    # Default: BIC (Schwarz) -- stronger complexity penalty, more conservative
    # about calling a difference (aligned with dryR and with avoiding the
    # over-splitting that AICc is prone to when rhythms are truly shared).
    return base + k * np.log(n)

#-------------------------------------------------------------------
# Rhythm model
#-------------------------------------------------------------------
def _rhythm_design(group_arr, cos, sin, all_groups, model):
    # Design matrix for a rhythm model: one intercept column per group (mesor
    # free per group = nuisance), plus two columns (shared cos, shared sin) for
    # each rhythmic block. Arrhythmic groups contribute no rhythm columns.
    rhythmic_set, blocks = model
    cols = [(group_arr == g).astype(float) for g in all_groups]
    for b in blocks:
        in_block = np.isin(group_arr, list(b)).astype(float)
        cols.append(cos * in_block)
        cols.append(sin * in_block)
    return np.column_stack(cols)

#-------------------------------------------------------------------
# Rythm label
#-------------------------------------------------------------------
def _rhythm_label(model, all_groups):
    # Human-readable grouping label, e.g. "{G1,G2} != {G3}" or
    # "{G1,G2,G3} (all equal)" or "{G1,G2} ; arrhythmic: G3".
    rhythmic_set, blocks = model
    arrhythmic = [g for g in all_groups if g not in rhythmic_set]
    if not rhythmic_set:
        return "Arrhythmic in all groups"
    parts = ["{" + ",".join(b) + "}" for b in blocks]
    if len(parts) == 1:
        core = parts[0] + (" (all equal)" if not arrhythmic else "")
    else:
        core = " != ".join(parts)
    if arrhythmic:
        core += " ; arrhythmic: " + ",".join(arrhythmic)
    return core

def _rhythm_code(model, all_groups):
    # Canonical, stable ID of a rhythm model ("M01".."M15" for 3 groups),
    # matching the enumeration order documented in the README legend.
    try:
        return "M%02d" % (enumerate_rhythm_models(all_groups).index(model) + 1)
    except ValueError:
        return ""

#-------------------------------------------------------------------
# Rhythm group
#-------------------------------------------------------------------
def select_rhythm_grouping(df_gene, all_groups, rhythmic_set, criterion='BIC',
                           time_col='Time', expr_col='Value', group_col='Group'):
    # Chooses how the RHYTHMIC groups (rhythmic_set, decided by the per-group
    # multi-criteria tier) share their rhythm, by model selection over the
    # partitions of rhythmic_set. Groups outside rhythmic_set are arrhythmic in
    # every candidate. Returns (label, confidence, ic_gap, model_code).
    d = df_gene[[time_col, expr_col, group_col]].dropna(subset=[expr_col])
    if d.empty:
        return "Undetermined", np.nan, np.nan, ""
    t = d[time_col].to_numpy(dtype=float)
    y = d[expr_col].to_numpy(dtype=float)
    group_arr = d[group_col].to_numpy()
    omega = 2 * np.pi / 24.0
    cos = np.cos(omega * t)
    sin = np.sin(omega * t)

    rset = frozenset(rhythmic_set)
    models = [(rset, tuple(sorted(tuple(sorted(b)) for b in part)))
              for part in set_partitions(list(rhythmic_set))]
    ics = np.array([_ic_from_design(_rhythm_design(group_arr, cos, sin, all_groups, m), y, criterion)
                    for m in models], dtype=float)
    if not np.any(np.isfinite(ics)):
        return "Undetermined", np.nan, np.nan, ""

    order = np.argsort(ics)
    best = int(order[0])
    ic_min = ics[best]
    weights = np.exp(-0.5 * (ics - ic_min))
    weights = weights / np.nansum(weights)
    ic_gap = float(ics[order[1]] - ics[order[0]]) if len(order) > 1 and np.isfinite(ics[order[1]]) else np.inf

    return _rhythm_label(models[best], all_groups), float(weights[best]), ic_gap, _rhythm_code(models[best], all_groups)

# -------------------------------------------------------------------
# MESOR axis -- a SEPARATE grouping, on the baseline (mesor) only.
# Every group has a mesor (rhythmic or not), so the models are simply the
# set-partitions of ALL groups by shared baseline (Bell(N): N=3 -> 5). The
# rhythm is a free-per-group nuisance here, so the mesor comparison is not
# confounded by rhythm. Same IC / weight machinery as the rhythm axis.
# -------------------------------------------------------------------
def enumerate_mesor_models(groups):
    # Each model is a partition of ALL groups into shared-mesor blocks.
    models = []
    for part in set_partitions(list(groups)):
        blocks = tuple(sorted(tuple(sorted(b)) for b in part))
        models.append(blocks)
    return models

#-------------------------------------------------------------------
# Mesor design
#-------------------------------------------------------------------
def _mesor_design(group_arr, cos, sin, all_groups, blocks):
    # One intercept column per mesor block (shared baseline within a block),
    # plus a free per-group rhythm (cos, sin per group) as nuisance.
    cols = []
    for b in blocks:
        cols.append(np.isin(group_arr, list(b)).astype(float))
    for g in all_groups:
        ing = (group_arr == g).astype(float)
        cols.append(cos * ing)
        cols.append(sin * ing)
    return np.column_stack(cols)

#-------------------------------------------------------------------
# Mesor Label
#-------------------------------------------------------------------
def _mesor_label(blocks):
    parts = ["{" + ",".join(b) + "}" for b in blocks]
    if len(parts) == 1:
        return parts[0] + " (all equal)"
    return " != ".join(parts)

def _mesor_code(blocks, all_groups):
    # Canonical, stable ID of a mesor model ("MM1".."MM5" for 3 groups).
    try:
        return "MM%d" % (enumerate_mesor_models(all_groups).index(blocks) + 1)
    except ValueError:
        return ""

#-------------------------------------------------------------------
# Select mesor
#-------------------------------------------------------------------
def select_mesor_grouping(df_gene, all_groups, criterion='BIC',
                          time_col='Time', expr_col='Value', group_col='Group'):
    # Same output contract as select_rhythm_grouping, but for the baseline.
    d = df_gene[[time_col, expr_col, group_col]].dropna(subset=[expr_col])
    if d.empty:
        return "Undetermined", np.nan, np.nan
    t = d[time_col].to_numpy(dtype=float)
    y = d[expr_col].to_numpy(dtype=float)
    group_arr = d[group_col].to_numpy()
    omega = 2 * np.pi / 24.0
    cos = np.cos(omega * t)
    sin = np.sin(omega * t)

    models = enumerate_mesor_models(all_groups)
    ics = np.array([_ic_from_design(_mesor_design(group_arr, cos, sin, all_groups, b), y, criterion)
                    for b in models], dtype=float)
    if not np.any(np.isfinite(ics)):
        return "Undetermined", np.nan, np.nan

    order = np.argsort(ics)
    best = int(order[0])
    weights = np.exp(-0.5 * (ics - ics[best]))
    weights = weights / np.nansum(weights)
    ic_gap = float(ics[order[1]] - ics[order[0]]) if len(order) > 1 and np.isfinite(ics[order[1]]) else np.inf

    return _mesor_label(models[best]), float(weights[best]), ic_gap, _mesor_code(models[best], all_groups)

#-------------------------------------------------------------------
# Mesor omnibus gate
#-------------------------------------------------------------------
def mesor_omnibus_p(df_gene, all_groups, time_col='Time', expr_col='Value', group_col='Group'):
    # Omnibus test for a baseline (mesor) difference across groups, controlling
    # for a free per-group rhythm. Nested F-test of the all-shared-mesor model
    # (restricted) against the all-free-mesor model (full) -> p_mesor_diff. This
    # is the mesor counterpart of p_rhythm_diff and gates the mesor grouping.
    d = df_gene[[time_col, expr_col, group_col]].dropna(subset=[expr_col])
    n = len(d)
    if n == 0 or len(all_groups) < 2:
        return np.nan
    t = d[time_col].to_numpy(dtype=float)
    y = d[expr_col].to_numpy(dtype=float)
    group_arr = d[group_col].to_numpy()
    omega = 2 * np.pi / 24.0
    cos = np.cos(omega * t)
    sin = np.sin(omega * t)

    all_together = (tuple(sorted(all_groups)),)             # one shared-mesor block
    all_separate = tuple((g,) for g in sorted(all_groups))  # each group its own mesor

    X_r = _mesor_design(group_arr, cos, sin, all_groups, all_together)
    X_f = _mesor_design(group_arr, cos, sin, all_groups, all_separate)
    df1 = X_f.shape[1] - X_r.shape[1]     # = (num groups - 1)
    df2 = n - X_f.shape[1]
    if df1 <= 0 or df2 <= 0:
        return np.nan
    beta_r, _, _, _ = np.linalg.lstsq(X_r, y, rcond=None)
    beta_f, _, _, _ = np.linalg.lstsq(X_f, y, rcond=None)
    rss_r = float(np.sum((y - X_r @ beta_r) ** 2))
    rss_f = float(np.sum((y - X_f @ beta_f) ** 2))
    if rss_f <= 0:
        return np.nan
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    return float(stats.f.sf(F, df1, df2)) if F > 0 else np.nan

#-------------------------------------------------------------------
# FDR correction of the pairwise p-values (per component, WITHIN each pair)
#-------------------------------------------------------------------
def add_pairwise_fdr(df_comparisons, alpha_method='fdr_bh'):
    # Adds a Benjamini-Hochberg FDR column next to each raw pairwise p-value.
    # Grouping: PER COMPONENT, WITHIN each Pair (option A). Each (Pair, component)
    # is its own family of tests. Empty strings ("" -> Cat 1/2/3 not tested) and
    # NaN (NLS failed) are ignored by the correction and stay empty in the output.
    if df_comparisons is None or df_comparisons.empty:
        return df_comparisons

    comp_cols = {
        'p_diff_mesor':     'p_diff_mesor_FDR',
        'p_diff_amplitude': 'p_diff_amplitude_FDR',
        'p_diff_phase':     'p_diff_phase_FDR',
    }

    for _, fdr_col in comp_cols.items():
        df_comparisons[fdr_col] = np.nan

    for pair_label, idx in df_comparisons.groupby('Pair').groups.items():
        block = df_comparisons.loc[idx]
        for raw_col, fdr_col in comp_cols.items():
            # Numeric, testable p-values only (skip "" and NaN).
            def _is_testable(v):
                if v == "" or pd.isna(v):
                    return False
                try:
                    float(v)
                    return True
                except (TypeError, ValueError):
                    return False

            mask = block[raw_col].apply(_is_testable)
            testable_idx = block.index[mask]
            if len(testable_idx) == 0:
                continue

            pvals = block.loc[testable_idx, raw_col].astype(float).values
            p_adj = multipletests(pvals, method=alpha_method)[1]
            df_comparisons.loc[testable_idx, fdr_col] = p_adj

    return df_comparisons

#-------------------------------------------------------------------
# Benjamini-Hochberg FDR for the genome-wide GLOBAL p-values
# (p_global_rhythm, p_rhythm_diff, p_mesor_diff). NaN p-values stay NaN and are
# excluded from the correction.
#-------------------------------------------------------------------
def add_global_fdr(df_global, cols):
    for col in cols:
        if col not in df_global.columns:
            continue
        vals = pd.to_numeric(df_global[col], errors='coerce')
        out = pd.Series(np.nan, index=df_global.index, dtype=float)
        mask = vals.notna()
        if mask.sum() > 0:
            out.loc[mask] = multipletests(vals[mask].values, method='fdr_bh')[1]
        df_global[col + '_FDR'] = out
    return df_global

#-------------------------------------------------------------------
# (Re)assign the biological categories from a chosen p-value source
#-------------------------------------------------------------------
def assign_categories(df_comparisons, p_source='RAW', alpha=0.05):
    # Recomputes the COMPARISON-driven outputs from a chosen p-value source:
    #   - Mesor_Change        (from p_diff_mesor / p_diff_mesor_FDR)
    #   - LossGain_Confidence (Cat 2/3, from p_diff_amplitude / _FDR)
    #   - Biological_Category (Cat 4-7 split, from amplitude & phase / _FDR)
    #   - Rhythm_Status
    # Must run AFTER add_pairwise_fdr so the *_FDR columns exist. 'RAW' reproduces
    # the previous validated behavior exactly. Only the DECISIONS switch source;
    # both the raw and FDR columns stay in the table.
    #
    # The rhythmicity structure (Cat 1 vs 2 vs 3 vs both) is fixed upstream and
    # does NOT depend on the pairwise-difference p-values.
    if df_comparisons is None or df_comparisons.empty:
        return df_comparisons
    if 'Rhythmicity_Structure' not in df_comparisons.columns:
        # Older tables without the marker: leave categories untouched.
        return df_comparisons

    use_fdr = str(p_source).strip().upper() in ('FDR', 'ADJ', 'ADJUSTED')
    mesor_col = 'p_diff_mesor_FDR'     if use_fdr else 'p_diff_mesor'
    amp_col   = 'p_diff_amplitude_FDR' if use_fdr else 'p_diff_amplitude'
    phase_col = 'p_diff_phase_FDR'     if use_fdr else 'p_diff_phase'
    if mesor_col not in df_comparisons.columns:
        mesor_col = 'p_diff_mesor'
    if amp_col not in df_comparisons.columns:
        amp_col = 'p_diff_amplitude'
    if phase_col not in df_comparisons.columns:
        phase_col = 'p_diff_phase'

    def _is_sig(v):
        # True only for a testable p-value below alpha; "" / NaN => not significant
        if v == "" or pd.isna(v):
            return False
        try:
            return float(v) < alpha
        except (TypeError, ValueError):
            return False

    def _mesor_change(v):
        # Mesor is always tested; "" / NaN means the NLS adjustment failed.
        if v == "" or pd.isna(v):
            return "Undetermined"
        try:
            return "Different" if float(v) < alpha else "Conserved"
        except (TypeError, ValueError):
            return "Undetermined"

    cats, statuses, confs, mesor_changes = [], [], [], []
    for _, row in df_comparisons.iterrows():
        structure = row['Rhythmicity_Structure']
        diff_A = _is_sig(row.get(amp_col, ""))
        diff_P = _is_sig(row.get(phase_col, ""))
        mesor_changes.append(_mesor_change(row.get(mesor_col, "")))

        if structure == "neither":
            cats.append("Cat 1: Arrhythmic")
            statuses.append("Neither rhythmic")
            confs.append("")
        elif structure == "g1_only":
            cats.append("Cat 2: rhythmic_group_1_only")
            statuses.append("Group 1 only")
            confs.append("High confidence" if diff_A else "Weak evidence")
        elif structure == "g2_only":
            cats.append("Cat 3: rhythmic_group_2_only")
            statuses.append("Group 2 only")
            confs.append("High confidence" if diff_A else "Weak evidence")
        else:  # both rhythmic
            if not diff_A and not diff_P:
                cats.append("Cat 4: rhythmic_both_unchanged")
            elif diff_A and not diff_P:
                cats.append("Cat 5: rhythmic_with_changes_only_amp")
            elif diff_P and not diff_A:
                cats.append("Cat 6: rhythmic_with_changes_only_phase")
            else:
                cats.append("Cat 7: rhythmic_with_changes_amp_phase")
            statuses.append("Both rhythmic")
            confs.append("")

    df_comparisons = df_comparisons.copy()
    df_comparisons['Biological_Category'] = cats
    df_comparisons['Rhythm_Status'] = statuses
    df_comparisons['LossGain_Confidence'] = confs
    df_comparisons['Mesor_Change'] = mesor_changes
    return df_comparisons

#-------------------------------------------------------------------
# Reconcile the pairwise Biological_Category with the grouping model.
# The grouping is authoritative for "changed vs unchanged": for a pair of
# BOTH-rhythmic groups, same rhythm-block -> Cat 4 (unchanged); different blocks
# -> Cat 5/6/7 (changed), with the specific component taken from the pairwise
# amplitude/phase p-values (and, in the rare case the grouping says different but
# neither p is significant, the smaller-p component). The raw Delta_* / p_diff_*
# columns are left untouched so the underlying tests stay visible.
#-------------------------------------------------------------------
def reconcile_categories_with_grouping(df_comparisons, df_global, groups,
                                       alpha=0.05, p_source='RAW'):
    if (df_comparisons is None or df_comparisons.empty or
            df_global is None or df_global.empty or
            'Grouping_Model' not in df_global.columns):
        return df_comparisons

    gene_model = dict(zip(df_global['Gene'], df_global['Grouping_Model']))
    models_cache = enumerate_rhythm_models(groups)

    use_fdr = str(p_source).strip().upper() in ('FDR', 'ADJ', 'ADJUSTED')
    amp_col = 'p_diff_amplitude_FDR' if use_fdr else 'p_diff_amplitude'
    phase_col = 'p_diff_phase_FDR' if use_fdr else 'p_diff_phase'
    if amp_col not in df_comparisons.columns:
        amp_col = 'p_diff_amplitude'
    if phase_col not in df_comparisons.columns:
        phase_col = 'p_diff_phase'

    CAT = {4: "Cat 4: rhythmic_both_unchanged",
           5: "Cat 5: rhythmic_with_changes_only_amp",
           6: "Cat 6: rhythmic_with_changes_only_phase",
           7: "Cat 7: rhythmic_with_changes_amp_phase"}

    def _pnum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    def _relation(code, gA, gB):
        if not isinstance(code, str) or not code.startswith('M'):
            return None
        try:
            rset, blocks = models_cache[int(code[1:]) - 1]
        except (ValueError, IndexError):
            return None
        if gA not in rset or gB not in rset:
            return None                      # not both rhythmic -> leave as is
        for b in blocks:
            if gA in b and gB in b:
                return 'same'
        return 'diff'

    genes = df_comparisons['Gene'].tolist()
    pairs = df_comparisons['Pair'].tolist()
    amps = df_comparisons[amp_col].tolist()
    phases = df_comparisons[phase_col].tolist()
    cats = df_comparisons['Biological_Category'].tolist()

    for i in range(len(cats)):
        pair = str(pairs[i])
        if " vs " not in pair:
            continue
        gA, gB = [x.strip() for x in pair.split(" vs ", 1)]
        rel = _relation(gene_model.get(genes[i], ""), gA, gB)
        if rel == 'same':
            cats[i] = CAT[4]
        elif rel == 'diff':
            ap, pp = _pnum(amps[i]), _pnum(phases[i])
            amp_sig = (not np.isnan(ap)) and ap < alpha
            phase_sig = (not np.isnan(pp)) and pp < alpha
            if amp_sig and phase_sig:
                cats[i] = CAT[7]
            elif amp_sig:
                cats[i] = CAT[5]
            elif phase_sig:
                cats[i] = CAT[6]
            elif np.isnan(ap) and np.isnan(pp):
                cats[i] = CAT[7]             # no info -> both changed
            elif np.isnan(pp) or (not np.isnan(ap) and ap <= pp):
                cats[i] = CAT[5]            # conflict -> smaller-p component
            else:
                cats[i] = CAT[6]
        # rel is None -> leave the category untouched

    df_comparisons = df_comparisons.copy()
    df_comparisons['Biological_Category'] = cats
    return df_comparisons

#-------------------------------------------------------------------
# Pairwise Post-hoc Comparison (Wald Tests)
#-------------------------------------------------------------------
def build_pairwise_comparisons(df_results, df_long, comparisons_to_run=None, rhythmicity_cutoff='HIGH'):
    comparison_results = []

    if not comparisons_to_run:
        all_groups = df_results['Group'].unique()
        comparisons_to_run = list(combinations(all_groups, 2))

    rank_map = {'ARRHYTHMIC': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'EXTREMELY HIGH': 4}

    # Grouping data
    grouped_genes = df_results.groupby('Gene')

    for gene, group_data in tqdm(grouped_genes, desc="Comparing groups", unit="target"):
        if len(group_data) < 2: continue

        df_gene_raw = df_long[df_long['Gene'] == gene]

        for g1_name, g2_name in comparisons_to_run:
            r1_df = group_data[group_data['Group'] == g1_name]
            r2_df = group_data[group_data['Group'] == g2_name]

            if r1_df.empty or r2_df.empty: continue

            r1 = r1_df.iloc[0]
            r2 = r2_df.iloc[0]

            prob1 = str(r1.Probability).strip().upper()
            prob2 = str(r2.Probability).strip().upper()
            rank1 = rank_map.get(prob1, 0)
            rank2 = rank_map.get(prob2, 0)

            # --- Step 1: is each group rhythmic? (cutoff, default HIGH) ---
            rhythm_thresh = rank_map.get(rhythmicity_cutoff.upper(), 3)
            g1_rhythmic = rank1 >= rhythm_thresh
            g2_rhythmic = rank2 >= rhythm_thresh

            p_ref = np.mean([r1.Period, r2.Period]) if not pd.isna(r1.Period) else 24.0

            # Mesor is compared ALWAYS (independent of rhythm) -> its own column.
            nls_pvals = run_nls_pairwise_test(df_gene_raw, g1_name, g2_name,
                                              {'Mesor': r1.Mesor, 'Amplitude': r1.Amplitude, 'Phase': r1.Phase},
                                              {'Mesor': r2.Mesor, 'Amplitude': r2.Amplitude, 'Phase': r2.Phase},
                                              period=p_ref)
            p_diff_mesor = nls_pvals['p_diff_mesor']
            delta_mesor = r1.Mesor - r2.Mesor

            # How many groups are rhythmic determines what can be compared.
            both_rhythmic = g1_rhythmic and g2_rhythmic
            one_rhythmic = (g1_rhythmic or g2_rhythmic) and not both_rhythmic

            # Stable marker of the rhythmicity structure. This does NOT depend on
            # the pairwise-difference p-values, so it lets us re-decide the
            # significance-driven sub-categories later using raw OR FDR values.
            if both_rhythmic:
                rhythm_structure = "both"
            elif one_rhythmic:
                rhythm_structure = "g1_only" if g1_rhythmic else "g2_only"
            else:
                rhythm_structure = "neither"

            if both_rhythmic:
                # Both rhythmic: amplitude AND phase are meaningful.
                p_diff_amplitude = nls_pvals['p_diff_amplitude']
                p_diff_phase = nls_pvals['p_diff_phase']
                delta_amp = r1.Amplitude - r2.Amplitude
                delta_phase = circular_phase_diff(r1.Phase, r2.Phase, period=p_ref)
            elif one_rhythmic:
                # Exactly one rhythmic (Cat 2/3): compute the amplitude difference
                # to GRADE the confidence of the rhythm loss/gain . Phase is
                # left out: it is not interpretable against a non-rhythmic group.
                p_diff_amplitude = nls_pvals['p_diff_amplitude']
                p_diff_phase = ""
                delta_amp = r1.Amplitude - r2.Amplitude
                delta_phase = ""
            else:
                # Neither rhythmic (Cat 1): nothing to compare.
                p_diff_amplitude = ""
                p_diff_phase = ""
                delta_amp = ""
                delta_phase = ""

            # Biological_Category, Rhythm_Status, LossGain_Confidence and
            # Mesor_Change are assigned later by assign_categories() -- the single
            # source of truth, which honors the chosen raw/FDR source. Here we only
            # store the structural marker and the raw per-component data.
            comparison_results.append({
                'Gene': gene,
                'Pair': f"{g1_name} vs {g2_name}",
                'Delta_Mesor': delta_mesor,
                'p_diff_mesor': p_diff_mesor,
                'Delta_Amplitude': delta_amp,
                'p_diff_amplitude': p_diff_amplitude,
                'Delta_Phase': delta_phase,
                'p_diff_phase': p_diff_phase,
                'Rhythmicity_Structure': rhythm_structure
            })

    return pd.DataFrame(comparison_results)

#-------------------------------------------------------------------
# Heatmap
#-------------------------------------------------------------------
def generate_heatmap_compare(results, df_long, n_observations,counter_threshold, base_dir, time_label='Time (ZT Hours)'):
    time.sleep(0.1)
    print("\n" + "=" * 70, flush=True)
    print('                    GENERATING HEATMAPS                ', flush=True)
    print("=" * 70, flush=True)

    df_res = pd.DataFrame(results)
    if df_res.empty: return None

    groups = sorted(df_res['group_name'].unique())
    rhythmic_genes_info = df_res[df_res['counter'] >= counter_threshold].copy()

    if rhythmic_genes_info.empty:
        return None

    tp_prefix = 'CT' if 'CT' in time_label else ('ZT' if 'ZT' in time_label else '')
    heatmaps_data = {}

    for grp in groups:
        matrix_rows = []

        genes_do_grupo = rhythmic_genes_info[rhythmic_genes_info['group_name'] == grp]
        if genes_do_grupo.empty: continue

        sorted_gene_names = genes_do_grupo.sort_values('phase')['gene'].unique().tolist()

        df_grp = df_long[df_long['Group'] == grp]
        timepoints = sorted(df_grp['Time'].unique())
        n_unique_tp = len(timepoints)
        time_labels = [f"{tp_prefix}{h}" for h in timepoints]

        # Grouping data
        df_grp_grouped = df_grp.groupby('Gene')

        # Progress bar
        for gene in tqdm(sorted_gene_names, desc=f"Heatmap {grp}", unit="target"):
            if gene in df_grp_grouped.groups:
                y = df_grp_grouped.get_group(gene)['Value'].values
            else:
                y = []

            if len(y) == 0:
                matrix_rows.append([np.nan] * n_unique_tp)
                continue

            means = []
            for i in range(0, len(y), n_observations):
                chunk = y[i: i + n_observations]
                means.append(np.nanmean(chunk) if len(chunk) > 0 else np.nan)

            matrix_rows.append(means)

        if not matrix_rows: continue

        matrix = np.array(matrix_rows, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_means = np.nanmean(matrix, axis=1, keepdims=True)
            row_stds = np.nanstd(matrix, axis=1, keepdims=True)

        row_stds[row_stds == 0] = 1
        matrix_z = np.clip((matrix - row_means) / row_stds, -3, 3)

        n_genes = len(sorted_gene_names)

        FIG_W = 10.0;
        FIG_H = 12.0;
        HM_H_FRAC = 0.82
        hm_h_in = FIG_H * HM_H_FRAC
        needed_dpi = int(np.ceil((n_genes * 3) / hm_h_in))
        out_dpi = max(150, min(300, needed_dpi))
        cell_px = (hm_h_in * out_dpi) / n_genes
        show_labels = (cell_px >= 6) and (n_genes <= 50)
        gene_fontsize = max(2.0, min(7.0, cell_px * 0.55))

        LABEL_W = 1.8 if show_labels else 0.2
        hm_w = n_unique_tp * 0.55
        fig_w = max(FIG_W, LABEL_W + hm_w + 1.5)

        fig = plt.figure(figsize=(fig_w, FIG_H))

        ax_hm = fig.add_axes([LABEL_W / fig_w, 0.1, hm_w / fig_w, 0.8])
        ax_hm.set_facecolor('lightgray')

        im = ax_hm.imshow(matrix_z, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)

        ax_hm.set_xticks(range(n_unique_tp))
        ax_hm.set_xticklabels(time_labels, rotation=45, ha='right')
        ax_hm.set_yticks([])

        plt.title(f'Heatmap - Group: {grp}\n({n_genes} strictly rhythmic genes in this group)',
                  fontsize=10, fontweight='bold')

        if show_labels:
            ax_lbl = fig.add_axes([0, 0.1, LABEL_W / fig_w, 0.8])
            ax_lbl.set_ylim(-0.5, n_genes - 0.5)
            ax_lbl.invert_yaxis()
            ax_lbl.axis('off')
            for i, name in enumerate(sorted_gene_names):
                ax_lbl.text(0.95, i, name, ha='right', va='center', fontsize=gene_fontsize, family='monospace')

        ax_cb = fig.add_axes([(LABEL_W + hm_w + 0.3) / fig_w, 0.3, 0.2 / fig_w, 0.4])
        plt.colorbar(im, cax=ax_cb, label='Z-score')

        out_path = os.path.join(base_dir, 'plots', f'heatmap_{grp}.png')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(out_path, dpi=out_dpi, bbox_inches='tight')

        plt.close(fig)
        heatmaps_data[grp] = pd.DataFrame(matrix_z, index=sorted_gene_names, columns=time_labels)

    return heatmaps_data

# -------------------------------------------------------------------
# Heatmap by biological category (7-model scheme)
# -------------------------------------------------------------------
def generate_heatmap_by_category(df_comparisons, results, df_long, n_observations,
                                 comparisons_to_run, base_dir, time_label='Time (ZT Hours)'):
    # Instead of organizing heatmaps by group pair, we organize them by biological
    # category. For each comparison (g1, g2) and each of the 7 categories, we take
    # the genes classified in that category and draw two panels (g1 and g2),
    # z-scored row-wise. Row z-scoring centers each gene at zero, which REMOVES the
    # mesor by construction -> what remains is purely amplitude + phase, as asked.
    # Genes are ordered by acrophase (preferring the rhythmic group) so the phase
    # "wave" is visible; its loss in the other panel flags arrhythmicity/remodeling.
    print("\n" + "=" * 70, flush=True)
    print('             GENERATING PER-CATEGORY HEATMAPS          ', flush=True)
    print("=" * 70, flush=True)

    if df_comparisons is None or df_comparisons.empty:
        print("  [skip] no comparison table available.")
        return None

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return None

    tp_prefix = 'CT' if 'CT' in time_label else ('ZT' if 'ZT' in time_label else '')

    # Acrophase lookup from the per-group fit: (gene, group) -> phase in hours
    phase_lookup = {(r['gene'], r['group_name']): r.get('phase', np.nan) for r in results}

    category_order = [
        "Cat 1: Arrhythmic",
        "Cat 2: rhythmic_group_1_only",
        "Cat 3: rhythmic_group_2_only",
        "Cat 4: rhythmic_both_unchanged",
        "Cat 5: rhythmic_with_changes_only_amp",
        "Cat 6: rhythmic_with_changes_only_phase",
        "Cat 7: rhythmic_with_changes_amp_phase",
    ]

    def build_matrix(grp, gene_order):
        df_grp = df_long[df_long['Group'] == grp]
        timepoints = sorted(df_grp['Time'].unique())
        time_labels = [f"{tp_prefix}{h}" for h in timepoints]
        grouped = df_grp.groupby('Gene')
        rows = []
        for gene in gene_order:
            y = grouped.get_group(gene)['Value'].values if gene in grouped.groups else []
            if len(y) == 0:
                rows.append([np.nan] * len(timepoints))
                continue
            means = [np.nanmean(y[i:i + n_observations]) if len(y[i:i + n_observations]) > 0 else np.nan
                     for i in range(0, len(y), n_observations)]
            means = (means + [np.nan] * len(timepoints))[:len(timepoints)]
            rows.append(means)
        matrix = np.array(rows, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rm = np.nanmean(matrix, axis=1, keepdims=True)
            rs = np.nanstd(matrix, axis=1, keepdims=True)
        rs[rs == 0] = 1
        matrix_z = np.clip((matrix - rm) / rs, -3, 3)   # row z-score => mesor removed
        return matrix_z, time_labels

    def draw_one(matrix_z, time_labels, gene_order, title, out_path, fig_w, fig_h, label_w):
        if matrix_z.size == 0 or matrix_z.shape[1] == 0:
            print(f"  [skip] no data to plot for: {title}")
            return
        n_genes = len(gene_order)
        show_labels = n_genes <= 50
        gene_fontsize = max(2.0, min(7.0, (1920.0 / max(n_genes, 1)) * 0.5))
        panel_w = len(time_labels) * 0.55
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([label_w / fig_w, 0.1, panel_w / fig_w, 0.8])
        ax.set_facecolor('lightgray')
        im = ax.imshow(matrix_z, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_xticks(range(len(time_labels)))
        ax.set_xticklabels(time_labels, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight='bold')
        if show_labels:
            ax_lbl = fig.add_axes([0, 0.1, label_w / fig_w, 0.8])
            ax_lbl.set_ylim(-0.5, n_genes - 0.5)
            ax_lbl.invert_yaxis()
            ax_lbl.axis('off')
            for i, name in enumerate(gene_order):
                ax_lbl.text(0.95, i, name, ha='right', va='center',
                            fontsize=gene_fontsize, family='monospace')
        ax_cb = fig.add_axes([(label_w + panel_w + 0.3) / fig_w, 0.3, 0.2 / fig_w, 0.4])
        plt.colorbar(im, cax=ax_cb, label='Z-score')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    if not comparisons_to_run:
        comparisons_to_run = list(combinations(sorted(df_res['group_name'].unique()), 2))

    for g1, g2 in tqdm(comparisons_to_run, desc="Per-category heatmaps", unit="pair"):
        pair_label = f"{g1} vs {g2}"
        df_pair = df_comparisons[df_comparisons['Pair'] == pair_label]
        if df_pair.empty:
            continue

        for cat in category_order:
            genes_cat = df_pair[df_pair['Biological_Category'] == cat]['Gene'].unique().tolist()
            if not genes_cat:
                continue

            # Cat 1 (Arrhythmic): acrophase is meaningless for arrhythmic genes,
            # so order by the mesor change (Delta_Mesor = g2 - g1), ascending.
            # Every other category keeps the acrophase ordering.
            if cat.startswith('Cat 1'):
                delta_lookup = dict(zip(
                    df_pair['Gene'],
                    pd.to_numeric(df_pair['Delta_Mesor'], errors='coerce')))

                def sort_key(gene):
                    d = delta_lookup.get(gene, np.nan)
                    return (np.isnan(d), d if not np.isnan(d) else 0.0)

                sort_criterion = 'mesor change'
            else:
                # Order by acrophase, preferring the rhythmic group:
                #   Cat 3 (group_2_only) -> order by g2; otherwise by g1, fallback g2.
                if 'group_2_only' in cat:
                    ref_for_sort, alt_for_sort = g2, g1
                else:
                    ref_for_sort, alt_for_sort = g1, g2

                def sort_key(gene):
                    p = phase_lookup.get((gene, ref_for_sort), np.nan)
                    if np.isnan(p):
                        p = phase_lookup.get((gene, alt_for_sort), np.nan)
                    return (np.isnan(p), p if not np.isnan(p) else 0.0)

                sort_criterion = 'acrophase'

            gene_order = sorted(genes_cat, key=sort_key)
            n_genes = len(gene_order)

            m1, tl1 = build_matrix(g1, gene_order)
            m2, tl2 = build_matrix(g2, gene_order)

            n_tp = max(len(tl1), len(tl2), 1)
            fig_h = 12.0
            label_w = 1.8 if n_genes <= 300 else 0.2
            fig_w = max(8.0, label_w + n_tp * 0.55 + 1.5)

            cat_tag = cat.split(':')[0].strip().replace(' ', '')   # "Cat1", "Cat2", ...
            safe = f"{g1}_vs_{g2}".replace(' ', '_').replace('/', '_')

            draw_one(m1, tl1, gene_order,
                     f'{cat}\n{g1} ({n_genes} genes, ordered by {sort_criterion})',
                     os.path.join(base_dir, 'plots', f'heatmap_{cat_tag}_{safe}__A_{g1}.png'),
                     fig_w, fig_h, label_w)
    return None

# ===================================================================
# MODEL-BASED HEATMAPS (CODAC_Multi)
# One heatmap per grouping model (M01..M15): all groups shown side by side as
# panels, row z-scored (mesor removed -> amplitude+phase remain), genes ordered
# by the acrophase of the model's rhythmic group (M01 has none -> ordered by the
# baseline/mesor difference). Replaces the old pairwise per-category heatmaps.
# ===================================================================

def _zmatrix(df_long, grp, gene_order, n_observations, tp_prefix):
    df_grp = df_long[df_long['Group'] == grp]
    timepoints = sorted(df_grp['Time'].unique())
    time_labels = [f"{tp_prefix}{h}" for h in timepoints]
    grouped = df_grp.groupby('Gene')
    rows = []
    for gene in gene_order:
        y = grouped.get_group(gene)['Value'].values if gene in grouped.groups else []
        if len(y) == 0:
            rows.append([np.nan] * len(timepoints)); continue
        means = [np.nanmean(y[i:i + n_observations]) if len(y[i:i + n_observations]) > 0 else np.nan
                 for i in range(0, len(y), n_observations)]
        means = (means + [np.nan] * len(timepoints))[:len(timepoints)]
        rows.append(means)
    matrix = np.array(rows, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rm = np.nanmean(matrix, axis=1, keepdims=True)
        rs = np.nanstd(matrix, axis=1, keepdims=True)
    rs[rs == 0] = 1
    return np.clip((matrix - rm) / rs, -3, 3), time_labels

def _rhythmic_from_code(code, groups):
    # Canonical rhythmic set for a model code ("M05" -> frozenset of rhythmic groups).
    try:
        return sorted(enumerate_rhythm_models(groups)[int(code[1:]) - 1][0])
    except (ValueError, IndexError, TypeError):
        return []

def _model_gene_order(genes, code, groups, phase_lookup, mesor_lookup):
    rset = _rhythmic_from_code(code, groups)
    if rset:
        sort_grp = next((g for g in groups if g in rset), rset[0])
        def key(gene):
            p = phase_lookup.get((gene, sort_grp), np.nan)
            return (np.isnan(p), p if not np.isnan(p) else 0.0)
    else:
        # M01 (arrhythmic in all): order by mesor of first group minus the rest.
        first = groups[0]; rest = groups[1:]
        def key(gene):
            m0 = mesor_lookup.get((gene, first), np.nan)
            mr = np.nanmean([mesor_lookup.get((gene, g), np.nan) for g in rest]) if rest else np.nan
            d = m0 - mr
            return (np.isnan(d), d if not np.isnan(d) else 0.0)
    return sorted(genes, key=key)

def generate_heatmap_by_model(df_global, results, df_long, n_observations, groups,
                              base_dir, time_label='Time (ZT Hours)'):
    print("\n" + "=" * 70, flush=True)
    print('             GENERATING MODEL-BASED HEATMAPS           ', flush=True)
    print("=" * 70, flush=True)
    if df_global is None or df_global.empty or 'Grouping_Model' not in df_global.columns:
        print("  [skip] no grouping table available."); return []

    tp_prefix = 'CT' if 'CT' in time_label else ('ZT' if 'ZT' in time_label else '')
    phase_lookup = {(r['gene'], r['group_name']): r.get('phase', np.nan) for r in results}
    mesor_lookup = {(r['gene'], r['group_name']): r.get('mesor', np.nan) for r in results}
    grp_colors = {g: plt.cm.tab10(i % 10) for i, g in enumerate(groups)}

    from collections import defaultdict
    by_model = defaultdict(list)
    for _, row in df_global.iterrows():
        code = row.get('Grouping_Model', "")
        if isinstance(code, str) and code.startswith('M'):
            by_model[code].append(row['Gene'])

    saved = []
    for code in tqdm(sorted(by_model), desc="Model heatmaps", unit="model"):
        genes = by_model[code]
        if not genes:
            continue
        gene_order = _model_gene_order(genes, code, groups, phase_lookup, mesor_lookup)
        n_genes = len(gene_order)
        rset = _rhythmic_from_code(code, groups)
        sort_desc = f"{next((g for g in groups if g in rset), '')} acrophase" if rset else "mesor difference"

        mats = [(_zmatrix(df_long, g, gene_order, n_observations, tp_prefix), g) for g in groups]
        n_tp = max((len(tl) for (_, tl), _ in mats), default=1)
        show_labels = n_genes <= 50
        label_w = 1.8 if show_labels else 0.25
        panel_w = n_tp * 0.5
        gap = 0.35
        fig_w = label_w + len(groups) * (panel_w + gap) + 1.2
        fig_h = 12.0
        fig = plt.figure(figsize=(fig_w, fig_h))
        x0 = label_w
        im = None
        for (mz, tl), g in mats:
            ax = fig.add_axes([x0 / fig_w, 0.08, panel_w / fig_w, 0.82])
            ax.set_facecolor('lightgray')
            im = ax.imshow(mz, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
            ax.set_xticks(range(len(tl))); ax.set_xticklabels(tl, rotation=45, ha='right', fontsize=7)
            ax.set_yticks([])
            ax.set_title(g, fontsize=11, fontweight='bold', color=grp_colors[g])
            x0 += panel_w + gap
        if show_labels:
            ax_lbl = fig.add_axes([0, 0.08, label_w / fig_w, 0.82])
            ax_lbl.set_ylim(-0.5, n_genes - 0.5); ax_lbl.invert_yaxis(); ax_lbl.axis('off')
            fs = max(2.0, min(7.0, (1920.0 / max(n_genes, 1)) * 0.5))
            for i, name in enumerate(gene_order):
                ax_lbl.text(0.95, i, name, ha='right', va='center', fontsize=fs, family='monospace')
        cax = fig.add_axes([(fig_w - 0.9) / fig_w, 0.08, 0.15 / fig_w, 0.82])
        fig.colorbar(im, cax=cax, label='row z-score')
        fig.suptitle(f"Model {code}  ({n_genes} genes, ordered by {sort_desc})",
                     fontsize=13, fontweight='bold')
        out = os.path.join(base_dir, 'plots', f'heatmap_model_{code}.png')
        fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
        saved.append(out)
    return saved

def generate_heatmap_consolidated(df_global, results, df_long, n_observations, groups,
                                  base_dir, time_label='Time (ZT Hours)'):
    # One tall figure: models M02..M15 stacked vertically (M01 excluded -- focus on
    # rhythmic targets), the groups side by side as column-panels, each model block
    # ordered internally by its rhythmic group's acrophase. No per-gene labels; each
    # model block is marked by its code on the left.
    if df_global is None or df_global.empty or 'Grouping_Model' not in df_global.columns:
        return None
    tp_prefix = 'CT' if 'CT' in time_label else ('ZT' if 'ZT' in time_label else '')
    phase_lookup = {(r['gene'], r['group_name']): r.get('phase', np.nan) for r in results}
    mesor_lookup = {(r['gene'], r['group_name']): r.get('mesor', np.nan) for r in results}
    grp_colors = {g: plt.cm.tab10(i % 10) for i, g in enumerate(groups)}

    from collections import defaultdict
    by_model = defaultdict(list)
    for _, row in df_global.iterrows():
        code = row.get('Grouping_Model', "")
        if isinstance(code, str) and code.startswith('M') and code != 'M01':
            by_model[code].append(row['Gene'])
    codes = [c for c in sorted(by_model) if by_model[c]]
    if not codes:
        return None

    # Ordered gene list (blocks concatenated) + block boundaries.
    ordered, blocks = [], []
    for code in codes:
        go = _model_gene_order(by_model[code], code, groups, phase_lookup, mesor_lookup)
        blocks.append((code, len(ordered), len(ordered) + len(go)))
        ordered.extend(go)
    total = len(ordered)

    mats = [(_zmatrix(df_long, g, ordered, n_observations, tp_prefix), g) for g in groups]
    n_tp = max((len(tl) for (_, tl), _ in mats), default=1)
    left_w = 0.9
    panel_w = n_tp * 0.5
    gap = 0.35
    fig_w = left_w + len(groups) * (panel_w + gap) + 1.2
    fig_h = max(6.0, min(40.0, total * 0.02 + 2))
    fig = plt.figure(figsize=(fig_w, fig_h))
    x0 = left_w
    im = None
    for (mz, tl), g in mats:
        ax = fig.add_axes([x0 / fig_w, 0.05, panel_w / fig_w, 0.88])
        ax.set_facecolor('lightgray')
        im = ax.imshow(mz, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_xticks(range(len(tl))); ax.set_xticklabels(tl, rotation=45, ha='right', fontsize=7)
        ax.set_yticks([])
        ax.set_title(g, fontsize=12, fontweight='bold', color=grp_colors[g])
        for _, s, e in blocks:                 # separator lines between models
            if s > 0:
                ax.axhline(s - 0.5, color='black', lw=0.8)
        x0 += panel_w + gap
    # Model labels + brackets on the left.
    ax_lbl = fig.add_axes([0, 0.05, left_w / fig_w, 0.88])
    ax_lbl.set_ylim(total - 0.5, -0.5); ax_lbl.set_xlim(0, 1); ax_lbl.axis('off')
    for code, s, e in blocks:
        mid = (s + e) / 2.0
        ax_lbl.text(0.5, mid, code, ha='center', va='center', fontsize=9, fontweight='bold', rotation=90)
    cax = fig.add_axes([(fig_w - 0.9) / fig_w, 0.05, 0.15 / fig_w, 0.88])
    fig.colorbar(im, cax=cax, label='row z-score')
    fig.suptitle(f"Consolidated grouping heatmap  (models M02-M15, {total} rhythmic targets)",
                 fontsize=13, fontweight='bold')
    out = os.path.join(base_dir, 'plots', 'heatmap_consolidated.png')
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    return out

def bundle_heatmaps_pdf(paths, base_dir):
    # Bundle the consolidated + per-model heatmap PNGs into a single PDF.
    valid = [p for p in paths if p and os.path.exists(p)]
    if not valid:
        return None
    out = os.path.join(base_dir, 'plots', 'CODAC_Multi_heatmaps.pdf')
    with PdfPages(out) as pdf:
        for p in valid:
            img = plt.imread(p)
            h, w = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(min(14, w / 130.0), min(18, h / 130.0)))
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis('off')
            pdf.savefig(fig); plt.close(fig)
    return out


# -------------------------------------------------------------------
# Polar Rose Plot
# -------------------------------------------------------------------
def generate_polar_plot(results, p_threshold, r2_threshold, counter_threshold, base_dir, time_label='Time'):
    time.sleep(0.1)
    print("\n" + "=" * 70)
    print('                       GENERATING POLAR PLOTS                      ')
    print("=" * 70)

    df_res = pd.DataFrame(results)
    if df_res.empty: return None

    # Filter only reliable rhythmic genes for the polar plot
    valid_genes = df_res[
        (df_res['p_value_final'] <= p_threshold) &
        (df_res['r_squared'] >= r2_threshold) &
        (df_res['counter'] >= counter_threshold)
        ]

    groups = sorted(df_res['group_name'].unique())
    polar_figs = {}

    # Progress bar
    for grp in tqdm(groups, desc="Polar Plots", unit="group"):
        grp_data = valid_genes[valid_genes['group_name'] == grp]
        if grp_data.empty:
            continue

        phases = grp_data['phase'].values
        phases_rad = phases * (2 * np.pi / 24)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)

        bins = np.linspace(0, 2 * np.pi, 25)
        counts, _ = np.histogram(phases_rad, bins=bins)

        bars = ax.bar(bins[:-1], counts, width=2 * np.pi / 24, bottom=0.0,
                      alpha=0.7, edgecolor='black', color='royalblue')

        ticks = np.linspace(0, 2 * np.pi, 24, endpoint=False)[::3]
        tick_labels = [f"ZT{int(x)}" if 'ZT' in time_label else f"{int(x)}h"
                       for x in np.linspace(0, 24, 24, endpoint=False)[::3]]

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=10, fontweight='bold')

        plt.title(f'Phase Distribution (Rose Plot) - Group: {grp}\nN = {len(phases)} rhythmic genes',
                  pad=20, fontsize=12, fontweight='bold')

        out_path = os.path.join(base_dir, 'plots', f'plot_polar_{grp}.png')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        polar_figs[grp] = fig

    return polar_figs

#-------------------------------------------------------------------
# Nested Nonlinear Models (Nested NLS)
#-------------------------------------------------------------------
def run_nls_pairwise_test(df_gene, g1, g2, r1_params, r2_params, period=24.0):
    # Uses Nested Nonlinear Models (Nested NLS) to isolate the origin of the circadian
    # difference by measuring the Sum of Squared Residuals (SSR).
    mask = df_gene['Group'].isin([g1, g2])
    t = df_gene.loc[mask, 'Time'].values
    y = df_gene.loc[mask, 'Value'].values
    g = np.where(df_gene.loc[mask, 'Group'] == g2, 1, 0)  # 0 = Ref, 1 = Treatment

    # Remove NaNs and ensure sufficient points
    valid = ~np.isnan(y)
    t, y, g = t[valid], y[valid], g[valid]
    n = len(y)

    out = {'p_diff_mesor': np.nan, 'p_diff_amplitude': np.nan, 'p_diff_phase': np.nan}
    if n < 6: return out

    omega = 2.0 * np.pi / period
    x_data = np.vstack((t, g))

    # ---- DEFINITION OF TRIGONOMETRIC MODELS ----
    # 1. Full (Independent Mesor, Amp and Phase)
    def mod_full(x, M, dM, A, dA, P, dP):
        return (M + dM * x[1]) + (A + dA * x[1]) * np.cos(omega * x[0] - (P + dP * x[1]))

    # 2. Restricted 1: Forces the same Mesor
    def mod_no_mesor(x, M, A, dA, P, dP):
        return M + (A + dA * x[1]) * np.cos(omega * x[0] - (P + dP * x[1]))

    # 3. Restricted 2: Force the same Amplitude
    def mod_no_amp(x, M, dM, A, P, dP):
        return (M + dM * x[1]) + A * np.cos(omega * x[0] - (P + dP * x[1]))

    # 4. Restricted 3: Force the same Phase
    def mod_no_phase(x, M, dM, A, dA, P):
        return (M + dM * x[1]) + (A + dA * x[1]) * np.cos(omega * x[0] - P)

    # Initial point
    p1_rad = r1_params['Phase'] * (2 * np.pi / period)
    p2_rad = r2_params['Phase'] * (2 * np.pi / period)

    p0_full = [
        r1_params['Mesor'], r2_params['Mesor'] - r1_params['Mesor'],
        r1_params['Amplitude'], r2_params['Amplitude'] - r1_params['Amplitude'],
        p1_rad, p2_rad - p1_rad
    ]

    def get_rss(func, p0_guess):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            try:
                popt, _ = curve_fit(func, x_data, y, p0=p0_guess, maxfev=10000)
                preds = func(x_data, *popt)
                return np.sum((y - preds) ** 2)
            except:
                return np.inf

    rss_full = get_rss(mod_full, p0_full)
    if rss_full == np.inf or rss_full == 0: return out

    df_full = n - 6  # Degrees of freedom

    # RSS from Restricted Models
    rss_no_mesor = get_rss(mod_no_mesor, [p0_full[0], p0_full[2], p0_full[3], p0_full[4], p0_full[5]])
    rss_no_amp = get_rss(mod_no_amp, [p0_full[0], p0_full[1], p0_full[2], p0_full[4], p0_full[5]])
    rss_no_phase = get_rss(mod_no_phase, [p0_full[0], p0_full[1], p0_full[2], p0_full[3], p0_full[4]])

    # ---- F-TEST (Error Penalty) ----
    def calc_p(rss_restricted):
        if rss_restricted == np.inf: return np.nan
        F = ((rss_restricted - rss_full) / 1) / (rss_full / df_full)
        return stats.f.sf(F, 1, df_full) if F > 0 else np.nan

    out['p_diff_mesor'] = calc_p(rss_no_mesor)
    out['p_diff_amplitude'] = calc_p(rss_no_amp)
    out['p_diff_phase'] = calc_p(rss_no_phase)

    return out

#-------------------------------------------------------------------
# To Excel (robust: formatted via xlsxwriter, plain fallback via openpyxl)
#-------------------------------------------------------------------
def _export_excel_plain(df, file_path):
    # Fallback export: no styling, but always works (openpyxl is always present).
    df.to_excel(file_path, sheet_name='Resultados', index=False, engine='openpyxl')


def export_excel_merged(df, file_path):
    # Try the fully formatted export (xlsxwriter). If xlsxwriter is not
    # installed in the environment, fall back to a plain openpyxl export so the
    # data is ALWAYS written to disk instead of failing silently.
    try:
        import xlsxwriter  # noqa: F401  (only to check availability)
        _export_excel_formatted(df, file_path)
    except ModuleNotFoundError:
        print("[WARNING] 'xlsxwriter' not found; exporting a plain (unformatted) "
              "Excel file via openpyxl. Install xlsxwriter for the formatted version.")
        _export_excel_plain(df, file_path)

#-------------------------------------------------------------------
# To Excel
#-------------------------------------------------------------------
def _export_excel_formatted(df, file_path):
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    df.to_excel(writer, sheet_name='Resultados', index=False)

    workbook = writer.book
    worksheet = writer.sheets['Resultados']

    # ==========================================
    # 1. Definition of Basic Styles
    # ==========================================
    base_format = {'align': 'center', 'valign': 'vcenter', 'border': 1}

    header_base = {**base_format, 'bold': True, 'bg_color': '#D3D3D3', 'size': 12}
    header_format = workbook.add_format(header_base)
    header_thick_left = workbook.add_format({**header_base, 'left': 2})

    fmt_normal = workbook.add_format(base_format)
    fmt_thick_left = workbook.add_format({**base_format, 'left': 2})

    fmt_normal_bottom = workbook.add_format({**base_format, 'bottom': 2})
    fmt_thick_left_bottom = workbook.add_format({**base_format, 'left': 2, 'bottom': 2})

    fmt_hl = workbook.add_format({**base_format, 'bg_color': '#C6EFCE', 'font_color': '#006100'})
    fmt_hl_bottom = workbook.add_format({**base_format, 'bottom': 2, 'bg_color': '#C6EFCE', 'font_color': '#006100'})

    fmt_merge = workbook.add_format({**base_format, 'bottom': 2})
    fmt_merge_highlight = workbook.add_format(
        {**base_format, 'bottom': 2, 'bg_color': '#C6EFCE', 'font_color': '#006100'})

    # ==========================================
    # 2. Write Headers and Auto-Adjust Width
    # ==========================================
    for col_num, col_name in enumerate(df.columns):
        if col_name in ['Group', 'Pair']:
            worksheet.write(0, col_num, col_name, header_thick_left)
        else:
            worksheet.write(0, col_num, col_name, header_format)

        max_len_data = df[col_name].apply(lambda x: len(str(x))).max()
        max_len_data = max_len_data if pd.notna(max_len_data) else 0
        max_len_header = len(str(col_name))
        tamanho_ideal = max(max_len_data, max_len_header)
        worksheet.set_column(col_num, col_num, max(tamanho_ideal + 3, 12))

    # ==========================================
    # 3. Write the Normal Data and Color
    # ==========================================
    #
    genes_list_short = df['Target'].astype(str).tolist()

    for row_idx in range(len(df)):
        is_last_row_of_gene = False
        if row_idx == len(df) - 1:
            is_last_row_of_gene = True
        elif genes_list_short[row_idx] != genes_list_short[row_idx + 1]:
            is_last_row_of_gene = True

        for col_idx, col_name in enumerate(df.columns):
            val = df.iloc[row_idx, col_idx]
            val_str = "" if pd.isna(val) or val == "" else val

            if col_name in ['Group', 'Pair']:
                cell_format = fmt_thick_left_bottom if is_last_row_of_gene else fmt_thick_left
            else:
                cell_format = fmt_normal_bottom if is_last_row_of_gene else fmt_normal

            if col_name in ['p_diff_mesor', 'p_diff_amplitude', 'p_diff_phase','p_diff_mesor_FDR', 'p_diff_amplitude_FDR', 'p_diff_phase_FDR']:
                try:
                    if val_str != "" and float(val) < 0.05:
                        cell_format = fmt_hl_bottom if is_last_row_of_gene else fmt_hl
                except ValueError:
                    pass

            worksheet.write(row_idx + 1, col_idx, val_str, cell_format)

    # ==========================================
    # 4. Merge Global Columns and Color
    # ==========================================
    columns_gene = ['Target', 'p_global_rhythm', 'p_global_rhythm_FDR', 'p_rhythm_diff', 'p_rhythm_diff_FDR', 'p_mesor_diff', 'p_mesor_diff_FDR',
                    'Grouping', 'Grouping_Model', 'Grouping_Confidence', 'Grouping_IC_Gap',
                    'Grouping_Mesor', 'Grouping_Mesor_Model', 'Grouping_Mesor_Confidence', 'Grouping_Mesor_IC_Gap']

    for col_name in columns_gene:
        if col_name not in df.columns: continue
        col_idx = df.columns.get_loc(col_name)
        valores_lista = df[col_name].tolist()
        start_idx = 0

        # The loop now runs freely, without 'with tqdm'
        while start_idx < len(df):
            current_gene = genes_list_short[start_idx]
            if current_gene == "" or current_gene == "nan":
                start_idx += 1
                continue

            end_idx = start_idx
            while end_idx + 1 < len(df) and genes_list_short[end_idx + 1] == current_gene:
                end_idx += 1

            val = valores_lista[start_idx]
            val_str = "" if pd.isna(val) else val
            current_format = fmt_merge

            if col_name == 'p_rhythm_diff':
                try:
                    if val_str != "" and float(val) < 0.05:
                        current_format = fmt_merge_highlight
                except ValueError:
                    pass

            if end_idx > start_idx:
                worksheet.merge_range(start_idx + 1, col_idx, end_idx + 1, col_idx, val_str, current_format)
            else:
                worksheet.write(start_idx + 1, col_idx, val_str, current_format)

            start_idx = end_idx + 1
    writer.close()

#-----------------------------------------------------------------------
#                             Main function
#-----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Path Settings and Initial Reading
    # ------------------------------------------------------------------
    base_dir = os.getcwd()
    # All outputs go into a single CODA_Results folder next to the data,
    # so the R and PyCharm runs produce the exact same structure.
    results_dir = os.path.join(base_dir, 'CODA_Results')
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    global df_comparisons, df_export, df_master

    import __main__
    is_r_injected = hasattr(__main__, 'df_input')

    if is_r_injected:
        print("\n[INFO] Perfect connection with R! Reading the injected parameters.")

        r_timepoints = [str(t).strip() for t in list(__main__.r_timepoints)]
        r_groups = [str(g).strip() for g in list(__main__.r_groups)]

        # Comparisons come from R as a list of pairs (group1, group2)
        comparisons = []
        raw_comparisons = getattr(__main__, 'r_comparisons', None)
        if raw_comparisons is not None:
            for pair in raw_comparisons:
                comparisons.append((str(pair[0]).strip(), str(pair[1]).strip()))

        # Genes to plot
        genes_to_plot = []
        raw_genes = getattr(__main__, 'r_targets_to_plot', None)
        if raw_genes is not None:
            if isinstance(raw_genes, str):
                genes_to_plot = [raw_genes.upper()]
            else:
                genes_to_plot = [str(g).upper() for g in raw_genes]

        input_config = {
            'n_timepoints': len(r_timepoints),
            'n_observations': int(__main__.r_n_obs),
            'timepoints': r_timepoints,
            'groups': r_groups,
            'comparisons': comparisons,
            'genes_to_plot': genes_to_plot,
        }

        df_long, df_raw = read_data_file(None, input_config, df_raw=__main__.df_input)
        result_file = os.path.join(results_dir, "CODAC_Multi_Results.csv")

    else:
        print("\n[INFO] R environment not detected. Running direct in Python.")
        input_file = os.path.join(base_dir, 'compare_plus_input.txt')
        name_file = None
        if os.path.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                for row in f:
                    if row.startswith('Data File:'):
                        name_file = row.split(':', 1)[1].strip()
                        break
        if name_file is None:
            print("\n[ERROR] compare_input.txt not found or missing 'Data File:'. Aborting.")
            return

        data_file = os.path.join(base_dir, name_file)
        output_name_file = f"output_{name_file.replace('.txt', '.csv')}"
        result_file = os.path.join(results_dir, output_name_file)

        input_config = read_input_file(input_file)
        df_long, df_raw = read_data_file(data_file, input_config)

    # Extracts the pairs and genes from the configuration.
    comparisons_to_run = input_config.get('comparisons', [])

    # --- Validation: reject comparisons of a group against itself ---
    # Comparing a group with itself is meaningless, so we stop early with a clear
    # message instead of producing nonsensical results.
    if comparisons_to_run:
        self_pairs = [pair for pair in comparisons_to_run
                      if len(pair) == 2 and str(pair[0]).strip() == str(pair[1]).strip()]
        if self_pairs:
            offending = ", ".join(f'("{p[0]}", "{p[1]}")' for p in self_pairs)
            print("\n" + "!" * 70)
            print("[ERROR] A comparison lists the same group twice: " + offending)
            print("\nA group cannot be compared with itself. Fix the 'comparisons'")
            print("\nargument (each pair must have two DIFFERENT group names) and run again.")
            print("!" * 70 + "\n")
            raise ValueError(f"Invalid comparison(s) with repeated group: {offending}")
    genes_to_plot = input_config.get('genes_to_plot', [])

    n_timepoints = input_config['n_timepoints']
    n_observations = input_config['n_observations']
    timepoints = input_config['timepoints']
    groups = input_config['groups']

    # ------------------------------------------------------------------
    # 2. Parameter Initialization (Replaces Terminal Inputs)
    # ------------------------------------------------------------------
    # These variables are safely parameterized via R reticulate or PyCharm.

    # Captures global scope variables if injected by R, otherwise uses the default.
    current_vars = globals()

    interval_var = current_vars.get('interval_var', 1)
    r2_threshold = current_vars.get('r2_threshold', 0.4)
    plot_flag = current_vars.get('plot_flag', 'Y')
    plot_all = current_vars.get('plot_all', 'N')
    amp_stringency = current_vars.get('amp_stringency', 0.5)
    amp_stringency = min(max(float(amp_stringency), 0.0), 1.0)  # clamp to [0, 1]

    # Fixed Period (24 hours)
    is_fixed_period = True
    fixed_period = True
    period_lower = 24.0
    period_upper = 24.0

    p_value_option = current_vars.get('p_value_option', 'FDR') # 'FDR' = adjusted p-value (Benjamini-Hochberg) | 'RAW' or 'O' = original p-value
    p_threshold = current_vars.get('p_threshold', 0.05)
    rhythmicity_cutoff = current_vars.get('rhythmicity_cutoff', 'HIGH')
    p_value_comparison = current_vars.get('p_value_comparison', 'RAW')

    # Information criterion for the multi-group GROUPING selection (CODAC_Multi).
    #   'BIC'  (default) -- stronger complexity penalty, more conservative about
    #                       calling a difference (aligned with dryR and with the
    #                       anti-over-calling philosophy). Recommended.
    #   'AICc'           -- more permissive; better sensitivity to subtle
    #                       differences, at a higher false-positive risk.
    selection_criterion = str(current_vars.get('selection_criterion', 'BIC')).strip().upper()
    if selection_criterion not in ('BIC', 'AICC'):
        print(f"[WARN] Unknown selection_criterion '{selection_criterion}'; falling back to 'BIC'.")
        selection_criterion = 'BIC'
    print(f"[INFO] Grouping selection criterion: {selection_criterion}.")

    # Which p-value drives the GLOBAL gates (p_rhythm_diff / p_mesor_diff):
    #   'FDR' (default) -- Benjamini-Hochberg corrected across all targets, the
    #                      right choice for a genome-wide screen (the gate tests
    #                      run once per target over thousands of targets).
    #   'RAW'           -- uncorrected per-target p-values.
    # Both raw and *_FDR columns are always exported; only the gate DECISION
    # switches. (In codac_compare() these globals are reported but do not gate.)
    p_value_global = str(current_vars.get('p_value_global', 'FDR')).strip().upper()
    if p_value_global not in ('RAW', 'FDR'):
        print(f"[WARN] Unknown p_value_global '{p_value_global}'; falling back to 'FDR'.")
        p_value_global = 'FDR'
    print(f"[INFO] Global-gate p-value source: {p_value_global}.")

    missing_data_action = current_vars.get('missing_data_action', 'KEEP')  # 'KEEP', 'IMPUTE' or 'REMOVE'
    exclude_medium = current_vars.get('exclude_medium', True)
    if isinstance(exclude_medium, str):
        exclude_medium = exclude_medium.strip().upper() in ('TRUE', 'T', 'Y', 'YES', '1')
    else:
        exclude_medium = bool(exclude_medium)

    valid_categories = ["ARRHYTHMIC", "LOW", "MEDIUM", "HIGH", "EXTREMELY HIGH"]
    if rhythmicity_cutoff not in valid_categories:
        rhythmicity_cutoff = "HIGH"

    time_label = current_vars.get('time_label', 'Time (ZT Hours)')

    rhythmicity_label = current_vars.get('min_rhythmicity', 'HIGH')
    rhythm_map = {'EXTREMELY HIGH': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'ARRHYTHMIC': 0}
    counter_threshold = rhythm_map.get(rhythmicity_label, 3)

    print("\n" + "=" * 70)
    print('                       PROCESSING RESULTS                          ')
    print("=" * 70)
    time.sleep(0.1)

    # ------------------------------------------------------------------
    # Storage and Prep
    # ------------------------------------------------------------------
    # Storage vectors
    results = []
    amp_limits = []
    global_results = []
    suggested_r2 = None

    # Validation of genes for plotting data read from Input.txt
    if plot_flag == 'Y' and plot_all == 'N':
        if not genes_to_plot:
            print("Warning: You chose to plot specific genes, but the [GenesToPlot] list in Input.txt is empty.")
    elif plot_flag == 'N':
        # If the user doesn't want to plot anything, we clear the list for safety reasons.
        genes_to_plot = []

    # Handle missing data
    df_long = handle_missing_data_df(df_long, missing_data_action=missing_data_action)

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print('                           EVALUATING TARGETS                      ')
    print("=" * 70)
    time.sleep(0.1)

    # ------------------------------------------------------------------
    # PART A: Beginning of the main loop by gene
    # ------------------------------------------------------------------
    for gene in tqdm(df_long['Gene'].unique(), desc="Processing targets",unit="target"):
        # We filtered all the data for this gene at once.
        gene_data_all = df_long[df_long['Gene'] == gene]
        plot_data_gene = []
        df_gene_rows = []
        groups_processed = 0
        gene_counters = {}   # per-group multi-criteria score (tier) for this gene

        # We use Pandas' groupby feature.
        for group_name in groups:
            # Filter the data only from this group for the current gene.
            series_data = gene_data_all[gene_data_all['Group'] == group_name]

            if series_data.empty: continue
            gene_label = f"{gene} | {group_name}"

            # We extracted X (Time) and y (Expression).
            X = series_data['Time'].values
            y = series_data['Value'].values

            # Sorting (important for the line graph later)
            order_idx = np.argsort(X)
            X = X[order_idx]
            y = y[order_idx]

            # We've saved this for later multi-group analysis.
            for x_val, y_val in zip(X, y):
                df_gene_rows.append({
                    'gene': gene,
                    'group': group_name,
                    'time': float(x_val),
                    'expr': float(y_val)
                })

            groups_processed += 1
            try:
                # Per-timepoint replicate chunks (used for stats and the outlier test).
                group_list = [y[i: i + n_observations] for i in range(0, len(y), n_observations)]

                # Outlier detection: IQR *within each timepoint* (across replicates),
                # so it flags a bad replicate without ever removing genuine rhythm
                # peaks/troughs. Feeds only the interval test below, not the fit.
                has_outliers, outliers = check_outliers(group_list)
                y_valid = y[~np.isnan(y)]
                values_filtered = y_valid[~np.isin(y_valid, outliers)]

                # Rhythmicity F-test (Cosinor Linear vs Nulo)
                F_stat, p_value_rhythm, _, _ = test_rhythmicity_cosinor(X, y)

                # Group Statistics (Average of Means and SD)
                means, mean_of_means, std_dev = calculate_full_metrics(group_list)

                # ------------------------------------------------------------------
                # PART B: Fine-tuning and Quality Criteria
                # ------------------------------------------------------------------
                mean_of_means = np.mean(means) if means else 0
                std_dev = np.std(means) if means else 0

                # Adaptive threshold (same shape as before), then scaled by the
                # user's stringency dial: 0.5 -> unchanged, 0 -> off, 1.0 -> 2x.
                adaptive_limit = max(min(_amp_mean_ratio * mean_of_means, _amp_std_ratio * std_dev), _amp_floor)
                amp_limit = (2.0 * amp_stringency) * adaptive_limit
                amp_limits.append(amp_limit)

                # Curve fitting principal
                if is_fixed_period:
                    params, r_squared, y_pred = perform_curve_fit(X, y, fixed_period=True)
                    # We forced the period to 24 hours for the subsequent calculation.
                    if params is not None:
                        params = (params[0], params[1], params[2], 24.0)
                        r_squared = calc_r2(y, circular_function(X, *params))
                else:
                    params, r_squared, y_pred = perform_curve_fit(
                        X, y, fixed_period=False, bounds=(period_lower, period_upper))

                if params is None:
                    continue

                # Extraction of biological parameters
                k_est, a_est, f_est, T_est = params

                # Interpolation to find the actual peak (Acrophase)
                X_interp_local = np.linspace(np.min(X), np.max(X), 500)
                y_interp_local = circular_function(X_interp_local, k_est, a_est, f_est, T_est)

                x_peak = X_interp_local[np.argmax(y_interp_local)]
                y_peak = np.max(y_interp_local)
                y_trough = np.min(y_interp_local)
                amplitude_corr = y_peak - k_est

                # Real decimal hours -- used for ALL math (Delta_Phase, NLS, polar,
                # heatmap sort). The reported 'phase' MUST be decimal.
                phase_decimal = x_peak
                period_decimal = T_est

                # Display-only "clock" format: the digits after the dot are MINUTES
                # (e.g. 13.45 = 13h45min). NEVER do arithmetic on this column.
                hour_phase = int(x_peak)
                minute_phase = int(round((x_peak - hour_phase) * 60)) / 100.0
                if minute_phase > 0.59:
                    minute_phase = 0.0
                    hour_phase += 1
                phase_hhmm = hour_phase + minute_phase

                # Application of the 4 Criteria of Rhythmicity (points = criteria MET)
                counter = 0
                if amplitude_corr >= amp_limit: counter += 1
                if r_squared >= r2_threshold: counter += 1

                # Interval Test (Percentiles)
                aux1 = 25 - 10 * (interval_var - 1)
                lower_bound = np.percentile(values_filtered, aux1)
                upper_bound = np.percentile(values_filtered, 100 - aux1)

                if (y_trough >= lower_bound) and (y_peak <= upper_bound):
                    interval_flag = 'In'
                else:
                    interval_flag = 'Out'
                    counter += 1

                # Definition of the Rhythmicity Class
                p_value_final = p_value_rhythm
                if r_squared > r2_threshold and p_value_final <= p_threshold and amplitude_corr >= amp_limit:
                    rhythm_class = 'Significant'
                elif r_squared > r2_threshold and p_value_final <= p_threshold and amplitude_corr < amp_limit:
                    rhythm_class = 'LowAmplitude'
                else:
                    rhythm_class = 'NonSignificant'

                result_row = {
                    'gene': gene,
                    'group_name': group_name,
                    'mean': float(mean_of_means),
                    'amplitude': float(amplitude_corr),
                    'phase': float(phase_decimal),
                    'phase_hhmm': float(phase_hhmm),
                    'period': float(period_decimal),
                    'r_squared': float(r_squared),
                    'p_value': float(p_value_rhythm),
                    'p_value_adjusted': np.nan,
                    'p_value_final': float(p_value_final),
                    'rhythm_class': rhythm_class,
                    'counter': int(counter),
                    'interval': interval_flag,
                    'probability': None,
                    'n_points': int(len(y)),
                    'n_outliers': int(len(outliers)),
                    'sd_values_filtered': float(np.std(values_filtered)) if len(values_filtered) > 1 else np.nan,
                    'amp_limit': float(amp_limit)
                }
                results.append(result_row)
                gene_counters[group_name] = int(counter)

                X_interp = np.linspace(float(np.min(X)), float(np.max(X)), 400)
                y_interp = circular_function(X_interp, *params)
                plot_data_gene.append({
                    'group_name': group_name,
                    'X': X,
                    'y': y,
                    'X_interp': X_interp,
                    'y_interp': y_interp,
                    'x_peak': x_peak
                })

            except Exception as e:
                print(f"Unexpected error for gene '{gene_label}': {e}")
                continue

        if len(df_gene_rows) >= 6 and groups_processed >= 2:
            try:
                df_gene = pd.DataFrame(df_gene_rows)
                df_gene = add_harmonic_terms(df_gene, 'time', 24.0)

                # 1. Performs the global linear fit.
                model0, model1, model2 = fit_multigroup_models(df_gene, 'expr', 'group')

                # 2. Extract the overall p-values and store them in the list
                global_test_row = get_global_tests(model0, model1, model2, 'group')
                global_test_row['gene'] = gene
                global_test_row['n_groups'] = int(df_gene['group'].nunique())
                global_test_row['n_points_total'] = int(len(df_gene))

                # 3. Multi-group GROUPING selection (two independent axes).
                groups_in_gene = sorted(df_gene['group'].unique())

                # Rhythm axis. The per-group multi-criteria tier decides WHICH
                # groups are rhythmic (R = groups reaching `rhythmicity_cutoff`,
                # the same bar CODA uses everywhere -- so a LOW group is NOT
                # rhythmic). Model selection then only decides HOW the rhythmic
                # groups share their rhythm:
                #   |R| == 0            -> "All groups arrhythmic".
                #   |R| == 1            -> that group rhythmic, the rest arrhythmic.
                #   |R| >= 2, p NaN     -> "Undetermined" (can't test the sharing).
                #   |R| >= 2, p <= a    -> search the partitions of R.
                #   |R| >= 2, p  > a    -> R shares one common rhythm.
                p_gate = global_test_row.get('p_rhythm_diff', np.nan)
                _cut_rank = {'ARRHYTHMIC': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'EXTREMELY HIGH': 4}.get(str(rhythmicity_cutoff).upper(), 3)
                R = [g for g in groups_in_gene if gene_counters.get(g, 0) >= _cut_rank]

                if len(R) >= 2 and pd.isna(p_gate):
                    g_lab, g_conf, g_gap, g_code = "Undetermined (insufficient data)", np.nan, np.nan, ""
                elif len(R) >= 2 and (p_gate <= p_threshold):
                    g_lab, g_conf, g_gap, g_code = select_rhythm_grouping(
                        df_gene, groups_in_gene, R, criterion=selection_criterion,
                        time_col='time', expr_col='expr', group_col='group')
                else:
                    # |R| <= 1, or |R| >= 2 with no significant difference:
                    # the winning model is simply R as a single rhythmic block
                    # (empty R => arrhythmic in all).
                    model = (frozenset(R), (tuple(sorted(R)),) if R else tuple())
                    if len(R) == 0:
                        g_lab = "All groups arrhythmic"
                    elif len(R) == len(groups_in_gene):
                        g_lab = "All groups rhythmic (shared rhythm)"
                    else:
                        g_lab = _rhythm_label(model, groups_in_gene)
                    g_conf, g_gap = np.nan, np.nan
                    g_code = _rhythm_code(model, groups_in_gene)
                global_test_row['Grouping'] = g_lab
                global_test_row['Grouping_Confidence'] = g_conf
                global_test_row['Grouping_IC_Gap'] = g_gap
                global_test_row['Grouping_Model'] = g_code
                # Kept for the post-loop FDR gate override (dropped before output).
                global_test_row['_R_rhythm'] = ",".join(sorted(R))
                global_test_row['_all_groups'] = ",".join(sorted(groups_in_gene))

                # Mesor axis (baseline), gated by its own omnibus test -- symmetric
                # to the rhythm axis, so a baseline grouping is only searched when
                # there is a significant mesor difference to explain.
                p_mesor = mesor_omnibus_p(
                    df_gene, groups_in_gene,
                    time_col='time', expr_col='expr', group_col='group')
                global_test_row['p_mesor_diff'] = p_mesor
                if pd.isna(p_mesor):
                    m_lab, m_conf, m_gap, m_code = "Undetermined (insufficient data)", np.nan, np.nan, ""
                elif p_mesor <= p_threshold:
                    m_lab, m_conf, m_gap, m_code = select_mesor_grouping(
                        df_gene, groups_in_gene, criterion=selection_criterion,
                        time_col='time', expr_col='expr', group_col='group')
                else:
                    m_lab, m_conf, m_gap = "All groups equal (same baseline)", np.nan, np.nan
                    m_code = _mesor_code((tuple(sorted(groups_in_gene)),), groups_in_gene)
                global_test_row['Grouping_Mesor'] = m_lab
                global_test_row['Grouping_Mesor_Confidence'] = m_conf
                global_test_row['Grouping_Mesor_IC_Gap'] = m_gap
                global_test_row['Grouping_Mesor_Model'] = m_code

                global_results.append(global_test_row)

            except Exception as e:
                print(f"Warning in multigroup comparison for gene '{gene}': {e}")

        # ------------------------------------------------------------------
        # PART C: Plotting System
        # ------------------------------------------------------------------

        # Clean the file name to avoid errors when saving.
        safe_gene_name = str(gene).replace("/", "_").replace("\\", "_").replace(":", "_")

        # Checks if the user wants the graph for this specific gene.
        should_plot_gene = (plot_flag == 'Y' and (plot_all == 'Y' or gene.upper() in genes_to_plot))

        if should_plot_gene and len(plot_data_gene) > 0:
            fig_g, ax = plt.subplots(figsize=(10, 6))

            # Accessible color palette (Color Blind Friendly)
            group_colors = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#882255']

            # Dark period shading (e.g., ZT12 to ZT24)
            # ax.axvspan(12, 24, color='lightgray', alpha=0.8, zorder=0)
            all_ymins = []
            all_ymaxs = []
            all_xmins = []
            all_xmaxs = []
            legend_handles = []

            for i, pdata in enumerate(plot_data_gene):
                color = group_colors[i % len(group_colors)]
                X_grp = np.array(pdata['X'])
                y_grp = np.array(pdata['y'])
                X_interp_plot = pdata['X_interp']
                y_interp_plot = pdata['y_interp']
                group_name = pdata['group_name']

                # Calculation of mean and error (SEM) for experimental data points
                x_unique = np.sort(np.unique(X_grp))
                y_mean = np.array([np.nanmean(y_grp[X_grp == zt]) for zt in x_unique])
                y_sem = np.array([stats.sem(y_grp[X_grp == zt], nan_policy='omit') if np.sum(X_grp == zt) > 1 else 0 for zt in x_unique])

                # Draw the points with error bars.
                ax.errorbar(x_unique, y_mean, yerr=y_sem, fmt='o', color=color,ecolor=color, elinewidth=1.8, capsize=4, capthick=1.8,markersize=7, alpha=0.8)

                # Draw the fitted curve (Cosnor)
                ax.plot(X_interp_plot, y_interp_plot, color=color, linewidth=2.5)
                all_ymins.append(np.nanmin(y_interp_plot))
                all_ymaxs.append(np.nanmax(y_interp_plot))
                all_xmins.append(np.nanmin(X_grp))
                all_xmaxs.append(np.nanmax(X_grp))

                # Create a custom caption.
                legend_handles.append(mlines.Line2D([], [], color=color, marker='o',linestyle='-', linewidth=2.5, markersize=7, label=group_name))

            # Axis Settings
            ax.set_xlabel(time_label)  # Use the label (ZT, CT, or Clock) chosen at the beginning.
            # X-limits: start where the observed data begins and end where it ends
            if all_xmins and all_xmaxs:
                x_data_min = min(all_xmins)
                x_data_max = max(all_xmaxs)
                span = x_data_max - x_data_min
                pad = 0.03 * span if span > 0 else 1.0
                ax.set_xlim(x_data_min - pad, x_data_max + pad)

            # Intelligent Y-scale adjustment with margin
            if all_ymins and all_ymaxs:
                y_min_total = min(all_ymins)
                y_max_total = max(all_ymaxs)
                margin = max(0.5, 0.1 * (y_max_total - y_min_total))
                ax.set_ylim(y_min_total - margin, y_max_total + margin)

            if all_xmins and all_xmaxs:
                step = 2 if span <= 24 else (4 if span <= 60 else 8)
                ax.set_xticks(np.arange(x_data_min, x_data_max + 1, step))
            ax.set_ylabel('Observed Variables')
            ax.set_title(f"{gene}", loc='left', fontweight='bold', fontstyle='italic')
            ax.legend(handles=legend_handles, loc='upper right', frameon=True)
            plt.tight_layout()
            # Save the file in the 'plots' folder.
            plt.savefig(os.path.join(results_dir, 'plots', f'gene_{safe_gene_name}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_g)

    # ------------------------------------------------------------------
    # END OF MAIN LOOP
    # ------------------------------------------------------------------

    # FDR correction (Benjamini-Hochberg)
    if results:
        p_raw = [r["p_value"] for r in results]
        # The multipletests returns a tuple, index [1] are the fitted p-values.
        p_adj = multipletests(p_raw, method='fdr_bh')[1]
        for i, r in enumerate(results):
            r["p_value_adjusted"] = p_adj[i]
            # Here we respect your choice (O or A) defined in the User Information.
            if str(p_value_option).upper() in ['O', 'RAW']:
                r['p_value_final'] = r['p_value']
            else:
                r['p_value_final'] = r['p_value_adjusted']

    # ------------------------------------------------------------------
    # SVM suggestion for R² threshold
    # ------------------------------------------------------------------
    time.sleep(0.1)
    print("\n" + "=" * 70)
    print('                         SVM Analysis                     ')
    print("=" * 70)

    if len(results) > 10:  # It only makes sense if we have a minimum sample size.
        try:
            r2_arr = np.array([r['r_squared'] for r in results])
            pv_arr = np.array([r['p_value_final'] for r in results])
            # We created the labels: 1 for rhythmic (p <= threshold), 0 for non-rhythmic.
            svm_labels = (pv_arr <= p_threshold).astype(int)
            # We only run the test if there is at least one gene in each category (rhythmic and non-rhythmic).
            if len(np.unique(svm_labels)) > 1:
                svm_model = SVC(kernel='linear')
                svm_model.fit(r2_arr.reshape(-1, 1), svm_labels)
                suggested_r2 = None
                for lim in np.linspace(0, 1, 100):
                    if svm_model.predict(np.array([[lim]])) == 1:
                        suggested_r2 = lim
                        break
                if suggested_r2:
                    print(
                        f"[SVM]: Based on your data, a suggested R² threshold would be: {suggested_r2:.3f}")
        except Exception as e:
            print(f"[SVM]: Could not calculate suggested R²: {e}")

    # ------------------------------------------------------------------
    # WRITING RESULTS (CSV)
    # ------------------------------------------------------------------
    time.sleep(0.1)
    print("\n" + "=" * 70)
    print('                         WRITING RESULTS (CSV)                     ')
    print("=" * 70)

    classification_map = {
        0: "ARRHYTHMIC",
        1: "LOW",
        2: "MEDIUM",
        3: "HIGH",
        4: "EXTREMELY HIGH"
    }

    if results:
        # We finalized the Counter and Probability logic before saving.
        for r in results:
            # Criterion 1: final p-value (raw or FDR according to the user's choice).
            # A point is awarded when the p-value PASSES the threshold.
            if not np.isnan(r['p_value_final']) and r['p_value_final'] <= p_threshold:
                r['counter'] += 1

            # The tier is awarded from the number of criteria MET (4 met = Extremely High).
            r['probability'] = classification_map.get(r['counter'], "ARRHYTHMIC")

        # Converting to DataFrame for easy manipulation.
        df_results = pd.DataFrame(results)

        # --- Optional: drop targets that are MEDIUM in ANY group (exclude_medium) ---
        # "less is more": removes ambiguous/borderline targets before comparisons.
        if exclude_medium:
            medium_targets = df_results.loc[
                df_results['probability'].astype(str).str.upper() == 'MEDIUM', 'gene'
            ].unique()
            if len(medium_targets) > 0:
                n_before = df_results['gene'].nunique()
                df_results = df_results[~df_results['gene'].isin(medium_targets)].copy()
                print(f"[INFO] exclude_medium: removed {len(medium_targets)} target(s) "
                      f"that were MEDIUM in \nat least one group "
                      f"({n_before} -> {df_results['gene'].nunique()} targets).")

        # Organizing columns for the CSV
        df_export = df_results[[
            'gene', 'group_name', 'mean', 'amplitude', 'amp_limit', 'phase', 'phase_hhmm',
            'p_value', 'p_value_adjusted', 'interval', 'r_squared',
            'period', 'probability'
        ]].copy()

        # Renaming to match your pattern
        df_export.columns = [
            'Gene', 'Group', 'Mesor', 'Amplitude', 'amp_limit', 'Phase', 'Phase (h:min)',
            'p_value', 'p_adj', 'Interval', 'R2', 'Period', 'Probability'
        ]

        # Applying rounding and scientific formatting
        df_export['p_adj'] = df_export['p_adj'].apply(lambda x: f"{x:.4e}")
        df_export['Mesor'] = df_export['Mesor'].round(4)
        df_export['Amplitude'] = df_export['Amplitude'].round(4)
        df_export['Phase'] = df_export['Phase'].round(2)
        df_export['Phase (h:min)'] = df_export['Phase (h:min)'].round(2)
        df_export['R2'] = df_export['R2'].round(4)
        df_export['Period'] = df_export['Period'].round(2)

        # ==================================================================
        # CALLING THE PAIRED ANALYSIS AND GENERATING THE MASTER TABLE
        # ==================================================================
        df_comparisons = build_pairwise_comparisons(df_export, df_long, comparisons_to_run,rhythmicity_cutoff=rhythmicity_cutoff)

        # Apply FDR per component, within each pair (raw values are kept).
        df_comparisons = add_pairwise_fdr(df_comparisons)

        # Decide the comparison-driven outputs (Mesor_Change, categories, loss/gain
        # confidence) from the chosen source: raw or FDR-adjusted pairwise p-values.
        # Default 'RAW' reproduces the previous, validated behavior exactly.
        df_comparisons = assign_categories(df_comparisons, p_source=p_value_comparison)
        print(f"[INFO] Comparison p-value source: "
              f"{'FDR-adjusted' if str(p_value_comparison).upper() in ('FDR','ADJ','ADJUSTED') else 'raw'} "
              f"(p_value_comparison = '{p_value_comparison}').")

        # ------------------------------------------------------------------
        # EXCEL EXPORT (Fault-tolerant for missing pairs and global models)
        # ------------------------------------------------------------------
        if results:
            # 1. Initialize dataframes tolerating that they might be empty
            df_global = pd.DataFrame(global_results).rename(
                columns={'gene': 'Gene'}) if global_results else pd.DataFrame()

            # Genome-wide FDR on the global gate p-values, then -- if the user
            # gates on FDR (default) -- re-close the gate for targets that pass
            # the raw threshold but fail the corrected one. Because BH is
            # monotone (FDR >= raw), a target can only move from "searched split"
            # back to "shared", never the other way, so this is a one-directional
            # override of the in-loop (raw-gated) decision.
            if not df_global.empty:
                df_global = add_global_fdr(df_global, ['p_global_rhythm', 'p_rhythm_diff', 'p_mesor_diff'])
                if p_value_global == 'FDR':
                    def _split(s):
                        s = "" if (s is None or (isinstance(s, float) and pd.isna(s))) else str(s)
                        return [x for x in s.split(",") if x]
                    for idx in df_global.index:
                        ag = _split(df_global.at[idx, '_all_groups']) if '_all_groups' in df_global.columns else []
                        # --- rhythm axis: search ran (conf not NaN) but FDR gate now closed ---
                        pr = df_global.at[idx, 'p_rhythm_diff_FDR'] if 'p_rhythm_diff_FDR' in df_global.columns else np.nan
                        if pd.notna(df_global.at[idx, 'Grouping_Confidence']) and pd.notna(pr) and pr > p_threshold:
                            R = _split(df_global.at[idx, '_R_rhythm'])
                            model = (frozenset(R), (tuple(sorted(R)),) if R else tuple())
                            df_global.at[idx, 'Grouping'] = ("All groups rhythmic (shared rhythm)"
                                                             if R and len(R) == len(ag) else _rhythm_label(model, ag))
                            df_global.at[idx, 'Grouping_Confidence'] = np.nan
                            df_global.at[idx, 'Grouping_IC_Gap'] = np.nan
                            df_global.at[idx, 'Grouping_Model'] = _rhythm_code(model, ag)
                        # --- mesor axis: search ran but FDR gate now closed ---
                        pm = df_global.at[idx, 'p_mesor_diff_FDR'] if 'p_mesor_diff_FDR' in df_global.columns else np.nan
                        if pd.notna(df_global.at[idx, 'Grouping_Mesor_Confidence']) and pd.notna(pm) and pm > p_threshold:
                            blocks = (tuple(sorted(ag)),) if ag else tuple()
                            df_global.at[idx, 'Grouping_Mesor'] = "All groups equal (same baseline)"
                            df_global.at[idx, 'Grouping_Mesor_Confidence'] = np.nan
                            df_global.at[idx, 'Grouping_Mesor_IC_Gap'] = np.nan
                            df_global.at[idx, 'Grouping_Mesor_Model'] = _mesor_code(blocks, ag)
                df_global = df_global.drop(columns=['_R_rhythm', '_all_groups'], errors='ignore')

            # Make the pairwise categories consistent with the final grouping.
            df_comparisons = reconcile_categories_with_grouping(
                df_comparisons, df_global, groups,
                alpha=p_threshold, p_source=p_value_comparison)

            df_export_grouped = df_export.groupby('Gene')
            df_comp_grouped = df_comparisons.groupby('Gene') if not df_comparisons.empty else None
            df_global_grouped = df_global.groupby('Gene') if not df_global.empty else None

            master_rows = []

            for gene in tqdm(df_export['Gene'].unique(), desc="Building the Excel", unit="target"):
                # Ensure extraction of individual data
                df_grp = df_export_grouped.get_group(gene).reset_index(drop=True)

                # Safe extraction of comparison data
                df_par = pd.DataFrame()
                if df_comp_grouped and gene in df_comp_grouped.groups:
                    df_par = df_comp_grouped.get_group(gene).reset_index(drop=True)

                # Safe extraction of global data
                df_gbl = pd.DataFrame()
                if df_global_grouped and gene in df_global_grouped.groups:
                    df_gbl = df_global_grouped.get_group(gene).reset_index(drop=True)

                # The Excel block grows based on the largest available table
                num_rows_block = max(len(df_grp), len(df_par))

                for i in range(num_rows_block):
                    row_dict = {}

                    # -- 1. Global Fill --
                    row_dict['Gene'] = gene
                    row_dict['p_global_rhythm'] = df_gbl.loc[0, 'p_global_rhythm'] if not df_gbl.empty else ""
                    row_dict['p_global_rhythm_FDR'] = df_gbl.loc[0, 'p_global_rhythm_FDR'] if (not df_gbl.empty and 'p_global_rhythm_FDR' in df_gbl.columns) else ""
                    row_dict['p_rhythm_diff'] = df_gbl.loc[0, 'p_rhythm_diff'] if not df_gbl.empty else ""
                    row_dict['p_rhythm_diff_FDR'] = df_gbl.loc[0, 'p_rhythm_diff_FDR'] if (not df_gbl.empty and 'p_rhythm_diff_FDR' in df_gbl.columns) else ""
                    row_dict['p_mesor_diff'] = df_gbl.loc[0, 'p_mesor_diff'] if (not df_gbl.empty and 'p_mesor_diff' in df_gbl.columns) else ""
                    row_dict['p_mesor_diff_FDR'] = df_gbl.loc[0, 'p_mesor_diff_FDR'] if (not df_gbl.empty and 'p_mesor_diff_FDR' in df_gbl.columns) else ""
                    _has_grp = (not df_gbl.empty) and ('Grouping' in df_gbl.columns)
                    row_dict['Grouping'] = df_gbl.loc[0, 'Grouping'] if _has_grp else ""
                    row_dict['Grouping_Model'] = df_gbl.loc[0, 'Grouping_Model'] if (_has_grp and 'Grouping_Model' in df_gbl.columns) else ""
                    row_dict['Grouping_Confidence'] = df_gbl.loc[0, 'Grouping_Confidence'] if _has_grp else ""
                    row_dict['Grouping_IC_Gap'] = df_gbl.loc[0, 'Grouping_IC_Gap'] if _has_grp else ""
                    row_dict['Grouping_Mesor'] = df_gbl.loc[0, 'Grouping_Mesor'] if _has_grp else ""
                    row_dict['Grouping_Mesor_Model'] = df_gbl.loc[0, 'Grouping_Mesor_Model'] if (_has_grp and 'Grouping_Mesor_Model' in df_gbl.columns) else ""
                    row_dict['Grouping_Mesor_Confidence'] = df_gbl.loc[0, 'Grouping_Mesor_Confidence'] if _has_grp else ""
                    row_dict['Grouping_Mesor_IC_Gap'] = df_gbl.loc[0, 'Grouping_Mesor_IC_Gap'] if _has_grp else ""

                    # -- 2. Individual Fill --
                    if i < len(df_grp):
                        row_dict['Group'] = df_grp.loc[i, 'Group']
                        row_dict['P-value'] = df_grp.loc[i, 'p_value']
                        row_dict['P-value (FDR)'] = df_grp.loc[i, 'p_adj']
                        row_dict['R2'] = df_grp.loc[i, 'R2']
                        row_dict['Mesor'] = df_grp.loc[i, 'Mesor']
                        row_dict['Amplitude'] = df_grp.loc[i, 'Amplitude']
                        row_dict['amp_limit'] = df_grp.loc[i, 'amp_limit']
                        row_dict['Phase'] = df_grp.loc[i, 'Phase']
                        row_dict['Phase (h:min)'] = df_grp.loc[i, 'Phase (h:min)']
                        row_dict['Interval'] = df_grp.loc[i, 'Interval']
                        row_dict['Period'] = df_grp.loc[i, 'Period']
                        row_dict['Probability'] = df_grp.loc[i, 'Probability']
                    else:
                        for col in ['Group', 'P-value', 'P-value (FDR)', 'R2', 'Mesor', 'Amplitude', 'amp_limit', 'Phase', 'Phase (h:min)', 'Interval', 'Period', 'Probability']:
                            row_dict[col] = ""

                    # -- 3. Comparative Fill --
                    if i < len(df_par):
                        row_dict['Pair'] = df_par.loc[i, 'Pair']
                        row_dict['Delta_Mesor'] = df_par.loc[i, 'Delta_Mesor']
                        row_dict['p_diff_mesor'] = df_par.loc[i, 'p_diff_mesor']
                        row_dict['p_diff_mesor_FDR'] = df_par.loc[i, 'p_diff_mesor_FDR']
                        row_dict['Mesor_Change'] = df_par.loc[i, 'Mesor_Change']
                        row_dict['Delta_Amplitude'] = df_par.loc[i, 'Delta_Amplitude']
                        row_dict['p_diff_amplitude'] = df_par.loc[i, 'p_diff_amplitude']
                        row_dict['p_diff_amplitude_FDR'] = df_par.loc[i, 'p_diff_amplitude_FDR']
                        row_dict['Delta_Phase'] = df_par.loc[i, 'Delta_Phase']
                        row_dict['p_diff_phase'] = df_par.loc[i, 'p_diff_phase']
                        row_dict['p_diff_phase_FDR'] = df_par.loc[i, 'p_diff_phase_FDR']
                        row_dict['Biological_Category'] = df_par.loc[i, 'Biological_Category']
                        row_dict['Rhythm_Status'] = df_par.loc[i, 'Rhythm_Status']
                        row_dict['LossGain_Confidence'] = df_par.loc[i, 'LossGain_Confidence']
                    else:
                        for col in ['Pair', 'Delta_Mesor', 'p_diff_mesor', 'p_diff_mesor_FDR', 'Mesor_Change',
                                    'Delta_Amplitude', 'p_diff_amplitude', 'p_diff_amplitude_FDR',
                                    'Delta_Phase', 'p_diff_phase', 'p_diff_phase_FDR',
                                    'Biological_Category', 'Rhythm_Status',
                                    'LossGain_Confidence']:
                            row_dict[col] = ""

                    master_rows.append(row_dict)

            df_master = pd.DataFrame(master_rows)

            # Final column sorting
            col_order = [
                'Gene', 'p_global_rhythm', 'p_global_rhythm_FDR', 'p_rhythm_diff', 'p_rhythm_diff_FDR', 'p_mesor_diff', 'p_mesor_diff_FDR',
                'Grouping', 'Grouping_Model', 'Grouping_Confidence', 'Grouping_IC_Gap',
                'Grouping_Mesor', 'Grouping_Mesor_Model', 'Grouping_Mesor_Confidence', 'Grouping_Mesor_IC_Gap',
                'Group', 'P-value', 'P-value (FDR)', 'R2', 'Mesor', 'Amplitude', 'amp_limit', 'Phase', 'Phase (h:min)', 'Interval', 'Period',
                'Probability',
                'Pair', 'Delta_Mesor', 'p_diff_mesor', 'p_diff_mesor_FDR', 'Mesor_Change',
                'Delta_Amplitude', 'p_diff_amplitude', 'p_diff_amplitude_FDR',
                'Delta_Phase', 'p_diff_phase', 'p_diff_phase_FDR',
                'Biological_Category', 'Rhythm_Status', 'LossGain_Confidence'
            ]
            df_master = df_master[[c for c in col_order if c in df_master.columns]]

            # Rename the output column Gene -> Target (internal code keeps 'Gene')
            df_master = df_master.rename(columns={'Gene': 'Target'})
            df_master = df_master.rename(columns={'amp_limit': 'Amp. Minimum'})

            # ------------------------------------------------------------------
            # KEEP df_master NUMERIC
            # ------------------------------------------------------------------
            # All p-value / delta / metric columns are coerced to real numbers so
            # the table handed to R (df_r, built later from df_master) can be
            # filtered directly without any as.numeric() gymnastics. The "not
            # tested" ("") and "NLS failed" (NaN) sentinels both become NaN, which
            # is the correct numeric representation of a missing value.
            numeric_cols = [
                'p_global_rhythm', 'p_global_rhythm_FDR', 'p_rhythm_diff', 'p_rhythm_diff_FDR', 'p_mesor_diff', 'p_mesor_diff_FDR',
                'Grouping_Confidence', 'Grouping_IC_Gap',
                'Grouping_Mesor_Confidence', 'Grouping_Mesor_IC_Gap',
                'P-value', 'P-value (FDR)',
                'R2', 'Mesor', 'Amplitude', 'Amp. Minimum', 'Phase', 'Phase (h:min)', 'Period',
                'Delta_Mesor', 'p_diff_mesor', 'p_diff_mesor_FDR',
                'Delta_Amplitude', 'p_diff_amplitude', 'p_diff_amplitude_FDR',
                'Delta_Phase', 'p_diff_phase', 'p_diff_phase_FDR',
            ]
            for _col in numeric_cols:
                if _col in df_master.columns:
                    df_master[_col] = pd.to_numeric(df_master[_col], errors='coerce')

            # ------------------------------------------------------------------
            # Excel gets a SEPARATE, display-formatted copy (numbers unchanged
            # in df_master). Only the p-value columns are rendered in scientific
            # notation; missing values render as blank instead of "nan".
            # ------------------------------------------------------------------
            def _sci(x):
                if x == "" or pd.isna(x):
                    return ""
                try:
                    return f"{float(x):.4e}"
                except (TypeError, ValueError):
                    return x

            df_excel = df_master.copy()
            for _pcol in ['P-value', 'P-value (FDR)',
                          'p_diff_mesor', 'p_diff_mesor_FDR',
                          'p_diff_amplitude', 'p_diff_amplitude_FDR',
                          'p_diff_phase', 'p_diff_phase_FDR']:
                if _pcol in df_excel.columns:
                    df_excel[_pcol] = df_excel[_pcol].apply(_sci)

            excel_file = result_file.replace('.csv', '_CODA.xlsx')
            try:
                print("Saving Excel, this may take a while...")
                export_excel_merged(df_excel, excel_file)
                print(f"\nExcel report generated successfully: \n{excel_file}")
            except Exception as e:
                print(f"Warning: Failed to generate Excel spreadsheet: {e}")

    # ------------------------------------------------------------------
    # SUMMARY PLOTS
    # ------------------------------------------------------------------
    time.sleep(0.1)
    print("\n" + "=" * 70)
    print('                      PLOTTING SUMMARY CHARTS                      ')
    print("=" * 70)
    time.sleep(0.1)

    # Vector preparation based on final results (post-FDR)
    # We need to extract the lists from the 'results' dictionary
    r2_values_all = [r['r_squared'] for r in results]
    amp_values_all = [r['amplitude'] for r in results]
    p_values_final = [r['p_value_final'] for r in results]

    # Auxiliary lists for specific charts.
    r2_high = [];
    r2_low = [];
    idx_high = [];
    idx_low = []
    colors_all = [];
    colors_amp = [];
    colors_pv = []
    df_results_temp = pd.DataFrame(results)
    processed_groups = sorted(df_results_temp['group_name'].unique()) if not df_results_temp.empty else []

    # Fixed colors for rhythmicity
    # Blue = Significant | Red = Not Significant
    for i, r in enumerate(results):
        label = f"{r['gene']} ({r['group_name']})"

        if r['p_value_final'] > p_threshold:
            r2_high.append(r['r_squared'])
            idx_high.append(i)
            colors_all.append('red')
        else:
            r2_low.append(r['r_squared'])
            idx_low.append(i)
            colors_all.append('blue')

        # Color for the amplitude graph based on rhythmicity
        colors_amp.append('blue' if r['p_value_final'] <= p_threshold else 'red')

    def _save_fig(filename):
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', filename), dpi=300)
        plt.close()

    num_groups = len(processed_groups)
    total_figuras = (num_groups * 2) + 1
    with tqdm(total=total_figuras, desc="Generating Figures", unit="figure") as pb:

        # ------------------------------------------------------------------
        # R² bar charts
        # ------------------------------------------------------------------
        for group_to_plot in processed_groups:
            plt.figure(figsize=(12, 6))
            group_res = [r for r in results if r['group_name'] == group_to_plot]
            if not group_res: continue
            r2_vals = [r['r_squared'] for r in group_res]
            group_colors_r2 = ['blue' if r['p_value_final'] <= p_threshold else 'red' for r in group_res]
            plt.bar(range(len(r2_vals)), r2_vals, color=group_colors_r2, alpha=0.8)
            plt.axhline(r2_threshold, color='black', linestyle='--', label='R² Threshold')
            if suggested_r2:
                plt.axhline(suggested_r2, color='orange', linestyle='--', label=f'Suggested R²: {suggested_r2:.2f}')
            plt.title(f'R² Distribution — Group: {group_to_plot}')
            plt.ylabel('R² Value')
            plt.xlabel('Genes')
            plt.xticks([])
            plt.ylim(0, 1.05)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            legend_elements = [
                mlines.Line2D([0], [0], color='blue', lw=4, label=f'p-value <= {p_threshold}'),
                mlines.Line2D([0], [0], color='red', lw=4, label=f'p-value > {p_threshold}'),
                mlines.Line2D([0], [0], color='black', linestyle='--', label='Manual R² Limit')]
            if suggested_r2:
                legend_elements.append(
                    mlines.Line2D([0], [0], color='orange', linestyle='--', label=f'SVM Suggested R² ({suggested_r2:.3f})'))
            plt.legend(handles=legend_elements, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plots', f'plot_r2_{group_to_plot}.png'), dpi=300)
            plt.close()
            pb.update(1)

        # ------------------------------------------------------------------
        # Scatter R² vs p-value
        # ------------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        plt.grid(True, linestyle=':', alpha=0.4, color='gray')
        plt.gca().set_axisbelow(True)
        plt.scatter(r2_values_all, p_values_final,color=colors_all,s=70,alpha=0.8,edgecolor='black',linewidth=0.5)
        plt.axhline(p_threshold, color='red', linestyle='--', label=f'p-threshold ({p_threshold})')
        plt.axvline(r2_threshold, color='black', linestyle='--', label=f'Manual R² Threshold ({r2_threshold})')
        if suggested_r2:
            plt.axvline(suggested_r2, color='orange', linestyle='--',label=f'SVM Suggested R² ({suggested_r2:.3f})')
        plt.title('R² vs p-value Scatter (All Genes & Groups)')
        plt.xlabel('R² (Quality of Fit)')
        plt.ylabel('p-value (Significance)')
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        legend_elements = [mlines.Line2D([0], [0], color='blue', marker='o', linestyle='None',markeredgecolor='black', markeredgewidth=0.5, label=f'p-value <= {p_threshold}'),
                           mlines.Line2D([0], [0], color='red', marker='o', linestyle='None',markeredgecolor='black', markeredgewidth=0.5, label=f'p-value > {p_threshold}'),
                           mlines.Line2D([0], [0], color='red', linestyle='--', label='p-threshold limit'),
                           mlines.Line2D([0], [0], color='black', linestyle='--', label='Manual R² limit')]
        if suggested_r2:
            legend_elements.append(mlines.Line2D([0], [0], color='orange', linestyle='--', label=f'SVM Suggested R² ({suggested_r2:.3f})'))
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize='small')
        _save_fig('plot_pvalue_versus_r2.png')
        pb.update(1)

        # ------------------------------------------------------------------
        # Amplitude bar charts
        # ------------------------------------------------------------------
        all_amps = [r['amplitude'] for r in results]
        global_max_amp = max(all_amps) if all_amps else 1.0
        y_limit_upper = global_max_amp * 1.1
        for group_to_plot in processed_groups:
            plt.figure(figsize=(12, 6))
            group_res = [r for r in results if r['group_name'] == group_to_plot]
            if not group_res: continue
            amps = [r['amplitude'] for r in group_res]
            group_colors = ['green' if r['interval'] == 'In' else 'orange' for r in group_res]
            plt.bar(range(len(amps)), amps, color=group_colors, alpha=0.8)
            for i, r in enumerate(group_res):
                lim = r.get('amp_limit', 0.15)
                plt.plot([i - 0.4, i + 0.4], [lim, lim], color='black', linestyle='-', lw=1)
            plt.title(f'Amplitude Distribution — Group: {group_to_plot}')
            plt.ylabel('Amplitude')
            plt.xlabel('Genes')
            plt.xticks([])
            plt.ylim(0, y_limit_upper)
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            legend_elements = [
                mlines.Line2D([0], [0], color='green', lw=4, label='Within Interval (In)'),
                mlines.Line2D([0], [0], color='orange', lw=4, label='Outside Interval (Out)'),
                mlines.Line2D([0], [0], color='black', lw=1, label='Individual Amp. Limit')]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plots', f'plot_amplitude_{group_to_plot}.png'), dpi=300)
            plt.close()
            pb.update(1)

    # ------------------------------------------------------------------
    # Polar and Heatmaps
    # ------------------------------------------------------------------
    fig_polar = generate_polar_plot(results, p_threshold, r2_threshold,counter_threshold, results_dir,time_label=time_label)
    heatmaps_results = generate_heatmap_compare(results, df_long, n_observations,counter_threshold, results_dir,time_label=time_label)
    # Model-based heatmaps (replace the old per-pair/per-category heatmaps).
    _dfg = df_global if 'df_global' in locals() else pd.DataFrame()
    model_heatmap_paths = generate_heatmap_by_model(_dfg, results, df_long, n_observations, groups, results_dir, time_label=time_label)
    consolidated_path = generate_heatmap_consolidated(_dfg, results, df_long, n_observations, groups, results_dir, time_label=time_label)
    # Bundle everything (consolidated first, then per-model) into one PDF (dryR-style).
    bundle_heatmaps_pdf(([consolidated_path] if consolidated_path else []) + (model_heatmap_paths or []), results_dir)
    # Save the data from the first group to maintain compatibility with Spyder.
    if heatmaps_results and len(heatmaps_results) > 0:
        first_group = list(heatmaps_results.keys())[0]
        df_expression = heatmaps_results[first_group]
    else:
        df_expression = None

    # ------------------------------------------------------------------
    # PREPARING A CLEAN TABLE FOR RSTUDIO
    # ------------------------------------------------------------------
    if results:
        # Enables future Pandas behavior to silence the warning.
        pd.set_option('future.no_silent_downcasting', True)
        df_clean_for_r = df_master.copy()
        df_clean_for_r = df_clean_for_r.replace('', float('nan'))

        # Fill in the blanks by repeating the values from the three global columns.
        # Only the plain global columns are ffilled. The Grouping columns are
        # already repeated on every row of a target (Global Fill) and their NaN
        # confidence on gated targets must stay NaN, so they are NOT ffilled here.
        globals_columns = ['Target', 'p_global_rhythm', 'p_rhythm_diff']
        for col in globals_columns:
            if col in df_clean_for_r.columns:
                df_clean_for_r[col] = df_clean_for_r[col].ffill()

        # Send the completed table to R.
        globals()['df_r'] = df_clean_for_r
    else:
        globals()['df_r'] = None

    # ------------------------------------------------------------------
    # EXPORT THE FIGURES TO GLOBAL MEMORY (SO THAT R CAN VIEW THEM)
    # ------------------------------------------------------------------
    if 'fig_polar' in locals(): globals()['fig_polar'] = fig_polar

if __name__ == "__main__":
    main()