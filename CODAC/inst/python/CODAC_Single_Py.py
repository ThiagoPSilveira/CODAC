import os
import csv
import numpy as np
import warnings
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import FormatStrFormatter
from sklearn.svm import SVC
from tqdm import tqdm

# ==================================================================
# FALLBACK VARIABLES (For direct testing in PyCharm)
# ==================================================================
if 'r2_threshold' not in globals():
    interval_var = None
    r2_threshold = None
    plot_flag = None
    plot_all = None
    p_value_option = None
    p_threshold = None
    time_label = None
    min_rhythmicity = None
    missing_data_action = None
    period_mode = None
    fixed_period = None
    period_lower = None
    period_upper = None
    amp_stringency = None

# ================================================================
# SPYDER VARIABLE EXPLORER — accessible objects after running
# ================================================================
df_results = None
df_expression = None
fig_r2_high = None
fig_r2_low = None
fig_r2_all = None
fig_r2_pvalue = None
fig_amplitude = None
fig_polar = None
fig_heatmap = None

# ==================================================================
# AMPLITUDE FILTER — single stringency dial (0.0 to 1.0)
# ==================================================================
# A rhythm is only trusted if its amplitude is large enough to stand out from
# noise. This one parameter sets how demanding that test is:
#   amp_stringency = 0.0  -> filter OFF (amplitude never rejects; noise passes)
#   amp_stringency = 0.5  -> DEFAULT, validated behavior (recommended)
#   amp_stringency = 1.0  -> most stringent (requires twice the default amplitude)
# Values in between scale linearly. The threshold is ADAPTIVE (adjusts to each
# target's expression level and variability); amp_stringency just scales it.
amp_stringency = 0.5

# --- Internal shape of the adaptive threshold (NOT user-facing) ---
_amp_floor      = 0.15    # absolute noise floor (amplitude below this = noise)
_amp_mean_ratio = 0.10    # >= 10% of mean expression (minimum fold-change ~1.22)
_amp_std_ratio  = 0.50    # >= 50% of the data's std dev (meaningful share of variance)

# ==================================================================
# MATHEMATICS
# ==================================================================
def circular_function(x, k, a, f, T):
    r = (x / T) * (2 * np.pi)
    return k + a * np.cos(r - f)

#-------------------------------------------------------------------
# Read the tab-delimited gene expression data file
#-------------------------------------------------------------------
def read_data_file(file_path: str, timepoints: list, n_observations: int) -> dict:
    data = {}
    expected_len = len(timepoints) * n_observations
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file) # Efficiently skips the header (line 0) without loading the entire file.
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= (expected_len + 1): # Checks if the row has at least the gene name (1) + expected data columns
                gene = parts[0]
                raw_values = parts[1:expected_len + 1] # Isolates exactly the expected number of values
                values = []
                for val in raw_values:
                    val = val.strip()
                    if not val or val.upper() in ['NA', 'NAN', 'NULL']: # If the string is empty or is a common missing data marker
                        values.append(np.nan)
                    else:
                        try:
                            values.append(float(val.replace(',', '.'))) # Ensures that decimal points are converted to periods.
                        except ValueError:
                            values.append(np.nan) # Any other textual garbage becomes NaN instead of breaking the line.
                if len(values) == expected_len: # Because we preserve the NaNs, the size will always be as expected.
                    data[gene] = (timepoints, np.array(values)) # We've already saved it as an np.array to optimize future mathematical calculations.
                else:
                    print(f"[DEBUG] The gene '{gene}' has an unexpected number of columns.")
    return data

#-------------------------------------------------------------------
# Calculate R-squared
#-------------------------------------------------------------------
def calc_r2(y_obs: np.ndarray, y_hat: np.ndarray) -> float:
    valid_mask = ~np.isnan(y_obs) & ~np.isnan(y_hat) # Creates a mask to only consider indices where BOTH arrays contain valid data.
    y_obs_valid = y_obs[valid_mask]
    y_hat_valid = y_hat[valid_mask]
    if len(y_obs_valid) < 2: # If fewer than 2 valid points remain, it is not possible to calculate the variance.
        return 0.0
    # Mathematical calculations using only the clean data.
    ss_res = np.sum((y_obs_valid - y_hat_valid) ** 2)
    ss_tot = np.sum((y_obs_valid - np.mean(y_obs_valid)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

#-------------------------------------------------------------------
# Fit linear cosinor model — used to compute p-value vs null model
#-------------------------------------------------------------------
def fit_linear_cosinor(X: np.ndarray, y: np.ndarray, period: float = 24.0) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Missing Data Filter
    valid_mask = ~np.isnan(X) & ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    # Safety lock: Multiple regression with 3 parameters requires at least 3 points.
    if len(y_valid) < 3:
        return {
            "M": np.nan, "B": np.nan, "C": np.nan,
            "amplitude": np.nan, "phase": np.nan,
            "y_pred": np.full_like(y, np.nan),
            "r2": np.nan, "ss_res": np.nan
        }

    # 2. Building the Design Matrix with valid data
    omega = 2.0 * np.pi / period
    X_design_valid = np.column_stack((np.ones_like(X_valid),np.cos(omega * X_valid),np.sin(omega * X_valid)))

    # Least Squares Adjustment (Normal Equation)
    coef, _, _, _ = np.linalg.lstsq(X_design_valid, y_valid, rcond=None)
    M, B, C = coef

    # 3. Continuous Projection (The trick to plot development)
    # We calculated y_pred for ALL of X, including the times when there was no data.
    X_design_full = np.column_stack((np.ones_like(X),np.cos(omega * X),np.sin(omega * X)))
    y_pred_full = X_design_full.dot(coef)

    # 4. Derivation of Biological Parameters
    amplitude = np.sqrt(B ** 2 + C ** 2)
    phase = np.arctan2(-C, B)  # Standard cosinor convention

    # 5. Statistical Validation (Only on actual observed data points)
    y_pred_valid = X_design_valid.dot(coef)
    ss_res = np.sum((y_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "M": M,
        "B": B,
        "C": C,
        "amplitude": amplitude,
        "phase": phase,
        "y_pred": y_pred_full,
        "r2": r2,
        "ss_res": ss_res
    }

#-------------------------------------------------------------------
# Rhythmicity test: cosinor vs. null (constant) model via F-test
#-------------------------------------------------------------------
def test_rhythmicity_cosinor(X: np.ndarray, y: np.ndarray, period: float = 24.0) -> tuple:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Missing Data Filter for accurate observation count
    valid_mask = ~np.isnan(X) & ~np.isnan(y)
    y_valid = y[valid_mask]
    n_valid = len(y_valid)

    # For the F-test with 3 cosine parameters, we need n >= 4 for df² > 0.
    if n_valid < 4:
        # A p-value of 1.0 avoids false positives in genes with almost no data.
        return np.nan, 1.0, None, None

    # 2. Null Model: y = mean(y_valid)
    k = np.mean(y_valid)
    ss_res_null = np.sum((y_valid - k) ** 2)

    # 3. Alternative Model (Linearized Cosinor)
    cosinor = fit_linear_cosinor(X, y, period=period)
    ss_res_cos = cosinor["ss_res"]

    # Protection: if the adjustment failed internally for some reason
    if np.isnan(ss_res_cos):
        return np.nan, 1.0, {"k": k, "ss_res": ss_res_null}, cosinor

    # 4. Degrees of Freedom
    # df1: difference of parameters (Cosine has 3: M, B, C. Null has 1: k. 3 - 1 = 2)
    df1 = 2
    df2 = n_valid - 3

    # If the cosine error is greater than or equal to zero, the model does not best explain the variance.
    if df2 <= 0 or ss_res_cos >= ss_res_null:
        return np.nan, 1.0, {"k": k, "ss_res": ss_res_null}, cosinor

    # Extreme mathematical protection: perfect fit prevents division by zero.
    if ss_res_cos < 1e-12:
        return np.inf, 0.0, {"k": k, "ss_res": ss_res_null}, cosinor

    # 5. F-statistic
    F = ((ss_res_null - ss_res_cos) / df1) / (ss_res_cos / df2)

    # Using the "Survival Function" (sf) instead of (1 - cdf) avoids precision errors.
    # in infinitesimally small p-values
    p_value = float(stats.f.sf(F, df1, df2))

    return F, p_value, {"k": k, "ss_res": ss_res_null}, cosinor

#-------------------------------------------------------------------
# Nonlinear curve fitting (based on the Parson function)
#-------------------------------------------------------------------
def perform_curve_fit(X: np.ndarray, y: np.ndarray, fixed_period: bool = False, period: float = 24.0, bounds: tuple = (18.0, 24.0)) -> tuple:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Strict masking of Missing Data
    valid_mask = ~np.isnan(X) & ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    # We need at least 4 points to adjust 3 or 4 parameters.
    if len(y_valid) < 4:
        return None, None, None

    # 2. Initial guesses (p0) and limits based strictly on clean data.
    k0 = np.mean(y_valid)
    a0 = (np.max(y_valid) - np.min(y_valid)) / 2.0
    idx_max = np.argmax(y_valid)
    t_peak = X_valid[idx_max]
    # Phase guess consistent with the period in use (period for fixed mode, else 24 as a neutral start).
    _f0_period = period if fixed_period else 24.0
    f0 = ((t_peak % _f0_period) / _f0_period) * (2 * np.pi)

    # Model 1: Period T fixed at 24.0
    if fixed_period:
        try:
            params, _ = curve_fit(
                lambda x, k, a, f: circular_function(x, k, a, f, period),
                X_valid,
                y_valid,
                p0=[k0, a0, f0],
                maxfev=10000
            )
            # Project the fitted curve onto the original X vector (all timepoints)
            y_pred_full = circular_function(X, *params, period)
            return (*params, period), calc_r2(y, y_pred_full), y_pred_full
        except (RuntimeError, ValueError):
            pass

    # Model 2: Free period T between the bounds -- MULTI-START to avoid local minima.
    # We keep the SAME curve_fit, but launch it from several period guesses spread
    # across the bounds and keep the best fit (highest R2). A local minimum (the
    # amplitude-collapses-to-zero problem) only wins if EVERY start lands in it,
    # which spreading the starts makes very unlikely.
    try:
        amplitude_max = np.ptp(y_valid) if np.ptp(y_valid) > 0 else 1e-5

        lower_bounds = [np.min(y_valid), 0.0, -2 * np.pi, bounds[0]]
        upper_bounds = [np.max(y_valid), amplitude_max, 2 * np.pi, bounds[1]]

        # Period starts: spread evenly inside the bounds (never on the border).
        n_starts = 5
        period_starts = np.linspace(bounds[0], bounds[1], n_starts + 2)[1:-1]

        best_params = None
        best_r2 = -np.inf

        for T_start in period_starts:
            # Phase guess consistent with THIS period start (not a fixed 24).
            f0_start = ((t_peak % T_start) / T_start) * (2 * np.pi)
            p0_start = [k0, a0, f0_start, T_start]

            # Clip the starting point into the bounds (curve_fit rejects x0 outside).
            p0_start = [min(max(v, lo), hi)
                        for v, lo, hi in zip(p0_start, lower_bounds, upper_bounds)]

            try:
                params_try, _ = curve_fit(
                    circular_function,
                    X_valid,
                    y_valid,
                    p0=p0_start,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=20000
                )
            except (RuntimeError, ValueError):
                continue  # this start failed; try the next one

            y_pred_try = circular_function(X, *params_try)
            r2_try = calc_r2(y, y_pred_try)

            if r2_try is not None and not np.isnan(r2_try) and r2_try > best_r2:
                best_r2 = r2_try
                best_params = params_try

        # If at least one start converged, return the best; otherwise fall through.
        if best_params is not None:
            y_pred_full = circular_function(X, *best_params)
            return best_params, calc_r2(y, y_pred_full), y_pred_full
    except (RuntimeError, ValueError):
        pass

    # Fallback in case the algorithms diverge or maxfev exceeds
    return None, None, None

#-------------------------------------------------------------------
# Calculate group means and standard deviation
#-------------------------------------------------------------------
def calculate_full_metrics(groups: list) -> tuple:
    means = []
    for g in groups:
        g_array = np.asarray(g, dtype=float)
        # Isolate only the valid data for this specific time.
        g_valid = g_array[~np.isnan(g_array)]

        if len(g_valid) > 0:
            means.append(np.mean(g_valid)) # Calculate the average using only the replicas that survived.
        else:
            means.append(np.nan) # If all 4 replicas of the time disappeared, we insert NaN to NOT shrink the list. This keeps the X-axis of the graph intact.

    # To calculate the global metrics (mean_of_means and std_dev), we ignore the times that were completely empty.
    means_array = np.asarray(means, dtype=float)
    valid_means = means_array[~np.isnan(means_array)]
    if len(valid_means) > 0:
        mean_of_means = np.mean(valid_means)
        std_dev = np.std(valid_means)
    else:
        mean_of_means = 0.0
        std_dev = 0.0
    return means, mean_of_means, std_dev

#-------------------------------------------------------------------
# Outlier detection using the IQR method
#-------------------------------------------------------------------
def check_outliers(data: list) -> tuple:
    outliers = []

    for group in data:
        g_array = np.asarray(group, dtype=float)
        g_valid = g_array[~np.isnan(g_array)] # Isolate the valid data to avoid contaminating the quartile calculation.

        # The IQR method requires a minimum of dispersion to make sense.
        # If fewer than 2 points remain (due to missing data), there is no way for there to be a mathematically detectable outlier in this group.
        if len(g_valid) < 2:
            continue

        # Calculate Q1 and Q3 using only the observed data.
        Q1 = np.percentile(g_valid, 25)
        Q3 = np.percentile(g_valid, 75)
        IQR = Q3 - Q1

        # Defines the boundaries (lower_bound and upper_bound)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # It extracts the values that escape the boundaries.
        group_outliers = g_valid[(g_valid < lower_bound) | (g_valid > upper_bound)]
        outliers.extend(group_outliers)

    return len(outliers) > 0, outliers
#-------------------------------------------------------------------
# Read Input.txt
#-------------------------------------------------------------------
def read_input_config(input_path: str) -> dict:
    raw_config = {}
    genes_to_plot = []
    current_section = 'config'

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Switch parsing section when the target header is found
            if line == '[GenesToPlot]':
                current_section = 'genes_to_plot'
                continue

            if current_section == 'config':
                if ':' in line:
                    key, value = line.split(':', 1)
                    raw_config[key.strip().lower()] = value.strip()
            elif current_section == 'genes_to_plot':
                # Force uppercase to avoid case-sensitivity issues later
                genes_to_plot.append(line.upper())

    # Consolidate and type-cast the extracted parameters
    try:
        data_file = raw_config['data file']
        n_tp = int(raw_config['number of timepoints'])
        n_obs = int(raw_config['number of observations'])

        # Capture, clean, and convert the timepoints list to floats
        timepoints = [float(x.strip()) for x in raw_config['timepoints'].split(',')]
    except KeyError as e:
        raise KeyError(f"Required key not found in the input file: {e}")
    except ValueError as e:
        raise ValueError(f"Data type conversion failed (e.g., timepoints must be numbers): {e}")

    # Safety check: ensure the declared number of timepoints matches the provided list
    if len(timepoints) != n_tp:
        raise ValueError(
            f"The number of listed times ({len(timepoints)}) "
            f"does not match the 'Number of Timepoints' ({n_tp})."
        )

    # Return a single, cleanly formatted dictionary
    return {
        "data_file": data_file,
        "n_timepoints": n_tp,
        "n_observations": n_obs,
        "timepoints": timepoints,
        "genes_to_plot": genes_to_plot
    }

#-------------------------------------------------------------------
# Handle missing data: impute with group mean OR remove gene
#-------------------------------------------------------------------
def handle_missing_data(data: dict, n_observations: int, missing_data_action: str = 'KEEP') -> dict:
    # Normalize to upper-case so KEEP / IMPUTE / REMOVE work in any case.
    action = str(missing_data_action).strip().upper()
    valid_actions = ['KEEP', 'IMPUTE', 'REMOVE']
    if action not in valid_actions:
        raise ValueError(f"Invalid missing_data_action. Must be one of: {valid_actions}")

    # If the user chooses to keep the missing data, we just pass the data forward.
    # Our downstream math functions (R2, F-test, curve_fit) are already NaN-proof.
    if action == 'KEEP':
        return data

    genes_to_remove = []

    for gene, (timepoints, y) in data.items():
        # Check if the gene array contains any NaNs
        if not np.isnan(y).any():
            continue

        if action == 'REMOVE':
            genes_to_remove.append(gene)
            continue

        if action == 'IMPUTE':
            # Reshape the 1D array into a 2D matrix of shape (num_timepoints, n_observations)
            # This allows us to calculate the mean per timepoint block automatically
            y_reshaped = y.reshape(-1, n_observations)

            for i in range(y_reshaped.shape[0]):
                group = y_reshaped[i]
                nan_mask = np.isnan(group)

                # If there is at least one missing value in this timepoint
                if nan_mask.any():
                    valid_values = group[~nan_mask]

                    # If there are surviving valid values, use their mean.
                    if len(valid_values) > 0:
                        # Partial gap: impute the mean of the surviving replicates.
                        group[nan_mask] = np.mean(valid_values)
                    # else: the WHOLE timepoint is missing -> leave as NaN.
                    # Inventing a value (0.0 or the global mean) would bias amplitude/
                    # phase; the downstream math is NaN-safe, so the point is simply
                    # excluded from the fit instead.

            # Flatten the array back to 1D and update the dictionary
            data[gene] = (timepoints, y_reshaped.flatten())

    # Safely remove discarded genes from the dictionary outside the loop
    if action == 'REMOVE':
        for gene in genes_to_remove:
            del data[gene]

    return data

#-------------------------------------------------------------------
# Polar rose plot
#-------------------------------------------------------------------
def generate_polar_plot(df_results: pd.DataFrame,counter_threshold: int,base_dir: str,time_label: str = 'Time (ZT Hours)') -> plt.Figure:
    print("\n" + "=" * 70)
    print('                         Generating Polar Plot                        ')
    print("=" * 70)

    # 1. Filter rhythmic genes using vectorized Pandas operations
    # Based on the rule: 0 (Arrhythmic) to 4 (Extremely High)
    rhythmic_df = df_results[df_results['counter'] >= counter_threshold]

    if rhythmic_df.empty:
        print(f"No genes found with counter >= {counter_threshold}. Polar plot skipped.")
        return None

    print(f"Polar plot: {len(rhythmic_df)} targets included.")

    # 2. Extract phases and amplitudes
    # Convert phase 24.0 to 0.0 to close the circle correctly
    phases = rhythmic_df['phase'].replace(24.0, 0.0).values
    amplitudes = rhythmic_df['amplitude'].values

    # 3. Bin data for the polar histogram
    bin_edges = np.arange(0, 25, 1)
    bin_centers = np.arange(0, 24, 1)

    # Split by amplitude threshold (0.5 is a common biological baseline)
    counts_low, _ = np.histogram(phases[amplitudes < 0.5], bins=bin_edges)
    counts_high, _ = np.histogram(phases[amplitudes >= 0.5], bins=bin_edges)

    # 4. Initialize Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    theta = (bin_centers / 24.0) * 2 * np.pi
    width = (2 * np.pi) / 24

    color_low = '#9E9AC8'
    color_high = '#54278F'

    # 5. Plot stacked bars
    ax.bar(
        theta, counts_low, width=width, bottom=0,
        color=color_low, alpha=0.85, linewidth=0.5,
        edgecolor='white', label='Amplitude < 0.5'
    )
    ax.bar(
        theta, counts_high, width=width, bottom=counts_low,
        color=color_high, alpha=0.85, linewidth=0.5,
        edgecolor='white', label='Amplitude >= 0.5'
    )

    # 6. Customize Polar Axes (Clockwise, Zero at North)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks((np.arange(0, 24) / 24.0) * 2 * np.pi)
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=9)

    # 7. Scale Y-axis dynamically
    max_stacked = int(np.max(counts_low + counts_high))
    y_max = max(50, int(np.ceil(max_stacked / 50.0) * 50))
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, max(50, y_max // 4)))
    ax.tick_params(axis='y', labelsize=11)

    # 8. Define Time Prefix Dynamically
    if 'CT' in time_label:
        tp_prefix = 'CT'
    elif 'Clock' in time_label:
        tp_prefix = 'h'
    else:
        tp_prefix = 'ZT'

    ax.set_xlabel(f'Acrophase ({tp_prefix})', labelpad=15, fontsize=13)
    ax.set_title('# of genes', y=1.08, fontsize=13)

    # 9. Legends and Annotations
    ax.legend(
        title='Amplitude', loc='upper right',
        bbox_to_anchor=(1.35, 1.15), fontsize=12, title_fontsize=13
    )

    # Corrected Labels corresponding to the actual rhythmicity logic
    labels_dict = {
        0: 'ARRHYTHMIC',
        1: 'LOW',
        2: 'MEDIUM',
        3: 'HIGH',
        4: 'EXTREMELY HIGH'
    }

    fig.text(
        0.97, 0.97,
        f'n = {len(rhythmic_df)} genes\n(>= {labels_dict.get(counter_threshold, "UNKNOWN")})',
        ha='right', va='top', fontsize=11, color=color_high, fontweight='bold'
    )

    # 10. Save Output
    plt.tight_layout()

    # Ensure directory exists before saving
    os.makedirs(os.path.join(base_dir, 'plots'), exist_ok=True)
    out_path = os.path.join(base_dir, 'plots', 'polar_plot.png')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Polar plot saved to: \n{out_path}")

    return fig

# -------------------------------------------------------------------
# Heatmap — gene names + blue-to-red expression matrix + colorbar.
# -------------------------------------------------------------------
def generate_heatmap(df_results: pd.DataFrame, data: dict, n_observations: int, counter_threshold: int, base_dir: str,
                     time_label: str = 'Time (ZT Hours)') -> tuple:
    print("\n" + "=" * 70)
    print('                         Generating Heatmap                          ')
    print("=" * 70)

    # Labels for the genes
    LABELS = {0: 'ARRHYTHMIC', 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', 4: 'EXTREMELY HIGH'}

    # Derive short time prefix for column labels
    if 'CT' in time_label:
        tp_prefix = 'CT'
    elif 'Clock' in time_label:
        tp_prefix = ''
    else:
        tp_prefix = 'ZT'

    # Name Protection: Checks if the column has been renamed to uppercase or not.
    col_counter = 'Counter' if 'Counter' in df_results.columns else 'counter'
    col_phase = 'Phase' if 'Phase' in df_results.columns else 'phase'
    col_gene = 'Gene' if 'Gene' in df_results.columns else 'gene'

    # 2. Filter and Sort using Pandas
    rhythmic_df = df_results[df_results[col_counter] >= counter_threshold].copy()

    if rhythmic_df.empty:
        print(f"No targets found (counter >= {counter_threshold}). Heatmap skipped.")
        return None, None

    # Sort rhythmic genes by acrophase (phase)
    rhythmic_df = rhythmic_df.sort_values(by=col_phase)

    # Extract the names from the correct column.
    sorted_gene_names = rhythmic_df[col_gene].tolist()

    # 3. Build Expression Matrix Using NumPy Vectorization
    gene_names = []
    matrix_rows = []
    time_labels = None

    for raw_gene in sorted_gene_names:
        # Clears gene name and forces it to String.
        gene = str(raw_gene).strip()

        if gene not in data:
            continue

        timepoints, values_array = data[gene]

        if time_labels is None:
            time_labels = [f"{tp_prefix}{int(h)}" for h in timepoints]

        # Ensures that the values are a NumPy Array.
        val_arr = np.array(values_array, dtype=float)
        reshaped_values = val_arr.reshape(-1, n_observations)
        means_per_tp = np.nanmean(reshaped_values, axis=1)

        gene_names.append(gene)
        matrix_rows.append(means_per_tp)

    if not gene_names:
        print("No valid expression data found. Heatmap skipped.")
        return None, None

    # 4. Z-score normalization per gene (row), clipped to [-3, 3]
    matrix = np.array(matrix_rows, dtype=float)
    row_means = np.nanmean(matrix, axis=1, keepdims=True)
    row_stds = np.nanstd(matrix, axis=1, keepdims=True)

    row_stds[row_stds == 0] = 1.0
    matrix_z = np.clip((matrix - row_means) / row_stds, -3.0, 3.0)

    n_genes = len(gene_names)
    n_tp = matrix_z.shape[1]
    print(f"{n_genes} targets  x  {n_tp} time points")

    # ------------------------------------------------------------------
    # Adaptive DPI Layout
    # ------------------------------------------------------------------
    FIG_W = 10.0;
    FIG_H = 12.0;
    HM_H_FRAC = 0.82
    MIN_PX_ROW = 3;
    BASE_DPI = 150;
    MAX_DPI = 300

    hm_h_in = FIG_H * HM_H_FRAC
    needed_dpi = int(np.ceil((n_genes * MIN_PX_ROW) / hm_h_in))
    out_dpi = max(BASE_DPI, min(MAX_DPI, needed_dpi))
    cell_px = (hm_h_in * out_dpi) / n_genes

    show_labels = (cell_px >= 6) and (n_genes <= 50)
    gene_fontsize = max(2.0, min(7.0, cell_px * 0.55))
    ax_fontsize = 9.0

    CBAR_W, GAP, R_PAD, BOT_PAD, TOP_PAD = 0.18, 0.10, 0.60, 0.55, 0.65
    LABEL_W = 1.8 if show_labels else 0.0
    LABEL_GAP = GAP if show_labels else 0.0
    hm_w = n_tp * 0.55

    fig_w = max(FIG_W, LABEL_W + LABEL_GAP + hm_w + GAP + CBAR_W + R_PAD)
    fig_h = FIG_H

    fig = plt.figure(figsize=(fig_w, fig_h))
    fw, fh = fig_w, fig_h

    def fr(v, t):
        return v / t

    hm_h = fig_h - TOP_PAD - BOT_PAD
    y0 = fr(BOT_PAD, fh)
    height = fr(hm_h, fh)

    x_hm = fr(LABEL_W + LABEL_GAP, fw)
    x_cbar = fr(LABEL_W + LABEL_GAP + hm_w + GAP, fw)
    w_hm = fr(hm_w, fw)
    w_cbar = fr(CBAR_W, fw)

    if show_labels:
        ax_lbl = fig.add_axes([fr(0, fw), y0, fr(LABEL_W, fw), height])
        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(-0.5, n_genes - 0.5)
        ax_lbl.invert_yaxis()
        ax_lbl.axis('off')
        for i, name in enumerate(gene_names):
            ax_lbl.text(0.98, i, name, ha='right', va='center',
                        fontsize=gene_fontsize, family='monospace')

    ax_hm = fig.add_axes([x_hm, y0, w_hm, height])
    im = ax_hm.imshow(
        matrix_z, aspect='auto', cmap=plt.cm.RdBu_r,
        vmin=-3, vmax=3, interpolation='nearest', origin='upper'
    )
    ax_hm.set_xticks(range(n_tp))
    ax_hm.set_xticklabels(time_labels, fontsize=ax_fontsize, rotation=45, ha='right')
    ax_hm.set_yticks([])
    ax_hm.tick_params(left=False, bottom=True)
    ax_hm.spines['left'].set_visible(False)

    ax_cb = fig.add_axes([x_cbar, y0 + height * 0.25, w_cbar, height * 0.50])
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('Z-score', fontsize=8, labelpad=4)
    cb.set_ticks([-3, -1.5, 0, 1.5, 3])
    cb.ax.tick_params(labelsize=7)

    fig.text(
        x_hm + w_hm / 2, y0 + height + fr(0.05, fh),
        f'Gene Expression Heatmap  |  {n_genes} genes  \n'
        f'(>= {LABELS.get(counter_threshold, "UNKNOWN")}, sorted by acrophase)',
        ha='center', va='bottom', fontsize=9, fontweight='bold'
    )

    if not show_labels:
        fig.text(
            x_hm - fr(0.012, fw), y0 + height / 2,
            f'{n_genes} genes', ha='right', va='center',
            fontsize=8, rotation=90, color='#555555'
        )

    os.makedirs(os.path.join(base_dir, 'plots'), exist_ok=True)
    out_path = os.path.join(base_dir, 'plots', 'heatmap.png')
    fig.savefig(out_path, dpi=out_dpi, bbox_inches='tight')
    print(f"Heatmap saved to: \n{out_path}")

    df_expr = pd.DataFrame(matrix_z, index=gene_names, columns=time_labels)
    df_expr.index.name = 'gene'

    return fig, df_expr

# -------------------------------------------------------------------
# Export do Excel format
# -------------------------------------------------------------------
def export_excel_formatted_single(df, file_path):
    # Writes the results to .xlsx with auto-fit column widths and all cells
    # centered. Falls back to a plain export if xlsxwriter is unavailable.
    try:
        import xlsxwriter  # noqa: F401
    except ModuleNotFoundError:
        df.to_excel(file_path, index=False, engine='openpyxl')
        return

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Results', index=False)
    workbook = writer.book
    worksheet = writer.sheets['Results']

    # Centered format for all cells (header and body)
    center = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    header_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True})

    for col_idx, col_name in enumerate(df.columns):
        # Auto-fit width: the longest of the header and the cell values, +2 padding
        col_values = df[col_name].astype(str)
        max_len = max([len(str(col_name))] + [len(str(v)) for v in col_values]) + 2
        worksheet.set_column(col_idx, col_idx, max_len, center)
        # Re-write the header with the bold+centered format
        worksheet.write(0, col_idx, col_name, header_fmt)

    writer.close()

# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def main():
    # Declaring all global variables for R's Spyder/reticulate
    global interval_var, r2_threshold, plot_flag, plot_all, p_value_option, \
        p_threshold, time_label, min_rhythmicity, missing_data_action, \
        period_mode, fixed_period, period_lower, period_upper, \
        counter_threshold, df_results, df_expression, \
        fig_r2_high, fig_r2_low, fig_r2_all, fig_r2_pvalue, \
        fig_amplitude, fig_polar, fig_heatmap, \
        colors, colors2

    base_dir = os.getcwd()
    results_dir = os.path.join(base_dir, 'CODA_Results')
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    data = {}
    input_config = {'n_observations': 1, 'genes_to_plot': []}

    import __main__

    # It detects R by the "signature" that the wrapper injects into Python's __main__ function.
    is_r_injected = hasattr(__main__, 'df_input')

    if is_r_injected:
        print("\n[INFO] Perfect connection with R! Reading the injected parameters....")

        # Read the values that the wrapper (coda_single.R) has already delivered to Python.
        df_input = __main__.df_input
        r_timepoints = list(__main__.r_timepoints)

        input_config['n_observations'] = int(__main__.r_n_obs)

        raw_genes = getattr(__main__, 'r_targets_to_plot', None)
        if raw_genes is not None:
            if isinstance(raw_genes, str):
                input_config['genes_to_plot'] = [raw_genes.upper()]
            else:
                input_config['genes_to_plot'] = [str(g).upper() for g in raw_genes]

        # ==================================================================
        # 1. Statistical parameters (read from __main__, NOT from the r object)
        # ==================================================================
        r2_threshold = float(getattr(__main__, 'r2_threshold', 0.4))
        p_threshold = float(getattr(__main__, 'p_threshold', 0.05))
        p_value_option = str(getattr(__main__, 'p_value_option', 'FDR'))
        amp_stringency = min(max(float(getattr(__main__, 'amp_stringency', 0.5)), 0.0), 1.0)
        min_rhythmicity = str(getattr(__main__, 'min_rhythmicity', 'HIGH'))
        missing_data_action = str(getattr(__main__, 'missing_data_action', 'KEEP'))
        interval_var = int(getattr(__main__, 'interval_var', 1))
        period_mode = str(getattr(__main__, 'period_mode', 'fixed'))
        fixed_period = float(getattr(__main__, 'fixed_period', 24.0))
        period_lower = float(getattr(__main__, 'period_lower', 20.0))
        period_upper = float(getattr(__main__, 'period_upper', 28.0))
        plot_flag = str(getattr(__main__, 'plot_flag', 'N'))
        plot_all = str(getattr(__main__, 'plot_all', 'Y'))
        time_label = str(getattr(__main__, 'time_label', 'ZT'))

        # Translates the R DataFrame into Python's internal dictionary.
        gene_col = df_input.columns[0]
        for idx, row in df_input.iterrows():
            gene = str(row[gene_col]).strip()
            values = pd.to_numeric(row.iloc[1:], errors='coerce').values
            data[gene] = (r_timepoints, values)

    else:
        print("\n[INFO] R environment not detected. Running direct in Python")
        input_file = os.path.join(base_dir, 'single_input.txt')
        if os.path.exists(input_file):
            input_config = read_input_config(input_file)
            data_file = os.path.join(base_dir, input_config['data_file'])
            data = read_data_file(
                file_path=data_file,
                timepoints=input_config['timepoints'],
                n_observations=input_config['n_observations']
            )
        else:
            print(f"\n[ERROR] Configuration file not found for local testing.: {input_file}")
            return

        # ==================================================================
        # 2. Local mode: R isn't here to inject parameters
        # ==================================================================
        if r2_threshold is None:
            interval_var = 1                # 1, 2 or 3 — width of the percentile band in the interval test (1=25-75%, 2=15-85%, 3=5-95%; higher = more permissive)
            r2_threshold = 0.4              # float 0.0-1.0 — minimum R² required to count as a good fit
            plot_flag = 'Y'                 # 'Y' or 'N' — whether to generate the individual per-gene plots
            plot_all = 'N'                  # 'Y' = plot every gene | 'N' = plot only the genes listed in genes_to_plot
            p_value_option = 'FDR'          # 'FDR' = adjusted p-value (Benjamini-Hochberg) | 'RAW' or 'O' = original p-value
            p_threshold = 0.05              # float 0.0-1.0 — significance cutoff applied to the chosen p-value (e.g. 0.05, 0.01)
            amp_stringency = 0.5            # 0.0 = off | 0.5 = default | 1.0 = strictest
            time_label = 'ZT'               # 'ZT', 'CT' or 'Clock' — time convention / X-axis label
            min_rhythmicity = 'HIGH'        # 'ARRHYTHMIC', 'LOW', 'MEDIUM', 'HIGH' or 'EXTREMELY HIGH' — minimum rhythmicity tier
            missing_data_action = 'KEEP'    # 'KEEP', 'IMPUTE' or 'REMOVE' — how to handle missing data (NaN)
            period_mode = 'fixed'           # 'fixed' = use the fixed_period value | 'variable' = fit the period within [period_lower, period_upper]
            fixed_period = 24.0             # float in hours — period used when period_mode = 'fixed' (e.g. 24.0)
            period_lower = 20.0             # float in hours — lower bound for the period when period_mode = 'variable'
            period_upper = 28.0             # float in hours — upper bound for the period when period_mode = 'variable'

    # Security against empty data
    if not data:
        print("\n[ERROR] Data processing failed. Check the input. Aborting....")
        return

    # ==================================================================
    # 3. Handle the rhythmicity threshold mapping for the counter
    # ==================================================================
    rhythm_map = {'EXTREMELY HIGH': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'ARRHYTHMIC': 0}

    if isinstance(min_rhythmicity, str):
        counter_threshold = rhythm_map.get(min_rhythmicity.upper(), 3)

    # Internal mapping for time conventions
    # 1: ZT, 2: CT, 3: Clock
    if time_label == 'ZT':
        convention = 1
    elif time_label == 'CT':
        convention = 2
    else:
        convention = 3

    # ==================================================================
    # 4. Handle Missing Data globally before entering the gene loop
    # ==================================================================
    data = handle_missing_data(
        data=data,
        n_observations=input_config['n_observations'],
        missing_data_action=missing_data_action
    )

    # ==================================================================
    # 5. PREPARE FOR PROCESSING
    # ==================================================================
    # Storage vectors for the main loop
    # PEP8 Best Practice: Avoid using semicolons to put multiple variables
    # on the same line. Vertical alignment is cleaner and easier to read.
    r2_values = []
    p_values = []
    amp_correct = []
    amp_limits = []
    results_list = []

    # Color vectors for global plots (if still required by downstream functions)
    colors = []
    colors2 = []

    # Get the list of genes explicitly marked for plotting
    # (Extracted automatically during the config read phase)
    genes_to_plot = input_config['genes_to_plot']

    if not genes_to_plot:
        print("[INFO] No specific targets to plot were provided in the config.")

    # ==================================================================
    # 6. MAIN GENE LOOP
    # ==================================================================
    print("\n" + "=" * 70)
    print('                         Evaluation in process                        ')
    print("=" * 70)
    time.sleep(0.1)

    for gene, (timepoints, y) in tqdm(data.items(), desc="Processing Targets", unit="target"):

        # 1. Convert timepoints to float array and align with observations
        X_base = np.array(timepoints, dtype=float)
        X = np.repeat(X_base, input_config['n_observations'])
        y_array = np.array(y, dtype=float)

        # 2. Isolate valid data for initial parameter guesses to avoid NaN errors
        valid_mask = ~np.isnan(y_array)
        y_valid = y_array[valid_mask]

        if len(y_valid) < 4:
            print(f"[DEBUG] Target '{gene}' has insufficient valid data. Skipping.")
            continue

        # Initial parameter guesses (NaN-safe)
        k0 = np.nanmean(y_valid)
        a0 = (np.nanmax(y_valid) - np.nanmin(y_valid)) / 2.0

        # Determine the phase guess based on the actual peak of expression
        idx_max = np.argmax(y_valid)
        t_peak = X[valid_mask][idx_max]
        f0 = (t_peak / 24.0) * (2 * np.pi)

        # 3. Outlier detection and filtering (for the interval confidence limits)
        groups = [y_array[i:i + input_config['n_observations']]
                  for i in range(0, len(y_array), input_config['n_observations'])]

        _, outliers = check_outliers(groups)
        values_filtered = y_valid[~np.isin(y_valid, outliers)]

        # 4. Rhythmicity F-test: compare cosinor vs null model
        F_stat, p_value_rhythm, _, _ = test_rhythmicity_cosinor(X, y_array)

        # 5. Group statistics
        means, mean_of_means, std_dev = calculate_full_metrics(groups)

        # Adaptive amplitude threshold, scaled by the stringency dial:
        # 0.5 -> unchanged (default), 0 -> off, 1.0 -> 2x.
        adaptive_limit = max(min(_amp_mean_ratio * mean_of_means, _amp_std_ratio * std_dev), _amp_floor)
        amp_limit = (2.0 * amp_stringency) * adaptive_limit
        amp_limits.append(amp_limit)

        # 6. Curve Fitting (Wrapped in warning catcher to avoid double-processing)
        params, r_squared, y_pred = None, None, None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Using our robust function
            if period_mode == 'fixed':
                params, r_squared, y_pred = perform_curve_fit(X, y_array, fixed_period=True, period=fixed_period)
            else:
                params, r_squared, y_pred = perform_curve_fit(X, y_array,fixed_period=False,bounds=(period_lower, period_upper))

            if w:
                print(f"Warning fitting gene '{gene}': {w[-1].message}")

        if params is None:
            continue  # Skip gene if the mathematical fit diverged completely

        # Extract fitted parameters
        k_est, a_est, f_est, T_est = params
        r2_values.append(r_squared)
        p_values.append(p_value_rhythm)

        # ------------------------------------------------------------------
        # FIX 1
        # ------------------------------------------------------------------
        x_data_min = np.min(X)
        x_data_max = np.max(X)
        X_interp = np.linspace(x_data_min, x_data_max, 500)
        y_interp = circular_function(X_interp, k_est, a_est, f_est, T_est)

        # Peak and trough values
        x_peak = X_interp[np.argmax(y_interp)]
        x_peak = x_peak % T_est
        y_peak = np.max(y_interp)
        y_trough = np.min(y_interp)

        amplitude_corr = y_peak - mean_of_means
        amp_correct.append(amplitude_corr)

        # Real decimal values (hours) -- use THESE for any math (means, deltas, etc.)
        phase_decimal = x_peak  # acrophase in real decimal hours (e.g. 13.75)
        period_decimal = T_est  # period in real decimal hours

        # Display-only "clock" format where the digits after the dot are MINUTES
        # (e.g. 13.45 means 13h45min). NEVER do arithmetic on these columns.
        hour_phase = int(x_peak)
        minute_phase = int(round((x_peak - hour_phase) * 60)) / 100.0
        phase_hhmm = hour_phase + minute_phase

        hour_T = int(T_est)
        minute_T = int(round((T_est - hour_T) * 60)) / 100.0
        if minute_T > 0.59:
            minute_T = 0
            hour_T += 1
        period_hhmm = hour_T + minute_T

        # ------------------------------------------------------------------
        # RHYTHMICITY SCORING (Criteria 1-3)
        # Criterion 4 (p-value) will be added after FDR correction
        # ------------------------------------------------------------------
        counter = 0

        # Criteria 1: Amplitude
        if amplitude_corr >= amp_limit:
            counter += 1

        # Criteria 2: Goodness of fit (R2)
        if r_squared is not None and r_squared >= r2_threshold:
            counter += 1

        # Criteria 3: Interval bounds check
        aux1 = 25 - 10 * (interval_var - 1)
        aux2 = 100 - aux1

        # Calculate bounds using only the clean, outlier-free data
        lower_bound = np.percentile(values_filtered, aux1)
        upper_bound = np.percentile(values_filtered, aux2)

        if (y_trough >= lower_bound) and (y_peak <= upper_bound):
            interval_flag = "In"
            inside_interval = True
        else:
            counter += 1
            interval_flag = "Out"
            inside_interval = False

        # Store gene summary
        results_list.append({
            "gene": gene,
            "mean": mean_of_means,
            "amplitude": amplitude_corr,
            "amp_limit": amp_limit,
            "phase": phase_decimal,
            "phase_hhmm": phase_hhmm,
            "p_value": p_value_rhythm,
            "p_value_adjusted": np.nan,  # Placeholder for FDR
            "r_squared": r_squared,
            "interval": interval_flag,
            "period": period_decimal,
            "period_hhmm": period_hhmm,
            "counter": counter
        })

        # ------------------------------------------------------------------
        # FIX 2, 3, 4: Individual Gene Plots (COMPARE Visual Style)
        # ------------------------------------------------------------------
        # Clean the file name to avoid errors when saving.
        safe_gene_name = str(gene).replace("/", "_").replace("\\", "_").replace(":", "_")

        if str(plot_flag).upper() == 'Y' and (plot_all == 'Y' or gene.upper() in genes_to_plot):
            fig_g, ax = plt.subplots(figsize=(10, 6))

            # Accessible color palette (Color Blind Friendly)
            # Using the first color for the Single version
            group_color = '#D55E00'

            # Dark period shading (Dynamically covering all days present in data)
            # Shades from 12 to 24, 36 to 48, etc., depending on data length
            # max_cycle_hours = int(np.ceil(x_data_max / 24.0) * 24)
            #for start_shade in range(12, max_cycle_hours, 24):
            #    ax.axvspan(start_shade, start_shade + 12, color='lightgray', alpha=0.8, zorder=0)

            # Calculation of mean and error (SEM) for experimental data points
            x_unique = np.sort(np.unique(X_base))
            y_mean = []
            y_sem = []

            for zt in x_unique:
                # Extract valid values for this specific timepoint (ignoring NaNs)
                zt_values = y_array[(X == zt) & ~np.isnan(y_array)]
                if len(zt_values) > 0:
                    y_mean.append(np.mean(zt_values))
                    y_sem.append(stats.sem(zt_values) if len(zt_values) > 1 else 0.0)
                else:
                    y_mean.append(np.nan)
                    y_sem.append(np.nan)

            y_mean = np.array(y_mean)
            y_sem = np.array(y_sem)

            # Draw the points with error bars
            ax.errorbar(
                x_unique, y_mean, yerr=y_sem,
                fmt='o', color=group_color, ecolor=group_color,
                elinewidth=1.8, capsize=4, capthick=1.8, markersize=7, alpha=0.8, zorder=5
            )

            # Draw the fitted curve (Cosinor)
            ax.plot(X_interp, y_interp, color=group_color, linewidth=2.5, zorder=4)

            # Optional: Add the interval thresholds from the Single version subtly
            if inside_interval:
                ax.axhline(lower_bound, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
                ax.axhline(upper_bound, color='black', linestyle='--', linewidth=0.8, alpha=0.4)

            #ax.axvline(x_peak, color='green', linestyle='--', alpha=0.6, label='Phase (acrophase)')

            # Axis Settings
            ax.set_xlabel(time_label)

            # X-Limits: start where the observed data begins and end where it ends
            span = x_data_max - x_data_min
            pad = 0.03 * span  # 3% at each side
            ax.set_xlim(x_data_min - pad, x_data_max + pad)
            step = 2 if span <= 24 else (4 if span <= 60 else 8)
            ax.set_xticks(np.arange(x_data_min, x_data_max + 1, step))

            # Intelligent Y-scale adjustment with margin
            # We consider both the curve max/min AND the error bar max/min
            y_min_total = np.nanmin(y_interp)
            y_max_total = np.nanmax(y_interp)

            valid_means = y_mean[~np.isnan(y_mean)]
            valid_sems = y_sem[~np.isnan(y_sem)]
            if len(valid_means) > 0:
                y_min_total = np.min(valid_means - valid_sems)
                y_max_total = np.max(valid_means + valid_sems)
            else:
                y_min_total = np.nanmin(y_interp)
                y_max_total = np.nanmax(y_interp)

            margin = max(0.5, 0.1 * (y_max_total - y_min_total))
            ax.set_ylim(y_min_total - margin, y_max_total + margin)

            ax.set_ylabel('Observed Variables')
            ax.set_title(f"{gene}", loc='left', fontweight='bold', fontstyle='italic')

            # Create a custom caption
            legend_handles = [
                mlines.Line2D([], [], color=group_color, marker='o', linestyle='-',
                              linewidth=2.5, markersize=7, label='Model & Data')
            ]

            ax.legend(handles=legend_handles, loc='upper right', frameon=True)

            # Preserve the annotation box on the outside right
            annotation = (
                f'Amplitude: {amplitude_corr:.2f}\n'
                f'Phase: {phase_hhmm:.2f} (h:min)\n'
                f'Mean: {mean_of_means:.2f}\n'
                f'Period: {period_hhmm:.2f} (h:min)'
            )
            ax.text(
                1.03, 0.5, annotation, transform=ax.transAxes, fontsize=12,
                va='center', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            fig_g.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.12)
            plt.savefig(os.path.join(results_dir, 'plots', f'gene_{safe_gene_name}.png'), dpi=300)
            plt.close(fig_g)
    # ------------------------------------------------------------------
    # END OF MAIN LOOP
    # ------------------------------------------------------------------

    # ==================================================================
    # 7. POST-LOOP PROCESSING: FDR, COUNTER
    # ==================================================================
    # 1. Convert the list of dictionaries into a Pandas DataFrame
    # This unlocks vectorized operations and prepares data for the plotting functions
    global df_results
    df_results = pd.DataFrame(results_list)

    if df_results.empty:
        print("\n[ERROR] No genes were successfully fitted. Analysis aborted.")
        return

    # 2. FDR Correction (Benjamini-Hochberg)
    # Ensure we only calculate FDR for genes that actually generated a valid p-value
    valid_p_mask = df_results['p_value'].notna()

    if valid_p_mask.any():
        _, p_adj, _, _ = multipletests(
            df_results.loc[valid_p_mask, 'p_value'],
            method='fdr_bh'
        )
        df_results.loc[valid_p_mask, 'p_value_adjusted'] = p_adj

    # 3. Final P-Value Selection
    # If option 'O' or 'RAW' is chosen, use raw p-value; otherwise use FDR
    if str(p_value_option).upper() in ['O', 'RAW']:
        df_results['p_value_final'] = df_results['p_value']
    else:
        df_results['p_value_final'] = df_results['p_value_adjusted']

    # 4. Finalize Rhythmicity Scoring (Criterion 4)
    # Increment the counter if the final p-value is below the threshold
    p_value_passed = (df_results['p_value_final'] <= p_threshold).astype(int)
    df_results['counter'] = df_results['counter'] + p_value_passed

    # Enforce maximum cap at 4 (just in case of overlapping logic)
    df_results['counter'] = df_results['counter'].clip(upper=4)

    # Apply the Rhythmicity Classification Label
    classification_map = {
        0: 'ARRHYTHMIC',
        1: 'LOW',
        2: 'MEDIUM',
        3: 'HIGH',
        4: 'EXTREMELY HIGH'
    }
    df_results['classification'] = df_results['counter'].map(classification_map)

    # ==================================================================
    # 8. SVM
    # ==================================================================
    print("\n" + "=" * 70)
    print('                         SVM - Analysis                      ')
    print("=" * 70)
    # SVM Suggestion for R-Squared Threshold
    # Only attempt to train if there are valid R2 and P-values
    valid_svm_mask = df_results['r_squared'].notna() & df_results['p_value_final'].notna()

    r2_arr = df_results.loc[valid_svm_mask, 'r_squared'].values.reshape(-1, 1)
    pv_arr = df_results.loc[valid_svm_mask, 'p_value_final'].values
    svm_labels = (pv_arr <= p_threshold).astype(int)

    suggested_r2 = np.nan

    # SVC.fit will crash if there is only one class (e.g., all 0s or all 1s).
    # We must ensure there is variance in the significance results.
    if len(np.unique(svm_labels)) > 1:
        try:
            svm_model = SVC(kernel='linear')
            svm_model.fit(r2_arr, svm_labels)

            # Vectorized threshold search
            test_limits = np.linspace(0, 1, 1000).reshape(-1, 1)
            predictions = svm_model.predict(test_limits)

            # Find the first R2 value where the prediction becomes 1 (Significant)
            significant_indices = np.where(predictions == 1)[0]
            if len(significant_indices) > 0:
                suggested_r2 = test_limits[significant_indices[0]][0]
                print(f"[INFO] SVM suggested R² threshold: {suggested_r2:.4f}")
            else:
                print("[INFO] SVM could not find a clear boundary for R².")
        except Exception as e:
            print(f"[WARNING] Failed to calculate SVM R² threshold: {e}")
    else:
        print("[INFO] SVM calculation skipped: All valid targets fall into a single \nsignificance category.")

    # ==================================================================
    # 9. EXPORTING RESULTS (EXCEL)
    # ==================================================================
    print("\n" + "=" * 70)
    print('                         Writing Rresults (Excel)                   ')
    print("=" * 70)
    time.sleep(0.1)

    # 1. Generate color vectors for downstream plots
    # We do this vectorized (fast) instead of using a loop.
    # Red if p-value fails threshold or is NaN, Blue otherwise
    failed_p_mask = df_results['p_value_final'].isna() | (df_results['p_value_final'] > p_threshold)

    colors = np.where(failed_p_mask, 'red', 'blue').tolist()
    colors2 = colors.copy()

    # 2. Format the DataFrame specifically for the final Excel presentation
    df_export = pd.DataFrame({
        'Target': df_results['gene'],
        'Mesor': df_results['mean'].round(4),
        'Amplitude': df_results['amplitude'].round(4),
        'Amp. Minimum': df_results['amp_limit'].round(4),
        'Phase': df_results['phase'].round(4),
        'Phase (h:min)': df_results['phase_hhmm'].round(2),

        # Format p-values using scientific notation, handling potential NaNs
        'p_value': df_results['p_value'].apply(
            lambda x: f"{x:.4e}" if pd.notnull(x) else "NaN"
        ),
        'p_adj': df_results['p_value_adjusted'].apply(
            lambda x: f"{x:.4e}" if pd.notnull(x) else "NaN"
        ),

        'Interval': df_results['interval'],
        'R2': df_results['r_squared'].round(4),
        'Period': df_results['period'].round(4),
        'Period (h:min)': df_results['period_hhmm'].round(2),
        'Probability': df_results['classification']
    })

    # 3. Save to Excel
    # Ensure the file extension is updated to .xlsx
    excel_file = os.path.join(results_dir, 'CODAC_Single_Results.xlsx')

    try:
        # Formatted export: auto-fit column widths + centered cells (xlsxwriter),
        # with a plain openpyxl fallback if xlsxwriter is missing.
        export_excel_formatted_single(df_export, excel_file)
        print(f"Results successfully saved to: \n{excel_file}")
    except ModuleNotFoundError:
        print("\n[ERROR] The 'openpyxl' library is required to save Excel files.")
        print("Please run: pip install openpyxl")

    # ==================================================================
    # 10. SUMMARY PLOTS & VARIABLE EXPLORER EXPORT
    # ==================================================================
    time.sleep(0.1)
    print("\n" + "=" * 70)
    print('                         Plotting Summary Charts                      ')
    print("=" * 70)
    time.sleep(0.1)

    # Use Pandas boolean masking for instant filtering
    mask_high_p = df_results['p_value_final'] > p_threshold
    df_high = df_results[mask_high_p]
    df_low = df_results[~mask_high_p]

    # Map colors for the amplitude plot dynamically based on the 'interval' column
    amp_colors = df_results['interval'].map({'In': 'yellow', 'Out': 'green'}).tolist()

    with tqdm(total=5, desc="Generating Figures", unit="figure") as pb:
        def _save_fig(filename):
            fig = plt.gcf()
            fig.savefig(os.path.join(results_dir, 'plots', filename), dpi=300)
            plt.close(fig)  # Crucial to prevent memory leaks!
            pb.update(1)
            return fig

        # Chart 1: R² — genes with p-value > threshold
        plt.figure(figsize=(10, 6))
        highlight_colors_low = ['red' if p > p_threshold else 'white'
                                for p in df_results['p_value_final']]
        plt.bar(df_results.index, df_results['r_squared'], color=highlight_colors_low)
        plt.title(f'R² — Genes with p-value > {p_threshold:.2f} (highlighted)')
        plt.xlabel('Gene Index')
        plt.ylabel('R²')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.axhline(r2_threshold, color='black', linestyle='--',
                    label=f'R² Threshold = {r2_threshold:.2f}')

        if suggested_r2 and not pd.isna(suggested_r2):
            plt.axhline(suggested_r2, color='orange', linestyle='--',
                        label=f'Suggested R² = {suggested_r2:.2f}')
        plt.legend()
        fig_r2_low = _save_fig('plot_r2_low.png')

        # Chart 2: R² — genes with p-value <= threshold
        plt.figure(figsize=(10, 6))
        highlight_colors = ['blue' if p <= p_threshold else 'white'
                            for p in df_results['p_value_final']]
        plt.bar(df_results.index, df_results['r_squared'], color=highlight_colors)
        plt.title(f'R² — Genes with p-value <= {p_threshold:.2f} (highlighted)')
        plt.xlabel('Gene Index')
        plt.ylabel('R²')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.axhline(r2_threshold, color='black', linestyle='--',
                    label=f'R² Threshold = {r2_threshold:.2f}')

        if suggested_r2 and not pd.isna(suggested_r2):
            plt.axhline(suggested_r2, color='orange', linestyle='--',
                        label=f'Suggested R² = {suggested_r2:.2f}')
        plt.legend()
        fig_r2_high = _save_fig('plot_r2_high.png')

        # Chart 3: R² — all genes
        plt.figure(figsize=(10, 6))
        # 'colors' was defined globally in the Excel export block
        plt.bar(df_results.index, df_results['r_squared'], color=colors)
        plt.title('R² — All Genes')
        plt.xlabel('Gene Index')
        plt.ylabel('R²')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.axhline(r2_threshold, color='black', linestyle='--')

        handles = [
            mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          label=f'p-value > {p_threshold:.2f}'),
            mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          label=f'p-value <= {p_threshold:.2f}'),
            plt.Line2D([0], [0], color='black', linestyle='--',
                       label=f'R² Threshold = {r2_threshold:.2f}')
        ]

        if suggested_r2 and not pd.isna(suggested_r2) and suggested_r2 > 0.01:
            plt.axhline(suggested_r2, color='orange', linestyle='--')
            handles.append(plt.Line2D([0], [0], color='orange', linestyle='--',
                                      label=f'Suggested R² = {suggested_r2:.2f}'))
        plt.legend(handles=handles)
        fig_r2_all = _save_fig('plot_r2_all.png')

        # Chart 4: R² vs p-value scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(df_results['r_squared'], df_results['p_value_final'],
                    color=colors, alpha=0.7)
        plt.title('R² vs p-value')
        plt.xlabel('R²')
        plt.ylabel('p-value')
        plt.xlim(0, 1)
        plt.ylim(-0.5, 1)
        plt.axhline(p_threshold, color='red', linestyle='--',
                    label=f'p-value threshold = {p_threshold:.2f}')
        plt.axvline(r2_threshold, color='black', linestyle='--',
                    label=f'R² threshold = {r2_threshold:.2f}')

        if suggested_r2 and not pd.isna(suggested_r2) and suggested_r2 > 0.01:
            plt.axvline(suggested_r2, color='orange', linestyle='--',
                        label=f'Suggested R² = {suggested_r2:.2f}')

        plt.legend(handles=handles, loc='upper right')
        fig_r2_pvalue = _save_fig('plot_r2_vs_pvalue.png')

        # Chart 5: Amplitude bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(df_results.index, df_results['amplitude'], color=amp_colors)
        plt.title('Amplitude — All Genes')
        plt.xlabel('Gene Index')
        plt.ylabel('Amplitude')
        plt.xticks(rotation=90)

        max_amp = df_results['amplitude'].max()
        plt.ylim(0, (max_amp + 0.1) if not pd.isna(max_amp) else 1)

        # Draw the individual amplitude limit lines per gene
        for i, lim in enumerate(amp_limits):
            plt.plot([i - 0.4, i + 0.4], [lim, lim], color='black', linestyle='--')

        plt.legend(handles=[
            mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          label='Within interval'),
            mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          label='Outside interval')
        ], loc='upper right')

        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        fig_amplitude = _save_fig('plot_amplitude.png')

    # ------------------------------------------------------------------
    # Generate Collaborator's Global Plots
    # ------------------------------------------------------------------
    fig_polar = generate_polar_plot(
        df_results=df_results,
        counter_threshold=counter_threshold,
        base_dir=results_dir,
        time_label=time_label
    )

    fig_heatmap, df_expression = generate_heatmap(
        df_results=df_results,
        data=data,
        n_observations=input_config['n_observations'],
        counter_threshold=counter_threshold,
        base_dir=results_dir,
        time_label=time_label
    )

    # ------------------------------------------------------------------
    # Finish and Report
    # ------------------------------------------------------------------
    # Ensure columns match the requested output explicitly for Spyder
    df_results = df_results.rename(columns={
        'gene': 'Target',
        'mean': 'Mesor',
        'amplitude': 'Amplitude',
        'amp_limit': 'Amp. Minimum',
        'phase': 'Phase',
        'phase_hhmm': 'Phase (h:min)',
        'p_value_adjusted': 'p_adj',
        'r_squared': 'R2',
        'interval': 'Interval',
        'period': 'Period',
        'period_hhmm': 'Period (h:min)',
        'classification': 'Probability',
        'counter': 'Counter'
    })

    print("\n" + "=" * 70)
    print("  Objects available in Variable Explorer:")
    print("    df_results     — results table")
    print("    df_expression  — z-scored expression matrix (rhythmic targets)")
    print("    fig_polar      — polar rose plot figure")
    print("    fig_heatmap    — heatmap figure")
    print("    fig_r2_high / fig_r2_low / fig_r2_all — R² charts")
    print("    fig_r2_pvalue  — R² vs p-value scatter")
    print("    fig_amplitude  — amplitude chart")
    print("=" * 70)
    print("\n[SUCCESS] Rhythmicity analysis completed perfectly!")

    return df_results


if __name__ == "__main__":
    main()