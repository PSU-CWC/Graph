"""
WEC Electronics graphing and CWC scoring utilities for Houston CSV logs.

Every function expects `Time` in SECONDS from the start of the run;
load_data() converts Houston's epoch milliseconds, so always load through it.

Colab:
    !wget -q https://raw.githubusercontent.com/PSU-CWC/Graph/main/graphutils.py
    from graphutils import *
    df = load_data("testing-5-13/tunnel_testing_01.csv")   # relative to DATA on Drive
    graph_2d(df, "Load Power", "Set Power", timestart=30, timeend=60)

PID gains live in pid_tuning.py.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

DATA = '/content/drive/MyDrive/WEC_Electronics/Data/'

# ---------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------

def load_data(f, time_unit='ms'):
    """
    Load a CSV, zero the Time column and convert it to SECONDS.

    Args:
        f: path relative to DATA (Drive), a local path, or a file-like object.
        time_unit: unit of the raw Time column, 'ms' (logger default) or 's'.
    """
    if isinstance(f, str) and not os.path.exists(f):
        f = os.path.join(DATA, f)
    df = pd.read_csv(f)
    df["Time"] = df["Time"] - df["Time"].iloc[0]
    if time_unit == 'ms':
        df["Time"] = df["Time"] / 1000.0
    elif time_unit != 's':
        raise ValueError("time_unit must be 'ms' or 's'")
    return df


def processed_csv(f, resample_interval='100ms'):
    """Load `f` via load_data and resample to a fixed interval. Time in seconds."""
    return processing_graph(load_data(f), resample_interval=resample_interval)


def processing_graph(df, resample_interval='100ms'):
    """Resample an already-loaded df (Time in seconds) to a fixed interval."""
    df_clean = df.copy()

    df_clean['Timedelta'] = pd.to_timedelta(df_clean['Time'], unit='s')
    df_clean = df_clean.set_index('Timedelta')
    df_clean = df_clean.drop(columns=['Time'])

    df_resampled = df_clean.resample(resample_interval).mean().interpolate()
    df_resampled['Time'] = df_resampled.index.total_seconds()
    df_resampled = df_resampled.reset_index(drop=True)

    cols = ['Time'] + [c for c in df_resampled.columns if c != 'Time']
    return df_resampled[cols]


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def graph_2d(df, *vars, timestart=0, timeend=np.inf):
    """Plot one or more columns against Time (seconds), optionally time-cropped."""
    missing = [c for c in vars if c not in df.columns]
    if missing:
        raise KeyError(f"{missing} not in this file. Columns: {list(df.columns)}")
    df = df[(df["Time"] >= timestart) & (df["Time"] <= timeend)]

    plt.figure(figsize=(12, 6))
    for col in vars:
        plt.plot(df["Time"], df[col], label=col, linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title(f"Time vs {', '.join(vars)}", fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def graph_2d_bar(df, *vars, resample=1):
    """Raw line + binned bar + smoothed spline overlay. `resample` in seconds."""
    plt.figure(figsize=(12, 6))
    proc_df = processing_graph(df, resample_interval=f"{int(resample * 1000)}ms")

    x = proc_df["Time"].values
    for col in vars:
        plt.plot(df["Time"], df[col], label=f"{col} (Raw)",
                 color='black', alpha=0.3, linewidth=1.2)
        y = proc_df[col].values
        plt.bar(x, y, width=resample, align='edge', alpha=0.3,
                label=f"{col} (Binned)", edgecolor="black", linewidth=0.5)
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 500)
            y_smooth = np.maximum(make_interp_spline(x, y, k=3)(x_smooth), 0.0)
            plt.plot(x_smooth, y_smooth, label=f"{col} (Smooth)", linewidth=2.5)
        else:
            plt.plot(x, y, marker='o', label=f"{col} (Line)", linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title(f"Time vs {', '.join(vars)}", fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def graph_3d(df, x_col, y_col, z_col):
    """Interactive 3D scatter (drag to rotate), colored by z_col."""
    import plotly.express as px
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col, color=z_col,
        color_continuous_scale='Viridis', opacity=0.7,
        title=f'3D Analysis: {x_col} vs {y_col} vs {z_col}'
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.show()


# ---------------------------------------------------------
# CWC scoring
# ---------------------------------------------------------

_BIN_CHOICES = [5, 6, 7, 8, 9, 10, 12]
_BIN_FACTORS = {5: 0.7, 6: 0.8, 7: 0.8, 8: 0.7, 9: 0.4, 10: 0.3}


def _bin_conditions(df):
    ws = df['Wind Speed']
    return [
        (ws >= 4.5) & (ws < 5.5),
        (ws >= 5.5) & (ws < 6.5),
        (ws >= 6.5) & (ws < 7.5),
        (ws >= 7.5) & (ws < 8.5),
        (ws >= 8.5) & (ws < 9.5),
        (ws >= 9.5) & (ws < 10.5),
        (ws >= 10.5) & (ws <= 13.5),  # rated power regulation range
    ]


def _rated_score(rp):
    """Official CWC tanh scoring for rated-power regulation. rp = P / P_10ms."""
    return 50 * ((np.tanh(-20 * np.abs(rp - 1) + np.pi) + 1) / (np.tanh(np.pi) + 1))


def realtime_score(df, window_size=50, ref_10ms_power=None, cv_threshold=0.05):
    """
    Add rolling statistics and instantaneous CWC scores. Does NOT modify `df`.

    Args:
        df: DataFrame with 'Wind Speed' and 'Load Power'.
        window_size: rolling window in samples (50 = 5s at 10Hz).
        ref_10ms_power: target stable power at 10 m/s. If None, found from data.
        cv_threshold: max coefficient of variation (std/mean) counted as stable.

    Returns:
        A new DataFrame with the monitoring columns added.
    """
    df = df.copy()

    conditions = _bin_conditions(df)
    factor_choices = [_BIN_FACTORS[b] if b in _BIN_FACTORS else 0.0 for b in _BIN_CHOICES]

    df['Bin'] = np.select(conditions, _BIN_CHOICES, default=0)
    df['Factor'] = np.select(conditions, factor_choices, default=0.0)

    roll = df['Load Power'].rolling(window=window_size, min_periods=1)
    df['Rolling_Mean_Power'] = roll.mean()
    df['Rolling_Std_Power'] = roll.std().fillna(0)

    # Coefficient of variation — dimensionless, so one threshold works at 5W and 40W.
    df['Realtime_CV'] = np.where(df['Rolling_Mean_Power'] > 0,
                                 df['Rolling_Std_Power'] / df['Rolling_Mean_Power'],
                                 np.inf)

    df['Instant_PowerCurve_Score'] = df['Rolling_Mean_Power'] * df['Factor']

    if ref_10ms_power is None:
        # Needs a full window of history, else the first samples look "stable" by default.
        # Positional, not df.index — a sliced df (df.iloc[6000:12000]) keeps its
        # original labels and the history guard would never bite.
        has_history = np.arange(len(df)) >= window_size
        valid = df[(df['Bin'] == 10) & (df['Realtime_CV'] < cv_threshold) & has_history]
        ref_10ms_power = valid['Rolling_Mean_Power'].iloc[-1] if not valid.empty else np.nan

    df['Ref_10ms_Power'] = ref_10ms_power

    if np.isfinite(ref_10ms_power) and ref_10ms_power > 0:
        rated = _rated_score(df['Rolling_Mean_Power'] / ref_10ms_power)
    else:
        rated = 0.0  # no stable 10 m/s baseline yet -> nothing to regulate against
    df['Instant_Rated_Score'] = np.where(df['Bin'] == 12, rated, 0.0)

    return df


def final_score(df, cv_threshold=0.05):
    """
    Final official CWC Turbine Testing scores, post-run.

    Args:
        df: DataFrame with 'Wind Speed' and 'Load Power'.
        cv_threshold: max coefficient of variation (std/mean) counted as stable.
                      NOTE: this is now a true CV, not a raw std in watts.

    Returns:
        Dict of scores, plus per-bin diagnostics explaining any zeros.
    """
    df_copy = df.copy()
    df_copy['Bin'] = np.select(_bin_conditions(df_copy), _BIN_CHOICES, default=0)

    power_curve_score = 0.0
    stable_power_10ms = None
    scores_per_bin = {}
    diagnostics = {}

    # Task 1: Power Curve Performance (5-10 m/s)
    for bin_ws in [5, 6, 7, 8, 9, 10]:
        bin_data = df_copy[df_copy['Bin'] == bin_ws]['Load Power']
        if len(bin_data) == 0:
            scores_per_bin[bin_ws] = 0.0
            diagnostics[bin_ws] = {"samples": 0, "reason": "no data"}
            continue

        mean_p, std_p = bin_data.mean(), bin_data.std(ddof=1)
        cv = std_p / mean_p if mean_p > 0 else np.inf
        stable = mean_p > 0 and cv <= cv_threshold

        scores_per_bin[bin_ws] = mean_p * _BIN_FACTORS[bin_ws] if stable else 0.0
        power_curve_score += scores_per_bin[bin_ws]
        if stable and bin_ws == 10:
            stable_power_10ms = mean_p

        diagnostics[bin_ws] = {
            "samples": int(len(bin_data)),
            "mean_W": round(float(mean_p), 4),
            "cv": round(float(cv), 4),
            "stable": bool(stable),
            "reason": "" if stable else f"CV {cv:.3f} > {cv_threshold}",
        }

    # Task 2: Control of Rated Power (11-13 m/s)
    rated_power_score = 0.0
    high_wind_data = df_copy[df_copy['Bin'] == 12]['Load Power']
    if stable_power_10ms is not None and len(high_wind_data) > 0:
        rated_power_score = float(_rated_score(high_wind_data.mean() / stable_power_10ms))

    return {
        "Total Power Curve Score (Max 50+)": round(power_curve_score, 5),
        "Control of Rated Power Score (Max 50)": round(rated_power_score, 5),
        "10m/s Reference Power (W)": round(stable_power_10ms, 5) if stable_power_10ms else None,
        "Detailed Power Curve Bin Scores": {k: float(round(v, 5)) for k, v in scores_per_bin.items()},
        "Bin Diagnostics": diagnostics,
    }



if __name__ == "__main__":
    import io

    # --- load_data normalises Time to seconds -------------------------
    csv = "Time,Load Power\n1000,10\n1100,11\n1200,12\n"
    d = load_data(io.StringIO(csv))
    assert np.allclose(d["Time"].values, [0.0, 0.1, 0.2]), d["Time"].values

    # --- realtime_score must not mutate its argument -------------------
    rng = np.random.default_rng(42)
    base = pd.DataFrame({
        'Time': np.arange(0, 100, 0.1),
        'Wind Speed': np.repeat([5.0, 10.0, 12.0, 7.0], 250),
        'Load Power': np.repeat([5.0, 20.0, 20.0, 9.0], 250),
    })
    cols_before = list(base.columns)
    scored = realtime_score(base)
    assert list(base.columns) == cols_before, "realtime_score mutated the input"
    assert 'Instant_PowerCurve_Score' in scored.columns

    # --- cv_threshold is a real CV, scale-independent ------------------
    stable = base.copy()
    noisy = base.copy()
    noisy['Load Power'] = noisy['Load Power'] * (1 + rng.normal(0, 0.30, len(noisy)))
    s_stable = final_score(stable)
    s_noisy = final_score(noisy)
    assert s_stable["Total Power Curve Score (Max 50+)"] > 0
    assert s_noisy["Total Power Curve Score (Max 50+)"] == 0.0, \
        "30% noise must fail a 5% CV gate"
    # a 5W bin and a 20W bin with the same *relative* noise must be judged alike
    assert s_stable["Bin Diagnostics"][5]["stable"] == s_stable["Bin Diagnostics"][10]["stable"]

    print("graphutils self-check passed.")
