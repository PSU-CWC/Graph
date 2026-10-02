"""
PID gain finder for the WEC load controller (WEC-2025-2026/main).

Extracted from PIDmaybe.ipynb and reworked to match what the firmware actually runs.

Workflow
    1. identify_fopdt(): fit a first-order-plus-dead-time plant (DAC voltage -> y)
       from MEASURED DAC output and MEASURED y. Best data: an open-loop DAC step
       (constant-current mode drives the DAC directly: OutputVoltage = set_current / 1.6).
    2. rule_based_gains(): IMC / Cohen-Coon PI gains, mapped into firmware form.
    3. tune_gains(): optimize gains on an exact replica of PID_Compute() + main.c.

Why the firmware form matters (PID.c / main.c):
    OutputVoltage += Kp*e + clamp(Ki*sum(e), +-0.1) + Kd*(e - e_prev)
    - It is INCREMENTAL: the output is added to the DAC voltage every loop.
      So "Kp" acts as integral action and "Kd" acts as proportional action.
      Textbook PI (Kc, Ti) maps to:  Kd_fw = Kc,  Kp_fw = Kc * Ts / Ti,  Ki_fw = 0.
    - Gains are per SAMPLE (no dt), so they depend on the loop period Ts.
    - OutputVoltage is clamped to [0, 1] V.
    - Resistance mode passes res_Kp * -0.001 as Kp; Ki and Kd come from PID_Init unscaled.

The notebook's approach fitted the closed-loop response of the OLD controller and
simulated a positional PID in seconds, so its gains could not be pasted into main.c.

Usage
    python pid_tuning.py data.csv --start 30 --end 45 --y "Load Resistance" --setpoint 500
    python pid_tuning.py runs/*.csv --y "Load Power" --setpoint 20   # compare files, one shared gain set
    python pid_tuning.py            # runs the self-check
"""

import argparse

import numpy as np
from scipy.optimize import minimize

I_CLAMP = 0.1          # `threshold` in PID.c
U_LIMITS = (0.0, 1.0)  # clamp(OutputVoltage, 0, 1) in main.c
RES_KP_SCALE = -0.001  # main.c: PID_Compute(..., res_Kp*-0.001, res_Ki, res_Kd)
# The same OutputVoltage was logged under different names over time:
# "DAC Output" (to Apr 2026), "Output Voltage" (Apr-Sep 2026), "DAC Output Voltage" (Sep 2026-).
DAC_COLUMNS = ('DAC Output Voltage', 'Output Voltage', 'DAC Output')
# ponytail: guess used when a log cannot reveal the loop period (open-loop step test).
# Measured 49 ms with 16 telemetry fields at 115200 baud; fewer fields -> faster loop.
DEFAULT_TS = 0.04
# Measurement latency the step fit cannot see: the ADC snapshot is already this old when
# published. Calibrated from Power_Mode.csv (Kp=0.01 closed loop, old idle-priority ADC
# task): 0.15 s reproduces the measured overshoot and ~5 s settling. Re-measure after
# firmware changes to the ADC task.
EXTRA_DELAY = 0.15


# ---------------------------------------------------------
# Plant identification
# ---------------------------------------------------------

def _fopdt_response(t, u, K, tau, theta):
    """Deviation response dy to input deviation u, exact ZOH on an uneven time grid."""
    # Input seen by the plant is u(t - theta), held from the previous sample.
    idx = np.searchsorted(t, t - theta, side='right') - 1
    u_del = np.where(idx >= 0, u[np.clip(idx, 0, None)], 0.0)
    y = np.zeros(len(t))
    for i in range(1, len(t)):
        a = np.exp(-max(t[i] - t[i - 1], 1e-9) / tau)
        y[i] = a * y[i - 1] + K * u_del[i - 1] * (1 - a)
    return y


def identify_fopdt(t, u, y, max_theta=None, n_theta=21, max_samples=2000):
    """
    Fit dy/dt = (K * u(t - theta) - y) / tau around the starting operating point.

    Args:
        t: time in seconds. u: DAC output voltage. y: measured output (resistance, power...).
    Returns:
        dict with K, tau, theta (s), r2, and the operating point y0, u0.
    """
    t, u, y = (np.asarray(a, float) for a in (t, u, y))
    if len(t) > max_samples:
        # ponytail: uniform decimation keeps the fit fast; may blur very short dead times.
        idx = np.linspace(0, len(t) - 1, max_samples).astype(int)
        t, u, y = t[idx], u[idx], y[idx]
    t = t - t[0]
    du = np.asarray(u, float) - u[0]
    dy = np.asarray(y, float) - y[0]
    if np.ptp(du) < 1e-9:
        raise ValueError("DAC output never moves in this window; nothing to identify from.")

    K0 = dy[-1] / du[-1] if abs(du[-1]) > 1e-9 else dy.std() / du.std()
    tau0 = max(t[-1] / 5, 1e-3)
    max_theta = t[-1] / 3 if max_theta is None else max_theta

    # Grid over dead time (non-smooth), simplex search over K and log(tau).
    # Coarse grid first, then a fine grid around the coarse winner.
    best = None
    step = max_theta / (n_theta - 1)
    for grid in (np.linspace(0, max_theta, n_theta), None):
        if grid is None:
            grid = np.linspace(max(best[3] - step, 0), best[3] + step, n_theta)
        for theta in grid:
            res = minimize(
                lambda p: np.mean((dy - _fopdt_response(t, du, p[0], np.exp(p[1]), theta)) ** 2),
                x0=[K0, np.log(tau0)], method='Nelder-Mead',
            )
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x[0], np.exp(res.x[1]), theta)

    mse, K, tau, theta = best
    return {'K': K, 'tau': tau, 'theta': theta, 'y0': y[0], 'u0': u[0],
            'r2': 1 - mse / dy.var() if dy.var() > 0 else float('nan')}


# ---------------------------------------------------------
# Firmware replica
# ---------------------------------------------------------

def simulate_firmware(gains, plant, Ts, n, y0, u0, setpoint):
    """
    Closed loop of the identified plant with the exact PID_Compute() + main.c update.

    gains are the EFFECTIVE values PID_Compute() receives (see firmware_lines()).
    plant['hold'] (s, optional): how often the ADC snapshot refreshes. The PID keeps
    re-applying its correction to the same stale reading until it does.
    Returns (y, u) arrays, one entry per control loop iteration.
    """
    Kp, Ki, Kd = gains
    a = np.exp(-Ts / plant['tau'])
    d = int(round(plant['theta'] / Ts))
    m = max(1, int(round(plant.get('hold', 0.0) / Ts)))
    y = np.full(n, float(y0))
    u = np.full(n, float(u0))
    dy, integral, last_e = 0.0, 0.0, 0.0  # PID_Init sets last_error = 0
    y_meas = float(y0)
    for k in range(1, n):
        dy = a * dy + plant['K'] * (u[max(k - 1 - d, 0)] - u0) * (1 - a)
        y[k] = y0 + dy
        if k % m == 0:
            y_meas = y[k]
        e = setpoint - y_meas
        integral += e
        i_out = min(max(Ki * integral, -I_CLAMP), I_CLAMP)
        out = Kp * e + i_out + Kd * (e - last_e)
        last_e = e
        u[k] = min(max(u[k - 1] + out, U_LIMITS[0]), U_LIMITS[1])
    return y, u


# ---------------------------------------------------------
# Gain selection
# ---------------------------------------------------------

def rule_based_gains(plant, Ts):
    """
    Classic PI tuning rules, converted to firmware (incremental, per-sample) gains.
    D is dropped: the firmware has no second-difference term to host it.
    """
    K, tau = plant['K'], plant['tau']
    theta = max(plant['theta'], Ts)  # rules break down at zero dead time
    r = theta / tau
    rules = {
        # lambda: closed-loop time constant. Larger = smoother, smaller = faster.
        'IMC': (tau / (K * (theta + max(0.5 * tau, 2.0 * theta))), tau),
        'IMC_fast': (tau / (K * (theta + max(0.1 * tau, 0.8 * theta))), tau),
        'Cohen-Coon': ((1 / (K * r)) * (0.9 + r / 12), theta * (30 + 3 * r) / (9 + 20 * r)),
    }
    return {name: (Kc * Ts / Ti, 0.0, Kc) for name, (Kc, Ti) in rules.items()}


def step_cost(gains, plant, Ts, y0, u0, setpoint, horizon, effort_weight=0.0):
    """Normalized ITAE of a setpoint step, plus optional DAC-chatter penalty."""
    n = max(int(horizon / Ts), 10)
    y, u = simulate_firmware(gains, plant, Ts, n, y0, u0, setpoint)
    if not np.all(np.isfinite(y)):
        return 1e12
    t = np.arange(n) * Ts
    cost = np.mean(t * np.abs(setpoint - y)) / (abs(setpoint - y0) * horizon + 1e-12)
    return cost + effort_weight * np.mean(np.diff(u) ** 2)


def tune_gains(plants, Ts, setpoint, horizon=None, use_ki=False,
               k_spread=(0.7, 1.0, 1.3), effort_weight=0.0):
    """
    Optimize ONE firmware gain set on the replica loop for one or more plants.

    Robust: minimizes the WORST cost over every plant (e.g. one per CSV) and over
    K * k_spread, because the fitted K is rough and the real plant gain changes
    with the operating point. Each plant is started from its own y0/u0.
    Search is in log space; the sign of every gain follows sign(K) so the loop
    pushes the right way (resistance falls as DAC voltage rises, so K < 0 there).
    """
    if isinstance(plants, dict):
        plants = [plants]
    signs = {np.sign(p['K']) for p in plants}
    if len(signs) != 1:
        raise ValueError("Plants disagree on the sign of K; check the data before tuning.")
    sign = signs.pop()
    horizon = horizon or 10 * max(p['tau'] + p['theta'] for p in plants)
    variants = [dict(p, K=p['K'] * s) for p in plants for s in k_spread]

    def unpack(p):
        return (sign * 10 ** p[0], sign * 10 ** p[2] if use_ki else 0.0, sign * 10 ** p[1])

    def cost(p):
        return max(step_cost(unpack(p), v, Ts, v['y0'], v['u0'], setpoint, horizon, effort_weight)
                   for v in variants)

    # Start from the best rule-based candidate over all plants.
    seeds = [g for p in plants for g in rule_based_gains(p, Ts).values()]
    seed = min(seeds, key=lambda g: cost(np.log10(np.abs([g[0], g[2], 1e-6]) + 1e-12)))
    x0 = np.log10(np.abs([seed[0], seed[2], 1e-6]) + 1e-12)
    if not use_ki:
        x0 = x0[:2]
    res = minimize(cost, x0, method='Nelder-Mead', options={'maxiter': 400, 'xatol': 1e-3})
    return unpack(res.x), res.fun


def firmware_lines(gains, mode):
    """C lines for main.c. Undoes the -0.001 that main.c applies to res_Kp."""
    Kp, Ki, Kd = gains
    if mode == 'resistance':
        return (f"float res_Kp = {Kp / RES_KP_SCALE:.6g}f;  // PID_Compute sees {Kp:.6g}\n"
                f"float res_Ki = {Ki:.6g}f;\nfloat res_Kd = {Kd:.6g}f;")
    return f"float pow_Kp = {Kp:.6g}f;\nfloat pow_Ki = {Ki:.6g}f;\nfloat pow_Kd = {Kd:.6g}f;"


# ---------------------------------------------------------
# CLI / self-check
# ---------------------------------------------------------

def load_run(path, start=0, end=None, y_col='Load Resistance', u_col=None):
    """
    Read one Houston CSV window [start, end) in seconds from the first row.
    Returns (t in s, u, y, loop period in s or None, measurement refresh period in s).
    """
    import pandas as pd
    df = pd.read_csv(path)
    sec = (df['Time'] - df['Time'].iloc[0]) / 1000.0  # Houston logs ms
    df = df[(sec >= start) & (sec < (np.inf if end is None else end))]
    if u_col is None:
        u_col = next((c for c in DAC_COLUMNS if c in df.columns), None)
        if u_col is None:
            raise KeyError(f"{path}: no DAC column (looked for {DAC_COLUMNS}); pass --u")
    # Resistance is logged as a huge value when current ~ 0; drop inf/NaN rows.
    df = df[['Time', u_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    # Houston writes one row per GUI frame, not per firmware loop, so most rows are
    # repeats. Drop rows where nothing changed.
    df = df[(df[[u_col, y_col]].diff().fillna(1) != 0).any(axis=1)]
    t = (df['Time'].values - df['Time'].values[0]) / 1000.0  # Houston logs ms
    # Loop period = typical gap between DAC updates (the PID moves it every loop).
    # In an open-loop step test the DAC barely moves, so it cannot be measured.
    gaps = np.diff(t[np.diff(df[u_col].values, prepend=np.nan) != 0])
    ts_est = float(np.median(gaps)) if len(gaps) > 50 else None
    # Measurement refresh = typical gap between changes of y.
    hold = float(np.median(np.diff(t[np.diff(df[y_col].values, prepend=np.nan) != 0])))
    return t, df[u_col].values, df[y_col].values, ts_est, hold


def plot_run(name, t, u, y, plant, gains, Ts, setpoint, y_label):
    import matplotlib.pyplot as plt
    n = int(t[-1] / Ts) + 1
    y_sim, u_sim = simulate_firmware(gains, plant, Ts, n, y[0], u[0], setpoint)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    fig.suptitle(name)
    ax1.plot(t, y, 'b.', ms=2, alpha=0.4, label='Measured')
    ax1.plot(t, y[0] + _fopdt_response(t, u - u[0], plant['K'], plant['tau'], plant['theta']),
             'g-', label='Fitted model (measured DAC)')
    ax1.plot(np.arange(n) * Ts, y_sim, 'r-', label='Simulated with tuned gains')
    ax1.axhline(setpoint, color='k', ls='--', label='Setpoint')
    ax1.set_ylabel(y_label)
    ax1.legend()
    ax2.plot(t, u, 'b-', alpha=0.4, label='Measured DAC')
    ax2.plot(np.arange(n) * Ts, u_sim, 'r-', label='Simulated DAC')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('DAC Output (V)')
    ax2.legend()
    plt.tight_layout()


def run(paths, y_col, setpoint, start=0, end=None, u_col=None, ts=None, min_r2=0.8,
        use_ki=False, plot=True, extra_delay=EXTRA_DELAY):
    """
    Identify a plant per CSV, print a comparison table, and tune ONE shared gain set.
    Returns the effective (Kp, Ki, Kd) that PID_Compute() receives.
    """
    if isinstance(paths, str):
        paths = [paths]
    mode = 'resistance' if 'Resist' in y_col else 'power'

    runs = []
    for path in paths:
        try:
            t, u, y, ts_est, hold = load_run(path, start, end, y_col, u_col)
            Ts = ts or ts_est
            if Ts is None:
                Ts = DEFAULT_TS
                print(f"{path}: DAC barely moves, loop period unknown -> assuming "
                      f"{DEFAULT_TS * 1000:.0f} ms. Pass ts / --ts from a closed-loop log.")
            pl = identify_fopdt(t, u, y)
            pl['hold'] = hold
            pl['theta'] += extra_delay
            runs.append((path, t, u, y, Ts, pl))
        except (KeyError, ValueError) as e:
            print(f"skip {path}: {e}")
    if not runs:
        raise ValueError("No usable files.")

    # Per-file table: plant fit + gains tuned on that file alone.
    print(f"\n{'file':<32}{'rows':>7}{'Ts ms':>7}{'hold ms':>8}{'K':>11}{'tau s':>9}{'theta s':>8}{'R2':>7}"
          f"{'Kp':>12}{'Kd':>12}")
    for path, t, u, y, Ts, pl in runs:
        g, _ = tune_gains(pl, Ts, setpoint, use_ki=use_ki)
        flag = '' if pl['r2'] >= min_r2 else '  <- poor fit, excluded'
        print(f"{path[-31:]:<32}{len(t):>7}{Ts * 1000:>7.1f}{pl['hold'] * 1000:>8.0f}{pl['K']:>11.4g}{pl['tau']:>9.3g}"
              f"{pl['theta']:>8.3g}{pl['r2']:>7.3f}{g[0]:>12.4g}{g[2]:>12.4g}{flag}")

    good = [r for r in runs if r[5]['r2'] >= min_r2]
    if not good:
        raise ValueError("No fit passed min_r2. Use open-loop DAC steps (constant-current mode).")
    Ts = ts or float(np.median([r[4] for r in good]))
    gains, cost = tune_gains([r[5] for r in good], Ts, setpoint, use_ki=use_ki)
    print(f"\nShared gains, robust over {len(good)} file(s) x K*0.7..1.3 (cost={cost:.4f}, Ts={Ts * 1000:.2f} ms):")
    print(firmware_lines(gains, mode))
    print('(Houston Live Data "Res Kp/Ki/Kd" or "Pow Kp/Ki/Kd" take the same values.)')

    if plot:
        import matplotlib.pyplot as plt
        for path, t, u, y, _, pl in good:
            plot_run(path, t, u, y, pl, gains, Ts, setpoint, y_col)
        plt.show()
    return gains


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+', help='one or more Houston CSV files')
    ap.add_argument('--start', type=float, default=0, help='window start in s (applies to every file)')
    ap.add_argument('--end', type=float, default=None, help='window end in s (applies to every file)')
    ap.add_argument('--y', default='Load Resistance', help="'Load Resistance' or 'Load Power'")
    ap.add_argument('--u', default=None, help='DAC column (default: auto-detect)')
    ap.add_argument('--setpoint', type=float, required=True)
    ap.add_argument('--ts', type=float, default=None,
                    help='control loop period in s (default: median interval between DAC updates)')
    ap.add_argument('--min-r2', type=float, default=0.8, help='skip fits worse than this when tuning')
    ap.add_argument('--use-ki', action='store_true', help='also tune Ki (double integral, usually off)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--extra-delay', type=float, default=EXTRA_DELAY,
                    help='measurement latency in s added to the fitted dead time')
    args = ap.parse_args()
    run(args.csv, args.y, args.setpoint, args.start, args.end, args.u, args.ts,
        args.min_r2, args.use_ki, not args.no_plot, args.extra_delay)


def _self_check():
    # Known resistance-like plant: more DAC voltage -> lower resistance.
    true = {'K': -600.0, 'tau': 0.4, 'theta': 0.05}
    Ts, u0, y0 = 0.01, 0.3, 500.0
    t = np.arange(300) * Ts
    u = np.where(t >= 0.5, 0.5, u0)
    y = y0 + _fopdt_response(t, u - u0, true['K'], true['tau'], true['theta'])

    est = identify_fopdt(t, u, y)
    assert abs(est['K'] / true['K'] - 1) < 0.05, est
    assert abs(est['tau'] / true['tau'] - 1) < 0.1, est
    assert abs(est['theta'] - true['theta']) < 0.02, est

    # Every rule-based candidate must at least be stable on the true plant.
    for g in rule_based_gains(est, Ts).values():
        y_cl, _ = simulate_firmware(g, true, Ts, 1000, y0, u0, 400.0)
        assert abs(y_cl[-1] - 400.0) < 2.0, (g, y_cl[-1])

    gains, _ = tune_gains(est, Ts, 400.0)
    assert gains[0] < 0 and gains[2] < 0, "gain sign must follow sign(K)"
    y_cl, u_cl = simulate_firmware(gains, true, Ts, 1000, y0, u0, 400.0)
    assert abs(y_cl[-1] - 400.0) < 1.0, y_cl[-1]
    assert u_cl.min() >= 0.0 and u_cl.max() <= 1.0
    assert 'res_Kp' in firmware_lines(gains, 'resistance')
    # Two runs with different plant gains -> one shared gain set that settles on both.
    other = dict(est, K=est['K'] * 1.8)
    shared, _ = tune_gains([est, other], Ts, 400.0)
    for pl in (true, dict(true, K=true['K'] * 1.8)):
        y_cl, _ = simulate_firmware(shared, pl, Ts, 1500, y0, u0, 400.0)
        assert abs(y_cl[-1] - 400.0) < 1.0, y_cl[-1]
    # A stale measurement (ADC refresh 0.25 s) must lead to gentler gains.
    slow, _ = tune_gains(dict(est, hold=0.25), Ts, 400.0)
    assert abs(slow[0]) < abs(gains[0]), (slow, gains)
    y_cl, _ = simulate_firmware(slow, dict(true, hold=0.25), Ts, 3000, y0, u0, 400.0)
    assert abs(y_cl[-1] - 400.0) < 1.0, y_cl[-1]
    print("pid_tuning self-check passed.")
    print(firmware_lines(gains, 'resistance'))


if __name__ == '__main__':
    import sys
    main() if len(sys.argv) > 1 else _self_check()
