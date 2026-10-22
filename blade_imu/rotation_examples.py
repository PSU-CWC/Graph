"""Worked examples of blade_torque.py: fake data with a known answer, then the real log."""
import numpy as np
from pathlib import Path
import pandas as pd
from blade_torque import load, spin, fit_radius, export, W_RANGE, G_RANGE

DATA = Path(__file__).with_name('data')

def fake_blade(r=0.40, rpm=30, hz=10, seed=0, elev_deg=0, pitch=None):
    """Simulate the WT901 on a blade: spin up 0->rpm, hold, coast down. elev_deg: axis above horizontal (0 = horizontal axis).
    pitch: (times, degrees) breakpoints of blade pitch, turning the sensor about the blade's span axis.
    Returns the true blade angle too: 0 = sensor at the bottom."""
    rng = np.random.default_rng(seed)
    fine = np.arange(0, 60, 0.001)                                       # "real world" at 1 kHz ...
    w_f = np.interp(fine, [0, 10, 40, 55], [0, rpm, rpm, 0]) * 2 * np.pi / 60   # rad/s
    t, w, theta = fine[::1000 // hz], w_f[::1000 // hz], (np.cumsum(w_f) * 0.001)[::1000 // hz]  # ... sensor sees 10 Hz
    beta = np.radians(np.interp(t, *pitch) if pitch else 0 * t)[:, None]
    alpha = np.gradient(w, t)
    u = np.array([0.9, 0.1, 0.4]); u /= np.linalg.norm(u)            # sensor mounted crooked
    e1 = np.cross(u, [0, 0, 1]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    # pitch turns the rotor frame about the span axis e1 as seen by the sensor; the sensor itself spins at -beta_dot about e1
    us, e2s = np.cos(beta) * u + np.sin(beta) * np.cross(e1, u), np.cos(beta) * e2 + np.sin(beta) * np.cross(e1, e2)
    gyro = np.degrees(w[:, None] * us - np.gradient(beta[:, 0], t)[:, None] * e1) + rng.normal(0, 0.1, (len(t), 3))
    kin = (np.outer(-w**2 * r, e1) + alpha[:, None] * r * e2s) / 9.81  # centripetal + tangential, in g
    el = np.radians(elev_deg)
    grav = np.sin(el) * us + np.cos(el) * (np.outer(np.cos(theta), e1) - np.sin(theta)[:, None] * e2s)  # 1 g "up" in sensor frame
    acc = kin + grav + rng.normal(0, 0.002, (len(t), 3))   # real log at rest: ~0.001 g
    gyro, acc = np.clip(gyro, -W_RANGE, W_RANGE), np.clip(acc, -G_RANGE, G_RANGE)   # sensor full scale
    return t, gyro, acc, u, (np.degrees(theta) + 180) % 360           # theta = 0 has the sensor (radius e1) pointing up

def run(name, t, gyro, acc):
    t, axis, theta, w, alpha = spin(t, gyro)
    rA, rB, coef, r2 = fit_radius(theta, w, alpha, acc)
    print(f'--- {name}')
    print(f'  axis u           = {axis.round(3)}')
    print(f'  max speed        = {np.abs(w).max() * 60 / (2 * np.pi):.1f} rpm, turned {np.degrees(np.ptp(theta)):.0f} deg')
    print(f'  r from w^2 term  = {rA:.3f} m   r from alpha term = {rB:.3f} m   fit R^2 = {r2:.2f}')
    print(f'  gravity fixed C  = {coef[2].round(3)} g   rotating |D|,|E| = {np.linalg.norm(coef[3]):.2f}, {np.linalg.norm(coef[4]):.2f} g')
    return axis, rA

if __name__ == '__main__':
    t, gyro, acc, u_true, _ = fake_blade()
    axis, r = run('Example 1: fake blade, true r = 0.400 m', t, gyro, acc)
    assert abs(abs(axis @ u_true) - 1) < 1e-3 and abs(r - 0.4) < 0.02, 'method failed on known data'
    run('Example 2: Limited Range Test.txt', *load(DATA / 'Limited Range Test.txt'))

    # Example 3: horizontal axis raised 3 deg, 900 rpm. Gyro saturates above ~333 rpm, so rpm comes from
    # centripetal accel there; that only fits the +-16 g accel range if the sensor sits within ~1.8 cm of the axis.
    t, gyro, acc, u_true, ang_true = fake_blade(r=0.015, rpm=900, elev_deg=3)
    rf = export(t, gyro, acc, 'fake_900rpm', {'r': 0.015})
    clip = rf['gyro_clip']
    w_acc = np.sqrt(rf['a_cent'][clip] * 9.81 / 0.015) * 60 / (2 * np.pi)
    err = lambda a, ref=None: np.abs((a - (ang_true if ref is None else ref) + 180) % 360 - 180)  # circular angle error, deg
    fused = pd.read_csv(Path(__file__).with_name('out') / 'fake_900rpm_rotor.csv').angle_deg.values
    print(f'--- Example 3: axis {rf["u0"].round(3)} (true {u_true.round(3)}), elevation {np.nanmedian(rf["elev"][~clip]):.2f} / '
          f'{np.nanmedian(rf["elev"][clip]):.2f} deg unclipped / clipped (true 3), fitted r {rf["r_fit"]:.4f} m (true 0.015)')
    near600 = abs(np.interp(t, [0, 10, 40, 55], [0, 900, 900, 0]) - 600) < 50   # rotation = sample rate: gravity aliases to DC
    print(f'  gyro clipped {clip.mean():.0%}, accel rpm while clipped {w_acc.min():.0f}..{w_acc.max():.0f}; '
          f'gravity angle error p95 {np.nanpercentile(err(rf["angle_grav"])[~near600], 95):.1f} deg away from 600 rpm '
          f'(near 600 rpm: max {np.nanmax(err(rf["angle_grav"])[near600]):.0f} deg); fused angle error p95 {np.percentile(err(fused), 95):.1f} deg')
    assert abs(rf['u0'] @ u_true) > 0.9999 and abs(np.median(w_acc[w_acc > 880]) - 900) < 10
    assert abs(np.nanmedian(rf['elev']) - 3) < 0.3 and abs(rf['r_fit'] - 0.015) < 0.002
    assert np.nanpercentile(err(rf['angle_grav'])[~near600], 95) < 10 and np.percentile(err(fused), 95) < 10

    # Example 4: 200 rpm, blade pitched 0 -> +20 deg at 20 s -> -10 deg at 35 s. Pitch must come out on its own,
    # and speed / elevation / blade angle must not be thrown off by the pitch change.
    pitch_true = ([0, 20, 21, 35, 36, 60], [0, 0, 20, 20, -10, -10])
    t, gyro, acc, u_true, ang_true = fake_blade(r=0.1, rpm=200, elev_deg=3, pitch=pitch_true)
    rf = export(t, gyro, acc, 'fake_pitch', {'r': 0.1})
    spinning = np.interp(t, [0, 10, 40, 55], [0, 200, 200, 0]) > 60
    p_err = abs(rf['pitch'] - np.interp(t, *pitch_true))[spinning]
    rpm_err = abs(rf['w_axis'] / 6 - np.interp(t, [0, 10, 40, 55], [0, 200, 200, 0]))[spinning]
    print(f'--- Example 4: pitch error p50 {np.nanpercentile(p_err, 50):.2f}, p95 {np.nanpercentile(p_err, 95):.2f} deg '
          f'(steps of 20 and 30 deg); rpm error p95 {np.percentile(rpm_err, 95):.2f}; elevation {np.nanmedian(rf["elev"]):.2f} (true 3); '
          f'blade angle error p95 {np.nanpercentile(err(rf["angle_grav"], ang_true)[spinning], 95):.2f} deg  -> out/fake_pitch_view.html, .csv')
    assert np.nanpercentile(p_err, 95) < 2 and np.percentile(rpm_err, 95) < 1 and abs(np.nanmedian(rf['elev']) - 3) < 0.3
