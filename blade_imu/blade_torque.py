"""WT901BLE IMU log -> rotor-frame motion (axis, spin, wobble, elevation, blade angle, torque) as an HTML viewer and a CSV.
Usage: python3 blade_torque.py data/<log>.txt [--r 0.1 --R 0.1 --n 3 --m 0.1 --I0 0 --offset 0 --smooth 0.5 --flip 0]
         -> out/<log>_view.html, out/<log>_rotor.csv
       python3 blade_torque.py
         -> out/blade_viewer.html: empty viewer, drop any log .txt on it (runs this file in the browser via Pyodide)"""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd

W_RANGE, G_RANGE = 2000, 16  # WT901 full scale (deg/s, g); readings at the rail are clipped
# r: sensor distance from axis (m), R: rotor radius (m), n: blades, m: mass per blade (kg, placeholder),
# I0: extra inertia (kg m^2), offset: angle reference (deg), smooth: alpha smoothing window (s),
# flip: 1 = view from the other end of the axis (u -> -u), which reverses the positive rotation direction,
# hub_m, hub_r: hub + generator rotor as a solid disk (kg, m), pitch0: blade pitch at the start of the record (deg),
# V: wind speed (m/s) for tip speed ratio, 0 = unknown
DEFAULTS = dict(r=0.1, R=0.1, n=3, m=0.1, hub_m=0.0, hub_r=0.02, I0=0.0, offset=0.0, pitch0=0.0, V=0.0, smooth=0.5, flip=0)

ANGLES = ['AngleX(°)', 'AngleY(°)', 'AngleZ(°)']   # the sensor's own Euler angles, shown for comparison only

def load(path, angles=False):
    d = pd.read_csv(path, sep='\t', parse_dates=['time']).drop_duplicates('time').reset_index(drop=True)  # BLE logger repeats timestamps
    t = (d.time - d.time[0]).dt.total_seconds().values
    out = t, d[['AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)']].values, d[['AccX(g)', 'AccY(g)', 'AccZ(g)']].values
    return out + (d[ANGLES].values if set(ANGLES) <= set(d.columns) else None,) if angles else out

def analyze(path, smooth_s=0.5):
    t, g, _ = load(path)
    return spin(t, g, smooth_s)

def spin(t, g, smooth_s=0.5):
    """gyro (N x 3, deg/s) -> axis, angle theta, speed w, accel alpha."""
    axis = np.linalg.svd(g, full_matrices=False)[2][0]   # dominant rotation axis in sensor frame
    w = np.radians(g @ axis)                              # 3D gyro -> 1D spin rate, rad/s
    theta = np.concatenate([[0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(t))])
    n = max(1, int(smooth_s / np.median(np.diff(t))))
    alpha = np.gradient(pd.Series(w).rolling(n, center=True, min_periods=1).mean(), t)  # derivative amplifies noise, so smooth first
    return t, axis, theta, w, alpha

def fit_radius(theta, w, alpha, acc):
    """acc (g) = w^2*A + alpha*B + C + cos(theta)*D + sin(theta)*E, solved by least squares.
    A: centripetal (|A| = r), B: tangential (|B| = r), C+D/E: gravity (fixed part / part rotating with blade)."""
    X = np.c_[w**2, alpha, np.ones_like(w), np.cos(theta), np.sin(theta)]
    coef = np.linalg.lstsq(X, acc * 9.81, rcond=None)[0]
    pred = X @ coef
    r2 = 1 - ((acc * 9.81 - pred)**2).sum() / ((acc * 9.81 - acc.mean(0) * 9.81)**2).sum()
    return np.linalg.norm(coef[0]), np.linalg.norm(coef[1]), coef / 9.81, r2

def _accel_split(t, g, acc, U, gyro_clip, acc_clip, sm, e_fallback):
    """Split accel into gravity / centripetal / tangential about the per-sample rotation axis U (N x 3)."""
    dot = lambda a, b: (a * b).sum(1)
    spin, a_axis = dot(g, U), dot(acc, U)
    acc_perp = acc - a_axis[:, None] * U
    a_perp = np.linalg.norm(acc_perp, axis=1)
    # accel = gravity + centripetal (scales with w^2) + tangential (scales with alpha). Fit the last two as vectors on
    # unclipped samples and subtract; that also removes their small leak into u (u error, accel/gyro axis misalignment).
    w2 = np.radians(spin) ** 2
    fit = ~gyro_clip & ~acc_clip
    k_axis, k_perp = np.polyfit(w2[fit], a_axis[fit], 1)[0], np.polyfit(w2[fit], a_perp[fit], 1)[0]
    w2 = np.where(gyro_clip, a_perp / max(k_perp, 1e-9), w2)          # gyro clipped: speed from centripetal size
    w = np.sign(spin) * np.sqrt(w2)
    alpha = np.gradient(sm(w), t)
    th = np.r_[0, np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(t))]
    # cos/sin terms soak up gravity rotating in the plane (horizontal axis) so it doesn't bias the centripetal fit
    X = np.c_[w2, alpha, np.cos(th), np.sin(th)]
    AB = np.linalg.lstsq(X[fit], acc_perp[fit], rcond=None)[0][:2]
    r_fit = np.linalg.norm(AB[0]) * 9.81                              # |centripetal per w^2| = sensor radius (m)
    known_r = fit.any() and w2[fit].max() > np.pi ** 2 and r_fit < 5   # needs >= 30 rpm to see the radius direction
    e_r = -AB[0] / np.linalg.norm(AB[0]) if known_r else e_fallback     # sensor's outward radius (else 0 deg = e1 at the bottom)
    hi = gyro_clip & ~acc_clip
    if known_r and hi.sum() > 10:
        # at high speed an e_r error of 0.5 deg leaks 13 g * 0.009 = 0.1 g into gravity (~7 deg of blade angle); the mean
        # in-plane accel there is pure centripetal (gravity averages out), so take the direction from it
        e_r = -acc_perp[hi].mean(0) / np.linalg.norm(acc_perp[hi].mean(0))
        AB[0] = -np.linalg.norm(AB[0]) * e_r
    # centripetal size from accel: its direction is fixed in the sensor frame while gravity spins, so averaging the
    # radial component roughly cancels gravity. shortcut: fails if rpm is a multiple of 60 * sample rate (gravity aliases to DC)
    a_cent = sm(acc_perp @ -e_r) if known_r else sm(a_perp)
    if known_r and hi.any():
        # per sample: in-plane accel = c * (-e_r) + gravity, and |gravity| is known, so c has two roots;
        # take the one nearer the averaged estimate (it says which side of the circle gravity is on)
        rest = np.linalg.norm(g, axis=1) < 1
        g_mag = np.median(np.linalg.norm(acc[rest], axis=1)) if rest.sum() > 10 else 1.0   # accel scale error
        x, y = acc_perp @ -e_r, dot(acc_perp, np.cross(U, e_r))
        root = np.sqrt(np.clip(g_mag ** 2 - a_axis ** 2 - y ** 2, 0, None))
        for _ in range(5):                                            # re-centre on the picked roots: true c is smooth
            c = np.where(abs(x - root - a_cent) < abs(x + root - a_cent), x - root, x + root)
            a_cent = np.where(hi, sm(np.where(hi, c, a_cent)), a_cent)
        a_cent = np.where(hi, c, a_cent)
    if known_r:                                                       # gyro clipped: speed from centripetal, then redo alpha
        w2 = np.where(gyro_clip, np.maximum(a_cent, 0) / np.linalg.norm(AB[0]), w2)
        alpha = np.gradient(sm(np.sign(spin) * np.sqrt(w2)), t)
    up = acc_perp - np.c_[w2, alpha] @ AB                             # in-plane gravity (accel reads +1 g upward)
    g_perp, g_axis = np.linalg.norm(up, axis=1), a_axis - k_axis * w2
    elev = np.degrees(np.arctan2(sm(g_axis), sm(g_perp)))             # accel scale error cancels in the ratio
    elev[acc_clip] = np.nan
    # blade angle from gravity: angle from "down" to the sensor's outward radius, measured about u
    down = -up
    angle_grav = np.degrees(np.arctan2(dot(np.cross(down, e_r), U), down @ e_r)) % 360
    angle_grav[acc_clip | (g_perp < 0.3) | (gyro_clip & ~known_r)] = np.nan   # g_perp small: axis near vertical
    grav = up + g_axis[:, None] * U                                   # gravity as the accel sees it (+1 g up), sensor x, y, z
    grav[acc_clip] = np.nan
    return dict(spin=spin, a_axis=a_axis, a_perp=a_perp, a_cent=a_cent, elev=elev, angle_grav=angle_grav, grav=grav,
                e_r=e_r, known_r=known_r, r_fit=r_fit if known_r else np.nan)

def rotor_frame(t, g, acc, win_s=2.0, smooth_s=0.5):
    """Sensor-frame gyro (deg/s) + accel (g) -> rotor frame.
    u: rotation axis per sample, corrected for blade pitch; pitch: blade pitch change since the start (deg), from the
    axis turning about the blade's span (radial) axis e1 in the sensor frame; w_axis: spin about u;
    w_e1/w_e2: rotation across the axis (wobble) about e1 (radial) and e2 = u x e1 (chord-wise), pitch removed;
    elev: axis elevation above horizontal (deg); drift: windowed axis vs the pitch-corrected axis (deg);
    angle_grav: absolute blade angle from gravity (deg, 0 = sensor at the bottom)."""
    N = len(t)
    gyro_clip = (abs(g) >= W_RANGE * 0.995).any(1)
    acc_clip = (abs(acc) >= G_RANGE * 0.995).any(1)
    ok = ~gyro_clip if (~gyro_clip).sum() > 10 else np.ones(N, bool)  # a clipped axis bends the measured direction
    u0 = np.linalg.svd(g[ok], full_matrices=False)[2][0]
    u0 = u0 if (g[ok] @ u0).sum() >= 0 else -u0                       # SVD sign is arbitrary; make the mean spin positive
    e1 = np.cross(u0, np.eye(3)[np.argmin(abs(u0))]); e1 /= np.linalg.norm(e1)
    k = max(1, int(smooth_s / np.median(np.diff(t))))
    sm = lambda x: pd.Series(x).rolling(k, center=True, min_periods=1).mean().values
    u_w = np.full((N, 3), np.nan)                                     # rotation direction over a short window
    for i in range(N):
        win = (abs(t - t[i]) <= win_s / 2) & ~gyro_clip
        if win.sum() > 3 and np.abs(g[win] @ u0).mean() > 5:         # need real spin in the window, else SVD fits noise
            v = np.linalg.svd(g[win], full_matrices=False)[2][0]
            u_w[i] = v if v @ u0 >= 0 else -v
    S = _accel_split(t, g, acc, np.tile(u0, (N, 1)), gyro_clip, acc_clip, sm, e1)
    U, pitch, dbeta = np.tile(u0, (N, 1)), np.full(N, np.nan), np.zeros(N)
    if S['known_r'] and np.isfinite(u_w[:, 0]).any():
        # Pitch turns the sensor about the blade's span axis e_r, so in the sensor frame the rotation axis turns about
        # e_r by the pitch angle. Gyro gives the rate (fast, drifts); the windowed axis gives the angle (slow, absolute).
        e_r = S['e_r'] - (S['e_r'] @ u0) * u0
        e_r /= np.linalg.norm(e_r)                                    # exactly perpendicular, or spin leaks into the pitch rate
        flat = lambda v: v - np.outer(v @ e_r, e_r)
        u_ref = flat(u_w[np.isfinite(u_w[:, 0])][:1])[0]
        u_ref /= np.linalg.norm(u_ref)                                # pitch 0 = axis at the start of the record
        uwp = flat(u_w)
        beta_u = np.degrees(np.arctan2(np.cross(u_ref, uwp) @ e_r, uwp @ u_ref))
        rate = np.where(gyro_clip, 0, -sm(g @ e_r))                   # sensor turns +beta -> gyro reads -beta_dot on e_r
        beta_g = np.r_[0, np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(t))]
        corr = pd.Series(beta_u - beta_g).rolling(2 * max(1, round(1 / np.median(np.diff(t)))) + 1, center=True,
                                                  min_periods=1).mean().ffill().bfill().fillna(0).values
        beta = beta_g + corr
        b = np.radians(beta)[:, None]
        U = np.cos(b) * u_ref + np.sin(b) * np.cross(e_r, u_ref)
        pitch = np.where(gyro_clip, np.nan, beta)                     # clipped gyro: pitch unseen (use the actuator log)
        dbeta = np.gradient(beta, t)
        e1 = e_r
        S = _accel_split(t, g, acc, U, gyro_clip, acc_clip, sm, e1)   # redo with the pitch-corrected axis
    E2 = np.cross(U, e1)
    drift = np.degrees(np.arccos(np.clip(abs((u_w * U).sum(1)), 0, 1)))  # NaN where no window axis
    wob = np.c_[g @ e1 + dbeta, (g * E2).sum(1)]                      # pitch rate removed from the radial part
    wob[gyro_clip] = np.nan                                           # clipped vector's sideways part is meaningless
    return dict(u0=u0, u=U, e1=e1, e2=E2, w_axis=S['spin'], w_e1=wob[:, 0], w_e2=wob[:, 1], a_axis=S['a_axis'],
                a_perp=S['a_perp'], a_cent=S['a_cent'], elev=S['elev'], drift=drift, angle_grav=S['angle_grav'],
                pitch=pitch, r_fit=S['r_fit'], grav=S['grav'],
                angle_ref='sensor at bottom' if S['known_r'] else 'e1 at bottom (radius direction unknown)',
                gyro_clip=gyro_clip, acc_clip=acc_clip)

def derive(t, rf, c):
    """Values that depend on the constants. Mirrors derive() in blade_view_template.html -- keep the two in sync."""
    w_gyro = np.radians(rf['w_axis'])
    w_acc = np.sqrt(np.maximum(rf['a_cent'] * 9.81 / c['r'], 0))      # centripetal = w^2 r  ->  w = sqrt(a / r)
    good = ~rf['gyro_clip'] & (abs(w_gyro) > 0.5)
    sign = pd.Series(np.where(good, np.sign(w_gyro), np.nan)).ffill().fillna(1).values  # direction from last good gyro sample
    w = np.where(rf['gyro_clip'], sign * w_acc, w_gyro)
    theta = np.degrees(np.r_[0, np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(t))])
    # gyro angle drifts; pull it onto the gravity angle (circular mean of the difference over a 2 s window)
    d = np.radians(rf['angle_grav'] - theta)
    if np.isfinite(d).any():
        k = 2 * max(1, round(1 / np.median(np.diff(t)))) + 1
        cs = [pd.Series(f(d)).rolling(k, center=True, min_periods=1).mean().ffill().bfill().values for f in (np.cos, np.sin)]
        theta = theta + np.degrees(np.arctan2(cs[1], cs[0]))
    theta = theta + c['offset']
    h = max(1, round(c['smooth'] / np.median(np.diff(t)))) // 2
    ws = pd.Series(w).rolling(2 * h + 1, center=True, min_periods=1).mean().values
    a, b = np.r_[0, np.arange(len(t) - 1)], np.r_[np.arange(1, len(t)), len(t) - 1]
    alpha = (ws[b] - ws[a]) / (t[b] - t[a])
    I = c['n'] * c['m'] * c['R'] ** 2 / 3 + c['hub_m'] * c['hub_r'] ** 2 / 2 + c['I0']  # blades: uniform rods to R; hub: disk
    source = np.where(~rf['gyro_clip'], 'gyro', np.where(~rf['acc_clip'], 'accel_estimate', 'unmeasurable'))
    return dict(omega_rad_s=w, omega_accel_rad_s=w_acc, speed_source=source, angle_deg=theta, alpha_rad_s2=alpha,
                inertia_kg_m2=I, torque_Nm=I * alpha, pitch_deg=rf['pitch'] + c['pitch0'],
                tsr=abs(w) * c['R'] / c['V'] if c['V'] > 0 else np.full(len(t), np.nan))   # tip speed ratio = wR / V

def to_csv(t, g, acc, rf, c, out):
    if c['flip']:   # u -> -u: e1 stays, e2 = u x e1 flips, and so does every signed spin value
        rf = {**rf, 'u': -rf['u'], 'e2': -rf['e2'], 'w_axis': -rf['w_axis'], 'w_e2': -rf['w_e2'],
              'elev': -rf['elev'], 'angle_grav': -rf['angle_grav'] % 360}
    dv = derive(t, rf, c)
    w = dv['omega_rad_s']
    df = pd.DataFrame({
        'time_s': t,
        'gyro_x_dps': g[:, 0], 'gyro_y_dps': g[:, 1], 'gyro_z_dps': g[:, 2],
        'acc_x_g': acc[:, 0], 'acc_y_g': acc[:, 1], 'acc_z_g': acc[:, 2],
        **{f'gravity_{ax}_g': rf['grav'][:, j] for j, ax in enumerate('xyz')},          # gravity part of the raw accel
        **{f'motion_{ax}_g': acc[:, j] - rf['grav'][:, j] for j, ax in enumerate('xyz')},  # raw minus gravity
        **{f'{k}_{ax}': rf[k][..., j] for k in ['u', 'e1', 'e2'] for j, ax in enumerate('xyz')},  # rotor frame in sensor coords
        'spin_dps': rf['w_axis'], 'wobble_e1_dps': rf['w_e1'], 'wobble_e2_dps': rf['w_e2'],
        'wobble_dps': np.hypot(rf['w_e1'], rf['w_e2']),
        'acc_axial_g': rf['a_axis'], 'acc_inplane_g': rf['a_perp'], 'centripetal_from_acc_g': rf['a_cent'],
        'axis_elevation_deg': rf['elev'], 'axis_drift_deg': rf['drift'], 'angle_gravity_deg': rf['angle_grav'],
        'pitch_deg': dv['pitch_deg'], 'tip_speed_ratio': dv['tsr'],
        'gyro_clipped': rf['gyro_clip'].astype(int), 'acc_clipped': rf['acc_clip'].astype(int),
        'speed_source': dv['speed_source'], 'omega_rad_s': w, 'rpm': w * 60 / (2 * np.pi),
        'omega_accel_rad_s': dv['omega_accel_rad_s'],
        'angle_deg': dv['angle_deg'], 'angle_wrapped_deg': dv['angle_deg'] % 360,
        'alpha_rad_s2': dv['alpha_rad_s2'], 'torque_Nm': dv['torque_Nm'],
        'sensor_speed_m_s': w * c['r'], 'centripetal_g': w ** 2 * c['r'] / 9.81, 'tangential_g': dv['alpha_rad_s2'] * c['r'] / 9.81,
        **{f'const_{k}': v for k, v in c.items()}, 'const_inertia_kg_m2': dv['inertia_kg_m2'],
    })
    return df.to_csv(out, index=False, float_format='%.6g')   # out=None -> CSV text

def view_data(t, g, acc, rf, name, consts=None, angles=None):
    """What the viewer needs; constant-dependent values are recomputed in the page (derive()).
    Raw gyro/accel/angles and the per-sample axes go along so the page can show the measured data and the transform."""
    r3 = lambda x: [None if np.isnan(v) else round(float(v), 4) for v in x]
    cols = lambda a: [r3(a[:, j]) for j in range(3)]                 # N x 3 -> three lists (x, y, z)
    return dict(name=name, consts=consts or {}, u=rf['u0'].round(4).tolist(), t=r3(t), angle_ref=rf['angle_ref'],
                gyro=cols(g), acc=cols(acc), grav=cols(rf['grav']), dev_angles=cols(angles) if angles is not None else None,
                axis_u=cols(rf['u']), e1=rf['e1'].round(5).tolist(),
                **{k: r3(rf[k]) for k in ['w_axis', 'w_e1', 'w_e2', 'a_axis', 'a_cent', 'elev', 'drift', 'angle_grav', 'pitch']},
                gyro_clip=rf['gyro_clip'].astype(int).tolist(), acc_clip=rf['acc_clip'].astype(int).tolist())

def write_view(out, data=None):
    """data=None writes the empty viewer. The page embeds this file's source so it can analyze dropped logs itself."""
    html = Path(__file__).with_name('blade_view_template.html').read_text()
    html = html.replace('__DATA__', json.dumps(data)).replace('__DEFAULTS__', json.dumps(DEFAULTS))
    html = html.replace('__PYSRC__', Path(__file__).read_text())   # last: the source contains the other placeholder names
    Path(out).write_text(html)

def browser_load(text, name):
    """Pyodide entry point: log text -> viewer data JSON."""
    t, g, acc, ang = load(io.StringIO(text), angles=True)
    return json.dumps(view_data(t, g, acc, rotor_frame(t, g, acc), name, angles=ang))

def browser_csv(text, consts_json):
    """Pyodide entry point: log text + the page's current constants -> CSV text."""
    t, g, acc = load(io.StringIO(text))
    return to_csv(t, g, acc, rotor_frame(t, g, acc), {**DEFAULTS, **json.loads(consts_json)}, None)

def export(t, g, acc, name, consts=None, out_dir=Path(__file__).with_name('out'), angles=None):
    """Write out/<name>_view.html and out/<name>_rotor.csv with the same constants."""
    c = {**DEFAULTS, **(consts or {})}
    out_dir.mkdir(exist_ok=True)
    rf = rotor_frame(t, g, acc)
    write_view(out_dir / f'{name}_view.html', view_data(t, g, acc, rf, name, c, angles))
    to_csv(t, g, acc, rf, c, out_dir / f'{name}_rotor.csv')
    return rf

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log', type=Path, nargs='?')
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', type=float, default=v)
    args = ap.parse_args()
    if args.log is None:
        out = Path(__file__).with_name('out') / 'blade_viewer.html'
        out.parent.mkdir(exist_ok=True)
        write_view(out)
        print('->', out)
    else:
        t, g, acc, ang = load(args.log, angles=True)
        rf = export(t, g, acc, args.log.stem, {k: getattr(args, k) for k in DEFAULTS}, angles=ang)
        print('axis u:', rf['u0'].round(3), ' elevation: %.1f deg' % np.nanmedian(rf['elev']), ' fitted sensor radius: %.3f m' % rf['r_fit'],
              ' gyro-clipped samples:', int(rf['gyro_clip'].sum()), ' -> out/%s_view.html, out/%s_rotor.csv' % ((args.log.stem,) * 2))
