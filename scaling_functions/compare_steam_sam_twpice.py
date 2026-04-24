"""
compare_steam_sam_twpice.py

Horizontal structure-function comparison between STEAM and SAM_TWPICE at
5000–6000 m altitude. One plot per variable, both datasets overlaid, scale
axis in km. STEAM uses the strip refinement subgroups (long axis = x).

Each SF is normalized by its own value at 1 km (log-log interpolated) so the
two datasets share a common anchor and slopes are directly comparable. A
thin dashed black reference line with slope = H_STEAM (fitted between
1–30 km) is drawn two decades above the normalized curves from 0.1 to 30 km.

Variables: T, MSE, q_v, q_t, p
  STEAM:  T, h, qv, qt, p
  SAM:    TABS, MSE, QV, (QV+QC+QI), PP
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import scaleinvariance as si
si.set_backend('torch')
si.set_device('cuda')
si.set_numerical_precision('float64')

FIGURES_DIR = _ROOT / 'Figures'
FIGURES_DIR.mkdir(exist_ok=True)

ALT_MIN = 5000.0      # m
ALT_MAX = 6000.0      # m
ORDER   = 1
NORM_SCALE_KM = 1.0   # normalization anchor
FIT_MIN_KM    = 1.0   # STEAM H fit range
FIT_MAX_KM    = 3.0
REF_MIN_KM    = 0.1   # reference-line extent
REF_MAX_KM    = 30.0
REF_OFFSET    = np.sqrt(2.0)  # log-halfway between data (=1) and the prior 2x reference

# Hand-picked muted pair: clay red + deep teal
STEAM_COLOR = '#bc4b3d'
SAM_COLOR   = '#3e6d7a'
REF_COLOR   = '#222222'


def _steam_seed_files():
    files = sorted((_ROOT / 'STEAM' / 'data').glob('nested_refine_seed*.nc'))
    return [f.resolve() for f in files]


def _load_steam(variable):
    from utils.steam_loader import load_steam_variable_interpolated
    z, data, dx = load_steam_variable_interpolated(
        variable, group='strips', seed_files=_steam_seed_files(),
        min_height=ALT_MIN, max_height=ALT_MAX)
    return z, data.astype(np.float64, copy=False), dx


def _load_sam(variable):
    from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
    z, data, dx = load_sam_twpice_variable_interpolated(
        variable, min_height=ALT_MIN, max_height=ALT_MAX)
    return z, data.astype(np.float64, copy=False), dx


def _sf(data, axis, order=ORDER):
    from scaleinvariance import structure_function
    from scaleinvariance.backend import to_numpy
    lags, vals = structure_function(
        data, order=order, axis=axis, lags='powers of 1.05')
    return np.asarray(to_numpy(lags)), np.asarray(to_numpy(vals))


def _sam_qt_total():
    z, qv, dx = _load_sam('QV')
    _, qc, _  = _load_sam('QC')
    _, qi, _  = _load_sam('QI')
    return z, qv + qc + qi, dx


def _interp_at(scales_km, vals, target_km):
    """Log-log linear interpolation of SF value at target_km."""
    mask = (vals > 0) & np.isfinite(vals)
    return float(np.exp(np.interp(
        np.log(target_km), np.log(scales_km[mask]), np.log(vals[mask]))))


def _fit_H(scales_km, vals, lo_km, hi_km, order=ORDER):
    """Fit slope in log-log between lo_km and hi_km. Returns H = slope/order."""
    mask = (scales_km >= lo_km) & (scales_km <= hi_km) & (vals > 0) & np.isfinite(vals)
    if mask.sum() < 2:
        return None
    slope, _ = np.polyfit(np.log(scales_km[mask]), np.log(vals[mask]), 1)
    return slope / order


def _compute(loader):
    z, data, dx = loader()
    lags, vals = _sf(data, axis=1)
    scales_km = lags * dx / 1e3
    anchor = _interp_at(scales_km, vals, NORM_SCALE_KM)
    return scales_km, vals / anchor, dx, data.shape


def _add_ref_label(ax, H_ref):
    """Place the 'Reference H=...' label rotated along the reference line,
    a touch above it, at a scale outside the plotted data range so it doesn't
    collide with the data curves."""
    # Anchor a bit past the low-scale end of the reference line.
    x_anchor = REF_MIN_KM * 1.15
    y_line   = REF_OFFSET * (x_anchor / NORM_SCALE_KM) ** H_ref
    y_label  = y_line * 1.25    # sit just above the line
    # Convert the data-space slope H_ref to display-space angle for rotation.
    ax.figure.canvas.draw()
    x0, x1 = x_anchor, x_anchor * 10.0
    y0, y1 = y_line, y_line * (10.0 ** H_ref)
    (u0, v0), (u1, v1) = ax.transData.transform([(x0, y0), (x1, y1)])
    angle_deg = float(np.degrees(np.arctan2(v1 - v0, u1 - u0)))
    ax.text(x_anchor, y_label, f'Reference H={H_ref:.3f}',
            rotation=angle_deg, rotation_mode='anchor',
            ha='left', va='bottom', fontsize=8, color=REF_COLOR)


def _plot_panel(ax, label, unit, steam_pack, sam_pack, H_ref):
    if steam_pack is not None:
        s, v, dx = steam_pack
        ax.loglog(s, v, '-', color=STEAM_COLOR, lw=1.8,
                  label=f'STEAM strips (dx={dx:.0f} m)')
    if sam_pack is not None:
        s, v, dx = sam_pack
        ax.loglog(s, v, '-', color=SAM_COLOR, lw=1.8,
                  label=f'SAM_TWPICE (dx={dx:.0f} m)')

    # Reference line: y = REF_OFFSET * (l / NORM_SCALE_KM)^H
    if H_ref is not None:
        ref_x = np.array([REF_MIN_KM, REF_MAX_KM])
        ref_y = REF_OFFSET * (ref_x / NORM_SCALE_KM) ** H_ref
        ax.loglog(ref_x, ref_y, '--', color=REF_COLOR, lw=0.9)
        _add_ref_label(ax, H_ref)

    ax.set_xlabel('Scale (km)')
    ax.set_ylabel(f'$S_{{{ORDER}}}(l)\\,/\\,S_{{{ORDER}}}(1\\,\\mathrm{{km}})$')
    ax.set_title(f'{label}, horizontal, '
                 f'{ALT_MIN/1e3:.0f}–{ALT_MAX/1e3:.0f} km altitude')
    ax.grid(False)
    ax.legend(frameon=False)


def _run_variable(label, steam_loader, sam_loader, unit, outfile):
    print(f'\n=== {label} ===')
    steam_pack = sam_pack = None
    H_ref = None

    if steam_loader is not None:
        try:
            s, v, dx, sh = _compute(steam_loader)
            steam_pack = (s, v, dx)
            print(f'  STEAM: shape {sh}, dx={dx:.1f} m')
            H_ref = _fit_H(s, v, FIT_MIN_KM, FIT_MAX_KM)
            print(f'  STEAM H (fit {FIT_MIN_KM}–{FIT_MAX_KM} km): {H_ref}')
        except Exception as e:
            print(f'  STEAM failed: {e}')

    if sam_loader is not None:
        try:
            s, v, dx, sh = _compute(sam_loader)
            sam_pack = (s, v, dx)
            print(f'  SAM:   shape {sh}, dx={dx:.1f} m')
        except Exception as e:
            print(f'  SAM failed: {e}')

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_panel(ax, label, unit, steam_pack, sam_pack, H_ref)
    plt.tight_layout()
    out = FIGURES_DIR / outfile
    plt.savefig(out, transparent=True)
    plt.close(fig)
    print(f'  saved {out}')


def main():
    print(f'backend={si.get_backend()}, device={si.get_device()}, '
          f'precision={si.get_numerical_precision()}')
    print(f'Altitude: {ALT_MIN:.0f}–{ALT_MAX:.0f} m, normalize @ {NORM_SCALE_KM} km, '
          f'fit H on STEAM over {FIT_MIN_KM}–{FIT_MAX_KM} km')

    _run_variable('T',   lambda: _load_steam('T'),
                  lambda: _load_sam('TABS'), 'K',
                  f'cmp_steam_sam_twpice_T_{int(ALT_MIN)}-{int(ALT_MAX)}m.pdf')

    _run_variable('MSE', lambda: _load_steam('h'),
                  lambda: _load_sam('MSE'), 'K',
                  f'cmp_steam_sam_twpice_MSE_{int(ALT_MIN)}-{int(ALT_MAX)}m.pdf')

    _run_variable('q_v', lambda: _load_steam('qv'),
                  lambda: _load_sam('QV'), 'g/kg',
                  f'cmp_steam_sam_twpice_qv_{int(ALT_MIN)}-{int(ALT_MAX)}m.pdf')

    _run_variable('q_t', lambda: _load_steam('qt'),
                  _sam_qt_total, 'g/kg',
                  f'cmp_steam_sam_twpice_qt_{int(ALT_MIN)}-{int(ALT_MAX)}m.pdf')

    _run_variable('p (SAM: PP perturbation)',
                  lambda: _load_steam('p'),
                  lambda: _load_sam('PP'), 'Pa',
                  f'cmp_steam_sam_twpice_p_{int(ALT_MIN)}-{int(ALT_MAX)}m.pdf')


if __name__ == '__main__':
    main()
