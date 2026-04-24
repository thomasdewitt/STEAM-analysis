"""
compare_steam_sam_twpice_vertical.py

Vertical first-order structure function comparison between STEAM,
SAM_TWPICE, and BEACH ORCESTRA dropsondes at 2–10 km altitude. One plot
per variable. `scaleinvariance.structure_function` handles NaNs natively
(uses `nanmean`), so BEACH gaps are passed through without pre-processing.

Variables: MSE, q_v, q_t. (T and p omitted — their vertical scaling sits
at H ≳ 1 where SF cannot resolve the exponent.)

q_t overlays STEAM and SAM only; sondes lack condensate so BEACH q_t
would just be q_v and is not plotted on that panel.

Each curve is normalized by its value at 1 km (log-log interpolated). A
thin dashed black reference line with slope = H_STEAM (fitted over 1–3 km)
sits sqrt(2)× above the normalized curves from 0.03 to 4 km.
"""

import sys
from pathlib import Path
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import scaleinvariance as si
si.set_backend('torch')
si.set_device('cuda')
si.set_numerical_precision('float64')

FIGURES_DIR = _ROOT / 'Figures'
FIGURES_DIR.mkdir(exist_ok=True)

BEACH_PATH = Path('/home/thomas/code-and-data/sonde-regrid/output/beach.nc')

ALT_MIN = 2000.0
ALT_MAX = 10000.0
ORDER   = 1

NORM_SCALE_KM = 1.0
FIT_MIN_KM    = 1.0
FIT_MAX_KM    = 3.0
REF_MIN_KM    = 0.03
REF_MAX_KM    = 4.0
REF_OFFSET    = np.sqrt(2.0)

BEACH_MIN_SCALE_KM = 0.1   # truncate BEACH SF below this scale (sonde inertia)

STEAM_COLOR = '#bc4b3d'    # clay red
SAM_COLOR   = '#3e6d7a'    # deep teal
BEACH_COLOR = '#6a5b7a'    # muted slate purple
REF_COLOR   = '#222222'


# ── Data loading ──────────────────────────────────────────────────────

def _steam_seed_files():
    return [f.resolve() for f in
            sorted((_ROOT / 'STEAM' / 'data').glob('nested_refine_seed*.nc'))]


def _load_steam(variable):
    from utils.steam_loader import load_steam_variable_interpolated
    z, data, _ = load_steam_variable_interpolated(
        variable, group='strips', seed_files=_steam_seed_files(),
        min_height=ALT_MIN, max_height=ALT_MAX)
    return z, data.astype(np.float64, copy=False)


def _load_sam(variable):
    from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
    z, data, _ = load_sam_twpice_variable_interpolated(
        variable, min_height=ALT_MIN, max_height=ALT_MAX)
    return z, data.astype(np.float64, copy=False)


def _sam_qt_total():
    z, qv = _load_sam('QV')
    _, qc = _load_sam('QC')
    _, qi = _load_sam('QI')
    return z, qv + qc + qi


_BEACH_MAP = {
    'T':  ('T',   1.0,       'K'),
    'h':  ('MSE', 1.0/1004., 'K'),
    'qv': ('q',   1e3,       'g/kg'),
    'qt': ('q',   1e3,       'g/kg'),   # sondes have no condensate
    'p':  ('p',   1.0,       'Pa'),
}


def _load_beach(variable):
    if variable not in _BEACH_MAP:
        return None
    native, factor, _ = _BEACH_MAP[variable]
    with nc.Dataset(BEACH_PATH, 'r') as ds:
        z = np.asarray(ds.variables['altitude'][:], dtype=np.float64)
        d = np.asarray(ds.variables[native][:], dtype=np.float64)
    mask = (z >= ALT_MIN) & (z <= ALT_MAX)
    z = z[mask]
    d = d[:, mask] * factor
    # Pass NaNs through — structure_function uses nanmean internally.
    return z, d[:, None, None, :]


# ── SFs ───────────────────────────────────────────────────────────────

def _sf(data, dz, order=ORDER):
    """First-order structure function along the last axis. NaN-aware via nanmean."""
    from scaleinvariance import structure_function
    from scaleinvariance.backend import to_numpy
    lags, vals = structure_function(
        data, order=order, axis=-1, lags='powers of 1.05')
    lags = np.asarray(to_numpy(lags))
    vals = np.asarray(to_numpy(vals))
    return lags * dz / 1e3, vals


def _interp_at(scales_km, vals, target_km):
    m = (vals > 0) & np.isfinite(vals)
    if m.sum() < 2:
        return None
    s = scales_km[m]
    if target_km < s.min() or target_km > s.max():
        return None
    return float(np.exp(np.interp(
        np.log(target_km), np.log(s), np.log(vals[m]))))


def _fit_H(scales_km, vals, lo_km, hi_km, order=ORDER):
    m = (scales_km >= lo_km) & (scales_km <= hi_km) & (vals > 0) & np.isfinite(vals)
    if m.sum() < 2:
        return None
    slope, _ = np.polyfit(np.log(scales_km[m]), np.log(vals[m]), 1)
    return slope / order


def _compute(loaded):
    if loaded is None:
        return None
    z, data = loaded
    if data.shape[-1] < 2 or data.shape[0] == 0:
        return None
    dz = float(np.median(np.diff(z)))
    scales_km, vals = _sf(data, dz)
    anchor = _interp_at(scales_km, vals, NORM_SCALE_KM)
    if anchor is None or anchor <= 0:
        return None
    return scales_km, vals / anchor, dz, data.shape


# ── Plotting ──────────────────────────────────────────────────────────

def _add_ref_label(ax, H_ref, anchor_x, anchor_y):
    x_lbl = REF_MIN_KM * 1.15
    y_line = anchor_y * (x_lbl / anchor_x) ** H_ref
    y_label = y_line * 1.25
    ax.figure.canvas.draw()
    x0, x1 = x_lbl, x_lbl * 10.0
    y0, y1 = y_line, y_line * (10.0 ** H_ref)
    (u0, v0), (u1, v1) = ax.transData.transform([(x0, y0), (x1, y1)])
    angle_deg = float(np.degrees(np.arctan2(v1 - v0, u1 - u0)))
    ax.text(x_lbl, y_label, f'Reference H={H_ref:.3f}',
            rotation=angle_deg, rotation_mode='anchor',
            ha='left', va='bottom', fontsize=8, color=REF_COLOR)


def _plot_panel(ax, label, steam_pack, sam_pack, beach_pack, H_ref,
                beach_note=None, ref_anchor=None):
    if steam_pack is not None:
        s, v, dz, _ = steam_pack
        ax.loglog(s, v, '-', color=STEAM_COLOR, lw=1.8,
                  label=f'STEAM strips (dz={dz:.0f} m)')
    if sam_pack is not None:
        s, v, dz, _ = sam_pack
        ax.loglog(s, v, '-', color=SAM_COLOR, lw=1.8,
                  label=f'SAM_TWPICE (dz={dz:.0f} m)')
    if beach_pack is not None:
        s, v, dz, _ = beach_pack
        keep = s >= BEACH_MIN_SCALE_KM
        lbl = f'BEACH sondes (dz={dz:.0f} m, ≥{BEACH_MIN_SCALE_KM*1e3:.0f} m)'
        if beach_note:
            lbl += f'  {beach_note}'
        ax.loglog(s[keep], v[keep], '-', color=BEACH_COLOR, lw=1.8, label=lbl)

    if H_ref is not None:
        anchor_x, anchor_y = ref_anchor if ref_anchor is not None \
                             else (NORM_SCALE_KM, REF_OFFSET)
        ref_x = np.array([REF_MIN_KM, REF_MAX_KM])
        ref_y = anchor_y * (ref_x / anchor_x) ** H_ref
        ax.loglog(ref_x, ref_y, '--', color=REF_COLOR, lw=0.9)
        _add_ref_label(ax, H_ref, anchor_x, anchor_y)

    ax.set_xlabel('Vertical scale (km)')
    ax.set_ylabel(f'$S_{{{ORDER}}}(\\Delta z)\\,/\\,S_{{{ORDER}}}(1\\,\\mathrm{{km}})$')
    ax.set_title(f'{label}, vertical, '
                 f'{ALT_MIN/1e3:.0f}–{ALT_MAX/1e3:.0f} km altitude')
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)


def _run_variable(label, steam_loader, sam_loader, beach_loader, outfile,
                  beach_note=None, ref_anchor=None):
    print(f'\n=== {label} ===')

    packs = {}
    for name, loader in [('STEAM', steam_loader),
                          ('SAM',   sam_loader),
                          ('BEACH', beach_loader)]:
        if loader is None:
            continue
        try:
            p = _compute(loader())
            if p is not None:
                print(f'  {name}: shape {p[3]}, dz={p[2]:.1f} m, '
                      f'n_scales={len(p[0])}')
                packs[name] = p
            else:
                print(f'  {name}: no usable result (too few levels / bad anchor)')
        except Exception as e:
            print(f'  {name} failed: {e}')

    H_ref = None
    if 'STEAM' in packs:
        H_ref = _fit_H(packs['STEAM'][0], packs['STEAM'][1],
                       FIT_MIN_KM, FIT_MAX_KM)
        print(f'  STEAM H (fit {FIT_MIN_KM}–{FIT_MAX_KM} km): {H_ref}')

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_panel(ax, label,
                packs.get('STEAM'), packs.get('SAM'), packs.get('BEACH'),
                H_ref, beach_note=beach_note, ref_anchor=ref_anchor)
    plt.tight_layout()
    out = FIGURES_DIR / outfile
    plt.savefig(out, transparent=True)
    plt.close(fig)
    print(f'  saved {out}')


def main():
    print(f'backend={si.get_backend()}, device={si.get_device()}, '
          f'precision={si.get_numerical_precision()}')
    print(f'Altitude: {ALT_MIN:.0f}–{ALT_MAX:.0f} m | normalize @ '
          f'{NORM_SCALE_KM} km | STEAM H fit {FIT_MIN_KM}–{FIT_MAX_KM} km')

    suffix = f'_{int(ALT_MIN)}-{int(ALT_MAX)}m.pdf'

    _run_variable('T',
                  lambda: _load_steam('T'),
                  lambda: _load_sam('TABS'),
                  lambda: _load_beach('T'),
                  f'cmp_steam_sam_twpice_vertical_T{suffix}',
                  ref_anchor=(0.1, 0.3))

    _run_variable('MSE',
                  lambda: _load_steam('h'),
                  lambda: _load_sam('MSE'),
                  lambda: _load_beach('h'),
                  f'cmp_steam_sam_twpice_vertical_MSE{suffix}')

    _run_variable('q_v',
                  lambda: _load_steam('qv'),
                  lambda: _load_sam('QV'),
                  lambda: _load_beach('qv'),
                  f'cmp_steam_sam_twpice_vertical_qv{suffix}')

    # BEACH sondes have no condensate → no meaningful q_t
    _run_variable('q_t',
                  lambda: _load_steam('qt'),
                  _sam_qt_total,
                  None,
                  f'cmp_steam_sam_twpice_vertical_qt{suffix}')


if __name__ == '__main__':
    main()
