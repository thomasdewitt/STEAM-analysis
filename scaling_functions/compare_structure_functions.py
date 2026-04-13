"""
compare_structure_functions.py

2×2 panel comparison of Haar/structure-function scaling across four model
datasets, with observation overlays (dropsonde vertical, airplane horizontal).

Usage:
    python scaling_functions/compare_structure_functions.py --variable qv
    python scaling_functions/compare_structure_functions.py --variable qv --no_show
    python scaling_functions/compare_structure_functions.py --variable T --alt_min 500 --alt_max 3000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

FIGURES_DIR = _ROOT / 'Figures'

# ── Airplane altitude distribution (from create_airplane_legs.py Step 0) ──
# Leg mean altitudes span ~55–7122 m; bulk of legs are boundary layer (<500m)
# and upper-troposphere (~5500–7100 m), with a modest mid-level cluster.
# Defaults below include all legs.
_DEFAULTS = dict(
    variable='qv',
    method='haar',
    alt_min=5500,    # airplane legs span ~55–7122 m
    alt_max=7200,  # airplane legs span ~55–7122 m
    orders=[1],
    fit_min=4,
    fit_max=64,
    no_show=False,
)

# ── Dataset layout (2×2 grid) ─────────────────────────────────────────
DATASETS = [
    # (dataset_key, experiment,       (row, col))
    ('STEAM',      None,              (0, 0)),
    ('CM1',        'RCE_large300',    (0, 1)),
    ('SAM_RCEMIP', 'RCE_large300',    (1, 0)),
    ('SAM_TWPICE', None,              (1, 1)),
]

# ── Colors ────────────────────────────────────────────────────────────
# One color per dataset (hand-picked, colorblind-friendly)
_DATASET_COLORS = {
    'STEAM':      '#2171b5',   # blue
    'CM1':        '#238b45',   # green
    'SAM_RCEMIP': '#d94801',   # orange-red
    'SAM_TWPICE': '#6a51a3',   # purple
}
_OBS_COLOR      = '#888888'   # gray for observations
_FIT_COLOR      = '#e63946'   # red dotted for power-law fits


# ── Data loading ──────────────────────────────────────────────────────

def _load_model(dataset, variable, experiment=None):
    if dataset == 'STEAM':
        from utils.steam_loader import load_steam_variable_interpolated
        return load_steam_variable_interpolated(variable)
    elif dataset == 'CM1':
        from utils.cm1_loader import load_cm1_variable_interpolated
        return load_cm1_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'SAM_RCEMIP':
        from utils.sam_rcemip_loader import load_sam_rcemip_variable_interpolated
        return load_sam_rcemip_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'SAM_TWPICE':
        from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
        return load_sam_twpice_variable_interpolated(variable)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _load_dropsonde(variable, alt_min, alt_max):
    from utils.dropsonde_loader import load_dropsonde_variable_interpolated
    return load_dropsonde_variable_interpolated(variable, min_height=alt_min,
                                                max_height=alt_max)


def _load_airplane(variable, alt_min, alt_max):
    from utils.airplane_loader import load_airplane_variable
    return load_airplane_variable(variable, alt_min=alt_min, alt_max=alt_max)


# ── Scaling analysis ──────────────────────────────────────────────────

def _compute_haar(data, axis, order):
    from scaleinvariance import haar_fluctuation_analysis
    return haar_fluctuation_analysis(data, order=order, axis=axis,
                                     lags='powers of 1.05', nan_behavior='ignore')


def _fit_power_law(lags, vals, fit_min, fit_max, order):
    """Fit log(vals) ~ slope * log(lags) + intercept; return (H, slope, intercept)."""
    mask = (lags >= fit_min) & (lags <= fit_max) & (vals > 0)
    if mask.sum() < 2:
        return None, None, None
    log_l = np.log(lags[mask])
    log_v = np.log(vals[mask])
    slope, intercept = np.polyfit(log_l, log_v, 1)
    H = slope / order if order != 0 else np.nan
    return H, slope, intercept


def _plot_sf(ax, lags, vals, spacing_m, ls, color, lw, label, zorder=2):
    scales = lags * spacing_m / 1e3   # km
    ax.loglog(scales, vals, ls, color=color, lw=lw, label=label, zorder=zorder)
    return scales


def _plot_fit(ax, lags, slope, intercept, spacing_m, fit_min, fit_max, color=_FIT_COLOR):
    mask = (lags >= fit_min) & (lags <= fit_max)
    if mask.sum() < 2:
        return
    fit_lags   = lags[mask]
    fit_vals   = np.exp(intercept) * fit_lags ** slope
    fit_scales = fit_lags * spacing_m / 1e3
    ax.loglog(fit_scales, fit_vals, ':', color=color, lw=1.5, zorder=5)


# ── Per-panel routine ─────────────────────────────────────────────────

def _panel(ax, dataset, experiment, variable, method, alt_min, alt_max,
           orders, fit_min, fit_max,
           ds_lags_v=None, ds_vals_v=None, ds_spacing_v=None,
           ap_lags_h=None, ap_vals_h=None, ap_spacing_h=None):
    """Draw one panel: model vertical+horizontal SFs + obs overlays."""

    color = _DATASET_COLORS[dataset]

    z, data, dx = _load_model(dataset, variable, experiment)
    print(f"  {dataset}: loaded shape {data.shape}")

    # ── Altitude subset ──────────────────────────────────────────────
    z_min = alt_min if alt_min is not None else z.min()
    z_max = alt_max if alt_max is not None else z.max()
    zmask = (z >= z_min) & (z <= z_max)
    data  = data[..., zmask]
    z     = z[zmask]
    if len(z) == 0:
        ax.text(0.5, 0.5, f"No data in\n{z_min:.0f}–{z_max:.0f} m",
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return

    vert_spacing = float(np.median(np.diff(z)))

    func = _compute_haar  # only haar supported in compare script

    # Vertical SF (axis=-1 for z)
    lags_v, vals_v = func(data, axis=-1, order=orders[0])
    if not np.any(np.isfinite(vals_v) & (vals_v > 0)):
        print(f"  WARNING: {dataset} vertical SF has no finite positive values — skipping")
        return
    H_v, slope_v, int_v = _fit_power_law(lags_v, vals_v, fit_min, fit_max, orders[0])
    h_str = f', H={H_v:.3f}' if H_v is not None else ''
    _plot_sf(ax, lags_v, vals_v, vert_spacing, '-', color, 1.8,
             f'{dataset} vertical{h_str}')
    if slope_v is not None:
        _plot_fit(ax, lags_v, slope_v, int_v, vert_spacing, fit_min, fit_max)

    # Horizontal SF (axis=1 for x)
    if dx is not None:
        lags_h, vals_h = func(data, axis=1, order=orders[0])
        if not np.any(np.isfinite(vals_h) & (vals_h > 0)):
            print(f"  WARNING: {dataset} horizontal SF has no finite positive values — skipping")
        else:
            H_h, slope_h, int_h = _fit_power_law(lags_h, vals_h, fit_min, fit_max, orders[0])
            h_str = f', H={H_h:.3f}' if H_h is not None else ''
            _plot_sf(ax, lags_h, vals_h, dx, '--', color, 1.8,
                     f'{dataset} horizontal{h_str}')
            if slope_h is not None:
                _plot_fit(ax, lags_h, slope_h, int_h, dx, fit_min, fit_max)

    # ── Observation overlays ─────────────────────────────────────────
    # Dropsonde vertical
    if ds_lags_v is not None and ds_vals_v is not None:
        _plot_sf(ax, ds_lags_v, ds_vals_v, ds_spacing_v, '-', _OBS_COLOR, 1.2,
                 'Dropsonde (obs, vert)', zorder=1)

    # Airplane horizontal
    if ap_lags_h is not None and ap_vals_h is not None:
        _plot_sf(ax, ap_lags_h, ap_vals_h, ap_spacing_h, '--', _OBS_COLOR, 1.2,
                 'Airplane (obs, horiz)', zorder=1)

    # ── Formatting ───────────────────────────────────────────────────
    from config import get_unit_label
    unit  = get_unit_label(variable)
    title = dataset
    if experiment:
        title += f'\n({experiment})'
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Scale (km)', fontsize=8)
    method_label = 'Haar fluctuation' if method == 'haar' else 'Structure function'
    ax.set_ylabel(f'{method_label} ({unit})', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)

    # Alt range annotation
    lo_km = z_min / 1e3
    hi_km = z_max / 1e3
    ax.annotate(f'{lo_km:.1f}–{hi_km:.1f} km', xy=(0.03, 0.05),
                xycoords='axes fraction', fontsize=7, color='#444444')

    ax.legend(fontsize=6, loc='lower right', framealpha=0.8)


# ── Main compare function ─────────────────────────────────────────────

def compare_structure_functions(variable='qv', method='haar',
                                alt_min=None, alt_max=None,
                                orders=None, fit_min=4, fit_max=64, show=True):
    if orders is None:
        orders = [1]
    if alt_min is None:
        alt_min = _DEFAULTS['alt_min']
    if alt_max is None:
        alt_max = _DEFAULTS['alt_max']

    print(f"\n{'='*60}")
    print(f"compare_structure_functions: {variable}, {method}")
    print(f"Altitude range: {alt_min}–{alt_max} m")
    print(f"{'='*60}\n")

    # ── Pre-compute observation SFs (reused in all panels) ────────────
    from scaleinvariance import haar_fluctuation_analysis

    # Dropsonde vertical
    ds_lags_v = ds_vals_v = ds_spacing_v = None
    print("Loading dropsonde ...")
    z_ds, data_ds, _ = _load_dropsonde(variable, alt_min, alt_max)
    if data_ds.shape[-1] > 1:
        ds_spacing_v = float(np.median(np.diff(z_ds)))
        ds_lags_v, ds_vals_v = haar_fluctuation_analysis(
            data_ds, order=orders[0], axis=-1, lags='powers of 1.05',
            nan_behavior='ignore')
        print(f"  Dropsonde shape: {data_ds.shape}, dz={ds_spacing_v:.1f}m")
    else:
        print("  Dropsonde: too few altitude levels after subsetting")

    # Airplane horizontal
    ap_lags_h = ap_vals_h = ap_spacing_h = None
    print("Loading airplane ...")
    altitudes_ap, data_ap, ap_dx = _load_airplane(variable, alt_min, alt_max)
    if len(altitudes_ap) == 0:
        print(f"  Airplane: no legs in {alt_min}–{alt_max} m")
    else:
        ap_spacing_h = ap_dx
        ap_lags_h, ap_vals_h = haar_fluctuation_analysis(
            data_ap, order=orders[0], axis=1, lags='powers of 1.05',
            nan_behavior='ignore')
        alt_info = (f"  Airplane: {len(altitudes_ap)} legs, "
                    f"alt {altitudes_ap.min():.0f}–{altitudes_ap.max():.0f} m, "
                    f"dx={ap_dx:.1f} m")
        print(alt_info)

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    for dataset, experiment, (row, col) in DATASETS:
        ax = axes[row][col]
        print(f"\nPanel ({row},{col}): {dataset} {experiment or ''}")
        _panel(ax, dataset, experiment, variable, method, alt_min, alt_max,
               orders, fit_min, fit_max,
               ds_lags_v=ds_lags_v, ds_vals_v=ds_vals_v, ds_spacing_v=ds_spacing_v,
               ap_lags_h=ap_lags_h, ap_vals_h=ap_vals_h, ap_spacing_h=ap_spacing_h)

    method_label = 'Haar' if method == 'haar' else 'SF'
    alt_tag = f'_{int(alt_min)}-{int(alt_max)}m'
    suptitle = (f'{method_label} scaling — {variable}  '
                f'({alt_min/1e3:.1f}–{alt_max/1e3:.1f} km)\n'
                f'Solid=vertical  Dashed=horizontal  '
                f'Gray=observations  Dotted red=power-law fit')
    fig.suptitle(suptitle, fontsize=9, y=0.98)

    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / f'compare_{method}_{variable}{alt_tag}.pdf'
    plt.savefig(out_path, transparent=True, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    if show:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='2×2 compare plot of Haar/SF scaling across four datasets.')
    p.add_argument('--variable', default=_DEFAULTS['variable'],
                   help='Standardized variable: qt, qv, h, T')
    p.add_argument('--method',   default=_DEFAULTS['method'],
                   choices=['haar', 'structure_function'])
    p.add_argument('--alt_min',  type=float, default=_DEFAULTS['alt_min'],
                   help='Min altitude (m). Default covers all airplane legs (~55 m)')
    p.add_argument('--alt_max',  type=float, default=_DEFAULTS['alt_max'],
                   help='Max altitude (m). Default covers all airplane legs (~7122 m)')
    p.add_argument('--orders',   type=float, nargs='+', default=_DEFAULTS['orders'])
    p.add_argument('--fit_min',  type=int,   default=_DEFAULTS['fit_min'])
    p.add_argument('--fit_max',  type=int,   default=_DEFAULTS['fit_max'])
    p.add_argument('--no_show',  action='store_true', default=_DEFAULTS['no_show'])
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    compare_structure_functions(
        variable=args.variable,
        method=args.method,
        alt_min=args.alt_min,
        alt_max=args.alt_max,
        orders=args.orders,
        fit_min=args.fit_min,
        fit_max=args.fit_max,
        show=not args.no_show,
    )
