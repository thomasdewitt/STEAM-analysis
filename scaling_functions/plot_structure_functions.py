import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'

# ── Defaults ──
# SAM_TWPICE:              QV, QT, QC, QI, MSE, TABS, U, V, W, PP
# SAM_RCEMIP RCE_large300: hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, QV, tntr, tntrs, tntrl
# SAM_RCEMIP RCE_small_les300: U, V, W, PP, QRAD, TABS, QV, QN, QP, LQRAD, SQRAD
# CM1:                     hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, tntr, tntrs, tntrl
# STEAM:                   qv, qt, qc, qi, h, T, p
# dropsonde:               q, ta, theta, rh, u, v, p, wspd, wdir
_DEFAULTS = dict(
    dataset='STEAM',
    experiment='RCE_large300',
    variable='qt',
    method='structure_function',
    alt_min=4000,
    alt_max=5000,
    orders=[1],
    fit_min=4,
    fit_max=64,
    steam_group='strips',
    no_show=False,
)

# Colors per order (one per moment)
_ORDER_COLORS = ['#2171b5', '#6a51a3', '#238b45', '#e6550d', '#cb181d', '#636363']


def _load_data(dataset, variable, experiment=None, steam_group=None):
    if dataset == 'SAM_TWPICE':
        from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
        return load_sam_twpice_variable_interpolated(variable)
    elif dataset == 'SAM_RCEMIP':
        from utils.sam_rcemip_loader import load_sam_rcemip_variable_interpolated
        return load_sam_rcemip_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'CM1':
        from utils.cm1_loader import load_cm1_variable_interpolated
        return load_cm1_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'STEAM':
        from utils.steam_loader import load_steam_variable_interpolated
        return load_steam_variable_interpolated(variable, group=steam_group)
    elif dataset == 'dropsonde':
        from utils.dropsonde_loader import load_dropsonde_variable_interpolated
        return load_dropsonde_variable_interpolated(variable)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def plot_structure_functions(dataset, variable, experiment=None, method='haar',
                              alt_min=None, alt_max=None, orders=None,
                              fit_min=4, fit_max=64, steam_group=None, show=True):
    """Plot Haar/structure-function scaling for both x and z directions on one plot."""
    if orders is None:
        orders = [1]

    from scaleinvariance import structure_function_analysis, haar_fluctuation_analysis
    from config import get_unit_label

    z, data, dx = _load_data(dataset, variable, experiment, steam_group=steam_group)
    unit = get_unit_label(variable)
    print(f"Loaded {dataset} {variable}, shape: {data.shape}")

    # ── Altitude subset ──
    if alt_min is not None or alt_max is not None:
        z_min = alt_min if alt_min is not None else z.min()
        z_max = alt_max if alt_max is not None else z.max()
        zmask = (z >= z_min) & (z <= z_max)
        data = data[..., zmask]
        z = z[zmask]
        print(f"  Altitude subset: {z_min:.0f}–{z_max:.0f} m, shape now: {data.shape}")

    vert_spacing = np.median(np.diff(z))

    # ── Directions to analyse ──
    # z always; horizontal only when spacing is available.
    # Horizontal SF is computed along x (axis=1); y (axis=2) is the strided
    # replication axis. Works for parent, strips (x=long, y=narrow), and cubes.
    # linestyle: solid for z, dashed for x; fit is always dotted
    directions = [('z', -1, vert_spacing, '-')]
    if dx is not None:
        directions.append(('x', 1, dx, '--'))

    fig, ax = plt.subplots(figsize=(7, 5))
    H_results = {}

    for direction, axis, spacing, data_ls in directions:
        for i, q in enumerate(orders):
            color = _ORDER_COLORS[i % len(_ORDER_COLORS)]

            if method == 'structure_function':
                lags, vals = structure_function_analysis(data, order=q, axis=axis,
                                                         lags='powers of 1.05')
            elif method == 'haar':
                lags, vals = haar_fluctuation_analysis(data, order=q, axis=axis,
                                                       lags='powers of 1.05')
            else:
                raise ValueError(f"Unknown method: {method}")

            scales = lags * spacing / 1e3  # km

            # Fit over [fit_min, fit_max] lags (computed first so H goes in label)
            H = None
            fitmask = (lags >= fit_min) & (lags <= fit_max) & (vals > 0)
            if fitmask.sum() >= 2:
                log_l = np.log(lags[fitmask])
                log_v = np.log(vals[fitmask])
                slope, intercept = np.polyfit(log_l, log_v, 1)
                H = slope / q if q != 0 else np.nan
                H_results[(direction, q)] = H
                print(f"  {direction} q={q:.1f}: xi={slope:.4f}, H={H:.4f}")

            h_str = f', H={H:.3f}' if H is not None else ''
            ax.loglog(scales, vals, data_ls, color=color, lw=1.5,
                      label=f'q={q:.1f} ({direction}{h_str})')

            if H is not None:
                fit_vals = np.exp(intercept) * lags[fitmask] ** slope
                fit_scales = lags[fitmask] * spacing / 1e3
                ax.loglog(fit_scales, fit_vals, '--', color='red', lw=1, alpha=1)

    method_label = 'Haar Fluctuation' if method == 'haar' else 'Structure Function'
    ax.set_xlabel('Scale (km)')
    ax.set_ylabel(f'{method_label} ({unit})')

    alt_str = ''
    if alt_min is not None or alt_max is not None:
        lo = f"{(alt_min or z.min()) / 1e3:.1f}"
        hi = f"{(alt_max or z.max()) / 1e3:.1f}"
        alt_str = f', {lo}–{hi} km'

    dir_label = '+'.join(d for d, _, _, _ in directions)
    ax.set_title(f'{dataset} {variable} {method_label} ({dir_label}{alt_str})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    alt_tag = ''
    if alt_min is not None or alt_max is not None:
        alt_tag = f'_{int(alt_min or 0)}-{int(alt_max or 99999)}m'
    plt.savefig(FIGURES_DIR / f'{method}_{dataset}_{variable}_{dir_label}{alt_tag}_orders.pdf',
                transparent=True)
    if show:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(
        description='Plot Haar/structure-function scaling from various datasets.')
    p.add_argument('--dataset', default=_DEFAULTS['dataset'],
                   choices=['SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM', 'dropsonde'])
    p.add_argument('--variable', default=_DEFAULTS['variable'])
    p.add_argument('--experiment', default=_DEFAULTS['experiment'],
                   help='SAM_RCEMIP/CM1 only: RCE_large300 or RCE_small_les300')
    p.add_argument('--method', default=_DEFAULTS['method'],
                   choices=['haar', 'structure_function'])
    p.add_argument('--alt_min', type=float, default=_DEFAULTS['alt_min'],
                   help='Minimum altitude in metres (None = use all)')
    p.add_argument('--alt_max', type=float, default=_DEFAULTS['alt_max'],
                   help='Maximum altitude in metres (None = use all)')
    p.add_argument('--orders', type=float, nargs='+', default=_DEFAULTS['orders'],
                   help='Moment orders, e.g. --orders 1 2 3')
    p.add_argument('--fit_min', type=int, default=_DEFAULTS['fit_min'],
                   help='Minimum lag for power-law fit')
    p.add_argument('--fit_max', type=int, default=_DEFAULTS['fit_max'],
                   help='Maximum lag for power-law fit')
    p.add_argument('--steam_group', default=_DEFAULTS['steam_group'],
                   help="STEAM only: netCDF group to read. 'parent', 'strips', "
                        "'cubes', or an explicit path. Default: 'strips'.")
    p.add_argument('--no_show', action='store_true', default=_DEFAULTS['no_show'],
                   help='Suppress plt.show() (useful in scripts/pipelines)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_structure_functions(
        dataset=args.dataset,
        variable=args.variable,
        experiment=args.experiment,
        method=args.method,
        alt_min=args.alt_min,
        alt_max=args.alt_max,
        orders=args.orders,
        fit_min=args.fit_min,
        fit_max=args.fit_max,
        steam_group=args.steam_group,
        show=not args.no_show,
    )
