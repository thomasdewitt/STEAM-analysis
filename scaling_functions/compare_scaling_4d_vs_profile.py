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
    method='haar',
    no_show=False,
)


def _load_data(dataset, variable, experiment=None):
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
        return load_steam_variable_interpolated(variable)
    elif dataset == 'dropsonde':
        from utils.dropsonde_loader import load_dropsonde_variable_interpolated
        return load_dropsonde_variable_interpolated(variable)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def compare_scaling_4d_vs_profile(dataset, variable, experiment=None,
                                   method='haar', show=True):
    """Compare Haar/structure-function scaling of the full 4D volume vs the mean profile."""
    from config import get_unit_label

    z, data, _ = _load_data(dataset, variable, experiment)
    unit = get_unit_label(variable)

    vert_spacing = np.median(np.diff(z))
    print(f"Loaded {dataset} {variable}, shape: {data.shape}, dz={vert_spacing:.1f}m")

    # ── Compute scaling for 4D volume and mean profile ──
    mean_profile = np.nanmean(data, axis=(0, 1, 2))

    if method == 'haar':
        from scaleinvariance import haar_fluctuation_analysis
        lags_4d,   vals_4d   = haar_fluctuation_analysis(data,         axis=3, lags='powers of 1.05', nan_behavior='ignore')
        lags_prof, vals_prof = haar_fluctuation_analysis(mean_profile,  axis=0, lags='powers of 1.05', nan_behavior='ignore')
        ylabel = f'Haar Fluctuation ({unit})'
    elif method == 'structure_function':
        from scaleinvariance import structure_function_analysis
        lags_4d,   vals_4d   = structure_function_analysis(data,        axis=3, lags='powers of 1.05')
        lags_prof, vals_prof = structure_function_analysis(mean_profile, axis=0, lags='powers of 1.05')
        ylabel = f'Structure Function $S_1(r)$ ({unit})'
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Plot ──
    scales_4d   = lags_4d   * vert_spacing / 1e3  # km
    scales_prof = lags_prof * vert_spacing / 1e3

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.loglog(scales_4d,   vals_4d,   '-', color='#2171b5', lw=1.5, label='4D Volume')
    ax.loglog(scales_prof, vals_prof, '-', color='#cb181d', lw=1.5, label='Mean Profile')

    # Reference slopes (in km)
    ref_x = np.array([0.4, 5.0])
    idx_4d = np.argmin(np.abs(scales_4d - 0.4))
    ref_y075 = vals_4d[idx_4d] * 1.5 * (ref_x / ref_x[0]) ** 0.6
    ax.loglog(ref_x, ref_y075, '--', color='gray', lw=1, label='slope 0.6')

    idx_prof = np.argmin(np.abs(scales_prof - 0.4))
    ref_y1 = vals_prof[idx_prof] * 0.7 * (ref_x / ref_x[0]) ** 1.0
    ax.loglog(ref_x, ref_y1, ':', color='gray', lw=1, label='slope 1')

    method_label = 'Haar' if method == 'haar' else 'Structure Function'
    ax.set_xlabel('Scale (km)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{dataset} {variable} {method_label}: 4D Volume vs Mean Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / f'{method}_{dataset}_{variable}_4d_vs_profile.pdf', transparent=True)
    if show:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(
        description='Compare 4D-volume vs mean-profile scaling for a dataset variable.')
    p.add_argument('--dataset', default=_DEFAULTS['dataset'],
                   choices=['SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM', 'dropsonde'])
    p.add_argument('--variable', default=_DEFAULTS['variable'])
    p.add_argument('--experiment', default=_DEFAULTS['experiment'],
                   help='SAM_RCEMIP/CM1 only: RCE_large300 or RCE_small_les300')
    p.add_argument('--method', default=_DEFAULTS['method'],
                   choices=['haar', 'structure_function'])
    p.add_argument('--no_show', action='store_true', default=_DEFAULTS['no_show'],
                   help='Suppress plt.show() (useful in scripts/pipelines)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    compare_scaling_4d_vs_profile(
        dataset=args.dataset,
        variable=args.variable,
        experiment=args.experiment,
        method=args.method,
        show=not args.no_show,
    )
