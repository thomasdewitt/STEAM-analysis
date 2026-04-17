import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (VARIABLE_MAP_SAM_TO_STEAM, VARIABLE_MAP_SAM_TO_DROPSONDE,
                    get_unit_label, get_unit_factor, resolve_variable_pair, resolve_to_sam)

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ── Defaults (also used when called with no args) ──
# SAM_TWPICE:              QV, QT, QC, QI, MSE, TABS, U, V, W, PP
# SAM_RCEMIP RCE_large300: hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, QV, tntr, tntrs, tntrl
# SAM_RCEMIP RCE_small_les300: U, V, W, PP, QRAD, TABS, QV, QN, QP, LQRAD, SQRAD
# CM1:                     hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, tntr, tntrs, tntrl
# STEAM:                   qv, qt, qc, qi, h, T, p
# dropsonde:               q, ta, theta, rh, u, v, p, wspd, wdir
# special:                 cloud_fraction
_DEFAULTS = dict(
    plot_type='per_dataset',
    dataset='STEAM',
    variable='cloud_fraction',
    experiment='steam',
    steam_data_dir=None,
    steam_file_pattern=None,
    steam_group='strips',
    no_show=False,
)

# Cloud fraction thresholds (g/kg)
CF_THRESHOLDS = (0.01, 0.1)

# ── Colors ──
SAM_TWPICE_COLOR = '#2171b5'
SAM_RCEMIP_COLOR = '#6a51a3'
CM1_COLOR = '#238b45'
STEAM_COLOR = '#e6550d'
DROPSONDE_COLOR = '#636363'


def _load_cloud_fraction(dataset, experiment=None, steam_data_dir=None,
                          steam_file_pattern=None, steam_group=None):
    """Load cloud condensate in g/kg for a dataset. Returns (z, qtotal) with 4D shape."""
    if dataset == 'SAM_TWPICE':
        from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
        z, qc, _ = load_sam_twpice_variable_interpolated('QC')
        _, qi, _ = load_sam_twpice_variable_interpolated('QI')
        qtotal = qc + qi
    elif dataset == 'SAM_RCEMIP':
        from utils.sam_rcemip_loader import load_sam_rcemip_variable_interpolated
        if experiment == 'RCE_small_les300':
            # QN = total non-precipitating condensate, already in g/kg
            z, qtotal, _ = load_sam_rcemip_variable_interpolated('QN', experiment=experiment)
        else:
            z, qc, _ = load_sam_rcemip_variable_interpolated('clw', experiment=experiment)
            _, qi, _ = load_sam_rcemip_variable_interpolated('cli', experiment=experiment)
            qtotal = qc + qi
    elif dataset == 'CM1':
        from utils.cm1_loader import load_cm1_variable_interpolated
        z, qc, _ = load_cm1_variable_interpolated('clw', experiment=experiment)
        _, qi, _ = load_cm1_variable_interpolated('cli', experiment=experiment)
        qtotal = qc + qi
    elif dataset == 'STEAM':
        from utils.steam_loader import load_steam_variable_interpolated
        z, qc, _ = load_steam_variable_interpolated('qc',
                                                     data_dir=steam_data_dir,
                                                     file_pattern=steam_file_pattern,
                                                     group=steam_group)
        _, qi, _ = load_steam_variable_interpolated('qi',
                                                    data_dir=steam_data_dir,
                                                    file_pattern=steam_file_pattern,
                                                    group=steam_group)
        qtotal = qc + qi
    else:
        raise ValueError(f"Cloud fraction not supported for dataset: {dataset}")
    return z, qtotal


def _load_profiles(dataset, variable, experiment=None, steam_data_dir=None,
                    steam_file_pattern=None, steam_group=None):
    """Load data and return (z, profiles_2d) in display units.

    profiles_2d has shape (n_profiles, nz).
    """
    if dataset == 'SAM_TWPICE':
        from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
        z, data, _ = load_sam_twpice_variable_interpolated(variable)
    elif dataset == 'SAM_RCEMIP':
        from utils.sam_rcemip_loader import load_sam_rcemip_variable_interpolated
        z, data, _ = load_sam_rcemip_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'CM1':
        from utils.cm1_loader import load_cm1_variable_interpolated
        z, data, _ = load_cm1_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'STEAM':
        from utils.steam_loader import load_steam_variable_interpolated
        z, data, _ = load_steam_variable_interpolated(variable,
                                                       data_dir=steam_data_dir,
                                                       file_pattern=steam_file_pattern,
                                                       group=steam_group)
    elif dataset == 'dropsonde':
        from utils.dropsonde_loader import load_dropsonde_variable_interpolated
        z, data, _ = load_dropsonde_variable_interpolated(variable)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Reshape to (n_profiles, nz)
    nz = data.shape[-1]
    profiles = data.reshape(-1, nz)
    return z, profiles


def _dataset_color(dataset):
    return {
        'SAM_TWPICE': SAM_TWPICE_COLOR,
        'SAM_RCEMIP': SAM_RCEMIP_COLOR,
        'CM1':        CM1_COLOR,
        'STEAM':      STEAM_COLOR,
        'dropsonde':  DROPSONDE_COLOR,
    }[dataset]


def _dataset_label(dataset, experiment=None):
    if dataset in ('SAM_RCEMIP', 'CM1') and experiment:
        return f'{dataset} {experiment}'
    return dataset


def plot_per_dataset(dataset, variable, experiment=None,
                     steam_data_dir=None, steam_file_pattern=None,
                     steam_group=None, show=True):
    """Plot individual profiles (thin) + mean (thick) for one dataset/variable."""
    if variable == 'cloud_fraction':
        _plot_cloud_fraction_per_dataset(dataset, experiment,
                                         steam_data_dir=steam_data_dir,
                                         steam_file_pattern=steam_file_pattern,
                                         steam_group=steam_group,
                                         show=show)
        return

    z, profiles = _load_profiles(dataset, variable, experiment,
                                  steam_data_dir=steam_data_dir,
                                  steam_file_pattern=steam_file_pattern,
                                  steam_group=steam_group)
    unit = get_unit_label(variable)
    z_km = z / 1e3
    mean_prof = np.nanmean(profiles, axis=0)
    label = _dataset_label(dataset, experiment)

    fig, ax = plt.subplots(figsize=(5, 7))

    n_show = min(5000, profiles.shape[0])
    lw = 0.05
    if n_show < 2000: lw = 0.2
    for i in np.random.default_rng(42).choice(profiles.shape[0], n_show, replace=False):
        ax.plot(profiles[i], z_km, color='black', alpha=0.05, lw=lw)

    ax.plot(mean_prof, z_km, color='k', lw=2, label='Mean')

    ax.set_ylabel('Height (km)')
    ax.set_xlabel(f'{variable} ({unit})')
    ax.set_title(f'{label} {variable} Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'profiles_{dataset}_{variable}.pdf', transparent=True)
    if show:
        plt.show()


def _plot_cloud_fraction_per_dataset(dataset, experiment=None,
                                     steam_data_dir=None, steam_file_pattern=None,
                                     steam_group=None, show=True):
    """Cloud fraction profiles for one dataset at multiple thresholds."""
    z, qtotal = _load_cloud_fraction(dataset, experiment,
                                     steam_data_dir=steam_data_dir,
                                     steam_file_pattern=steam_file_pattern,
                                     steam_group=steam_group)
    z_km = z / 1e3
    color = _dataset_color(dataset)
    label = _dataset_label(dataset, experiment)

    fig, ax = plt.subplots(figsize=(5, 7))
    for thresh in CF_THRESHOLDS:
        cf = np.nanmean(qtotal > thresh, axis=(0, 1, 2))
        ax.plot(cf, z_km, lw=2, color=color, label=f'{thresh} g/kg',
                linestyle='-' if thresh == CF_THRESHOLDS[0] else '--')

    ax.set_ylabel('Height (km)')
    ax.set_xlabel('Cloud Fraction')
    ax.set_title(f'{label} Cloud Fraction Profiles')
    ax.set_xlim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'profiles_{dataset}_cloud_fraction.pdf', transparent=True)
    if show:
        plt.show()


def plot_cross_dataset(variable, datasets=('SAM_TWPICE', 'STEAM', 'dropsonde'),
                       steam_data_dir=None, steam_file_pattern=None,
                       steam_group=None, show=True):
    """Plot mean profiles across datasets. Accepts any variable name.

    Only datasets that have a mapping for the variable are plotted.
    """
    if variable == 'cloud_fraction':
        _plot_cloud_fraction_cross(steam_data_dir=steam_data_dir,
                                   steam_file_pattern=steam_file_pattern,
                                   steam_group=steam_group, show=show)
        return

    sam_var = resolve_to_sam(variable)

    # Build {dataset: variable_name} for datasets that have this variable
    dataset_vars = {}
    if 'SAM_TWPICE' in datasets:
        dataset_vars['SAM_TWPICE'] = sam_var
    if 'STEAM' in datasets and sam_var in VARIABLE_MAP_SAM_TO_STEAM:
        dataset_vars['STEAM'] = VARIABLE_MAP_SAM_TO_STEAM[sam_var]
    if 'dropsonde' in datasets and sam_var in VARIABLE_MAP_SAM_TO_DROPSONDE:
        dataset_vars['dropsonde'] = VARIABLE_MAP_SAM_TO_DROPSONDE[sam_var]

    unit = get_unit_label(sam_var)

    fig, ax = plt.subplots(figsize=(5, 7))
    for ds, var in dataset_vars.items():
        z, profiles = _load_profiles(ds, var,
                                     steam_data_dir=steam_data_dir,
                                     steam_file_pattern=steam_file_pattern,
                                     steam_group=steam_group)
        mean_prof = np.nanmean(profiles, axis=0)
        ax.plot(mean_prof, z / 1e3, color=_dataset_color(ds), lw=2, label=f'{ds} ({var})')

    ax.set_ylabel('Height (km)')
    ax.set_xlabel(f'{sam_var} ({unit})')
    ax.set_title(f'{sam_var}: Cross-Dataset Mean Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'profiles_cross_{sam_var}.pdf', transparent=True)
    if show:
        plt.show()


def _plot_cloud_fraction_cross(steam_data_dir=None, steam_file_pattern=None,
                                steam_group=None, show=True):
    """Cloud fraction: SAM_TWPICE vs STEAM at multiple thresholds."""
    z_sam, qt_sam = _load_cloud_fraction('SAM_TWPICE')
    z_steam, qt_steam = _load_cloud_fraction('STEAM',
                                             steam_data_dir=steam_data_dir,
                                             steam_file_pattern=steam_file_pattern,
                                             steam_group=steam_group)

    fig, ax = plt.subplots(figsize=(5, 7))
    for thresh in CF_THRESHOLDS:
        ls = '-' if thresh == CF_THRESHOLDS[0] else '--'
        cf_sam = np.nanmean(qt_sam > thresh, axis=(0, 1, 2))
        cf_steam = np.nanmean(qt_steam > thresh, axis=(0, 1, 2))
        ax.plot(cf_sam, z_sam / 1e3, color=SAM_TWPICE_COLOR, lw=2, linestyle=ls,
                label=f'SAM_TWPICE ({thresh} g/kg)')
        ax.plot(cf_steam, z_steam / 1e3, color=STEAM_COLOR, lw=2, linestyle=ls,
                label=f'STEAM ({thresh} g/kg)')

    ax.set_ylabel('Height (km)')
    ax.set_xlabel('Cloud Fraction')
    ax.set_title('Cloud Fraction: SAM_TWPICE vs STEAM')
    ax.set_xlim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'profiles_cross_cloud_fraction.pdf', transparent=True)
    if show:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(description='Plot atmospheric profiles from various datasets.')
    p.add_argument('--plot_type', default=_DEFAULTS['plot_type'],
                   choices=['per_dataset', 'cross_dataset'])
    p.add_argument('--dataset', default=_DEFAULTS['dataset'],
                   choices=['SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM', 'dropsonde'])
    p.add_argument('--variable', default=_DEFAULTS['variable'])
    p.add_argument('--experiment', default=_DEFAULTS['experiment'])
    p.add_argument('--steam_data_dir', default=_DEFAULTS['steam_data_dir'],
                   help='Directory for STEAM output files (overrides config default)')
    p.add_argument('--steam_file_pattern', default=_DEFAULTS['steam_file_pattern'],
                   help='Glob pattern for STEAM files, e.g. "steam_SAM_small_seed_*.nc"')
    p.add_argument('--steam_group', default=_DEFAULTS['steam_group'],
                   help="netCDF group(s) to read from nested STEAM files. "
                        "Shortcuts: 'parent' (root domain), 'strips' (both "
                        "strip_center + strip_edge, averaged together), 'cubes' "
                        "(both cube_center + cube_edge). Also accepts an explicit "
                        "path like 'refinements/strip_center'. Default: 'strips'.")
    p.add_argument('--no_show', action='store_true', default=_DEFAULTS['no_show'],
                   help='Suppress plt.show() (useful in scripts/pipelines)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    show = not args.no_show
    kwargs = dict(steam_data_dir=args.steam_data_dir,
                  steam_file_pattern=args.steam_file_pattern,
                  steam_group=args.steam_group,
                  show=show)
    if args.plot_type == 'per_dataset':
        plot_per_dataset(args.dataset, args.variable, args.experiment, **kwargs)
    elif args.plot_type == 'cross_dataset':
        plot_cross_dataset(args.variable, **kwargs)
