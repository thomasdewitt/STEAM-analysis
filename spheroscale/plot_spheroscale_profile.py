import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spheroscale.compute_spheroscale import load_and_compute, N_HAAR

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'

# Fixed spheroscale bin edges: 1e-5 m to 1e5 m
LS_EDGES = np.logspace(np.log10(1e-5), np.log10(1e5), 101)

_DEFAULTS = dict(
    dataset='STEAM',
    experiment='RCE_large300',
    variable='qt',
    alt_min=100,
    alt_max=20000,
    no_show=False,
)


def plot_spheroscale_profile(dataset, variable, experiment=None,
                              alt_min=None, alt_max=None, n_haar=N_HAAR, show=True):
    """2D histogram: spheroscale vs height for a single dataset."""
    l_s_v, z_v, z_bd = load_and_compute(
        dataset, variable, experiment,
        alt_min=alt_min, alt_max=alt_max, n_haar=n_haar,
    )

    # ── Bins ──────────────────────────────────────────────────────────────────
    dz_bd   = float(np.median(np.diff(z_bd)))
    z_edges = np.append(z_bd - dz_bd / 2.0, z_bd[-1] + dz_bd / 2.0)

    counts, _, _ = np.histogram2d(z_v, l_s_v, bins=[z_edges, LS_EDGES])
    counts = np.where(counts > 0, counts, np.nan)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    Xm, Ym = np.meshgrid(LS_EDGES, z_edges / 1e3)
    mesh = ax.pcolormesh(
        Xm, Ym, counts,
        # norm=mcolors.LogNorm(vmin=1),
        cmap='plasma',
        shading='flat',
    )
    ax.set_xscale('log')
    ax.set_xlabel(r'Spheroscale $l_s$ (m)')
    ax.set_ylabel('Height (km)')

    exp_str = f' {experiment}' if experiment else ''
    ax.set_title(f'{dataset}{exp_str} {variable} — Spheroscale profile (n={n_haar} Haar)')

    plt.colorbar(mesh, ax=ax, label='Count')
    ax.grid(True, alpha=0.2, which='both', axis='x')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()

    alt_tag = ''
    if alt_min is not None or alt_max is not None:
        alt_tag = f'_{int(alt_min or 0)}-{int(alt_max or 99999)}m'
    FIGURES_DIR.mkdir(exist_ok=True)
    fname = FIGURES_DIR / f'spheroscale_profile_{dataset}_{variable}{alt_tag}.pdf'
    plt.savefig(fname, transparent=True)
    print(f"  Saved: {fname}")

    if show:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(description='Plot spheroscale height-profile histogram.')
    p.add_argument('--dataset',    default=_DEFAULTS['dataset'],
                   choices=['SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM'])
    p.add_argument('--variable',   default=_DEFAULTS['variable'])
    p.add_argument('--experiment', default=_DEFAULTS['experiment'],
                   help='SAM_RCEMIP/CM1 only')
    p.add_argument('--alt_min',    type=float, default=_DEFAULTS['alt_min'])
    p.add_argument('--alt_max',    type=float, default=_DEFAULTS['alt_max'])
    p.add_argument('--n_haar',     type=int,   default=N_HAAR,
                   help='Haar scale: number of points (even integer)')
    p.add_argument('--no_show',    action='store_true', default=_DEFAULTS['no_show'])
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_spheroscale_profile(
        dataset=args.dataset,
        variable=args.variable,
        experiment=args.experiment,
        alt_min=args.alt_min,
        alt_max=args.alt_max,
        n_haar=args.n_haar,
        show=not args.no_show,
    )
