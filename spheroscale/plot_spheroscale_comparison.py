"""
Compare spheroscale distributions across multiple datasets on a single 1D histogram.

Each dataset/variable/experiment is specified as a colon-separated token:
    DATASET:VARIABLE[:EXPERIMENT]

Example:
    python plot_spheroscale_comparison.py \
        STEAM:qt \
        SAM_TWPICE:QT \
        CM1:hus:RCE_large300
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spheroscale.compute_spheroscale import load_and_compute, N_HAAR

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'

# Fixed spheroscale bin edges shared with profile plot
LS_EDGES = np.logspace(np.log10(1e-7), np.log10(1e5), 101)

_COLORS = ['#2171b5', '#e6550d', '#238b45', '#6a51a3', '#cb181d', '#636363']

_DEFAULTS = dict(
    alt_min=100,
    alt_max=20000,
    no_show=False,
)


def plot_spheroscale_comparison(entries, alt_min=None, alt_max=None,
                                 n_haar=N_HAAR, show=True):
    """Overlay normalised 1D spheroscale histograms for multiple datasets.

    Parameters
    ----------
    entries : list of str
        Each entry is 'DATASET:VARIABLE' or 'DATASET:VARIABLE:EXPERIMENT'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ls_centres = np.sqrt(LS_EDGES[:-1] * LS_EDGES[1:])   # geometric midpoints

    for i, entry in enumerate(entries):
        parts = entry.split(':')
        dataset    = parts[0]
        variable   = parts[1]
        experiment = parts[2] if len(parts) > 2 else None

        l_s_v, _, _ = load_and_compute(
            dataset, variable, experiment,
            alt_min=alt_min, alt_max=alt_max, n_haar=n_haar,
        )

        counts, _ = np.histogram(l_s_v, bins=LS_EDGES, density=False)
        color = _COLORS[i % len(_COLORS)]
        label = f'{dataset} {variable}'
        if experiment:
            label += f' ({experiment})'

        ax.plot(ls_centres, counts, color=color, lw=1.8, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Spheroscale $l_s$ (m)')
    ax.set_ylabel('Count')
    ax.set_title(f'Spheroscale comparison (n={n_haar} Haar)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which='both')
    plt.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    tag = '_'.join(e.replace(':', '-') for e in entries)
    fname = FIGURES_DIR / f'spheroscale_comparison_{tag}.pdf'
    plt.savefig(fname, transparent=True)
    print(f"Saved: {fname}")

    if show:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(
        description='Compare spheroscale distributions across datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('entries', nargs='+',
                   help='One or more DATASET:VARIABLE[:EXPERIMENT] tokens')
    p.add_argument('--alt_min',  type=float, default=_DEFAULTS['alt_min'])
    p.add_argument('--alt_max',  type=float, default=_DEFAULTS['alt_max'])
    p.add_argument('--n_haar',   type=int,   default=N_HAAR,
                   help='Haar scale: number of points (even integer)')
    p.add_argument('--no_show',  action='store_true', default=_DEFAULTS['no_show'])
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_spheroscale_comparison(
        entries=args.entries,
        alt_min=args.alt_min,
        alt_max=args.alt_max,
        n_haar=args.n_haar,
        show=not args.no_show,
    )
