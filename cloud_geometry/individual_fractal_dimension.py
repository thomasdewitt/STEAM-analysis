"""Individual (perimeter-area) fractal dimension of cloud fields.

Loads LWC + IWC for each of four LES datasets, computes vertically-integrated
optical depth, binarizes at a threshold, then computes the perimeter-area
fractal dimension via objscale.individual_fractal_dimension().

Output: Figures/cloud_individual_fractal_dimension_tau{threshold}.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_geometry.cloud_utils import (
    FIGURES_DIR, DATASET_COLORS, DATASET_LABELS, DATASETS, _load_binary_arrays
)


def plot_individual_fractal_dimension(tau_threshold=1.0, show=True):
    import objscale

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for ax, dataset in zip(axes.flat, DATASETS):
        color = DATASET_COLORS[dataset]
        label = DATASET_LABELS[dataset]

        print(f"\nLoading {dataset}...")
        binary_arrays, dx, _ = _load_binary_arrays(dataset, tau_threshold)

        shape = binary_arrays[0].shape
        x_sizes = np.full(shape, dx)
        y_sizes = np.full(shape, dx)

        min_area_m2 = 10 * dx**2   # 10 pixels in physical units
        print(f"  [{dataset}] computing individual fractal dimension "
              f"({len(binary_arrays)} arrays, shape {shape}, min_area={min_area_m2:.0f} m²)...")
        Df, uncertainty, log10_sqrta, log10_p = objscale.individual_fractal_dimension(
            binary_arrays, x_sizes=x_sizes, y_sizes=y_sizes,
            min_a=min_area_m2, return_values=True
        )
        print(f"  [{dataset}] Df = {Df:.3f} ± {uncertainty:.3f}")

        # Bin log10(sqrt(area)) into log-spaced bins, plot mean log10(perimeter) per bin
        mask = log10_sqrta >= 0.5 * np.log10(min_area_m2)
        x = log10_sqrta[mask]
        y = log10_p[mask]
        bins = np.linspace(x.min(), x.max(), 40)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        counts, _ = np.histogram(x, bins=bins)
        y_sums, _ = np.histogram(x, bins=bins, weights=y)
        valid = counts > 0
        ax.scatter(bin_centers[valid], y_sums[valid] / counts[valid],
                   s=14, alpha=0.85, color=color, linewidths=0)

        # Fit line via linear_regression (raw points, not binned means)
        (slope, intercept), _ = objscale.linear_regression(x, y)
        x_fit = np.array([x.min(), x.max()])
        ax.plot(x_fit, slope * x_fit + intercept, color=color, lw=1.5)

        ax.set_title(label, fontsize=9)
        ax.set_xlabel(r'$\log_{10}(\sqrt{A})$', fontsize=8)
        ax.set_ylabel(r'$\log_{10}(P)$', fontsize=8)
        ax.annotate(f'$D_f$ = {Df:.2f} ± {uncertainty:.2f}',
                    xy=(0.05, 0.92), xycoords='axes fraction',
                    fontsize=9, color=color, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    fig.suptitle(f'Individual Fractal Dimension  (τ threshold = {tau_threshold})',
                 fontsize=10, y=1.01)
    plt.tight_layout()

    outpath = FIGURES_DIR / f'cloud_individual_fractal_dimension_tau{tau_threshold}.pdf'
    plt.savefig(outpath, bbox_inches='tight', transparent=True)
    print(f"\nSaved: {outpath}")

    if show:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(description='Individual fractal dimension of cloud fields.')
    p.add_argument('--tau_threshold', type=float, default=1.0,
                   help='Optical depth threshold for binarization (default 1.0)')
    p.add_argument('--no_show', action='store_true',
                   help='Suppress plt.show() (useful in scripts/pipelines)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_individual_fractal_dimension(
        tau_threshold=args.tau_threshold,
        show=not args.no_show,
    )
