"""Ensemble correlation dimension of cloud fields.

Loads LWC + IWC for each of four LES datasets, computes vertically-integrated
optical depth, binarizes at a threshold, then computes the ensemble correlation
dimension via objscale.ensemble_correlation_dimension().

Output: Figures/cloud_correlation_dimension_tau{threshold}.pdf
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


def plot_correlation_dimension(tau_threshold=1.0, show=True):
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

        print(f"  [{dataset}] computing correlation dimension "
              f"({len(binary_arrays)} arrays, shape {shape})...")
        dim, error, bins, C_l = objscale.ensemble_correlation_dimension(
            binary_arrays, x_sizes=x_sizes, y_sizes=y_sizes, return_C_l=True
        )
        print(f"  [{dataset}] D_corr = {dim:.3f} ± {error:.3f}")

        # Log-log line: C(l) vs scale
        ax.loglog(bins, C_l, color=color, alpha=0.9, lw=1.5)

        # Fit the lower and upper thirds of the valid scale range separately
        log10_bins = np.log10(bins)
        log10_C_l  = np.log10(C_l)
        valid = np.isfinite(log10_bins) & np.isfinite(log10_C_l)
        valid_idx = np.where(valid)[0]

        annot_lines = []
        if len(valid_idx) >= 9:
            n3 = len(valid_idx) // 3
            for thirds, shade, label_prefix in [
                (valid_idx[:n3],        0.45, 'low'),
                (valid_idx[n3:2*n3],    0.70, 'mid'),
                (valid_idx[-n3:],       0.95, 'high'),
            ]:
                x = log10_bins[thirds]
                y = log10_C_l[thirds]
                (slope, intercept), (slope_err, _) = objscale.linear_regression(x, y)
                C_fit = 10 ** (slope * x + intercept)
                ax.loglog(bins[thirds], C_fit, '--', color=color, lw=1.5, alpha=shade)
                annot_lines.append(
                    f'$D_{{\\rm {label_prefix}}}$ = {slope:.2f} ± {slope_err:.2f}'
                )
                print(f"  [{dataset}] D_{label_prefix} = {slope:.3f} ± {slope_err:.3f}  "
                      f"(scales {bins[thirds[0]]:.0f}–{bins[thirds[-1]]:.0f} m)")

        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Scale (m)', fontsize=8)
        ax.set_ylabel('C(l)', fontsize=8)
        annot_text = '\n'.join(annot_lines) if annot_lines else f'$D$ = {dim:.2f} ± {error:.2f}'
        ax.annotate(annot_text,
                    xy=(0.05, 0.93), xycoords='axes fraction',
                    fontsize=8, color=color, fontweight='bold',
                    va='top', linespacing=1.8)
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(labelsize=7)

    fig.suptitle(f'Correlation Dimension  (τ threshold = {tau_threshold})',
                 fontsize=10, y=1.01)
    plt.tight_layout()

    outpath = FIGURES_DIR / f'cloud_correlation_dimension_tau{tau_threshold}.pdf'
    plt.savefig(outpath, bbox_inches='tight', transparent=True)
    print(f"\nSaved: {outpath}")

    if show:
        plt.show()


def _parse_args():
    p = argparse.ArgumentParser(description='Ensemble correlation dimension of cloud fields.')
    p.add_argument('--tau_threshold', type=float, default=1.0,
                   help='Optical depth threshold for binarization (default 1.0)')
    p.add_argument('--no_show', action='store_true',
                   help='Suppress plt.show() (useful in scripts/pipelines)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    plot_correlation_dimension(
        tau_threshold=args.tau_threshold,
        show=not args.no_show,
    )
