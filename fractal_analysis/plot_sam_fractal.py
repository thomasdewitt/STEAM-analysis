#!/usr/bin/env python3
"""Plot the SAM fractal metrics from compute_sam_fractal.py.

The same four panels as plot_fractal_metrics.py, built from the same panel
spec so the two figures are constructed identically and can be read against
each other. Both SAM cases share every panel: colour is the albedo threshold
(darker with increasing R) and line style is which LES, matching the
convention in the matched-LES figures.

Read the scaling functions before quoting any exponent. Each rests on a
single snapshot, and objscale warns where a size distribution had too few
populated bins -- at high cloud cover in particular the mask percolates and
the size distribution stops describing separate clouds at all.

Usage: python plot_sam_fractal.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from albedo import threshold_tag
from plot_fractal_metrics import (PANELS, THRESHOLD_COLORS, LABEL, positive,
                                  series, guide, style, write_table)

HERE = Path(__file__).resolve().parent
DATA = HERE / "sam_fractal_metrics.npz"
OUT = HERE / "sam_fractal_metrics"

CASE_NAME = {"twpice": "SAM-TWPICE", "gate": "SAM-GATE"}
CASE_STYLE = {"twpice": "-", "gate": (0, (4, 2))}

# Must match compute_sam_fractal.py. False drops the two size-distribution
# panels, which one snapshot per case supports poorly. If this is True but
# the npz was computed with it False, main() says so rather than guessing.
RUN_SAM_DISTRIBUTIONS = True


def draw_panel(ax, d, cases, thresholds, letter, metric, xk, yk, xlabel,
               ylabel, marker, rising):
    for case in cases:
        for R, color in zip(thresholds, THRESHOLD_COLORS):
            tag = f"{case}_{threshold_tag(R)}"
            x, y = positive(series(d, tag, xk), series(d, tag, yk))
            if x.size == 0:
                continue
            slope = float(d[f"{tag}_{metric}"])
            ax.plot(x, y, color=color, lw=1.3, ls=CASE_STYLE[case],
                    solid_capstyle="round", alpha=0.9)
            guide(ax, x, y, slope if rising else -slope, color)
    style(ax, letter, xlabel, ylabel)


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA.name} not found -- run "
                         f"compute_sam_fractal.py first")
    d = np.load(DATA, allow_pickle=False)
    cases = [str(c) for c in d["cases"]]
    thresholds = [float(R) for R in d["thresholds"]]

    computed = bool(d["distributions"]) if "distributions" in d else True
    if RUN_SAM_DISTRIBUTIONS and not computed:
        raise SystemExit(
            "RUN_SAM_DISTRIBUTIONS is True here but sam_fractal_metrics.npz "
            "was computed with it False, so tau_area and tau_per are not in "
            "the file. Set the flag the same way in both scripts, and rerun "
            "compute_sam_fractal.py if you want the distributions.")
    panels = PANELS if RUN_SAM_DISTRIBUTIONS else PANELS[:2]
    metrics = (("D_e", "D_f", "tau_area", "tau_per") if RUN_SAM_DISTRIBUTIONS
               else ("D_e", "D_f"))

    if RUN_SAM_DISTRIBUTIONS:
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    for ax, spec in zip(axes.ravel(), panels):
        draw_panel(ax, d, cases, thresholds, *spec)

    handles = [Line2D([0], [0], color=c, lw=1.6, label=f"$R > {R:g}$")
               for R, c in zip(thresholds, THRESHOLD_COLORS)]
    handles += [Line2D([0], [0], color=LABEL, lw=1.4, ls=CASE_STYLE[c],
                       label=CASE_NAME[c]) for c in cases]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               handlelength=1.8, bbox_to_anchor=(0.5, 1.05))

    header = (f"SAM cases at dx = {float(d['dx_km']):g} km, one snapshot "
              f"per case\nobjscale {str(d['objscale_version'])}, "
              f"point_reduction_factor "
              f"{int(d['point_reduction_factor'])}"
              + ("" if RUN_SAM_DISTRIBUTIONS
                 else "\nsize distributions not computed"))
    rows = [(f"{CASE_NAME[c]} R>{R:g}", f"{c}_{threshold_tag(R)}")
            for c in cases for R in thresholds]
    write_table(f"{OUT}.txt", header, rows, d, metrics)

    fig.tight_layout()
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.name}.pdf and {OUT.name}.png")


if __name__ == "__main__":
    main()
