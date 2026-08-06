#!/usr/bin/env python3
"""Plot the MODIS fractal metrics from compute_modis_fractal.py.

The same four panels as plot_fractal_metrics.py and plot_sam_fractal.py,
built from the same panel spec so all three figures are constructed
identically and can be read against each other.

Both solar-zenith conventions share every panel where both have been
computed: colour is the reflectance threshold (darker with increasing R),
line style is the convention. Drawing them together is the point -- the
correction shifts cloud cover substantially at a fixed threshold, and the
question the figure answers is whether it also moves the exponents, which
is a different and much weaker claim.

Usage: python plot_modis_fractal.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from albedo import threshold_tag
from compute_modis_fractal import out_path
from plot_fractal_metrics import (PANELS, THRESHOLD_COLORS, LABEL, positive,
                                  series, guide, style, write_table)

HERE = Path(__file__).resolve().parent
OUT = HERE / "modis_fractal_metrics"

CONVENTIONS = ((False, "as stored", "-"),
               (True, r"$/\cos\theta_0$", (0, (4, 2))))


def draw_panel(ax, loaded, thresholds, letter, metric, xk, yk, xlabel,
               ylabel, marker, rising):
    for correction, _, ls in CONVENTIONS:
        d = loaded.get(correction)
        if d is None:
            continue
        for R, color in zip(thresholds, THRESHOLD_COLORS):
            tag = threshold_tag(R)
            x, y = positive(series(d, tag, xk), series(d, tag, yk))
            if x.size == 0:
                continue
            slope = float(d[f"{tag}_{metric}"])
            if marker is None:
                ax.plot(x, y, color=color, lw=1.3, ls=ls,
                        solid_capstyle="round", alpha=0.9)
            else:
                ax.plot(x, y, lw=0, marker="o", ms=marker, mfc=color,
                        mec="none", alpha=0.55 if correction else 0.85)
            guide(ax, x, y, slope if rising else -slope, color)
    style(ax, letter, xlabel, ylabel)


def main():
    loaded = {}
    for correction, _, _ in CONVENTIONS:
        path = out_path(correction)
        if path.exists():
            loaded[correction] = np.load(path, allow_pickle=False)
    if not loaded:
        raise SystemExit("no modis_fractal_metrics*.npz -- run "
                         "compute_modis_fractal.py first")

    any_d = next(iter(loaded.values()))
    thresholds = [float(R) for R in any_d["thresholds"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2))
    for ax, spec in zip(axes.ravel(), PANELS):
        draw_panel(ax, loaded, thresholds, *spec)

    handles = [Line2D([0], [0], color=c, lw=1.6, label=f"$R > {R:g}$")
               for R, c in zip(thresholds, THRESHOLD_COLORS)]
    handles += [Line2D([0], [0], color=LABEL, lw=1.4, ls=ls, label=name)
                for correction, name, ls in CONVENTIONS if correction in loaded]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               handlelength=1.8, bbox_to_anchor=(0.5, 1.05))

    lo, hi = any_d["x_size_km_range"]
    header = (f"MODIS band 1, {int(any_d['n_granules'])} granules pooled; "
              f"sensor zenith <= {float(any_d['max_sensor_zenith']):g} deg, "
              f"footprints {float(lo):.2f}-{float(hi):.2f} km along scan\n"
              f"objscale {str(any_d['objscale_version'])}, "
              f"point_reduction_factor "
              f"{int(any_d['point_reduction_factor'])}")
    rows, table = [], {}
    for correction, name, _ in CONVENTIONS:
        d = loaded.get(correction)
        if d is None:
            continue
        prefix = "sza" if correction else "raw"
        for R in thresholds:
            tag, key = threshold_tag(R), f"{prefix}_{threshold_tag(R)}"
            rows.append((f"{'cos-corrected' if correction else 'as stored'} "
                         f"R>{R:g}", key))
            for m in ("cover", "D_e", "D_f", "tau_area", "tau_per"):
                table[f"{key}_{m}"] = d[f"{tag}_{m}"]
    write_table(f"{OUT}.txt", header, rows, table,
                ("D_e", "D_f", "tau_area", "tau_per"))

    fig.tight_layout()
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.name}.pdf and {OUT.name}.png")


if __name__ == "__main__":
    main()
