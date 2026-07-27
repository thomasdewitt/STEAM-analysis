#!/usr/bin/env python3
"""Correlation dimension and size distributions for the SQUARE-domain runs.

Same methodology as make_fractal.py (tau > 1 masks from vertically
integrated optical depth, objscale ensemble correlation dimension +
finite-domain area-distribution exponent), applied to the 2-member
square ensemble steam_square_icon_lem_snap{0,1} (6144 x 6144 km at
dx = 3 km, outer scale 1536 km, constant 10 m spheroscale, c = 0.1010).

The square geometry is the point: the channel is too narrow for size
distributions (DeWitt 2024b), so these domains supply the paper's cloud
size distributions and large-domain fractal metrics. Correlation lags
run from 2 px (6 km, pixel-scale bias accepted as in make_fractal.py)
to 0.33x the 6144 km domain.

Writes stats/fractal_square.npz and figs/square/fractal_square.png.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import objscale

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
TAU_THRESHOLD = 1.0
DX = 3000.0
MODEL = "icon_lem"
SNAPS = (0, 1)

spec = importlib.util.spec_from_file_location(
    "cv_optical_depth",
    Path.home() / "code-and-data/cloudyview/cloudyview/optical_depth.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def square_masks():
    import netCDF4
    masks = []
    for i in SNAPS:
        ds = netCDF4.Dataset(RUNS / f"steam_square_{MODEL}_snap{i}.nc")
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(np.float64)
        lwc = ds.variables["qc"][:] * 1000.0   # (x, y, z), g/kg
        iwc = ds.variables["qi"][:] * 1000.0
        ds.close()
        tau = cv.vertically_integrated_optical_depth(lwc, z, iwc=iwc)
        masks.append((tau > TAU_THRESHOLD).astype(np.float32))
    return masks


def compute():
    masks = square_masks()
    sizes = [np.full(m.shape, DX) for m in masks]
    dim, bins, C_l = objscale.ensemble_correlation_dimension(
        masks, x_sizes=sizes[0], y_sizes=sizes[0], minlength=2 * DX,
        point_reduction_factor=40, return_C_l=True)
    exponent, (log_bins, log_counts) = objscale.finite_array_powerlaw_exponent(
        masks, "area", x_sizes=sizes[0], y_sizes=sizes[0],
        return_counts=True)
    cover = float(np.mean([m.mean() for m in masks]))
    np.savez(STATS / "fractal_square.npz",
             dim=dim, C_bins=bins, C_l=C_l, exponent=exponent,
             sd_log_bins=log_bins, sd_log_counts=log_counts, cover=cover)
    print(f"square {MODEL}: D2={dim:.2f}, area exp={exponent:.2f}, "
          f"cover={cover:.2f}")
    print("wrote stats/fractal_square.npz")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
        "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })
    d = np.load(STATS / "fractal_square.npz")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    axes[0].loglog(d["C_bins"] / 1000.0, d["C_l"], color="#1764ab", lw=1.4,
                   label=f"$D_2$ = {float(d['dim']):.2f}")
    axes[0].set(xlabel="r [km]", ylabel="correlation integral $C(r)$",
                title=f"$\\tau>1$ mask correlation integral "
                      f"(square {MODEL}, {len(SNAPS)} members)")
    axes[1].plot(d["sd_log_bins"] - 6.0, d["sd_log_counts"],
                 color="#1764ab", lw=1.4,
                 label=f"slope = {float(d['exponent']):.2f}")
    axes[1].set(xlabel="log$_{10}$ area [km$^2$]", ylabel="log$_{10}$ counts",
                title=f"area distribution, truncation-corrected "
                      f"(cover = {float(d['cover']):.2f})")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figs" / "square").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figs" / "square" / "fractal_square.png")
    print("wrote figs/square/fractal_square.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["compute", "figure"]
    if "compute" in wanted:
        compute()
    if "figure" in wanted:
        figure()
