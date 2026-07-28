#!/usr/bin/env python3
"""Size distributions + fractal dimensions from the large-domain STEAM ensemble.

The channel geometry is too narrow for size-distribution estimation
(DeWitt & Garrett 2024b), so these statistics come from the nine
2048 x 1024 STEAM runs (run_steam_large.py), with no host comparison.
All nine tau > 1 masks are passed to objscale at once:
  - finite-domain size-distribution exponents for area and perimeter
    ('summed perimeter'),
  - individual fractal dimension (filled perimeter vs filled area),
  - ensemble correlation dimension.
Domains are periodic but the finite_* truncation correction treats edges
as truncating -- conservative, same convention as make_fractal.py.

Writes stats/large.npz and figs/large.png.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import netCDF4
import objscale

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
TAU_THRESHOLD = 1.0
DX = 3000.0

spec = importlib.util.spec_from_file_location(
    "cv_optical_depth",
    Path.home() / "code-and-data/cloudyview/cloudyview/optical_depth.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def masks_and_models():
    masks, models = [], []
    for path in sorted(RUNS.glob("steam_large_*.nc")):
        ds = netCDF4.Dataset(path)
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(np.float64)
        lwc = ds.variables["qc"][:] * 1000.0   # (x, y, z), g/kg
        iwc = ds.variables["qi"][:] * 1000.0
        ds.close()
        tau = cv.vertically_integrated_optical_depth(lwc, z, iwc=iwc)
        masks.append((tau > TAU_THRESHOLD).astype(np.float32))
        models.append(path.stem.replace("steam_large_", ""))
    return masks, models


def compute():
    masks, models = masks_and_models()
    print(f"{len(masks)} masks, cover: "
          + " ".join(f"{m.mean():.2f}" for m in masks))
    sizes = np.full(masks[0].shape, DX)
    out = {"models": np.array(models),
           "cover": np.array([m.mean() for m in masks])}

    for variable, key in (("area", "area"), ("summed perimeter", "perim")):
        exponent, (log_bins, log_counts) = objscale.finite_array_powerlaw_exponent(
            masks, variable, x_sizes=sizes, y_sizes=sizes, return_counts=True)
        out[f"{key}_exponent"] = exponent
        out[f"{key}_log_bins"] = log_bins
        out[f"{key}_log_counts"] = log_counts
        print(f"{variable} exponent: {exponent:.2f}")

    d_i, log_l, log_p = objscale.individual_fractal_dimension(
        masks, x_sizes=sizes, y_sizes=sizes, return_values=True)
    out.update(individual_dim=d_i, ifd_log_l=log_l, ifd_log_p=log_p)
    print(f"individual fractal dimension: {d_i:.2f}")

    d_2, bins, C_l = objscale.ensemble_correlation_dimension(
        masks, x_sizes=sizes, y_sizes=sizes,
        point_reduction_factor=100, return_C_l=True)
    out.update(ensemble_dim=d_2, C_bins=bins, C_l=C_l)
    print(f"ensemble correlation dimension: {d_2:.2f}")

    np.savez(STATS / "large.npz", **out)
    print("wrote stats/large.npz")


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
    d = np.load(STATS / "large.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].plot(d["area_log_bins"] - 6.0, d["area_log_counts"],
                 color="#3B6EA5", lw=1.3,
                 label=f"area, slope {d['area_exponent']:.2f}")
    axes[0].plot(d["perim_log_bins"] - 3.0, d["perim_log_counts"],
                 color="#C1683C", lw=1.3,
                 label=f"perimeter, slope {d['perim_exponent']:.2f}")
    axes[0].set(xlabel="log$_{10}$ area [km$^2$] / perim [km]",
                ylabel="log$_{10}$ counts",
                title="size distributions, truncation-corrected")
    axes[1].plot(d["ifd_log_l"], d["ifd_log_p"], "o", ms=2.5,
                 color="#3B6EA5",
                 label=f"$D_i$ = {d['individual_dim']:.2f}")
    axes[1].set(xlabel="log$_{10}$ length scale [m]",
                ylabel="log$_{10}$ filled perimeter [m]",
                title="individual fractal dimension")
    axes[2].loglog(d["C_bins"] / 1000.0, d["C_l"], color="#3B6EA5", lw=1.3,
                   label=f"$D_2$ = {d['ensemble_dim']:.2f}")
    axes[2].set(xlabel="r [km]", ylabel="correlation integral $C(r)$",
                title="ensemble correlation dimension")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "figs" / "large.png")
    print("wrote figs/large.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["compute", "figure"]
    if "compute" in wanted:
        compute()
    if "figure" in wanted:
        figure()
