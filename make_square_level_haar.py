#!/usr/bin/env python3
"""Single-level horizontal Haar fluctuation across the 10-member squares.

Short diagnostic (Thomas, 2026-07-27): for each production case
(icon_lem, ukmo_ra1t), extract one height level from all 10 square
members (6144 km, dx = 3 km, c = 0.101), stack the members along a new
axis, and compute the mean absolute Haar fluctuation along x in ONE
scaleinvariance call (members and y pooled as independent realizations,
periodic in x). Panels: h, qt, flux; reference slopes H_h = 0.45 on the
scalars and 0 on the flux.

Writes figs/square/haar_level.png and stats/square_level_haar.npz.

Usage: python make_square_level_haar.py [level_metres]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance as si

HERE = Path(__file__).parent
RUNS = HERE / "runs"
LEVEL_M = float(sys.argv[1]) if len(sys.argv) > 1 else 7000.0
MODELS = ("icon_lem", "ukmo_ra1t")
N_MEMBERS = 3   # trimmed from 10 (Thomas, 2026-07-28: runtime)
DX = 3000.0
H_H = 0.45

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def stacked_level(model, name):
    """(members, x, y) at the level nearest LEVEL_M, plus the actual z."""
    slabs = []
    for m in range(N_MEMBERS):
        with netCDF4.Dataset(RUNS / f"steam_sq10_{model}_m{m:02d}.nc") as ds:
            ds.set_auto_mask(False)
            z = ds.variables["z"][:]
            k = int(np.argmin(np.abs(z - LEVEL_M)))
            slabs.append(np.asarray(ds.variables[name][:, :, k],
                                    dtype=np.float32))
    return np.stack(slabs, axis=0), float(z[k])


def main():
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    colors = {"icon_lem": "#1764ab", "ukmo_ra1t": "#e76f51"}
    z_used = None
    for name, ax in zip(("h", "qt", "flux"), axes):
        for model in MODELS:
            data, z_used = stacked_level(model, name)
            lags, F = si.haar_fluctuation(data, order=1.0, axis=1,
                                          periodic=True)
            out[f"{model}_{name}_lags"] = lags
            out[f"{model}_{name}_F"] = F
            ax.loglog(lags * DX / 1000, F, color=colors[model], lw=1.3,
                      label=model)
        # Reference slope pegged to the icon_lem central value.
        ref_lags = out[f"icon_lem_{name}_lags"] * DX / 1000
        ref_F = out[f"icon_lem_{name}_F"]
        mid = len(ref_lags) // 2
        slope = H_H if name != "flux" else 0.0
        span = 10 ** 0.9
        xs = np.array([ref_lags[mid] / span, ref_lags[mid] * span])
        ax.loglog(xs, ref_F[mid] * (xs / ref_lags[mid]) ** slope,
                  color="0.4", lw=0.9, ls="-.")
        ax.annotate("$H_h$" if name != "flux" else "$H=0$",
                    (xs[1], ref_F[mid] * span ** slope), fontsize=6.5,
                    color="0.4", xytext=(2, -2), textcoords="offset points")
        ax.set(xlabel="lag [km]", ylabel="$\\hat{M}_1(\\ell)$", title=name)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"Horizontal Haar fluctuation at z = {z_used:.0f} m "
                 f"({N_MEMBERS} square members each, pooled in one call)", fontsize=10)
    fig.tight_layout()
    (HERE / "figs" / "square").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figs" / "square" / "haar_level.png")
    np.savez(HERE / "stats" / "square_level_haar.npz",
             level_m=z_used, **out)
    print(f"wrote figs/square/haar_level.png (z = {z_used:.0f} m)")


if __name__ == "__main__":
    main()
