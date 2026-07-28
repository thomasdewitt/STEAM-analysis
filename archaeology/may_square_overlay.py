#!/usr/bin/env python3
"""Overlay: May checkpoint at production config vs tonight's production.

Single-level horizontal Haar at ~7 km, May members pooled in ONE
scaleinvariance call (same estimator as make_square_level_haar.py),
overlaid on the regenerated production icon_lem curves from
stats/square_level_haar.npz. Legend carries the 9-150 km fit slopes.

Run with the turbulon-analysis venv AFTER may_production_benchmark.py.
Writes figs/archaeology/may_production_benchmark.png and
stats/may_production_benchmark.npz.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance as si

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs" / "archaeology"
LEVEL_M = 7000.0
DX = 3000.0
N_MEMBERS = 3
FIT_KM = (9.0, 150.0)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def fit_slope(lags_km, F):
    m = (lags_km >= FIT_KM[0]) & (lags_km <= FIT_KM[1]) & np.isfinite(F)
    return float(np.polyfit(np.log(lags_km[m]), np.log(F[m]), 1)[0])


def main():
    slabs = {"h": [], "qt": []}
    z_used = None
    for m in range(N_MEMBERS):
        with netCDF4.Dataset(RUNS / f"may_sq_m{m:02d}.nc") as ds:
            ds.set_auto_mask(False)
            z = ds.variables["z"][:]
            k = int(np.argmin(np.abs(z - LEVEL_M)))
            z_used = float(z[k])
            for name in slabs:
                slabs[name].append(
                    np.asarray(ds.variables[name][:, :, k], dtype=np.float32))

    prod = np.load(HERE / "stats" / "square_level_haar.npz")
    out = {"level_m": z_used}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, name in zip(axes, ("h", "qt")):
        data = np.stack(slabs[name], axis=0)
        lags, F = si.haar_fluctuation(data, order=1.0, axis=1, periodic=True)
        lags_km = lags * DX / 1000
        s_may = fit_slope(lags_km, F)
        out[f"{name}_lags"], out[f"{name}_F"] = lags, F

        pl = prod[f"icon_lem_{name}_lags"] * DX / 1000
        pF = prod[f"icon_lem_{name}_F"]
        s_prod = fit_slope(pl, pF)

        ax.loglog(lags_km, F, color="#6b6b6b", lw=1.4,
                  label=f"May ac16b35 ({s_may:+.2f})")
        ax.loglog(pl, pF, color="#c0392b", lw=1.4,
                  label=f"production, fixed HEAD ({s_prod:+.2f})")
        mid = len(lags_km) // 2
        span = 10.0 ** 0.9
        xs = np.array([lags_km[mid] / span, lags_km[mid] * span])
        ax.loglog(xs, F[mid] * (xs / lags_km[mid]) ** 0.45,
                  color="0.35", lw=0.9, ls="-.")
        ax.annotate("$H_h=0.45$", (xs[1], F[mid] * span ** 0.45),
                    fontsize=6.5, color="0.35", xytext=(2, -2),
                    textcoords="offset points")
        ax.set(xlabel="lag [km]", ylabel="$\\hat{M}_1(\\ell)$", title=name)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"May checkpoint vs fixed HEAD at the production config "
                 f"(z = {z_used:.0f} m; slopes fit {FIT_KM[0]:.0f}-"
                 f"{FIT_KM[1]:.0f} km)", fontsize=10)
    fig.tight_layout()
    figdir = HERE / "figs" / "archaeology"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "may_production_benchmark.png")
    np.savez(HERE / "stats" / "may_production_benchmark.npz", **out)
    print("wrote figs/archaeology/may_production_benchmark.png")


if __name__ == "__main__":
    main()
