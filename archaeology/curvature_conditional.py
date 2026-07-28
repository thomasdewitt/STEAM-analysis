#!/usr/bin/env python3
"""Clip-conditioned Haar: is the flattening confined to near-bound regions?

Thomas 2026-07-28: small scales should be unaffected by the clip in
regions where the mean is far from the clip. Two tests on the existing
production HEAD m00 vs the May-checkpoint m00 (identical config):

1. Slope-vs-height sweep: Haar slope (9-150 km, plus 9-40 sub-band) at a
   ladder of z levels, h and qt, both codes. If HEAD's qt flattening
   tracks the dry upper levels (field pinned at qt_min = 0) but matches
   May at moist low levels, the bounds machinery is the driver.

2. Row-conditioned Haar at ~7 km: y-rows quartiled by dryness (fraction
   of cells with qt < 1e-5); pooled Haar per quartile (rows stacked, one
   scaleinvariance call). Wet-quartile HEAD vs May isolates the far-from-
   clip small-scale behavior.

Writes figs/archaeology/curvature_conditional.png and
stats/curvature_conditional.npz.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance as si

HERE = Path(__file__).resolve().parent.parent
FILES = {
    "HEAD": HERE / "runs" / "steam_sq10_icon_lem_m00.nc",
    "May": HERE / "runs" / "archaeology" / "may_sq_m00.nc",
}
DX = 3000.0
LEVELS_M = (1000, 2000, 4000, 7000, 10000, 12000)
BAND_FULL = (9.0, 150.0)
BAND_SMALL = (9.0, 40.0)
DRY_THRESH = 1e-5

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
COLORS = {"HEAD": "#c0392b", "May": "#6b6b6b"}


def fit_slope(lags_km, F, band):
    m = (lags_km >= band[0]) & (lags_km <= band[1]) & np.isfinite(F) & (F > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(lags_km[m]), np.log(F[m]), 1)[0])


def haar_x(data2d_or_3d, axis):
    lags, F = si.haar_fluctuation(data2d_or_3d, order=1.0, axis=axis,
                                  periodic=True)
    return lags * DX / 1000, F


def main():
    out = {}
    # ---- Test 1: slope vs height -----------------------------------
    slopes = {}   # (code, name, band) -> list over levels
    for code, path in FILES.items():
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            z = ds.variables["z"][:]
            for name in ("h", "qt"):
                for band_key in ("full", "small"):
                    slopes[(code, name, band_key)] = []
                for lev in LEVELS_M:
                    iz = int(np.argmin(np.abs(z - lev)))
                    slab = np.asarray(ds.variables[name][:, :, iz],
                                      dtype=np.float32)
                    lags_km, F = haar_x(slab, axis=0)
                    slopes[(code, name, "full")].append(
                        fit_slope(lags_km, F, BAND_FULL))
                    slopes[(code, name, "small")].append(
                        fit_slope(lags_km, F, BAND_SMALL))
    for key, val in slopes.items():
        out["slope_" + "_".join(key)] = np.array(val)
    out["levels_m"] = np.array(LEVELS_M, dtype=float)

    # ---- Test 2: dryness-quartile rows at ~7 km --------------------
    quart = {}    # (code, q) -> (lags_km, F); dryness from each code's own field
    for code, path in FILES.items():
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            z = ds.variables["z"][:]
            iz = int(np.argmin(np.abs(z - 7000)))
            qt = np.asarray(ds.variables["qt"][:, :, iz], dtype=np.float32)
        dryness = (qt < DRY_THRESH).mean(axis=0)          # per y-row
        order = np.argsort(dryness)
        nq = qt.shape[1] // 4
        for qi, sl in enumerate((order[:nq], order[-nq:])):  # wettest, driest
            rows = qt[:, np.sort(sl)].T                    # (nrows, nx)
            lags_km, F = haar_x(rows, axis=1)
            qname = "wet" if qi == 0 else "dry"
            quart[(code, qname)] = (lags_km, F)
            out[f"q_{code}_{qname}_lags_km"] = lags_km
            out[f"q_{code}_{qname}_F"] = F
            out[f"dryfrac_{code}_{qname}"] = float(
                dryness[sl].mean())

    # ---- figure ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    ax = axes[0]
    for code in FILES:
        for name, ls in (("h", "-"), ("qt", "--")):
            ax.plot(slopes[(code, name, "full")], np.array(LEVELS_M) / 1000,
                    ls, color=COLORS[code], marker="o", ms=3,
                    label=f"{code} {name}")
    ax.axvline(0.45, color="0.4", lw=0.8, ls=":")
    ax.set(xlabel=f"Haar slope, {BAND_FULL[0]:.0f}-{BAND_FULL[1]:.0f} km",
           ylabel="z [km]", title="slope vs height (full band)")
    ax.grid(True, alpha=0.5)
    ax.legend(fontsize=7)

    ax = axes[1]
    for code in FILES:
        for name, ls in (("h", "-"), ("qt", "--")):
            ax.plot(slopes[(code, name, "small")], np.array(LEVELS_M) / 1000,
                    ls, color=COLORS[code], marker="o", ms=3,
                    label=f"{code} {name}")
    ax.axvline(0.45, color="0.4", lw=0.8, ls=":")
    ax.set(xlabel=f"Haar slope, {BAND_SMALL[0]:.0f}-{BAND_SMALL[1]:.0f} km",
           ylabel="z [km]", title="slope vs height (small-scale band)")
    ax.grid(True, alpha=0.5)

    ax = axes[2]
    for code in FILES:
        for qname, ls in (("wet", "-"), ("dry", ":")):
            lags_km, F = quart[(code, qname)]
            s = fit_slope(lags_km, F, BAND_FULL)
            ax.loglog(lags_km, F, ls, color=COLORS[code], lw=1.3,
                      label=f"{code} {qname} rows ({s:+.2f})")
    ax.set(xlabel="lag [km]", ylabel="$\\hat{M}_1(\\ell)$",
           title="qt at 7 km by dryness quartile")
    ax.grid(True, which="both", alpha=0.5)
    ax.legend(fontsize=7)

    fig.suptitle("Clip-conditioned curvature: HEAD m00 vs May m00 "
                 "(identical config, seed 2000)", fontsize=10)
    fig.tight_layout()
    figdir = HERE / "figs" / "archaeology"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "curvature_conditional.png")
    np.savez(HERE / "stats" / "curvature_conditional.npz", **out)
    print("wrote figs/archaeology/curvature_conditional.png")
    for code in FILES:
        for name in ("h", "qt"):
            print(f"{code:5s} {name:2s} full-band slopes by z:",
                  np.round(slopes[(code, name, 'full')], 3))


if __name__ == "__main__":
    main()
