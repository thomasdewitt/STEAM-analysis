#!/usr/bin/env python3
"""ampproj (per-class 3-condition op) vs hybrid (final strict op):
horizontal Haar fluctuation functions at 2, 6, 12 km, h and qt, with
H = 0.45 reference slopes anchored at each curve's Nyquist point.

Writes figs/archaeology/ampproj_vs_hybrid.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance as si

from curvature_operator_fig import amp_op_level, BOUNDS, SRC

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs" / "archaeology"
LEVELS = (2000, 6000, 12000)
H_DESIGN = 0.45
DX_KM = 3.0

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def haar(slab):
    lags, F = si.haar_fluctuation(slab, order=1.0, axis=0, periodic=True)
    return lags * DX_KM, F


def main():
    curves = {}   # (variant, name, z) -> (lags_km, F)
    with netCDF4.Dataset(RUNS / "curv_ampproj.nc") as ds:
        ds.set_auto_mask(False)
        z = ds.variables["z"][:]
        for name in ("h", "qt"):
            for L in LEVELS:
                iz = int(np.argmin(abs(z - L)))
                slab = np.asarray(ds.variables[name][:, :, iz],
                                  dtype=np.float32)
                curves[("ampproj", name, L)] = haar(slab)

    with netCDF4.Dataset(RUNS / "curv_renormtaper_noproj.nc") as ds:
        ds.set_auto_mask(False)
        z = ds.variables["z"][:]
        for name in ("h", "qt"):
            lo, hi, prof = BOUNDS[name]
            for L in LEVELS:
                iz = int(np.argmin(abs(z - L)))
                slab = np.asarray(ds.variables[name][:, :, iz],
                                  dtype=np.float32)
                mz = float(np.interp(z[iz], SRC["z_profile"], prof))
                d, _ = amp_op_level(slab - mz, lo, hi, mz)
                curves[("hybrid", name, L)] = haar(d + mz)

    colors = {"ampproj": "#2471a3", "hybrid": "#d4ac0d"}
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    for row, name in enumerate(("h", "qt")):
        for col, L in enumerate(LEVELS):
            ax = axes[row, col]
            for variant in ("ampproj", "hybrid"):
                lags_km, F = curves[(variant, name, L)]
                ax.loglog(lags_km, F, color=colors[variant], lw=1.4,
                          label=variant)
                # H = 0.45 reference anchored at the Nyquist point
                ref = F[0] * (lags_km / lags_km[0]) ** H_DESIGN
                ax.loglog(lags_km, ref, color=colors[variant], lw=0.8,
                          ls="--", alpha=0.55)
            ax.set_title(f"{name} at {L/1000:.0f} km")
            if row == 1:
                ax.set_xlabel("lag [km]")
            if col == 0:
                ax.set_ylabel("$\\hat{M}_1(\\ell)$")
            ax.grid(True, which="both", alpha=0.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Per-class (ampproj) vs final-only (hybrid) 3-condition "
                 "operator — dashed: $\\ell^{0.45}$ anchored at Nyquist",
                 fontsize=10.5)
    fig.tight_layout()
    out = HERE / "figs" / "archaeology" / "ampproj_vs_hybrid.png"
    fig.savefig(out)
    print("wrote", out.relative_to(HERE))


if __name__ == "__main__":
    main()
