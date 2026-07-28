#!/usr/bin/env python3
"""Analyze the curvature probes (curvature_probe.py outputs).

Figure 1 (curvature_variants.png): final-field Haar at ~7 km, h and qt,
for every completed variant, overlaid on the May m00 curve — which
toggle recovers May's straightness?

Figures 2-3 (curvature_internals_{h,qt}.png), stock variant:
  A: realized deposit amplitude ladder <|W|> at ~7 km vs class scale k
     (taper+C_k+retention included) with the design k^H guide.
  B: per-class deposited-increment Haar curves on the class's own grid.
  C: partial-sum (post-projection) Haar buildup, class by class.

Prints taper activity and projection displacement per class.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance as si

HERE = Path(__file__).resolve().parent.parent
STATS = HERE / "stats"
RUNS = HERE / "runs" / "archaeology"
VARIANTS = ("stock", "nocomp", "nobounds", "nocomp_nobounds",
            "renormtaper", "b1", "b2")
VCOLORS = {"stock": "#c0392b", "nocomp": "#2471a3",
           "nobounds": "#229954", "nocomp_nobounds": "#af7ac5",
           "renormtaper": "#e67e22", "b1": "#16a085", "b2": "#7f8c8d"}
BAND_FULL = (9.0, 150.0)
BAND_SMALL = (9.0, 40.0)
H_DESIGN = 0.45

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def fit_slope(lags_km, F, band):
    m = (lags_km >= band[0]) & (lags_km <= band[1]) & np.isfinite(F) & (F > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(lags_km[m]), np.log(F[m]), 1)[0])


def haar(slab, dx_m):
    lags, F = si.haar_fluctuation(slab, order=1.0, axis=0, periodic=True)
    return lags * dx_m / 1000, F


def main():
    done = [v for v in VARIANTS if (STATS / f"curv_{v}.npz").exists()
            and (RUNS / f"curv_{v}.nc").exists()]
    print("variants:", done)

    may = np.load(STATS / "may_production_benchmark.npz")

    # ---------------- Figure 1: variant final fields ----------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, name in zip(axes, ("h", "qt")):
        ml = may[f"{name}_lags"] * 3000.0 / 1000
        mF = may[f"{name}_F"]
        ax.loglog(ml, mF, color="#6b6b6b", lw=1.6,
                  label=f"May ({fit_slope(ml, mF, BAND_FULL):+.2f} / "
                        f"{fit_slope(ml, mF, BAND_SMALL):+.2f})")
        for v in done:
            with netCDF4.Dataset(RUNS / f"curv_{v}.nc") as ds:
                ds.set_auto_mask(False)
                z = ds.variables["z"][:]
                iz = int(np.argmin(np.abs(z - 7000)))
                slab = np.asarray(ds.variables[name][:, :, iz],
                                  dtype=np.float32)
                dxm = float(ds.dx)
            lags_km, F = haar(slab, dxm)
            ax.loglog(lags_km, F, color=VCOLORS[v], lw=1.2,
                      label=f"{v} ({fit_slope(lags_km, F, BAND_FULL):+.2f} / "
                            f"{fit_slope(lags_km, F, BAND_SMALL):+.2f})")
        mid = np.searchsorted(lags_km, 30.0)
        xs = np.array([lags_km[mid] / 3, lags_km[mid] * 3])
        ax.loglog(xs, F[mid] * (xs / lags_km[mid]) ** H_DESIGN,
                  color="0.35", lw=0.9, ls="-.")
        ax.set(xlabel="lag [km]", ylabel="$\\hat{M}_1(\\ell)$", title=name)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=6.5, title="(9-150 / 9-40 km slopes)",
                  title_fontsize=6.5)
    fig.suptitle("Machinery toggles at the production config, ~7 km "
                 "(seed 2000; May reference)", fontsize=10)
    fig.tight_layout()
    figdir = HERE / "figs" / "archaeology"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "curvature_variants.png")
    print("wrote figs/archaeology/curvature_variants.png")

    # ---------------- Figures 2-3: stock internals ------------------
    if "stock" not in done:
        return
    d = np.load(STATS / "curv_stock.npz")
    k = d["k_values"]
    dx_k = d["dx_k"]
    n = len(k)
    for name in ("h", "qt"):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
        cmap = plt.cm.viridis(np.linspace(0, 0.9, n))

        ax = axes[0]
        amp = d[f"amp_{name}"]
        ax.loglog(k / 1000, amp, "o-", color="#c0392b", ms=4)
        ref = amp[n // 2] * (k / k[n // 2]) ** H_DESIGN
        ax.loglog(k / 1000, ref, "-.", color="0.35", lw=0.9,
                  label=f"$k^{{{H_DESIGN}}}$")
        slope_amp = np.polyfit(np.log(k), np.log(amp), 1)[0]
        ax.set(xlabel="class scale k [km]", ylabel="$\\langle|W|\\rangle$",
               title=f"realized deposit ladder ({slope_amp:+.2f})")
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=7)

        ax = axes[1]
        for i in range(n):
            lags_km, F = haar(d[f"inc_{name}_{i}"], dx_k[i])
            ax.loglog(lags_km, F, color=cmap[i], lw=1.0,
                      label=f"k={k[i]/1000:.0f} km")
            ax.axvline(k[i] / 1000, color=cmap[i], lw=0.5, alpha=0.3)
        ax.set(xlabel="lag [km]", ylabel="$\\hat{M}_1(\\ell)$",
               title="per-class deposited increments")
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=5.5, ncol=2)

        ax = axes[2]
        for i in range(n):
            lags_km, F = haar(d[f"ps_post_{name}_{i}"], dx_k[i])
            ax.loglog(lags_km, F, color=cmap[i], lw=1.0)
        xs = np.array([10.0, 300.0])
        ax.loglog(xs, F[np.searchsorted(lags_km, 30.0)]
                  * (xs / 30.0) ** H_DESIGN, "-.", color="0.35", lw=0.9)
        ax.set(xlabel="lag [km]", ylabel="$\\hat{M}_1(\\ell)$",
               title="partial-sum buildup (post-projection)")
        ax.grid(True, which="both", alpha=0.5)

        fig.suptitle(f"stock internals at ~7 km: {name}", fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / f"curvature_internals_{name}.png")
        print(f"wrote figs/archaeology/curvature_internals_{name}.png")

        taper = d[f"taper_{name}"]
        print(f"\n{name}: taper at ~7 km per class "
              f"(mean g, frac g<1, frac g=0); proj rms move")
        for i in range(n):
            pre = d[f"ps_pre_{name}_{i}"]
            post = d[f"ps_post_{name}_{i}"]
            rms = float(np.sqrt(np.mean((post - pre) ** 2)))
            print(f"  k={k[i]/1000:7.0f} km  g={taper[i][0]:.3f}  "
                  f"<1:{taper[i][1]:.3f}  =0:{taper[i][2]:.3f}  "
                  f"proj_rms={rms:.3e}")


if __name__ == "__main__":
    main()
