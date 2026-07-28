#!/usr/bin/env python3
"""The 3-condition operator: slope sweeps + where the clip actually fires.

Panels A/B: Haar slope (9-150 km) vs height for qt and h — May, stock,
per-class ampproj, rt_noproj + plain final projection, and the hybrid
(rt_noproj + final amplitude-preserving op).

Panels C/D: per-level diagnostics of the FINAL amplitude-preserving op
applied to the rt_noproj cascade: target amplitude A0 (pre-clip
mean-abs), feasibility ceiling 2*min(<phi>-lo, hi-<phi>), delivered
amplitude <|delta|>, and the fraction of cells sitting ON a cap after
the final clip ("clip-active").

Writes figs/archaeology/curvature_operator.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance as si

import importlib
sm = importlib.import_module("steam.simulate")
from steam.thermodynamics import _saturation_mixing_ratio

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
SRC = np.load(HERE / "stats" / "icon_lem_snap0.npz")
CP, LV = 1004.0, 2.5e6
SP = float(SRC["surface_pressure"])
QTS = float(_saturation_mixing_ratio(300.0, SP))
H_UP = max(CP * 300.0 + LV * QTS, float(SRC["h_profile"].max()) + 1.0)
H_LO = float(SRC["h_profile"].min()) - 10.0 * CP
BOUNDS = {"h": (H_LO, H_UP, SRC["h_profile"]),
          "qt": (0.0, QTS, SRC["qt_profile"])}
SLOPE_LEVELS = (1000, 2000, 4000, 7000, 10000, 12000)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def slope(slab):
    lags, F = si.haar_fluctuation(slab, order=1.0, axis=0, periodic=True)
    lk = lags * 3.0
    m = (lk >= 9) & (lk <= 150)
    return float(np.polyfit(np.log(lk[m]), np.log(F[m]), 1)[0])


def amp_op_level(pert, lo, hi, mz, n_iter=10):
    """Final amplitude-preserving bounded op on one level; diagnostics."""
    cl = np.float32(lo - mz)
    ch = np.float32(hi - mz)
    d = pert.astype(np.float32).copy()
    a0 = float(np.abs(d).mean(dtype=np.float64))
    if a0 <= 0:
        return d, dict(a0=0.0, delivered=0.0, clip_frac=0.0)
    for _ in range(n_iter):
        d -= np.float32(d.mean(dtype=np.float64))
        np.clip(d, cl, ch, out=d)
        m_abs = float(np.abs(d).mean(dtype=np.float64))
        if m_abs <= 0:
            break
        s = min(a0 / m_abs, 2.0)
        if abs(s - 1.0) < 1e-4:
            break
        d *= np.float32(s)
    # STRICT zero mean: demean-clip to convergence (condition 1 exact;
    # near the ceiling this is what caps delivered amplitude).
    for _ in range(40):
        mu = float(d.mean(dtype=np.float64))
        if abs(mu) < 1e-9 * max(a0, 1e-30):
            break
        d -= np.float32(mu)
        np.clip(d, cl, ch, out=d)
    delivered = float(np.abs(d).mean(dtype=np.float64))
    clip_frac = float(((d == cl) | (d == ch)).mean())
    return d, dict(a0=a0, delivered=delivered, clip_frac=clip_frac)


def main():
    # ---------------- slope sweeps -------------------------------
    sweeps = {}
    plain = {
        "May": RUNS / "archaeology" / "may_sq_m00.nc",
        "stock": RUNS / "archaeology" / "curv_stock.nc",
        "ampproj (per-class)": RUNS / "archaeology" / "curv_ampproj.nc",
    }
    for tag, path in plain.items():
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            z = ds.variables["z"][:]
            for name in ("h", "qt"):
                sweeps[(tag, name)] = [
                    slope(np.asarray(
                        ds.variables[name][:, :, int(np.argmin(abs(z - L)))],
                        dtype=np.float32))
                    for L in SLOPE_LEVELS]

    with netCDF4.Dataset(RUNS / "archaeology" / "curv_renormtaper_noproj.nc") as ds:
        ds.set_auto_mask(False)
        z = ds.variables["z"][:]
        for name in ("h", "qt"):
            lo, hi, prof = BOUNDS[name]
            plainproj, hybrid = [], []
            for L in SLOPE_LEVELS:
                iz = int(np.argmin(abs(z - L)))
                slab = np.asarray(ds.variables[name][:, :, iz],
                                  dtype=np.float32)
                mz = float(np.interp(z[iz], SRC["z_profile"], prof))
                pert = (slab - mz)[:, :, None].copy()
                sm._project_onto_bounds(pert, np.array([np.float32(mz)]),
                                        lo, hi)
                plainproj.append(slope(pert[:, :, 0] + mz))
                d, _ = amp_op_level(slab - mz, lo, hi, mz)
                hybrid.append(slope(d + mz))
            sweeps[("final plain proj", name)] = plainproj
            sweeps[("hybrid (final amp-op)", name)] = hybrid

        # ------------- operator diagnostics, ALL levels ----------
        diag = {}
        for name in ("h", "qt"):
            lo, hi, prof = BOUNDS[name]
            rows = []
            for iz in range(z.size):
                slab = np.asarray(ds.variables[name][:, :, iz],
                                  dtype=np.float32)
                mz = float(np.interp(z[iz], SRC["z_profile"], prof))
                _, dd = amp_op_level(slab - mz, lo, hi, mz)
                ceiling = 2.0 * min(mz - lo, hi - mz)
                rows.append((dd["a0"], dd["delivered"], ceiling,
                             dd["clip_frac"]))
            diag[name] = np.array(rows)

    # ---------------- figure -------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    zs = np.array(SLOPE_LEVELS) / 1000
    colors = {"May": "#6b6b6b", "stock": "#c0392b",
              "ampproj (per-class)": "#2471a3",
              "final plain proj": "#7f8c8d",
              "hybrid (final amp-op)": "#d4ac0d"}
    for ax, name in zip(axes[0], ("qt", "h")):
        for tag in colors:
            ax.plot(sweeps[(tag, name)], zs, "o-", ms=3, lw=1.3,
                    color=colors[tag], label=tag)
        ax.axvline(0.45, color="0.4", lw=0.8, ls=":")
        ax.set(xlabel="Haar slope 9-150 km", ylabel="z [km]",
               title=f"{name}: slope vs height")
        ax.grid(True, alpha=0.5)
        if name == "qt":
            ax.legend(fontsize=7, loc="lower right")

    zf = z / 1000
    for ax, name in zip(axes[1], ("qt", "h")):
        d = diag[name]
        ax.semilogx(d[:, 0], zf, color="#2471a3", lw=1.3, label="target $A_0$")
        ax.semilogx(d[:, 1], zf, color="#d4ac0d", lw=1.3, ls="--",
                    label="delivered")
        ax.semilogx(d[:, 2], zf, color="#c0392b", lw=1.1, ls=":",
                    label="ceiling $2\\min(\\bar\\varphi{-}lo,\\,hi{-}\\bar\\varphi)$")
        ax.set(xlabel="amplitude", ylabel="z [km]",
               title=f"{name}: final amp-op diagnostics")
        ax.grid(True, which="both", alpha=0.5)
        ax2 = ax.twiny()
        ax2.plot(d[:, 3], zf, color="#229954", lw=1.0, alpha=0.8)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("clip-active fraction (green)", fontsize=8,
                       color="#229954")
        ax2.tick_params(colors="#229954", labelsize=7)
        ax.legend(fontsize=7, loc="center right")

    fig.suptitle("The 3-condition operator: slopes and where the clip fires "
                 "(rt_noproj cascade, seed 2000)", fontsize=10.5)
    fig.tight_layout()
    out = HERE / "figs" / "archaeology" / "curvature_operator.png"
    fig.savefig(out)
    print("wrote", out.relative_to(HERE))
    for name in ("h", "qt"):
        d = diag[name]
        lim = d[:, 1] < 0.98 * d[:, 0]
        print(f"{name}: levels with delivered < 0.98*A0: {lim.sum()}/{len(d)} "
              f"(z = {', '.join(f'{v:.1f}' for v in zf[lim][:12])} km)")


if __name__ == "__main__":
    main()
