#!/usr/bin/env python3
"""Do the nested refinements deliver the designed scaling?

Order-1 Haar fluctuations + local exponents for both refinement levels of
runs/render_nests_steam_sq10_<model>_m00.nc, horizontal and vertical.

Estimator: Haar (scaleinvariance.haar_fluctuation, order 1, periodic=False).
The nests are aperiodic (periodic_x = periodic_y = 0), and the Mexican-hat
kernel used by make_local_slope.py is ~5.8r wide, so on 512 aperiodic cells
it sheds every lag past ~90 cells -- most of render_a's range. Haar's kernel
is exactly r wide, keeps the full lag range, and is what the parent-square
cache (stats/square_level_haar.npz) already holds, so the parent curves can
be overlaid on the same axes. Haar is valid for -1 < H < 1, which covers
H_h = 0.45 and H_v = 5/9.

Vertical grid: the brief expected stretching, but BOTH nests are uniform --
dz = 17.331 m (render_a) and 2.9283 m (render_b), constant to float32
rounding (ptp/dz ~ 1e-4). No regridding is needed; the fluctuation is taken
on the native z axis and lags are converted with that single dz.

Perturbations: horizontal fluctuations are taken on the raw level with its
mean removed (Haar removes the mean anyway -- the subtraction is only so the
float64 sums are well conditioned). Vertical fluctuations are of the
departure from the full-level horizontal mean profile, computed over all
nx*ny points before column subsampling, so the base h(z)/qt(z) gradient
cannot dominate.

Bands and levels are set below. Horizontal lags run to nx//2, vertical to
nband//4 (aperiodic: past that the position count collapses).

figs/nest_slope_<model>_<var>.png : F_1 top row, local slope bottom row,
horizontal left, vertical right. Writes stats/nest_slope.npz; restartable
via that cache.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scaleinvariance as si

HERE = Path(__file__).parent
RUNS = HERE / "runs"
STATS = HERE / "stats"
FIGS = HERE / "figs"

MODELS = ("ukmo_ra1t", "icon_nwp")
GROUPS = ("render_a", "render_b")
VARS = ("h", "qt", "flux")

LEVELS = {"render_a": (1000.0, 5000.0, 10000.0),   # m, horizontal panels
          "render_b": (500.0, 1000.0)}
VBAND = {"render_a": (2000.0, 8000.0),             # m, vertical panels
         "render_b": (200.0, 1800.0)}
STRIDE = {"render_a": 4, "render_b": 8}            # column subsample for z

H_H = 0.45
H_V = 5.0 / 9.0
SPHEROSCALE = 10.0                                 # m, constant in these runs
OUTER = {"render_a": 6000.0, "render_b": 93.75}    # nest's own cascade outer scale
DX = {"render_a": 46.875, "render_b": 2.9296875}
PARENT_DX = 3000.0

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
A_COLORS = ["#7A2418", "#C4442A", "#E08A72"]
B_COLORS = ["#123F5E", "#2E86C1"]
GREY = "#B9B3AC"


def local_exponent(lags, vals):
    """OLS slope of log10 F vs log10 r, half-decade window centred on each r."""
    logr, logf = np.log10(lags), np.log10(vals)
    out = np.full(lags.size, np.nan)
    for i in range(lags.size):
        w = (np.abs(logr - logr[i]) <= 0.25) & np.isfinite(logf)
        if w.sum() >= 3:
            out[i] = np.polyfit(logr[w], logf[w], 1)[0]
    return out


def curves():
    si.set_numerical_precision("float64")
    out = {}
    for model in MODELS:
        ds = netCDF4.Dataset(RUNS / f"render_nests_steam_sq10_{model}_m00.nc")
        ds.set_auto_mask(False)
        for g in GROUPS:
            G = ds.groups[g]
            z = G.variables["z"][:].astype(np.float64)
            dz = float(np.mean(np.diff(z)))
            s = STRIDE[g]
            k0 = int(np.argmin(np.abs(z - VBAND[g][0])))
            k1 = int(np.argmin(np.abs(z - VBAND[g][1])))
            for v in VARS:
                full = G.variables[v][:]                      # (x, y, z) float32
                if not np.isfinite(full).all():
                    print(f"!! non-finite samples in {model}/{g}/{v}", flush=True)

                for zt in LEVELS[g]:
                    k = int(np.argmin(np.abs(z - zt)))
                    f = full[:, :, k].astype(np.float64)
                    f -= f.mean()
                    both = np.stack([f, f.T])                 # lags along y, then x
                    lags, F = si.haar_fluctuation(
                        both, order=1.0, axis=2, periodic=False,
                        max_sep=both.shape[2] // 2, lags="powers of 1.1")
                    key = f"{model}_{g}_{v}_horiz_{zt:.0f}"
                    out[key + "_lags"] = np.asarray(lags, float) * DX[g]
                    out[key + "_F"] = np.asarray(F, float)
                    out[key + "_z"] = np.array(z[k])
                    print(f"{key}: z = {z[k]:.0f} m, {len(lags)} lags, "
                          f"{np.isfinite(F).sum()} finite", flush=True)

                prof = full.mean(axis=(0, 1), dtype=np.float64)
                col = (full[::s, ::s, k0:k1].astype(np.float64)
                       - prof[k0:k1])
                lags, F = si.haar_fluctuation(
                    col, order=1.0, axis=2, periodic=False,
                    max_sep=col.shape[2] // 4, lags="powers of 1.1")
                key = f"{model}_{g}_{v}_vert"
                out[key + "_lags"] = np.asarray(lags, float) * dz
                out[key + "_F"] = np.asarray(F, float)
                print(f"{key}: band {z[k0]:.0f}-{z[k1]:.0f} m, "
                      f"{col.shape[0]}x{col.shape[1]} columns, {len(lags)} lags, "
                      f"{np.isfinite(F).sum()} finite", flush=True)
                del full, col
        ds.close()
    return out


def panel(ax, axs, lags, F, color, label, lw=1.3):
    good = np.isfinite(F) & (F > 0)
    ax.loglog(lags[good], F[good], color=color, lw=lw, label=label)
    axs.semilogx(lags[good], local_exponent(lags[good], F[good]),
                 color=color, lw=lw)


def main():
    FIGS.mkdir(exist_ok=True)
    cache = STATS / "nest_slope.npz"
    if cache.exists():
        blob = dict(np.load(cache))
        print(f"loaded {cache}", flush=True)
    else:
        blob = curves()
        np.savez(cache, **blob)
        print(f"wrote {cache}", flush=True)

    parent = STATS / "square_level_haar.npz"
    par = dict(np.load(parent)) if parent.exists() else None
    if par is None:
        print("!! no stats/square_level_haar.npz -- no parent overlay", flush=True)
    else:
        print(f"parent overlay from {parent} at z = {par['level_m']:.0f} m",
              flush=True)

    for model in MODELS:
        for v in VARS:
            fig, ax = plt.subplots(2, 2, figsize=(9.2, 6.2), sharex="col")
            for g, colors in (("render_a", A_COLORS), ("render_b", B_COLORS)):
                for i, zt in enumerate(LEVELS[g]):
                    key = f"{model}_{g}_{v}_horiz_{zt:.0f}"
                    panel(ax[0, 0], ax[1, 0], blob[key + "_lags"], blob[key + "_F"],
                          colors[i], f"{g} ({DX[g]:.3g} m), "
                          f"z = {blob[key + '_z']/1e3:.1f} km")
                key = f"{model}_{g}_{v}_vert"
                panel(ax[0, 1], ax[1, 1], blob[key + "_lags"], blob[key + "_F"],
                      colors[1], f"{g} ({DX[g]:.3g} m), "
                      f"{VBAND[g][0]/1e3:.1f}-{VBAND[g][1]/1e3:.1f} km")

            if par is not None:
                panel(ax[0, 0], ax[1, 0], par[f"{model}_{v}_lags"] * PARENT_DX,
                      par[f"{model}_{v}_F"], "#6E6862",
                      f"parent square (3 km), z = {par['level_m']/1e3:.1f} km",
                      lw=1.0)

            ref = H_H if v != "flux" else 0.0
            for j, (name, refs) in enumerate(
                    [("horizontal", [(ref, "$H_h$")]),
                     ("vertical", [(H_V if v != "flux" else 0.0, "$H_v$"),
                                   (ref, "$H_h$")])]):
                ax[0, j].set(ylabel="$F_1(r)$", title=f"{model}  {v}  {name}")
                ax[1, j].set(xlabel="r [m]", ylabel="local exponent")
                for h, lab in refs:
                    ax[1, j].axhline(h, color=GREY, ls="--", lw=0.9)
                    ax[1, j].annotate(lab, (0.985, h), xycoords=("axes fraction",
                                      "data"), fontsize=7, color="#6E6862",
                                      ha="right", va="bottom")
                ax[1, j].set_ylim(-0.1, 1.1)
                for a in (ax[0, j], ax[1, j]):
                    a.grid(True, which="both", alpha=0.5)
                    lo, hi = a.get_xlim()
                    for x, c in ((SPHEROSCALE, "#C4442A"),
                                 (OUTER["render_a"], GREY),
                                 (OUTER["render_b"], GREY)):
                        if lo < x < hi:
                            a.axvline(x, color=c, ls=":", lw=0.9)
            ax[0, 0].legend(fontsize=6.5)
            ax[0, 1].legend(fontsize=6.5)
            fig.suptitle("dotted: 10 m spheroscale (red) and the nests' own "
                         "cascade outer scales 93.75 m / 6 km (grey)",
                         fontsize=8, color="#6E6862")
            fig.tight_layout()
            p = FIGS / f"nest_slope_{model}_{v}.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
