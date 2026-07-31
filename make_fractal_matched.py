#!/usr/bin/env python3
"""Cloud geometry of the matched STEAM-vs-LES pairs (half set).

One STEAM sim per RCEMIP RCE_small_les300 host, each on that host's domain
extent (run_steam_matched.py). tau > 1 masks from vertically integrated
optical depth, then the half set Thomas ruled for non-square domains
(2026-07-29): ensemble correlation dimension and individual fractal
dimension only -- no size distributions. These domains are ~100 km across
at ~200 m (outer scale L = L_x/4 = 24-27 km), so a single snapshot holds
far too few resolved objects for a finite-domain size distribution to mean
anything (DeWitt & Garrett 2024); the squares carry that part.

SAM-TWPICE is deliberately not here: its STEAM counterpart runs at
1024^2 x 200 m against a 2048^2 x 100 m host (see run_steam_matched.py),
so host and STEAM would be compared at different pixel scales.

Only one snapshot exists per host, so each case is a single mask -- the
correlation integral is correspondingly noisy and no uncertainty is quoted
(objscale returns point estimates by design).

Writes stats/fractal_matched.npz and figs/matched/fractal_matched.png.
"""

import sys
from pathlib import Path

import numpy as np
import objscale
import netCDF4

from make_fractal import cv, TAU_THRESHOLD

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
MODELS = ("les_cm1", "les_sam", "les_dales", "les_icon_lem")


def host_mask(model):
    from extract_stats import ADAPTERS
    z, T, qv, qc, qi, _ = ADAPTERS[model](0)
    del T, qv
    # adapter fields are (nz, ny, nx) kg/kg; cloudyview wants (nx, ny, nz) g/kg
    lwc = np.moveaxis(qc, 0, -1) * 1000.0
    iwc = np.moveaxis(qi, 0, -1) * 1000.0
    tau = cv.vertically_integrated_optical_depth(
        lwc, np.asarray(z, dtype=np.float64), iwc=iwc)
    return (tau > TAU_THRESHOLD).astype(np.float32)


def steam_mask(model):
    ds = netCDF4.Dataset(RUNS / f"steam_matched_{model}.nc")
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    x = ds.variables["x"][:].astype(np.float64)
    lwc = ds.variables["qc"][:] * 1000.0   # (x, y, z) already
    iwc = ds.variables["qi"][:] * 1000.0
    ds.close()
    tau = cv.vertically_integrated_optical_depth(lwc, z, iwc=iwc)
    return (tau > TAU_THRESHOLD).astype(np.float32), float(x[1] - x[0])


def analyze(mask, dx):
    sizes = np.full(mask.shape, dx)
    # minlength 2 px as in make_fractal.py: the default 8 px leaves less than
    # a decade against the 0.33 x domain cap on these ~500-cell grids. Pixel
    # scale bias at the small end is accepted and flagged in the caption.
    dim, bins, C_l = objscale.ensemble_correlation_dimension(
        [mask], x_sizes=sizes, y_sizes=sizes, minlength=2 * dx,
        point_reduction_factor=10, return_C_l=True)
    ind_dim, log_l, log_p = objscale.individual_fractal_dimension(
        [mask], x_sizes=sizes, y_sizes=sizes, return_values=True)
    return dict(dim=dim, C_bins=bins, C_l=C_l, ind_dim=ind_dim,
                ind_log_l=log_l, ind_log_p=log_p, cover=float(mask.mean()))


def compute():
    out = {"models": np.array(MODELS)}
    for model in MODELS:
        smask, dx = steam_mask(model)
        hmask = host_mask(model)
        hdx = 200.0   # all four RCE_small_les300 hosts are on a 200 m grid
        for prefix, mask, d in ((model, hmask, hdx),
                                (f"steam_{model}", smask, dx)):
            r = analyze(mask, d)
            for k, v in r.items():
                out[f"{prefix}_{k}"] = v
            print(f"{prefix}: D2={r['dim']:.2f}, Di={r['ind_dim']:.2f}, "
                  f"cover={r['cover']:.3f}, dx={d:.1f} m", flush=True)
    np.savez(STATS / "fractal_matched.npz", **out)
    print("wrote stats/fractal_matched.npz")


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
    d = np.load(STATS / "fractal_matched.npz", allow_pickle=True)
    colors = dict(zip(MODELS, ["#e76f51", "#1764ab", "#2a9d8f", "#8e5ea2"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for model in MODELS:
        c = colors[model]
        for prefix, ls, lw in ((model, "-", 1.4),
                               (f"steam_{model}", "--", 1.0)):
            label = model if ls == "-" else None
            axes[0].loglog(d[f"{prefix}_C_bins"] / 1000.0, d[f"{prefix}_C_l"],
                           color=c, lw=lw, ls=ls, label=label)
            axes[1].plot(d[f"{prefix}_ind_log_l"] - 3.0,   # m -> km
                         d[f"{prefix}_ind_log_p"] - 3.0,
                         color=c, lw=lw, ls=ls, label=label)
        axes[2].scatter([d[f"{model}_dim"]], [d[f"{model}_ind_dim"]],
                        color=c, s=32, marker="o", label=model)
        axes[2].scatter([d[f"steam_{model}_dim"]],
                        [d[f"steam_{model}_ind_dim"]], color=c, s=32,
                        marker="x")
        axes[2].plot([d[f"{model}_dim"], d[f"steam_{model}_dim"]],
                     [d[f"{model}_ind_dim"], d[f"steam_{model}_ind_dim"]],
                     color=c, lw=0.7, alpha=0.6)
    axes[0].set(xlabel="r [km]", ylabel="correlation integral $C(r)$",
                title="$\\tau>1$ mask correlation integral\n"
                      "(solid host, dashed STEAM)")
    axes[1].set(xlabel="log$_{10}$ length scale [km]",
                ylabel="log$_{10}$ filled perimeter [km]",
                title="individual perimeter-area scaling")
    axes[2].set(xlabel="ensemble correlation dimension $D_2$",
                ylabel="individual fractal dimension $D_i$",
                title="summary (o host, x STEAM)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.5)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    (HERE / "figs" / "matched").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figs" / "matched" / "fractal_matched.png")
    print("wrote figs/matched/fractal_matched.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["compute", "figure"]
    if "compute" in wanted:
        compute()
    if "figure" in wanted:
        figure()
