#!/usr/bin/env python3
"""Anomaly PDFs of qt' and h' at two heights, matched STEAM vs each host.

Same construction as make_pdfs.py -- level nearest 4 km and 10 km, per-level
anomalies (value minus that level's mean), histogrammed over the full pooled
range with no percentile capping -- but the comparison hosts are single
snapshots and each has exactly one matched STEAM member
(runs/steam_matched_<model>.nc from run_steam_matched.py).

TWPICE is plotted apart from the four LES: it is an observed-case SAM run of
the TWP-ICE campaign, not RCE, so it does not belong on the same axes as the
RCEMIP RCE_small_les300 ensemble.

Writes stats/pdfs_matched.npz, figs/matched/pdfs_twpice.png and
figs/matched/pdfs_les.png. Run "compute" before the .nc are deleted.

Usage: python make_pdfs_matched.py [compute] [figure]
"""

import sys
from pathlib import Path

import numpy as np

from make_pdfs import pooled_pdf, LEVELS_M
from run_steam_matched import MATCHED

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
FIGS = HERE / "figs" / "matched"
TWPICE = ["twpice"]
LES = ["les_cm1", "les_dales", "les_icon_lem", "les_sam"]
# Host solid / STEAM dashed in one colour per model (make_pdfs.py's
# convention). Muted, evenly spaced hues against the warm-grey axes.
COLORS = {"twpice": "#B0532E", "les_cm1": "#1F6F8B", "les_dales": "#C9772F",
          "les_icon_lem": "#6B4C9A", "les_sam": "#3E8E5A"}
RCPARAMS = {
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
}


def host_pdfs(model, out):
    from extract_stats import ADAPTERS
    z, T, qv, qc, qi, _ = ADAPTERS[model](0)
    z = np.asarray(z, dtype=np.float64)
    for target in LEVELS_M:
        k = int(np.argmin(np.abs(z - target)))
        qtk = (qv[k].astype(np.float64) + qc[k].astype(np.float64)
               + qi[k].astype(np.float64))
        hk = (cp * T[k].astype(np.float64) + g * z[k]
              + Lv * qv[k].astype(np.float64))
        out[f"{model}_z{target:.0f}"] = z[k]
        for name, anom in (("qt", qtk - qtk.mean()), ("h", hk - hk.mean())):
            density, edges = pooled_pdf([anom])
            out[f"{model}_{name}_{target:.0f}_density"] = density
            out[f"{model}_{name}_{target:.0f}_edges"] = edges
    print(f"host {model} done", flush=True)


def steam_pdfs(model, out):
    import netCDF4
    ds = netCDF4.Dataset(RUNS / f"steam_matched_{model}.nc")
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    for target in LEVELS_M:
        k = int(np.argmin(np.abs(z - target)))
        qtk = ds.variables["qt"][:, :, k].astype(np.float64)
        hk = ds.variables["h"][:, :, k].astype(np.float64)
        out[f"steam_{model}_z{target:.0f}"] = z[k]
        for name, anom in (("qt", qtk - qtk.mean()), ("h", hk - hk.mean())):
            density, edges = pooled_pdf([anom])
            out[f"steam_{model}_{name}_{target:.0f}_density"] = density
            out[f"steam_{model}_{name}_{target:.0f}_edges"] = edges
    ds.close()
    print(f"steam {model} done", flush=True)


def compute():
    out = {}
    for model in MATCHED:
        host_pdfs(model, out)
        steam_pdfs(model, out)
    np.savez(STATS / "pdfs_matched.npz", **out)
    print("wrote stats/pdfs_matched.npz", flush=True)


def panel(ax, d, models, name, target, scale):
    for m in models:
        for prefix, ls, alpha in ((m, "-", 1.0), (f"steam_{m}", "--", 0.95)):
            edges = d[f"{prefix}_{name}_{target:.0f}_edges"] * scale
            density = d[f"{prefix}_{name}_{target:.0f}_density"] / scale
            ax.plot(0.5 * (edges[:-1] + edges[1:]), density, color=COLORS[m],
                    lw=1.2, ls=ls, alpha=alpha, label=m if ls == "-" else None)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.5)


def sheet(d, models, out_name, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharey="row")
    for row, name, symbol, scale, unit in [
        (0, "qt", "q_t'", 1000.0, "g kg$^{-1}$"),
        (1, "h", "h'/c_p", 1.0 / cp, "K"),
    ]:
        for col, target in enumerate(LEVELS_M):
            ax = axes[row, col]
            panel(ax, d, models, name, target, scale)
            ax.set_title(f"${symbol}$  at  z $\\approx$ {target / 1000:.0f} km")
            ax.set_xlabel(f"${symbol}$ [{unit}]")
            if col == 0:
                ax.set_ylabel("PDF")
    axes[0, 1].legend(fontsize=7, title="solid host / dash STEAM",
                      title_fontsize=7)
    fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/matched/{out_name}", flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RCPARAMS)
    FIGS.mkdir(parents=True, exist_ok=True)
    d = np.load(STATS / "pdfs_matched.npz")
    sheet(d, TWPICE, "pdfs_twpice.png",
          "matched STEAM vs SAM-TWPICE, anomaly PDFs")
    sheet(d, LES, "pdfs_les.png",
          "matched STEAM vs RCEMIP RCE_small_les300, anomaly PDFs")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["compute", "figure"]
    if "compute" in wanted:
        compute()
    if "figure" in wanted:
        figure()
