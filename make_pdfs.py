#!/usr/bin/env python3
"""Anomaly PDFs of qt' and h' at two heights, all RCEMIP hosts vs STEAM.

For each model: level nearest 4 km and 10 km, per-level anomalies (value
minus level mean, per snapshot), all three snapshots pooled, histogrammed
over the pooled full data range (no percentile capping). Host from the
channel files via the extract_stats adapters; STEAM from runs/*.nc.
Writes stats/pdfs.npz and figs/pdfs.png (solid host / dashed STEAM).

Run the compute step under the analysis venv; figure-only via "figure".
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
LEVELS_M = (4000.0, 10000.0)
BINS = 150

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)


def pooled_pdf(samples):
    """density, edges over the pooled full range."""
    pooled = np.concatenate([s.ravel() for s in samples])
    density, edges = np.histogram(pooled, bins=BINS, density=True)
    return density, edges


def host_pdfs(model, out):
    from extract_stats import ADAPTERS
    qt_samples = {t: [] for t in LEVELS_M}
    h_samples = {t: [] for t in LEVELS_M}
    for i in range(3):
        z, T, qv, qc, qi, _ = ADAPTERS[model](i)
        z = np.asarray(z, dtype=np.float64)
        for target in LEVELS_M:
            k = int(np.argmin(np.abs(z - target)))
            qtk = (qv[k].astype(np.float64) + qc[k].astype(np.float64)
                   + qi[k].astype(np.float64))
            hk = (cp * T[k].astype(np.float64) + g * z[k]
                  + Lv * qv[k].astype(np.float64))
            qt_samples[target].append(qtk - qtk.mean())
            h_samples[target].append(hk - hk.mean())
            out[f"{model}_z{target:.0f}"] = z[k]
    for target in LEVELS_M:
        for name, samples in (("qt", qt_samples), ("h", h_samples)):
            density, edges = pooled_pdf(samples[target])
            out[f"{model}_{name}_{target:.0f}_density"] = density
            out[f"{model}_{name}_{target:.0f}_edges"] = edges
    print(f"host {model} done")


def steam_pdfs(model, out):
    import netCDF4
    qt_samples = {t: [] for t in LEVELS_M}
    h_samples = {t: [] for t in LEVELS_M}
    # 2026-07-29 ensemble: pool all 12 members (3 snaps x 4 configs).
    for path in sorted(RUNS.glob(f"steam_{model}_snap*_C1*.nc")):
        ds = netCDF4.Dataset(path)
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(np.float64)
        for target in LEVELS_M:
            k = int(np.argmin(np.abs(z - target)))
            qtk = ds.variables["qt"][:, :, k].astype(np.float64)
            hk = ds.variables["h"][:, :, k].astype(np.float64)
            qt_samples[target].append(qtk - qtk.mean())
            h_samples[target].append(hk - hk.mean())
        ds.close()
    for target in LEVELS_M:
        for name, samples in (("qt", qt_samples), ("h", h_samples)):
            density, edges = pooled_pdf(samples[target])
            out[f"steam_{model}_{name}_{target:.0f}_density"] = density
            out[f"steam_{model}_{name}_{target:.0f}_edges"] = edges
    print(f"steam {model} done")


def compute():
    from extract_stats import CHANNEL_HOSTS
    models = sorted(CHANNEL_HOSTS)
    out = {"models": np.array(models)}
    for model in models:
        host_pdfs(model, out)
        steam_pdfs(model, out)
    np.savez(STATS / "pdfs.npz", **out)
    print("wrote stats/pdfs.npz")


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
    d = np.load(STATS / "pdfs.npz", allow_pickle=True)
    models = list(d["models"])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    for row, name, symbol, scale, unit in [
        (0, "qt", "q_t'", 1000.0, "g/kg"),
        (1, "h", "h'/c_p", 1.0 / cp, "K"),
    ]:
        for col, target in enumerate(LEVELS_M):
            ax = axes[row, col]
            for m, c in zip(models, colors):
                for prefix, ls in ((m, "-"), (f"steam_{m}", "--")):
                    edges = d[f"{prefix}_{name}_{target:.0f}_edges"] * scale
                    density = d[f"{prefix}_{name}_{target:.0f}_density"] / scale
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.plot(centers, density, color=c, lw=1.0, ls=ls,
                            label=m if ls == "-" else None)
            ax.set_yscale("log")
            ax.set_title(f"${symbol}$  at  z $\\approx$ {target / 1000:.0f} km")
            ax.set_xlabel(f"${symbol}$ [{unit}]")
            ax.grid(True, alpha=0.5)
            if col == 0:
                ax.set_ylabel("PDF")
    axes[0, 1].legend(fontsize=6.5, title="solid host / dash STEAM",
                      title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(HERE / "figs" / "pdfs.png")
    print("wrote figs/pdfs.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["compute", "figure"]
    if "compute" in wanted:
        compute()
    if "figure" in wanted:
        figure()
