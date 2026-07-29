#!/usr/bin/env python3
"""Fluctuation functions + local exponents at one mid-level (~5 km).

Mexican-hat wavelet fluctuations (order 1, scaleinvariance) along the
long (x) dimension at the level nearest 5 km, periodic. Hosts: one curve
per model (all 3 snapshots stacked and passed at once). STEAM: every
ensemble member computed separately (same grid -> same lags), plotted as
median + 5-95% band. Local exponent = OLS slope of log10 F vs log10 r
over the half-decade window centred on each r (main.tex definition).

figs/local_slope_prognostic.png : h, qt
figs/local_slope_diagnostic.png : T, qc
Writes stats/local_slope.npz. Restartable via that cache.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import scaleinvariance

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
FIGS = HERE / "figs"
Z_TARGET = 5000.0
DX = 3000.0
VARS = ["h", "qt", "T", "qc"]

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
HOST_CMAP = plt.cm.viridis
STEAM_COLOR = "#C4442A"


def fluct(arr2d):
    """Mexican-hat order-1 fluctuation along the last axis (periodic)."""
    lags, vals = scaleinvariance.wavelet_fluctuation(
        arr2d, wavelet="mexican_hat", order=1, axis=arr2d.ndim - 1,
        periodic=True, nan_behavior="ignore")
    return np.asarray(lags, float), np.asarray(vals, float)


def host_curves():
    from extract_stats import ADAPTERS
    from steam.constants import (specific_heat_dry_air as cp,
                                 latent_heat_vaporization as Lv,
                                 gravity as g)
    out = {}
    models = sorted({p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
                     if not p.name.startswith(("steam_", "diag_"))})
    for m in models:
        fields = {v: [] for v in VARS}
        for i in range(3):
            z, T, qv, qc, qi, _ = ADAPTERS[m](i)      # (nz, ny, nx)
            z = np.asarray(z, float)
            k = int(np.argmin(np.abs(z - Z_TARGET)))
            qt = qv[k] + qc[k] + qi[k]
            h = cp * T[k] + g * z[k] + Lv * qv[k]
            fields["h"].append(h)
            fields["qt"].append(qt)
            fields["T"].append(T[k])
            fields["qc"].append(qc[k])
        out[m] = {}
        for v in VARS:
            stack = np.stack([np.asarray(a, np.float64) for a in fields[v]])
            lags, vals = fluct(stack)                 # along x
            out[m][v] = (lags * DX, vals)             # physical lags [m]
    return out


def steam_curves():
    """Per-member curves; all members share the 2048-cell x grid."""
    from steam.constants import gravity as g  # noqa: F401 (h stored directly)
    curves = {v: [] for v in VARS}
    lags_out = {}
    for p in sorted(RUNS.glob("steam_*_snap*_C1*.nc")):
        ds = netCDF4.Dataset(p)
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(float)
        k = int(np.argmin(np.abs(z - Z_TARGET)))
        for v in VARS:
            f = ds.variables[v][:, :, k].astype(np.float64)   # (x, y)
            lags, vals = fluct(f.T)                           # along x
            lags_out[v] = lags * DX
            curves[v].append(vals)
        ds.close()
        print(f"slope {p.stem}", flush=True)
    return {v: (lags_out[v], np.array(curves[v])) for v in VARS}


def local_exponent(lags, vals):
    logr, logf = np.log10(lags), np.log10(vals)
    out = np.full(lags.size, np.nan)
    for i in range(lags.size):
        w = (np.abs(logr - logr[i]) <= 0.25) & np.isfinite(logf)
        if w.sum() >= 3:
            out[i] = np.polyfit(logr[w], logf[w], 1)[0]
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    cache = STATS / "local_slope.npz"
    if cache.exists():
        blob = dict(np.load(cache, allow_pickle=True))
        hosts, steam = blob["hosts"].item(), blob["steam"].item()
    else:
        hosts = host_curves()
        steam = steam_curves()
        np.savez(cache, hosts=np.array(hosts, dtype=object),
                 steam=np.array(steam, dtype=object))

    for name, pair in [("prognostic", ["h", "qt"]), ("diagnostic", ["T", "qc"])]:
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.4))
        for col, v in enumerate(pair):
            axf, axs = axes[0, col], axes[1, col]
            for j, (m, d) in enumerate(hosts.items()):
                lags, vals = d[v]
                color = HOST_CMAP(j / max(len(hosts) - 1, 1))
                good = np.isfinite(vals) & (vals > 0)
                axf.loglog(lags[good] / 1e3, vals[good], lw=0.8, color=color,
                           label=m)
                axs.semilogx(lags[good] / 1e3,
                             local_exponent(lags[good], vals[good]),
                             lw=0.8, color=color)
            lags, members = steam[v]
            lo, med, hi = np.nanpercentile(members, [5, 50, 95], axis=0)
            axf.fill_between(lags / 1e3, lo, hi, color=STEAM_COLOR, alpha=0.2,
                             lw=0)
            axf.loglog(lags / 1e3, med, color=STEAM_COLOR, lw=1.6,
                       label="STEAM median")
            slopes = np.array([local_exponent(lags, m_) for m_ in members])
            slo, smed, shi = np.nanpercentile(slopes, [5, 50, 95], axis=0)
            axs.fill_between(lags / 1e3, slo, shi, color=STEAM_COLOR,
                             alpha=0.2, lw=0)
            axs.semilogx(lags / 1e3, smed, color=STEAM_COLOR, lw=1.6)
            axf.set_title(v)
            axf.set_ylabel("$F_1(r)$")
            axs.set_ylabel("local exponent")
            axs.set_xlabel("r [km]")
            axs.axhline(0, color="#B9B3AC", lw=0.6)
            for ax in (axf, axs):
                ax.grid(True)
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(FIGS / f"local_slope_{name}.png", bbox_inches="tight")
        plt.close(fig)
    print("local slope figures written", flush=True)


if __name__ == "__main__":
    main()
