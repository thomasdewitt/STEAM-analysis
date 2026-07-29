#!/usr/bin/env python3
"""Ensemble profile figures: hosts as lines, STEAM as percentile shading.

STEAM ensemble = all models x 3 snapshots x 4 configs (C1 in {0.03, 0.1},
l_s in {3, 10} m), ~108 members. Per panel: each host model is a thin
line (snapshot-averaged); STEAM is a median line + 5-95% and 25-75%
percentile bands across all members interpolated to a common z grid.

figs/profiles_prognostic.png : std h', std qt', cloud fraction
figs/profiles_diagnostic.png : mean and std of qc, qi, T
(pressure omitted: host adapters carry no 3D pressure field)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
STATS = HERE / "stats"
FIGS = HERE / "figs"
Z = np.arange(0.0, 20000.0 + 100.0, 100.0)
ZKM = Z / 1000.0

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
HOST_CMAP = plt.cm.viridis
STEAM_COLOR = "#C4442A"


def models():
    return sorted({p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
                   if not p.name.startswith(("steam_", "diag_"))})


def interp(z, v):
    good = np.isfinite(v)
    return np.interp(Z, z[good], v[good], left=np.nan, right=np.nan)


def load(name, key, transform=lambda d, k: d[k]):
    d = np.load(STATS / f"{name}.npz")
    return interp(d["z"].astype(float), transform(d, key))


def gather(pattern, key, transform=lambda d, k: d[k]):
    """Stack profiles for every stats file matching pattern."""
    rows = [load(p.stem, key, transform) for p in sorted(STATS.glob(pattern))]
    return np.array(rows)


def host_mean_profiles(key, transform, diag):
    out = {}
    for m in models():
        prefix = "diag_" if diag else ""
        rows = gather(f"{prefix}{m}_snap*.npz", key, transform)
        if rows.size:
            out[m] = np.nanmean(rows, axis=0)
    return out


def panel(ax, hosts, steam_members, xlabel, log=False):
    for j, (m, prof) in enumerate(hosts.items()):
        ax.plot(prof, ZKM, lw=0.8, color=HOST_CMAP(j / max(len(hosts) - 1, 1)),
                label=m, zorder=2)
    lo, q1, med, q3, hi = np.nanpercentile(steam_members, [5, 25, 50, 75, 95],
                                           axis=0)
    ax.fill_betweenx(ZKM, lo, hi, color=STEAM_COLOR, alpha=0.15, lw=0, zorder=1)
    ax.fill_betweenx(ZKM, q1, q3, color=STEAM_COLOR, alpha=0.25, lw=0, zorder=1)
    ax.plot(med, ZKM, color=STEAM_COLOR, lw=1.6, label="STEAM median", zorder=3)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 20)
    ax.grid(True)


def std_of(d, k):
    return np.sqrt(d[k])


def main():
    FIGS.mkdir(exist_ok=True)

    # --- prognostic ---
    specs = [("h_var", std_of, r"std $h'$ [J kg$^{-1}$]", False),
             ("qt_var", std_of, r"std $q_t'$ [kg kg$^{-1}$]", False),
             ("cloud_fraction", lambda d, k: d[k], "cloud fraction", False)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), sharey=True)
    for ax, (key, tr, lab, log) in zip(axes, specs):
        hosts = host_mean_profiles(key, tr, diag=False)
        steam = gather(f"steam_*_snap*_C1*.npz", key, tr)
        panel(ax, hosts, steam, lab, log)
    axes[0].set_ylabel("z [km]")
    axes[-1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "profiles_prognostic.png", bbox_inches="tight")
    plt.close(fig)

    # --- diagnostic (from diag_*.npz caches) ---
    specs = [("qc_mean", r"mean $q_c$ [kg kg$^{-1}$]"),
             ("qi_mean", r"mean $q_i$ [kg kg$^{-1}$]"),
             ("T_mean", "mean T [K]"),
             ("qc_std", r"std $q_c$ [kg kg$^{-1}$]"),
             ("qi_std", r"std $q_i$ [kg kg$^{-1}$]"),
             ("T_std", "std T [K]")]
    fig, axes = plt.subplots(2, 3, figsize=(9, 7.0), sharey=True)
    for ax, (key, lab) in zip(axes.ravel(), specs):
        hosts = host_mean_profiles(key, lambda d, k: d[k], diag=True)
        steam = gather(f"diag_steam_*_snap*_C1*.npz", key)
        panel(ax, hosts, steam, lab)
    for row in axes:
        row[0].set_ylabel("z [km]")
    axes[0, -1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "profiles_diagnostic.png", bbox_inches="tight")
    plt.close(fig)
    print("profiles written", flush=True)


if __name__ == "__main__":
    main()
