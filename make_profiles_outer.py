#!/usr/bin/env python3
"""Outer-scale comparison figures: STEAM 96 km vs STEAM L/4 = 1536 km.

Both sets under current code (2026-07-27; the 2026-07-21-era output was
deleted — Thomas's ruling, "no reason to build up crud").
Same layout family as make_deltas.py. Two figures:
  figs/profiles_outer.png — raw profiles: host (solid), STEAM 96 km (dashed),
    STEAM 1536 km (dotted), per model color.
  figs/deltas_outer.png — STEAM-minus-host deltas for both outer scales on
    the same inter-LES pairwise spread band (96 km thin, 1536 km thick).
Requires stats/steam_96_* and steam_L4_* from run_steam_96.py / run_steam_L4.py.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
STATS = HERE / "stats"
Z = np.arange(0.0, 20000.0 + 100.0, 100.0)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})

QUANTITIES = [
    ("cloud_fraction", lambda d: d["cloud_fraction"], "cloud fraction"),
    ("h_std", lambda d: np.sqrt(d["h_var"]) / 1004.0, "std $h'/c_p$ [K]"),
    ("qt_std", lambda d: np.sqrt(d["qt_var"]) * 1000.0, "std $q_t'$ [g/kg]"),
]


def snapshot_mean(prefix):
    out = {}
    files = sorted(STATS.glob(f"{prefix}_snap*.npz"))
    if len(files) < 3:
        return None
    for key, transform, _ in QUANTITIES:
        curves = []
        for f in files:
            d = np.load(f)
            curves.append(np.interp(Z, d["z"], transform(d),
                                    left=np.nan, right=np.nan))
        out[key] = np.nanmean(curves, axis=0)
    return out


models = sorted({p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
                 if not p.name.startswith("steam")})
host = {m: snapshot_mean(m) for m in models}
steam96 = {m: snapshot_mean(f"steam_96_{m}") for m in models}
steamL4 = {m: snapshot_mean(f"steam_L4_{m}") for m in models}
models = [m for m in models
          if host[m] is not None and steam96[m] is not None
          and steamL4[m] is not None]
print("models:", models)

colors = plt.cm.tab10(np.linspace(0, 1, 10))

# ── raw profiles ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8), sharey=True)
for ax, (key, _, label) in zip(axes, QUANTITIES):
    for m, c in zip(models, colors):
        ax.plot(host[m][key], Z / 1000, color=c, lw=1.2, label=m)
        ax.plot(steam96[m][key], Z / 1000, color=c, lw=1.0, ls="--")
        ax.plot(steamL4[m][key], Z / 1000, color=c, lw=1.0, ls=":")
    ax.set(xlabel=label, ylim=(0, 20))
    if key != "cloud_fraction":
        ax.set_xscale("log")
    ax.grid(True, alpha=0.6)
axes[0].set_ylabel("z [km]")
axes[2].legend(fontsize=6.5, loc="upper right",
               title="solid host / dash 96 km / dot 1536 km")
fig.suptitle("Outer scale: 96 km vs L/4 = 1536 km (same seeds, frozen config)",
             fontsize=10)
fig.tight_layout()
(HERE / "figs").mkdir(exist_ok=True)
fig.savefig(HERE / "figs" / "profiles_outer.png")
plt.close(fig)

# ── deltas, both outer scales on the inter-LES band ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8), sharey=True)
for ax, (key, _, label) in zip(axes, QUANTITIES):
    ensemble = np.array([host[m][key] for m in models])
    n = len(models)
    pairs = np.array([ensemble[a] - ensemble[b]
                      for a in range(n) for b in range(n) if a != b])
    ax.fill_betweenx(Z / 1000, np.nanmin(pairs, axis=0),
                     np.nanmax(pairs, axis=0), color="#E9E6E1",
                     label="LES$-$LES full range", lw=0)
    ax.fill_betweenx(Z / 1000, np.nanpercentile(pairs, 25, axis=0),
                     np.nanpercentile(pairs, 75, axis=0), color="#CFCAC2",
                     label="LES$-$LES 25$-$75%", lw=0)
    for m, c in zip(models, colors):
        ax.plot(steam96[m][key] - host[m][key], Z / 1000, color=c, lw=0.8,
                alpha=0.6)
        ax.plot(steamL4[m][key] - host[m][key], Z / 1000, color=c, lw=1.6,
                label=m)
    ax.axvline(0, color="#9A938B", lw=0.8)
    ax.set(xlabel=f"$\\Delta$ {label}", ylim=(0, 20))
    ax.grid(True, alpha=0.6)
axes[0].set_ylabel("z [km]")
axes[2].legend(fontsize=6.5, loc="upper right",
               title="thin 96 km / thick 1536 km")
fig.suptitle("STEAM $-$ host LES: outer scale 96 km vs 1536 km", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "figs" / "deltas_outer.png")
plt.close(fig)

print("wrote figs/profiles_outer.png, figs/deltas_outer.png")
