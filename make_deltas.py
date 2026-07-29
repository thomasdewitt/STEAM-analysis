#!/usr/bin/env python3
"""The delta figure: (STEAM - host LES) profiles vs the inter-LES spread.

For each RCEMIP channel model, snapshot-averaged profiles of cloud fraction,
std h', and std qt' are interpolated to a common z grid; the figure shows
STEAM-minus-host deltas per model (colored lines) on top of a grey band =
the inter-LES spread (each host minus the ensemble mean, min-to-max
envelope). STEAM "in distribution" <=> the colored lines live inside the
grey band. A companion figure shows the raw profiles.
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

QUANTITIES = [  # (key, transform, label)
    ("cloud_fraction", lambda d: d["cloud_fraction"], "cloud fraction"),
    ("h_std", lambda d: np.sqrt(d["h_var"]) / 1004.0, "std $h'/c_p$ [K]"),
    ("qt_std", lambda d: np.sqrt(d["qt_var"]) * 1000.0, "std $q_t'$ [g/kg]"),
]


def snapshot_mean(prefix):
    """Snapshot-averaged profiles on the common grid, or None if missing."""
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
                 if not p.name.startswith(("steam_", "diag_"))})
host = {m: snapshot_mean(m) for m in models}
steam = {m: snapshot_mean(f"steam_{m}") for m in models}
models = [m for m in models if host[m] is not None and steam[m] is not None]
print("models:", models)

colors = plt.cm.tab10(np.linspace(0, 1, 10))

# ── deltas ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8), sharey=True)
for ax, (key, _, label) in zip(axes, QUANTITIES):
    ensemble = np.array([host[m][key] for m in models])
    # Signed pairwise host-host differences (i != j): the apples-to-apples
    # yardstick for the pairwise STEAM-minus-host deltas.
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
        ax.plot(steam[m][key] - host[m][key], Z / 1000, color=c, lw=1.2,
                label=m)
    ax.axvline(0, color="#9A938B", lw=0.8)
    ax.set(xlabel=f"$\\Delta$ {label}", ylim=(0, 20))
    ax.grid(True, alpha=0.6)
axes[0].set_ylabel("z [km]")
axes[2].legend(fontsize=6.5, loc="upper right")
fig.suptitle("STEAM $-$ host LES (snapshot-mean), vs inter-LES spread",
             fontsize=10)
fig.tight_layout()
(HERE / "figs").mkdir(exist_ok=True)
fig.savefig(HERE / "figs" / "deltas.png")
plt.close(fig)

# ── raw profiles (hosts solid, STEAM dashed) ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8), sharey=True)
for ax, (key, _, label) in zip(axes, QUANTITIES):
    for m, c in zip(models, colors):
        ax.plot(host[m][key], Z / 1000, color=c, lw=1.2, label=m)
        ax.plot(steam[m][key], Z / 1000, color=c, lw=1.0, ls="--")
    ax.set(xlabel=label, ylim=(0, 20))
    if key != "cloud_fraction":
        ax.set_xscale("log")
    ax.grid(True, alpha=0.6)
axes[0].set_ylabel("z [km]")
axes[2].legend(fontsize=6.5, loc="upper right", title="solid host / dash STEAM")
fig.tight_layout()
fig.savefig(HERE / "figs" / "profiles.png")
plt.close(fig)

print("wrote figs/deltas.png, figs/profiles.png")
