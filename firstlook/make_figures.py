#!/usr/bin/env python3
"""First-look figures: SAM-TWPICE vs two STEAM cases (spheroscale profiles).

Cases: SAM; STEAM with log-linear spheroscale 1000 m -> 10 m over the 20 km
domain ("STEAM ls-var"); STEAM with constant 10 m spheroscale ("STEAM ls10").
Reads {sam,steam,steam_ls10}_stats.npz from compute_stats.py and writes
figures to figs/. Wavelet for the fluctuation functions: Haar, along x,
shown at order q = 1 only.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DX = 100.0  # both grids
Z_TOP_KM = 20.0

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})

C_REF = "#9A938B"
CASES = [  # (npz tag, label, color)
    ("sam", "SAM", "#C2410C"),
    ("steam", "STEAM ls 1000$\\to$10 m", "#1268A3"),
    ("steam_ls10", "STEAM ls 10 m", "#3FA34D"),
]
data = [(np.load(f"{tag}_stats.npz"), label, color)
        for tag, label, color in CASES]


def zkm(d):
    return d["z"] / 1000.0


# ── (a) cloud fraction + condensate partition ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(8.6, 4.6), sharey=True)
for d, label, color in data:
    axes[0].plot(d["cloud_fraction"], zkm(d), color=color, lw=1.6, label=label)
    axes[1].plot(d["qc_mean"] * 1000, zkm(d), color=color, lw=1.4)
    axes[2].plot(d["qi_mean"] * 1000, zkm(d), color=color, lw=1.4)
axes[0].set(xlabel="cloud fraction (cond. > 0.01 g/kg)", ylabel="z [km]",
            ylim=(0, Z_TOP_KM))
axes[1].set(xlabel="mean $q_c$ [g/kg]")
axes[2].set(xlabel="mean $q_i$ [g/kg]")
for ax in axes:
    ax.grid(True, alpha=0.6)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figs/cloud_fraction.png")
plt.close(fig)

# ── (b) variance and skewness profiles ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(10, 4.2), sharey=True)
panels = [
    ("h_var", "var $h'$ [J$^2$kg$^{-2}$]", "log"),
    ("qt_var", "var $q_t'$ [kg$^2$kg$^{-2}$]", "log"),
    ("h_skew", "skew $h'$", "linear"),
    ("qt_skew", "skew $q_t'$", "linear"),
]
for ax, (key, xlabel, xscale) in zip(axes, panels):
    for d, label, color in data:
        ax.plot(d[key], zkm(d), color=color, lw=1.4, label=label)
    if xscale == "log":
        ax.set_xscale("log")
    else:
        ax.axvline(0, color=C_REF, lw=0.8)
    ax.set(xlabel=xlabel, ylim=(0, Z_TOP_KM))
    ax.grid(True, alpha=0.6)
axes[0].set_ylabel("z [km]")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figs/variance_skewness.png")
plt.close(fig)

# ── (c) Haar fluctuation functions (q = 1) at three levels ───────────────────
FIT_RANGE_M = (400.0, 20_000.0)  # inside the scaling range of both models


def fit_slope(lags_m, F1):
    sel = (lags_m >= FIT_RANGE_M[0]) & (lags_m <= FIT_RANGE_M[1])
    return np.polyfit(np.log(lags_m[sel]), np.log(F1[sel]), 1)[0]


fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
levels = data[0][0]["levels"]
for row, name, symbol in [(0, "qt", "q_t"), (1, "h", "h")]:
    for col, target in enumerate(levels):
        ax = axes[row, col]
        text = []
        for d, label, color in data:
            lags = d[f"fluct_{name}_{target:.0f}_lags"] * DX
            F1 = d[f"fluct_{name}_{target:.0f}_F"][0]
            ax.loglog(lags, F1, color=color, lw=1.4, label=label)
            text.append(f"{label}: H={fit_slope(lags, F1):.2f}")
        ax.text(0.03, 0.97, "\n".join(text), transform=ax.transAxes,
                va="top", fontsize=7)
        z_used = data[0][0][f"fluct_{name}_{target:.0f}_z"]
        ax.set_title(f"${symbol}$  at  z $\\approx$ {z_used / 1000:.1f} km")
        ax.grid(True, which="both", alpha=0.5)
        if row == 1:
            ax.set_xlabel("lag [m]")
        if col == 0:
            ax.set_ylabel(f"$F_1$ (Haar) of ${symbol}'$")
axes[0, 2].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("figs/fluctuation_functions.png")
plt.close(fig)

# ── (d) 1D PDFs of h and qt at 4 and 10 km ──────────────────────────────────
pdf_levels = data[0][0]["pdf_levels"]
fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
for row, name, symbol, scale, unit in [
    (0, "qt", "q_t", 1000.0, "g/kg"),
    (1, "h", "h/c_p", 1.0 / 1004.0, "K"),
]:
    for col, target in enumerate(pdf_levels):
        ax = axes[row, col]
        for d, label, color in data:
            edges = d[f"pdf_{name}_{target:.0f}_edges"] * scale
            density = d[f"pdf_{name}_{target:.0f}_density"] / scale
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, density, color=color, lw=1.3, label=label,
                    drawstyle="steps-mid")
        ax.set_yscale("log")
        z_used = data[0][0][f"pdf_{name}_{target:.0f}_z"]
        ax.set_title(f"${symbol}$  at  z $\\approx$ {z_used / 1000:.1f} km")
        ax.set_xlabel(f"${symbol}$ [{unit}]")
        ax.grid(True, alpha=0.5)
        if col == 0:
            ax.set_ylabel("PDF")
axes[0, 1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("figs/pdfs.png")
plt.close(fig)

# ── (e) qt' cross-section and column condensate image rows ──────────────────
# Extents from the array shapes (dx = 100 m): STEAM is 204.8 x 102.4 km,
# SAM 204.8 x 204.8 km.


def extent_km(a):
    return [0, a.shape[0] * DX / 1000, 0, a.shape[1] * DX / 1000]


fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
limit = np.percentile(np.abs(np.concatenate(
    [d["qt_slice"].ravel() for d, _, _ in data])), 99.5)
for ax, (d, label, _) in zip(axes, data):
    im = ax.imshow(d["qt_slice"].T * 1000, cmap="BrBG", vmin=-limit * 1000,
                   vmax=limit * 1000, origin="lower", extent=extent_km(d["qt_slice"]))
    ax.set(title=f"{label}  $q_t'$  at z $\\approx$ {float(d['qt_slice_z']) / 1000:.1f} km",
           xlabel="x [km]")
axes[0].set_ylabel("y [km]")
fig.colorbar(im, ax=axes, label="$q_t'$ [g/kg]", shrink=0.8)
fig.savefig("figs/qt_cross_section.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
vmax = np.percentile(np.concatenate(
    [d["column_condensate"].ravel() for d, _, _ in data]), 99.5)
for ax, (d, label, _) in zip(axes, data):
    im = ax.imshow(d["column_condensate"].T * 1000, cmap="Blues", vmin=0,
                   vmax=vmax * 1000, origin="lower",
                   extent=extent_km(d["column_condensate"]))
    ax.set(title=f"{label}  column condensate", xlabel="x [km]")
axes[0].set_ylabel("y [km]")
fig.colorbar(im, ax=axes, label=r"$\int q_{cond}\,dz$ [g/kg $\cdot$ m]", shrink=0.8)
fig.savefig("figs/column_condensate.png", bbox_inches="tight")
plt.close(fig)

print("wrote figs/cloud_fraction.png, figs/variance_skewness.png,")
print("      figs/fluctuation_functions.png, figs/pdfs.png,")
print("      figs/qt_cross_section.png, figs/column_condensate.png")
