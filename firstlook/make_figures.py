#!/usr/bin/env python3
"""First-look figures: STEAM (canonical, TWPICE profiles) vs SAM-TWPICE.

Reads sam_stats.npz / steam_stats.npz from compute_stats.py and writes four
figures to figs/. Wavelet for the fluctuation functions: Haar, along x.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_STEAM = "#1268A3"
C_SAM = "#C2410C"
C_REF = "#9A938B"
DX = 100.0  # both grids
Z_TOP_KM = 18.0  # cloud-relevant depth for profile panels

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})

sam = np.load("sam_stats.npz")
steam = np.load("steam_stats.npz")


def zkm(d):
    return d["z"] / 1000.0


# ── (a) cloud fraction ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.6, 4.6))
ax.plot(sam["cloud_fraction"], zkm(sam), color=C_SAM, lw=1.6, label="SAM")
ax.plot(steam["cloud_fraction"], zkm(steam), color=C_STEAM, lw=1.6, label="STEAM")
ax.set(xlabel="cloud fraction (cond. > 0.01 g/kg)", ylabel="z [km]",
       ylim=(0, Z_TOP_KM))
ax.grid(True, alpha=0.6)
ax.legend()
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
    ax.plot(sam[key], zkm(sam), color=C_SAM, lw=1.4, label="SAM")
    ax.plot(steam[key], zkm(steam), color=C_STEAM, lw=1.4, label="STEAM")
    if xscale == "log":
        ax.set_xscale("log")
    else:
        ax.axvline(0, color=C_REF, lw=0.8)
    ax.set(xlabel=xlabel, ylim=(0, Z_TOP_KM))
    ax.grid(True, alpha=0.6)
axes[0].set_ylabel("z [km]")
axes[0].legend()
fig.tight_layout()
fig.savefig("figs/variance_skewness.png")
plt.close(fig)

# ── (c) Haar fluctuation functions at three levels ───────────────────────────
FIT_RANGE_M = (400.0, 20_000.0)  # inside the scaling range of both models


def fit_slope(lags_m, F, order):
    sel = (lags_m >= FIT_RANGE_M[0]) & (lags_m <= FIT_RANGE_M[1])
    return np.polyfit(np.log(lags_m[sel]), np.log(F[order - 1, sel]), 1)[0]


fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
levels = sam["levels"]
for row, name, symbol in [(0, "qt", "q_t"), (1, "h", "h")]:
    for col, target in enumerate(levels):
        ax = axes[row, col]
        text = []
        for d, color, label in [(sam, C_SAM, "SAM"), (steam, C_STEAM, "STEAM")]:
            lags = d[f"fluct_{name}_{target:.0f}_lags"] * DX
            F = d[f"fluct_{name}_{target:.0f}_F"]
            ax.loglog(lags, F[0], color=color, lw=1.4, label=f"{label} q=1")
            ax.loglog(lags, F[1], color=color, lw=1.1, ls="--", label=f"{label} q=2")
            z1 = fit_slope(lags, F, 1)
            z2 = fit_slope(lags, F, 2)
            text.append(f"{label}: H={z1:.2f}, 2$\\zeta_1$$-$$\\zeta_2$={2 * z1 - z2:.2f}")
        ax.text(0.03, 0.97, "\n".join(text), transform=ax.transAxes,
                va="top", fontsize=7)
        z_used = sam[f"fluct_{name}_{target:.0f}_z"]
        ax.set_title(f"${symbol}$  at  z $\\approx$ {z_used / 1000:.1f} km")
        ax.grid(True, which="both", alpha=0.5)
        if row == 1:
            ax.set_xlabel("lag [m]")
        if col == 0:
            ax.set_ylabel(f"$F_q$ (Haar) of ${symbol}'$")
axes[0, 2].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("figs/fluctuation_functions.png")
plt.close(fig)

# ── (d) qt' cross-section and column condensate image pairs ─────────────────
# Extents from the array shapes (dx = 100 m): STEAM is 204.8 x 102.4 km,
# SAM 204.8 x 204.8 km.


def extent_km(a):
    return [0, a.shape[0] * DX / 1000, 0, a.shape[1] * DX / 1000]


fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
limit = np.percentile(np.abs(np.concatenate([
    steam["qt_slice"].ravel(), sam["qt_slice"].ravel()])), 99.5)
for ax, d, label in [(axes[0], steam, "STEAM"), (axes[1], sam, "SAM")]:
    im = ax.imshow(d["qt_slice"].T * 1000, cmap="BrBG", vmin=-limit * 1000,
                   vmax=limit * 1000, origin="lower", extent=extent_km(d["qt_slice"]))
    ax.set(title=f"{label}  $q_t'$  at z $\\approx$ {float(d['qt_slice_z']) / 1000:.1f} km",
           xlabel="x [km]")
axes[0].set_ylabel("y [km]")
fig.colorbar(im, ax=axes, label="$q_t'$ [g/kg]", shrink=0.85)
fig.savefig("figs/qt_cross_section.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
vmax = np.percentile(np.concatenate([
    steam["column_condensate"].ravel(), sam["column_condensate"].ravel()]), 99.5)
for ax, d, label in [(axes[0], steam, "STEAM"), (axes[1], sam, "SAM")]:
    im = ax.imshow(d["column_condensate"].T * 1000, cmap="Blues", vmin=0,
                   vmax=vmax * 1000, origin="lower",
                   extent=extent_km(d["column_condensate"]))
    ax.set(title=f"{label}  column condensate", xlabel="x [km]")
axes[0].set_ylabel("y [km]")
fig.colorbar(im, ax=axes, label=r"$\int q_{cond}\,dz$ [g/kg $\cdot$ m]", shrink=0.85)
fig.savefig("figs/column_condensate.png", bbox_inches="tight")
plt.close(fig)

print("wrote figs/cloud_fraction.png, figs/variance_skewness.png,")
print("      figs/fluctuation_functions.png, figs/qt_cross_section.png,")
print("      figs/column_condensate.png")
