"""Experiment: height-varying spheroscale grid consistency check.

For each scale class k (from STEAM config), builds a vertical grid
iteratively using dz_i = k_z(z_i) / (2 * s_z), where

    k_z(z) = ls(z)^(1 - H_z) * k^H_z,   H_z = 5/9

and ls(z) increases logarithmically from ls_top=1 at domain top
to ls_bot=100 at z=0 (bottom).

Reports how close sum(dz_i) is to DOMAIN_HEIGHT for every scale class.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── STEAM config (from STEAM/steam_simulate.py) ───────────────────────────────
NX, NY           = 512, 512
DX, DY           = 5000.0, 5000.0   # m
OUTER_SCALE      = DX * 256         # m  = 1 280 000 m
DOMAIN_HEIGHT    = 20000.0          # m
S_X, S_Y, S_Z   = 1, 1, 1          # sparsity factors
K_FACTOR         = 2

# ── Physics ───────────────────────────────────────────────────────────────────
H_Z = 5.0 / 9.0                     # hurst_vertical_anisotropy

# ── Height-varying spheroscale ────────────────────────────────────────────────
LS_TOP = 1.0    # at z = DOMAIN_HEIGHT
LS_BOT = 100.0  # at z = 0

def ls_of_z(z):
    """Spheroscale at height z: log-linear from LS_BOT (z=0) to LS_TOP (z=H)."""
    frac = np.asarray(z) / DOMAIN_HEIGHT          # 0 at bottom, 1 at top
    return np.exp((1 - frac) * np.log(LS_BOT) + frac * np.log(LS_TOP))

def k_z_of_z(k, z):
    """Vertical turbulon scale at height z for horizontal scale k."""
    ls = ls_of_z(z)
    return ls * (k / ls) ** H_Z                   # = ls^(1-H_Z) * k^H_Z

def dz_exact(k, z):
    return k_z_of_z(k, z) / (2.0 * S_Z)

# ── Scale classes ─────────────────────────────────────────────────────────────
n_classes = int(round(np.log(OUTER_SCALE / (2 * DX)) / np.log(K_FACTOR))) + 1
k_values  = OUTER_SCALE / K_FACTOR ** np.arange(n_classes)

print(f"n_classes = {n_classes}")
print(f"k_values  = {[f'{k:.0f}' for k in k_values]} m\n")

# ── Per-class grid experiment ─────────────────────────────────────────────────
print(f"{'k (m)':>12}  {'nz':>6}  {'sum_dz (m)':>12}  {'error (m)':>12}  {'error (%)':>10}")
print("-" * 60)

results = []
for k in k_values:
    z     = 0.0
    dzs   = []
    while z < DOMAIN_HEIGHT:
        dz = dz_exact(k, z)
        dzs.append(dz)
        z += dz

    sum_dz = sum(dzs)
    error  = sum_dz - DOMAIN_HEIGHT
    pct    = 100.0 * error / DOMAIN_HEIGHT
    nz     = len(dzs)
    results.append((k, nz, sum_dz, error, pct, dzs))
    print(f"{k:12.0f}  {nz:6d}  {sum_dz:12.2f}  {error:12.6f}  {pct:10.6f}")

# ── Plot: dz vs height for each scale class ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_classes))

ax = axes[0]
for (k, nz, sum_dz, error, pct, dzs), color in zip(results, colors):
    z_centers = np.cumsum(dzs) - np.array(dzs) / 2
    ax.plot(dzs, z_centers / 1e3, color=color, lw=1.5, label=f'k={k/1e3:.0f} km')
ax.set_xlabel('dz (m)')
ax.set_ylabel('Height (km)')
ax.set_title('Grid spacing dz vs height\n(each scale class)')
ax.legend(fontsize=7, loc='upper right')
ax.axhline(DOMAIN_HEIGHT / 1e3, color='k', ls='--', lw=0.8, label='Domain top')
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for (k, nz, sum_dz, error, pct, dzs), color in zip(results, colors):
    z_centers = np.cumsum(dzs) - np.array(dzs) / 2
    ax2.plot(dzs, z_centers / 1e3, color=color, lw=1.5)
ax2.set_xlabel('dz (m)')
ax2.set_xscale('log')
ax2.set_ylabel('Height (km)')
ax2.set_title('Same (log x-axis)')
ax2.axhline(DOMAIN_HEIGHT / 1e3, color='k', ls='--', lw=0.8)
ax2.grid(True, alpha=0.3, which='both')

# overlay ls profile
ax3 = ax2.twiny()
z_fine = np.linspace(0, DOMAIN_HEIGHT, 500)
ax3.plot(ls_of_z(z_fine), z_fine / 1e3, 'k:', lw=1.2, alpha=0.5)
ax3.set_xlabel('ls (m)', color='gray', fontsize=9)
ax3.tick_params(axis='x', labelcolor='gray', labelsize=8)

plt.suptitle(
    f'Height-varying spheroscale: ls={LS_BOT} (bottom) → {LS_TOP} (top)\n'
    f'STEAM config: OUTER_SCALE={OUTER_SCALE/1e3:.0f} km, DOMAIN_HEIGHT={DOMAIN_HEIGHT/1e3:.0f} km, '
    f'H_z={H_Z:.4f}, s_z={S_Z}',
    fontsize=9,
)
plt.tight_layout()
out = Path('experiment_varying_spheroscale.png')
plt.savefig(out, dpi=150)
print(f'\nPlot saved → {out}')
plt.show()
