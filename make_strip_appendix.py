#!/usr/bin/env python3
"""Multifractal-parameter diagnostics on the strip nest (appendix figures).

Thomas's 2026-07-27 spec, for the not-yet-written multifractal appendix:

Figure 1 (normalization test; qt and h only): vertical and horizontal
mean-absolute Haar (M-hat) fluctuation functions of the nested strip,
for a mid-level band (4-8 km window centers) and the full domain,
alongside the vertical M-hat fluctuation function of the input MEAN
PROFILE. The crossover of the field's vertical M-hat toward the
profile's at the vertical outer scale is the direct test of the C_L
Haar normalization.

Figure 2 (all fields h, qt, flux; horizontal only): fluctuation
functions of orders q = 1, 2, 3 vs lag, and the moment-scaling
exponents xi(q) fitted over the nest's scaling range, plus the flux's
box-mean K(q) (the UM object) from 1D coarse-graining along x.

The Haar fluctuation at lag l is mean(upper half-window) - mean(lower
half-window); M-hat_q(l) = <|Haar|^q>. Along x the strip is periodic
(it spans its periodic parent), so windows wrap. Everything is computed
in z-chunks with float64 accumulators.

Writes figs/strip-appx/mhat_normalization.png, kq_fluctuations.png and
stats/strip_appendix.npz.

Usage: python make_strip_appendix.py [parent.nc] [group]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = Path(__file__).parent
PARENT = (Path(sys.argv[1]) if len(sys.argv) > 1
          else HERE / "runs" / "steam_square_icon_lem_snap0.nc")
GROUP = sys.argv[2] if len(sys.argv) > 2 else "refinements/r0"
BAND = (4000.0, 8000.0)          # mid-level window-center band [m]
Z_CHUNK = 16
ORDERS = (1.0, 2.0, 3.0)
XI_ORDERS = np.arange(0.2, 3.01, 0.2)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def haar_x_moments(var, lags_cells, orders, z_select=None):
    """<|Haar_x|^q> per lag along the periodic x axis, pooled over y, z.

    var: netCDF variable (x, y, z). lags_cells: even ints. Returns
    (n_lags, n_orders) float64.
    """
    nx, ny, nz = var.shape
    sums = np.zeros((len(lags_cells), len(orders)))
    counts = np.zeros(len(lags_cells))
    z_indices = np.arange(nz) if z_select is None else np.nonzero(z_select)[0]
    for z0 in range(0, len(z_indices), Z_CHUNK):
        zi = z_indices[z0:z0 + Z_CHUNK]
        field = np.asarray(var[:, :, zi[0]:zi[-1] + 1], dtype=np.float64)
        field = field[:, :, zi - zi[0]] if len(zi) > 1 else field
        # Periodic cumulative sum along x: wrap by tiling the needed head.
        for j, lag in enumerate(lags_cells):
            half = lag // 2
            # upper-half mean minus lower-half mean, windows starting at
            # every x (periodic wrap via np.roll on block sums).
            csum = np.cumsum(
                np.concatenate([field, field[:lag - 1]], axis=0), axis=0)
            block = np.empty((nx, field.shape[1], field.shape[2]))
            block[0] = csum[half - 1]
            block[1:] = csum[half:half + nx - 1] - csum[:nx - 1]
            upper = np.empty_like(block)
            upper[0] = csum[lag - 1] - csum[half - 1]
            upper[1:] = csum[lag:lag + nx - 1] - csum[half:half + nx - 1]
            haar = (upper - block) / half
            for oi, q in enumerate(orders):
                sums[j, oi] += float(np.sum(np.abs(haar) ** q))
            counts[j] += haar.size
    return sums / counts[:, None]


def haar_z_moments(var, z, lags_m, orders, band=None):
    """<|Haar_z|^q> per physical lag, window CENTERS restricted to band.

    The vertical grid is uniform for constant spheroscale; verified below.
    """
    nx, ny, nz = var.shape
    dz = float(np.mean(np.diff(z)))
    assert np.allclose(np.diff(z), dz, rtol=1e-3), "z grid not uniform"
    sums = np.zeros((len(lags_m), len(orders)))
    counts = np.zeros(len(lags_m))
    for x0 in range(0, nx, 1024):
        field = np.asarray(var[x0:x0 + 1024, :, :], dtype=np.float64)
        csum = np.cumsum(field, axis=2)
        for j, lag_m in enumerate(lags_m):
            half = max(1, int(round(lag_m / dz / 2)))
            lag = 2 * half
            if lag >= nz:
                sums[j] = np.nan
                continue
            lower = (csum[:, :, half - 1:nz - half - 1]
                     - np.concatenate([
                         np.zeros_like(csum[:, :, :1]),
                         csum[:, :, :nz - lag - 1]], axis=2))
            upper = (csum[:, :, lag - 1:nz - 1]
                     - csum[:, :, half - 1:nz - half - 1])
            haar = (upper - lower) / half
            if band is not None:
                centers = z[half:nz - half]
                keep = (centers >= band[0]) & (centers <= band[1])
                haar = haar[:, :, keep]
            if haar.size == 0:
                sums[j] = np.nan
                continue
            for oi, q in enumerate(orders):
                sums[j, oi] += float(np.sum(np.abs(haar) ** q))
            counts[j] += haar.size
    return sums / np.where(counts > 0, counts, 1)[:, None]


def profile_mhat(profile, z, lags_m):
    """Mean absolute Haar of the 1D mean profile at each lag."""
    out = np.full(len(lags_m), np.nan)
    dz = float(np.mean(np.diff(z)))
    csum = np.cumsum(np.asarray(profile, dtype=np.float64))
    n = profile.size
    for j, lag_m in enumerate(lags_m):
        half = max(1, int(round(lag_m / dz / 2)))
        lag = 2 * half
        if lag >= n:
            continue
        lower = csum[half - 1:n - half - 1] - np.concatenate(
            [[0.0], csum[:n - lag - 1]])
        upper = csum[lag - 1:n - 1] - csum[half - 1:n - half - 1]
        out[j] = float(np.mean(np.abs(upper - lower) / half))
    return out


def flux_box_kq(var, box_cells, orders):
    """<F_lambda^q> per box size: 1D coarse-graining along x, pooled y,z."""
    nx, ny, nz = var.shape
    moments = np.zeros((len(box_cells), len(orders)))
    counts = np.zeros(len(box_cells))
    for z0 in range(0, nz, Z_CHUNK):
        field = np.asarray(var[:, :, z0:z0 + Z_CHUNK], dtype=np.float64)
        for j, box in enumerate(box_cells):
            coarse = field[:nx - nx % box].reshape(
                nx // box, box, field.shape[1], field.shape[2]).mean(axis=1)
            for oi, q in enumerate(orders):
                moments[j, oi] += float(np.sum(coarse ** q))
            counts[j] += coarse.size
    return moments / counts[:, None]


def main():
    ds = netCDF4.Dataset(PARENT)
    grp = ds
    for part in GROUP.split("/"):
        grp = grp.groups[part]
    grp.set_auto_mask(False)
    x = grp.variables["x"][:]
    z = grp.variables["z"][:].astype(np.float64)
    dx = float(np.mean(np.diff(x)))
    nx = x.size

    # Horizontal dyadic lags: 2 cells (2*375 m) up to nx/4.
    lags_cells = [2 ** j for j in range(1, int(np.log2(nx // 4)) + 1)]
    lags_x = np.array([lag * dx for lag in lags_cells])
    # Vertical lags: dyadic in metres from ~2 dz up to 16 km.
    dz = float(np.mean(np.diff(z)))
    lags_z = np.array([dz * 2 * 2 ** j for j in range(0, 8)])
    lags_z = lags_z[lags_z < 16000.0]

    out = {"lags_x": lags_x, "lags_z": lags_z}
    band_mask = (z >= BAND[0]) & (z <= BAND[1])

    for name in ("h", "qt"):
        var = grp.variables[name]
        print(f"{name}: horizontal M-hat (full, band)...", flush=True)
        out[f"{name}_mhat_x_full"] = haar_x_moments(var, lags_cells, ORDERS)
        out[f"{name}_mhat_x_band"] = haar_x_moments(
            var, lags_cells, ORDERS, z_select=band_mask)
        print(f"{name}: vertical M-hat (full, band)...", flush=True)
        out[f"{name}_mhat_z_full"] = haar_z_moments(var, z, lags_z, ORDERS)
        out[f"{name}_mhat_z_band"] = haar_z_moments(
            var, z, lags_z, ORDERS, band=BAND)
        profile = grp.variables[f"{name}_profile"][:]
        z_profile = grp.variables["z_profile"][:].astype(np.float64)
        out[f"{name}_profile_mhat_z"] = profile_mhat(profile, z_profile, lags_z)
        print(f"{name}: xi(q) fit...", flush=True)
        out[f"{name}_mhat_x_xi"] = haar_x_moments(var, lags_cells, XI_ORDERS)

    print("flux: horizontal M-hat + box K(q)...", flush=True)
    fvar = grp.variables["flux"]
    out["flux_mhat_x_full"] = haar_x_moments(fvar, lags_cells, ORDERS)
    out["flux_mhat_x_xi"] = haar_x_moments(fvar, lags_cells, XI_ORDERS)
    box_cells = lags_cells
    out["flux_box_moments"] = flux_box_kq(fvar, box_cells, XI_ORDERS)
    ds.close()

    np.savez(HERE / "stats" / "strip_appendix.npz", **out)

    figdir = HERE / "figs" / "strip-appx"
    figdir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: normalization test (q = 1 M-hat), h and qt ──
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    for ax, name, unit in zip(axes, ("h", "qt"),
                              ("J kg$^{-1}$", "kg kg$^{-1}$")):
        ax.loglog(lags_x / 1000, out[f"{name}_mhat_x_full"][:, 0],
                  color="#1764ab", lw=1.4, label="horizontal, full domain")
        ax.loglog(lags_x / 1000, out[f"{name}_mhat_x_band"][:, 0],
                  color="#1764ab", lw=1.1, ls="--", label="horizontal, 4-8 km")
        ax.loglog(lags_z / 1000, out[f"{name}_mhat_z_full"][:, 0],
                  color="#e76f51", lw=1.4, label="vertical, full domain")
        ax.loglog(lags_z / 1000, out[f"{name}_mhat_z_band"][:, 0],
                  color="#e76f51", lw=1.1, ls="--", label="vertical, 4-8 km")
        ax.loglog(lags_z / 1000, out[f"{name}_profile_mhat_z"],
                  color="0.25", lw=1.4, ls=":", label="mean profile, vertical")
        ax.set(xlabel="lag [km]", ylabel=f"$\\hat{{M}}_1$ [{unit}]",
               title=name)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=7)
    fig.suptitle("Strip-nest M-hat fluctuation functions vs the mean profile "
                 "(normalization test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "mhat_normalization.png")
    plt.close(fig)

    # ── Figure 2: orders 1,2,3 + xi(q)/K(q), horizontal ──
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    colors = {"h": "#1764ab", "qt": "#e76f51", "flux": "#2a9d8f"}
    for name in ("h", "qt", "flux"):
        m = out[f"{name}_mhat_x_full"]
        for oi, q in enumerate(ORDERS):
            axes[0].loglog(lags_x / 1000, m[:, oi] / m[0, oi],
                           color=colors[name], lw=1.2, alpha=1 - 0.3 * oi,
                           label=f"{name} q={q:g}" if oi == 0 else None)
    axes[0].set(xlabel="lag [km]",
                ylabel="$\\hat{M}_q(\\ell)\\,/\\,\\hat{M}_q(\\ell_0)$",
                title="horizontal fluctuation functions, q = 1, 2, 3")

    # xi(q) from log-log slope over the nest's own classes (< 3 km) plus
    # one octave of inherited scales; flux K(q) from box moments over the
    # same range.
    fit = lags_x <= 6000.0
    for name in ("h", "qt", "flux"):
        m = out[f"{name}_mhat_x_xi"]
        xi = [np.polyfit(np.log(lags_x[fit]), np.log(m[fit, oi]), 1)[0]
              for oi in range(len(XI_ORDERS))]
        axes[1].plot(XI_ORDERS, xi, color=colors[name], lw=1.3,
                     label=f"{name} $\\xi(q)$")
    box = out["flux_box_moments"]
    kq = [-np.polyfit(np.log(lags_x[fit]), np.log(box[fit, oi]), 1)[0]
          for oi in range(len(XI_ORDERS))]
    axes[1].plot(XI_ORDERS, kq, color=colors["flux"], lw=1.3, ls="--",
                 label="flux $K(q)$ (box)")
    axes[1].axhline(0, color="#9A938B", lw=0.7)
    axes[1].set(xlabel="q", ylabel="exponent",
                title="moment-scaling exponents (fit: lags $\\leq$ 6 km)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=7)
    fig.suptitle("Strip-nest horizontal moment scaling (h, $q_t$, flux)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "kq_fluctuations.png")
    plt.close(fig)

    print("wrote figs/strip-appx/mhat_normalization.png, kq_fluctuations.png")


if __name__ == "__main__":
    main()
