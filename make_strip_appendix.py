#!/usr/bin/env python3
"""Multifractal-parameter diagnostics on the strip nest (appendix figures).

Thomas's 2026-07-27 spec, for the not-yet-written multifractal appendix:

Figure 1 (normalization test; qt and h only): vertical and horizontal
mean-absolute Haar (M-hat) fluctuation functions of the nested strip,
for a mid-level band (6-8 km window centers) and the full domain,
alongside the vertical M-hat fluctuation function of the input MEAN
PROFILE. The crossover of the field's vertical M-hat toward the
profile's at the vertical outer scale is the direct test of the C_L
Haar normalization.

Figure 2 (all fields h, qt, flux; horizontal only): fluctuation
functions of orders q = 1, 2, 3 vs lag, and the moment-scaling
exponents xi(q) fitted over the nest's scaling range, plus the flux's
box-mean K(q) (the UM object) from 1D coarse-graining along x.

All fluctuation statistics use the CANONICAL estimators from Thomas's
scaleinvariance package (haar_fluctuation with explicit dyadic lags;
L1 fluctuation normalization; periodic along x since the strip spans
its periodic parent). The mid-level band restricts to the 6-8 km slab
(windows within it). The flux box-mean K(q) keeps the calibration
estimator for direct comparability with C20.

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
BAND = (6000.0, 8000.0)          # mid-level window-center band [m]
Z_CHUNK = 16
ORDERS = (1.0, 2.0, 3.0)
XI_ORDERS = np.arange(0.2, 3.01, 0.2)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def haar_moments(field, lags_cells, orders, axis, periodic):
    """Canonical mean-absolute-Haar moments via scaleinvariance.

    Returns (n_lags, n_orders) to preserve this script's npz layout
    (scaleinvariance returns (n_orders, n_lags)).
    """
    import scaleinvariance as si
    lags_used, F = si.haar_fluctuation(
        field, order=np.asarray(orders, dtype=float), axis=axis,
        lags=np.asarray(lags_cells, dtype=int), periodic=periodic)
    if not np.array_equal(lags_used, np.asarray(lags_cells)):
        raise ValueError(f"lags altered by estimator: {lags_used}")
    F = np.atleast_2d(F)
    return np.asarray(F).T


def profile_mhat(profile, z, lags_m):
    """Mean absolute Haar of the 1D mean profile at each physical lag."""
    import scaleinvariance as si
    dz = float(np.mean(np.diff(z)))
    lags_cells = np.unique(np.maximum(
        2, (np.round(np.asarray(lags_m) / dz / 2) * 2).astype(int)))
    _, F = si.haar_fluctuation(
        np.asarray(profile, dtype=np.float64), order=1.0, axis=0,
        lags=lags_cells)
    out = np.full(len(lags_m), np.nan)
    for j, lag_m in enumerate(lags_m):
        cell = max(2, int(round(lag_m / dz / 2)) * 2)
        i = np.searchsorted(lags_cells, cell)
        if i < len(lags_cells) and lags_cells[i] == cell:
            out[j] = F[i] if F.ndim == 1 else F[0, i]
    return out


def flux_box_kq_array(field, box_cells, orders):
    """<F_lambda^q> per box size: 1D coarse-graining along x, pooled y,z.

    Kept alongside the scaleinvariance estimators deliberately: this is the
    UM box-mean object, the same estimator as the C1 calibration
    (calibration/flux_c1_calibration.py), for direct comparability.
    """
    nx = field.shape[0]
    moments = np.zeros((len(box_cells), len(orders)))
    for j, box in enumerate(box_cells):
        coarse = field[:nx - nx % box].astype(np.float64).reshape(
            nx // box, box, field.shape[1], field.shape[2]).mean(axis=1)
        for oi, q in enumerate(orders):
            moments[j, oi] = float(np.mean(coarse ** q))
    return moments


def reference_slope(ax, x_center, y_center, slope, span_decades=1.8,
                    color="0.4", label=None):
    """Short reference-slope segment pegged to (x_center, y_center)."""
    half = 10 ** (span_decades / 2)
    xs = np.array([x_center / half, x_center * half])
    ys = y_center * (xs / x_center) ** slope
    ax.plot(xs, ys, color=color, lw=0.9, ls="-.", alpha=0.9)
    if label:
        ax.annotate(label, (xs[1], ys[1]), fontsize=6.5, color=color,
                    xytext=(2, -2), textcoords="offset points")


H_H = 0.45                  # production horizontal Hurst
H_V = 0.45 / (5.0 / 9.0)    # vertical Hurst = H_h / H_z anisotropy = 0.81


def main():
    if "--figures-only" in sys.argv:
        d = np.load(HERE / "stats" / "strip_appendix.npz")
        out = {k: d[k] for k in d.files}
        lags_x = out["lags_x"]
        lags_z = out["lags_z"]
        make_figures(out, lags_x, lags_z)
        return
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
    # Vertical lags: dyadic in CELLS from 2 dz up to ~14 km.
    dz = float(np.mean(np.diff(z)))
    lags_z_cells = np.array([2 * 2 ** j for j in range(0, 8)])
    lags_z_cells = lags_z_cells[lags_z_cells * dz < 16000.0]
    lags_z = lags_z_cells * dz

    out = {"lags_x": lags_x, "lags_z": lags_z}
    band_mask = (z >= BAND[0]) & (z <= BAND[1])

    slab = np.nonzero(band_mask)[0]
    all_orders = np.asarray(sorted(set(ORDERS) | set(np.round(XI_ORDERS, 3))))
    idx_orders = [int(np.searchsorted(all_orders, q)) for q in ORDERS]
    idx_xi = [int(np.searchsorted(all_orders, round(q, 3)))
              for q in XI_ORDERS]

    for name in ("h", "qt", "flux"):
        print(f"{name}: loading...", flush=True)
        field = np.asarray(grp.variables[name][:], dtype=np.float32)
        print(f"{name}: horizontal M-hat (full, band)...", flush=True)
        m = haar_moments(field, lags_cells, all_orders, axis=0, periodic=True)
        out[f"{name}_mhat_x_full"] = m[:, idx_orders]
        out[f"{name}_mhat_x_xi"] = m[:, idx_xi]
        if name != "flux":
            out[f"{name}_mhat_x_band"] = haar_moments(
                field[:, :, slab], lags_cells, ORDERS, axis=0, periodic=True)
            print(f"{name}: vertical M-hat (full, band)...", flush=True)
            out[f"{name}_mhat_z_full"] = haar_moments(
                field, lags_z_cells, ORDERS, axis=2, periodic=False)
            band_lags = lags_z_cells[lags_z_cells < slab.size]
            band_m = haar_moments(
                field[:, :, slab], band_lags, ORDERS, axis=2, periodic=False)
            padded = np.full((len(lags_z_cells), len(ORDERS)), np.nan)
            padded[:len(band_lags)] = band_m
            out[f"{name}_mhat_z_band"] = padded
            profile = grp.variables[f"{name}_profile"][:]
            z_profile = grp.variables["z_profile"][:].astype(np.float64)
            out[f"{name}_profile_mhat_z"] = profile_mhat(
                profile, z_profile, lags_z)
        else:
            print("flux: box K(q)...", flush=True)
            out["flux_box_moments"] = flux_box_kq_array(field, lags_cells,
                                                        XI_ORDERS)
        del field
    ds.close()

    np.savez(HERE / "stats" / "strip_appendix.npz", **out)
    make_figures(out, lags_x, lags_z)


def make_figures(out, lags_x, lags_z):
    figdir = HERE / "figs" / "strip-appx"
    figdir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: normalization test (q = 1 M-hat), h and qt ──
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    for ax, name, unit in zip(axes, ("h", "qt"),
                              ("J kg$^{-1}$", "kg kg$^{-1}$")):
        ax.loglog(lags_x / 1000, out[f"{name}_mhat_x_full"][:, 0],
                  color="#1764ab", lw=1.4, label="horizontal, full domain")
        ax.loglog(lags_x / 1000, out[f"{name}_mhat_x_band"][:, 0],
                  color="#1764ab", lw=1.1, ls="--", label="horizontal, 6-8 km")
        ax.loglog(lags_z / 1000, out[f"{name}_mhat_z_full"][:, 0],
                  color="#e76f51", lw=1.4, label="vertical, full domain")
        ax.loglog(lags_z / 1000, out[f"{name}_mhat_z_band"][:, 0],
                  color="#e76f51", lw=1.1, ls="--", label="vertical, 6-8 km")
        ax.loglog(lags_z / 1000, out[f"{name}_profile_mhat_z"],
                  color="0.25", lw=1.4, ls=":", label="mean profile, vertical")
        # Reference slopes pegged to each line's central value.
        for curve, lags, slope, lab in (
                (out[f"{name}_mhat_x_full"][:, 0], lags_x, H_H, "$H_h$"),
                (out[f"{name}_mhat_x_band"][:, 0], lags_x, H_H, None),
                (out[f"{name}_mhat_z_full"][:, 0], lags_z, H_V, "$H_v$"),
                (out[f"{name}_mhat_z_band"][:, 0], lags_z, H_V, None)):
            good = np.isfinite(curve)
            mid = np.nonzero(good)[0][good.sum() // 2]
            reference_slope(ax, lags[mid] / 1000, curve[mid], slope,
                            label=lab)
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
    ref = out["h_mhat_x_full"][:, 0] / out["h_mhat_x_full"][0, 0]
    mid = len(lags_x) // 2
    reference_slope(axes[0], lags_x[mid] / 1000, ref[mid], H_H,
                    label="$H_h$")
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
