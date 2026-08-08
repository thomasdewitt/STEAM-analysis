#!/usr/bin/env python3
"""Shared pieces of the hydrodynamic comparison: the matching rule, the
per-level reduction, and the figure styling.

The matching rule is here rather than duplicated because it is the
methodological content of the comparison -- if the TWPICE and RCEMIP sides
ever disagreed about how two grids are brought together, neither figure would
mean what its caption said.
"""

import numpy as np
import matplotlib.pyplot as plt

VARS = ("h", "qt", "qc", "qi")
CLOUD_KGKG = 0.01e-3           # condensate for a cell to count as cloudy
PDF_LEVELS = (5000.0, 10000.0)

# variable -> (label, unit scale, unit name)
UNITS = {
    "h":  (r"$h$",   1e-3, "kJ kg$^{-1}$"),
    "qt": (r"$q_t$", 1e3,  "g kg$^{-1}$"),
    "qc": (r"$q_c$", 1e3,  "g kg$^{-1}$"),
    "qi": (r"$q_i$", 1e3,  "g kg$^{-1}$"),
}

# paper/concept-figs/turblib.py
INK = "#111111"
RULE = "#e3e3e3"
LABEL = "#7a7a7a"
# The hosts are the reference and take the ink; the STEAM amplitudes take
# three of turblib's palette entries -- ochre, deep teal, sienna. Ochre for
# c002 rather than the palette's slate blue: on the scaling figures the host
# lines are thin ink, and slate read as a washed-out black beside them.
COLOR = {"host": INK, "c002": "#C08A2D", "c005": "#1F6E6B",
         "c017": "#B5502A"}


def rcparams():
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def style(ax, panel, xlabel, ylabel):
    ax.set_title(panel, color=INK, pad=6, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(which="both", length=3, width=0.8, colors=LABEL)
    ax.tick_params(which="minor", length=1.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def coarsen_xy(a, f):
    """Block-mean over the two trailing axes by a factor f, in float64."""
    nz, ny, nx = a.shape
    return a.reshape(nz, ny // f, f, nx // f, f).mean(axis=(2, 4),
                                                      dtype=np.float64)


def coarsen_factor(steam_dx, host_dx):
    """Horizontal block factor that puts the host on STEAM's spacing.

    Derived from the run's own dx rather than written down, for the same
    reason match_factors derives the vertical factors from the two z axes:
    these figures claim the two grids are on one horizontal spacing, and a
    constant here would keep claiming it after a domain in
    run_steam_simulations.py moved. A non-integer ratio is fatal -- block
    averaging cannot express it, and silently rounding would mismatch the
    grids by whatever the rounding threw away.
    """
    f = steam_dx / host_dx
    if abs(f - round(f)) > 1e-9 or round(f) < 1:
        raise SystemExit(
            f"STEAM dx = {steam_dx:g} m is not an integer multiple of the "
            f"host's {host_dx:g} m (ratio {f:g}); the two grids cannot be "
            f"matched by block averaging")
    return int(round(f))


def match_factors(z_host, z_steam):
    """Per-level coarsening factors, host and STEAM, under the standing rule.

    At each host level, the finer of the two grids is averaged over the
    nearest integer number of its own levels that brings the two spacings
    closest together. Only one factor exceeds 1 at any level; the residual
    mismatch is whatever the nearest integer leaves, typically under 1.5x.
    """
    dz_host = np.gradient(z_host)
    dz_steam = np.gradient(z_steam)
    n_steam = np.empty(z_host.size, dtype=int)
    n_host = np.empty(z_host.size, dtype=int)
    for j, z in enumerate(z_host):
        ds = dz_steam[np.argmin(np.abs(z_steam - z))]
        ratio = dz_host[j] / ds
        n_steam[j] = max(1, int(round(ratio)))
        n_host[j] = max(1, int(round(1.0 / ratio)))
    return n_steam, n_host


def level_slice(field, z_axis, z_target, n):
    """Mean of the n levels centred on z_target, as a 2D field.

    field is indexed with z LAST. n is the coarsening factor for this level.
    """
    i0 = int(np.argmin(np.abs(z_axis - z_target))) - n // 2
    i0 = min(max(i0, 0), z_axis.size - n)
    return field[..., i0:i0 + n].mean(axis=-1, dtype=np.float64)


def reduce_source(z_axis, fields, z_levels, factors):
    """Per-level std and cloud fraction, plus the fields the PDFs use.

    Cloud fraction is thresholded AFTER coarsening: the coarsened cell value
    is what the resolved field says is there.
    """
    std = {v: np.empty(z_levels.size) for v in VARS}
    cf = np.empty(z_levels.size)
    for j, (z, n) in enumerate(zip(z_levels, factors)):
        for v in VARS:
            std[v][j] = level_slice(fields[v], z_axis, z, n).std()
        cond = (level_slice(fields["qc"], z_axis, z, n)
                + level_slice(fields["qi"], z_axis, z, n))
        cf[j] = float((cond >= CLOUD_KGKG).mean())
    slices = {}
    for z_pdf in PDF_LEVELS:
        j = int(np.argmin(np.abs(z_levels - z_pdf)))
        for v in VARS:
            slices[f"{v}_{z_pdf / 1000:.0f}km"] = level_slice(
                fields[v], z_axis, z_levels[j], factors[j]).astype(np.float32)
    return std, cf, slices
