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
# The hosts are the reference and take the ink. The STEAM amplitudes use the
# same three turblib entries as before, rotated so they run green -> yellow
# -> red with increasing c and the ladder reads in order without the legend:
# deep teal, ochre, sienna. No new colours -- teal is the green end of the
# palette already in use, and a true yellow would be illegible on white.
COLOR = {"host": INK, "c002": "#1F6E6B", "c005": "#C08A2D",
         "c017": "#B5502A"}

# The envelope figures name this explicitly rather than borrowing an
# amplitude's colour: there the amplitudes are pooled into one band, so the
# band's colour is not about c, and rotating the ladder above must not
# repaint it.
STEAM_BAND = "#1F6E6B"

# Amplitude tags in increasing c, so legends read in order whatever order
# the npz happened to store them in.
AMPLITUDE_ORDER = ("c002", "c005", "c017")


def by_amplitude(tags):
    """The given amplitude tags, ordered by increasing c.

    Unknown tags keep their original order and go last, so a set added to
    a run script but not to AMPLITUDE_ORDER still plots.
    """
    known = [t for t in AMPLITUDE_ORDER if t in tags]
    return known + [t for t in tags if t not in AMPLITUDE_ORDER]


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


def coarsen_xyz(a, f):
    """Block-mean by f along EVERY axis, in float64.

    The hydrodynamic outputs are coarsened in f x f x f blocks before any
    one-point statistic is taken, to keep the standard deviations off the
    host's own grid scale where numerical artifacts live (main.tex, one-point
    statistics). Until 2026-08-10 only the two horizontal axes were
    coarsened.

    Takes any rank, so the same call coarsens a (z, y, x) field and the (z,)
    coordinate that indexes it. That is the point of one function rather than
    a horizontal one and a vertical one: a field and its own axis cannot end
    up on different grids.

    Cells that do not fill a whole block are dropped from the end of each
    axis. TWPICE has 255 levels, so its topmost level goes; it sits above
    19.8 km and outside the clip to STEAM's top in any case, and half a
    block would be a level of a different thickness inside a rule that
    claims uniform blocks.
    """
    a = np.asarray(a)
    trimmed = a[tuple(slice(0, (n // f) * f) for n in a.shape)]
    blocked = []
    for n in trimmed.shape:
        blocked += [n // f, f]
    return trimmed.reshape(blocked).mean(
        axis=tuple(range(1, 2 * a.ndim, 2)), dtype=np.float64)


def coarsen_factor(steam_dx, host_dx):
    """Horizontal block factor that puts the host on STEAM's spacing.

    Derived from the run's own dx rather than written down, for the same
    reason match_factors derives the vertical factors from the two z axes:
    these figures claim the two grids are on one horizontal spacing, and a
    constant here would keep claiming it after a domain in
    run_rcemip_simulations.py moved. A non-integer ratio is fatal -- block
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


def level_planes(z_axis, fields, z_levels, factors):
    """Every comparison level as a 2-D plane: {var: (nlev, ny, nx)}.

    Keeping the planes rather than reducing them on the spot is what lets a
    caller with several snapshots hand the whole pooled sample to numpy in
    one call. The planes are small -- a coarsened RCEMIP host is ~64 x 992
    per level, so all levels of all four variables for one snapshot come to
    ~90 MB, against gigabytes for the field they were sliced from.
    """
    return {v: np.stack([level_slice(fields[v], z_axis, z, n)
                         for z, n in zip(z_levels, factors)])
            for v in VARS}


def reduce_source(z_axis, fields, z_levels, factors):
    """Per-level std and cloud fraction, plus the fields the PDFs use.

    Cloud fraction is thresholded AFTER coarsening: the coarsened cell value
    is what the resolved field says is there.

    The std here is over one source's horizontal plane, which is the right
    quantity only when there is one source. Where several snapshots or
    realizations are being combined, the wanted quantity is the statistic of
    the pooled sample: keep level_planes and reduce them together.
    """
    std = {v: np.empty(z_levels.size) for v in VARS}
    cf = np.empty(z_levels.size)
    for j, (z, n) in enumerate(zip(z_levels, factors)):
        for v in VARS:
            std[v][j] = level_slice(fields[v], z_axis, z, n).std()
        cond = (level_slice(fields["qc"], z_axis, z, n)
                + level_slice(fields["qi"], z_axis, z, n))
        cf[j] = float((cond >= CLOUD_KGKG).mean())
    return std, cf, pdf_slices(z_axis, fields, z_levels, factors)


def pdf_slices(z_axis, fields, z_levels, factors):
    """The single-level fields the PDFs are drawn from.

    Separate from reduce_source so a caller that pools its own level_planes
    can take these without reducing every level a second time.
    """
    slices = {}
    for z_pdf in PDF_LEVELS:
        j = int(np.argmin(np.abs(z_levels - z_pdf)))
        for v in VARS:
            slices[f"{v}_{z_pdf / 1000:.0f}km"] = level_slice(
                fields[v], z_axis, z_levels[j], factors[j]).astype(np.float32)
    return slices
