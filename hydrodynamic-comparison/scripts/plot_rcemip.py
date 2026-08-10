#!/usr/bin/env python3
"""Plot the matched RCEMIP-channel comparison from compute_rcemip_stats.py.

Same two figures as the TWPICE side, across all nine hosts at once:

  rcemip_profiles  per-level standard deviation of h, qt, T, p, qc and qi,
                   plus cloud fraction
  rcemip_pdfs      single-level distributions of h, qt, qc and qi at 5 and
                   10 km

T AND p JOINED THE PROFILE FIGURE ON 2026-08-10, drawn exactly like the four
that were already there. They are on the profiles only; the PDFs keep the
original four.

The pressure panel is over FEWER HOSTS than the rest, and both bands in it
are restricted to the same reduced set: scale and ucla archive no 3-D
pressure, so nine hosts become seven. Restricting only the host band would
compare seven models' spread against twenty-seven STEAM runs driven by nine,
which is not the comparison the panel claims. The panel is labelled with its
own host count and the omission is printed at run time.

AND IT IS NOT LIKE-FOR-LIKE AT THE BOTTOM. The hosts report their own full
pressure, surface variability included -- ~40 Pa of it at the lowest levels.
STEAM marches every column from ONE surface pressure, so its std(p) is
identically zero there and can only accumulate upward. The low-level
non-overlap in that panel is that boundary condition, not a result. What is
a result is aloft, where STEAM reaches ~400 Pa against the hosts' ~20-50 --
and even that is largely the std(T) excess in panel c integrated up the
column rather than an independent measurement.

ONE FIGURE PER OUTER SCALE (2026-08-08), so each stem carries the tag:
rcemip_profiles_Llong, rcemip_profiles_Lshort, and likewise for the PDFs.
L is set to the channel's long axis (6144 km) or its short one (384 km),
and the two are not pooled -- see members().

Both figures show ENVELOPES rather than individual trajectories: at each
level (or each bin) the minimum and maximum over all nine hosts, and the
minimum and maximum over all twenty-seven STEAM runs at that outer scale,
drawn as two shaded regions. No individual model was identifiable in the old
spaghetti version and none was meant to be -- the figure is about whether
STEAM falls inside the spread the hydrodynamic models already show among
themselves, which is exactly what the overlap of the two bands says. The
three flux amplitudes are pooled into the single STEAM envelope for the same
reason the nine hosts are pooled into one: the amplitude is a STEAM knob the
way model identity is a host knob, and the question is the width of the
spread it produces.

The hosts sit on three different vertical grids (the 47-level RCEMIP grid,
UKMO's 98-level grid cut to 78, and ICON's 46), so a min/max across hosts
needs one grid to be taken on. COMMON_Z is the coarsest host grid clipped to
the height range every host covers, and every profile is linearly
interpolated onto it before the envelope is formed.

Beside each profile panel is a bar keyed to height: green where the two
envelopes overlap at that level, red where they are non-overlapping.

Condensate panels are cut at the 0.01 g/kg cloud threshold, as on the TWPICE
side.

Usage: python plot_rcemip.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

from common import (VARS, STD_VARS, UNITS, INK, COLOR, STEAM_BAND, rcparams,
                    style)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
DATA = OUTPUT / "rcemip_stats.npz"

# The two envelopes: display name, edge colour, fill opacity. "host" is the
# reference and keeps the ink; STEAM takes the teal of the pair.
GROUPS = {
    "host": ("RCEMIP hosts", INK, 0.16),
    "steam": ("STEAM", STEAM_BAND, 0.28),
}
# The flux amplitudes pooled into the single STEAM band. An envelope is a
# claim about what varies WITHIN it, so a new axis in the npz has to be
# answered here rather than swept into the existing band.
STEAM_SETS = ("c002", "c005", "c017")
# The outer scale is the axis added on 2026-08-08, and it is answered by
# SEPARATING rather than pooling: one figure per outer scale, each carrying
# the STEAM band for that L alone. Pooling the two would have merged the
# very distinction the axis exists to show, and a 6144 km cascade and a
# 384 km one are not two samples of one spread.
OVERLAP = "#2E7D4F"
DISJOINT = "#B03A2E"

EDGE_LW = 0.9
BAR_X = (1.025, 1.075)             # bar extent in axes-fraction x

rcparams()


def common_z(d, hosts):
    """The coarsest host grid, clipped to the range every host covers.

    A min/max across hosts is only defined once they share a grid, and the
    three RCEMIP grids here differ by more than rounding. Taking the coarsest
    rather than inventing a uniform one keeps the levels at heights some host
    actually reported, and clipping to the common range keeps the envelope
    from being an extrapolation at either end.
    """
    z = min((d[f"z_{h}"] for h in hosts), key=lambda a: a.size)
    lo = max(d[f"z_{h}"][0] for h in hosts)
    hi = min(d[f"z_{h}"][-1] for h in hosts)
    # Tolerance: the two ICON grids differ in the third decimal of a metre,
    # and a strict comparison spends the surface level and the top level on
    # that. Levels agreeing to within a metre are the same level.
    return z[(z >= lo - 1.0) & (z <= hi + 1.0)]


def members(sources):
    """Which npz source tags feed each envelope.

    The one seam between the file on disk and the figure. Pooling the flux
    amplitudes into a single STEAM band is a deliberate reading of what the
    figure asks -- the amplitude is a STEAM knob the way model identity is a
    host knob -- but it is only defensible for the tags it was decided for.
    Sweeping up whatever else appears would quietly merge a distinction
    someone added the axis to show, and the band would still look like an
    answer. So an unfamiliar tag stops the figure instead.

    STEAM tags are `<set>_<lscale>`. Every tag reaching one figure must carry
    the same outer scale: a band spanning both would be the silent merge this
    check exists to prevent, and it is main() that separates them.
    """
    steam = [s for s in sources if s != "host"]
    unknown = [s for s in steam if s.rsplit("_", 1)[0] not in STEAM_SETS]
    if unknown:
        raise SystemExit(
            f"rcemip_stats.npz carries STEAM source tags {unknown} that this "
            f"figure has no grouping for (knows amplitudes "
            f"{list(STEAM_SETS)}). Decide whether they belong in the existing "
            f"STEAM envelope or want one of their own, then say so in "
            f"GROUPS/STEAM_SETS/members().")
    scales = {s.rsplit("_", 1)[1] for s in steam}
    if len(scales) > 1:
        raise SystemExit(
            f"one figure was handed more than one outer scale "
            f"({sorted(scales)}); these get a figure each, not a shared band")
    return {"host": ["host"], "steam": steam}


def group_names(L_km):
    """Display names for one figure.

    The STEAM band is labelled with the outer scale it was actually run at,
    in kilometres, rather than with the internal tag -- `Llong` means nothing
    to a reader of the paper, and the number is the thing being varied.
    """
    return {"host": GROUPS["host"][0],
            "steam": rf"STEAM  (all $c$),  $L = {L_km:g}$ km"}


def envelope(d, hosts, tags, key, z, x_of):
    """Per-level min and max over every (host, tag) curve, on the grid z."""
    curves = [np.interp(z, d[f"z_{h}"], x_of(d[f"{key}_{h}_{t}"]))
              for t in tags for h in hosts]
    stack = np.stack(curves)
    return stack.min(axis=0), stack.max(axis=0)


def draw(ax, d, hosts, sources, key, x_of, z):
    """The two envelopes, and the overlap bar beside them."""
    bands = {}
    for group, tags in members(sources).items():
        lo, hi = envelope(d, hosts, tags, key, z, x_of)
        _, colour, alpha = GROUPS[group]
        ax.fill_betweenx(z / 1000.0, lo, hi, color=colour, alpha=alpha,
                         lw=0)
        for edge in (lo, hi):
            ax.plot(edge, z / 1000.0, color=colour, lw=EDGE_LW,
                    solid_capstyle="round")
        bands[group] = (lo, hi)

    (h_lo, h_hi), (s_lo, s_hi) = bands["host"], bands["steam"]
    overlap_bar(ax, z / 1000.0,
                np.maximum(h_lo, s_lo) <= np.minimum(h_hi, s_hi))


def overlap_bar(ax, z_km, ok):
    """A vertical bar just outside the right spine, one colour per level.

    Blended transform: x in axes fractions so the bar keeps its width and
    position whatever the panel's units, y in data so it stays registered
    with the profiles it describes.
    """
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    # Level midpoints, so the segments meet rather than leaving gaps.
    edges = np.concatenate([[z_km[0]],
                            0.5 * (z_km[1:] + z_km[:-1]),
                            [z_km[-1]]])
    x0, x1 = BAR_X
    start = 0
    for j in range(1, ok.size + 1):
        if j == ok.size or ok[j] != ok[start]:
            ax.add_patch(Rectangle(
                (x0, edges[start]), x1 - x0, edges[j] - edges[start],
                transform=tr, clip_on=False, lw=0,
                color=OVERLAP if ok[start] else DISJOINT))
            start = j


def legend_handles(sources, names):
    handles = [Patch(facecolor=GROUPS[g][1], edgecolor=GROUPS[g][1],
                     alpha=max(GROUPS[g][2], 0.35), label=names[g])
               for g in members(sources)]
    handles += [Patch(facecolor=OVERLAP, edgecolor="none",
                      label="envelopes overlap"),
                Patch(facecolor=DISJOINT, edgecolor="none",
                      label="envelopes non-overlapping")]
    return handles


def var_hosts(d, hosts, v):
    """The hosts behind one panel: everyone, except p's reduced set.

    Read off the file rather than named here, because which hosts archive a
    3-D pressure is a property of the datasets and the compute step is what
    looked.
    """
    if v != "p":
        return hosts
    return [h for h in [str(x) for x in d["p_hosts"]] if h in hosts]


NCOL = 4                           # 6 std panels + cloud fraction + legend


def profiles(d, hosts, sources, names):
    z = common_z(d, hosts)
    fig, axes = plt.subplots(2, NCOL, figsize=(11.6, 6.0), sharey=True)
    flat = axes.ravel()
    panels = iter("abcdefg")

    for ax, v in zip(flat, STD_VARS):
        label, scale, unit = UNITS[v]
        vh = var_hosts(d, hosts, v)
        draw(ax, d, vh, sources, f"std_{v}", lambda a, s=scale: a * s, z)
        # Only a panel drawn over a reduced host set says how many; the rest
        # are the figure's stated nine and repeating it would be noise.
        note = "" if len(vh) == len(hosts) else f"  ({len(vh)} hosts)"
        style(ax, next(panels), f"std({label})  [{unit}]{note}",
              "height  [km]" if ax in (flat[0], flat[NCOL]) else "")

    cf_ax = flat[len(STD_VARS)]
    draw(cf_ax, d, hosts, sources, "cf", lambda a: a, z)
    style(cf_ax, next(panels), "cloud fraction", "")
    cf_ax.set_xlim(left=0)

    used = len(STD_VARS) + 1
    if used >= flat.size:
        raise SystemExit(
            f"{used} panels do not leave a slot for the legend in a "
            f"2 x {NCOL} grid; widen the grid")
    for ax in flat[used:]:
        ax.axis("off")
    flat[used].legend(handles=legend_handles(sources, names), loc="center",
                      handlelength=1.6)

    for ax in flat[:used]:
        ax.set_ylim(0, z[-1] / 1000.0)
    fig.tight_layout()
    return fig


def pdf_panel(ax, d, hosts, sources, v, z_km, panel, show_ylabel, threshold):
    """One variable at one level: the two envelopes over every model's PDF.

    Condensate is shown only above the cloud threshold, for the same reason
    as on the TWPICE side -- the hosts carry a large trace population well
    below anything STEAM produces.
    """
    label, scale, unit = UNITS[v]
    fields = {(host, s): d[f"pdf_{v}_{z_km:.0f}km_{host}_{s}"].ravel() * scale
              for s in ("host", *[t for t in sources if t != "host"])
              for host in hosts}
    condensate = v in ("qc", "qi")

    if condensate:
        cut = threshold * scale
        fields = {k: f[f >= cut] for k, f in fields.items()}
        hi = max((f.max() for f in fields.values() if f.size), default=1.0)
        bins = np.logspace(np.log10(cut), np.log10(hi), 51)
        ax.set_xscale("log")
    else:
        allv = np.concatenate([f for f in fields.values() if f.size])
        # A few hosts carry far-outlying tails; clipping the bin range at the
        # 0.01-99.99 percentile keeps the bands visible instead of squeezing
        # the bulk into two bins. Outliers still land in the end bins.
        lo, hi = np.percentile(allv, [0.01, 99.99])
        bins = np.linspace(lo, hi, 81)

    centres = 0.5 * (bins[1:] + bins[:-1])
    curves = {}
    for (host, s), f in fields.items():
        if f.size == 0:
            continue
        curves[(host, s)] = np.histogram(f, bins=bins, density=True)[0]

    positive = [c[c > 0].min() for c in curves.values() if np.any(c > 0)]
    # Empty bins are exact zeros, which have no place on a log axis; floored
    # to half a decade below the smallest density any model produced, so the
    # band closes at the bottom of the panel instead of vanishing.
    floor = 0.5 * min(positive) if positive else 1e-12

    for group, tags in members(sources).items():
        stack = np.stack([c for (host, s), c in curves.items() if s in tags])
        if stack.size == 0:
            continue
        lo = np.maximum(stack.min(axis=0), floor)
        hi = np.maximum(stack.max(axis=0), floor)
        _, colour, alpha = GROUPS[group]
        ax.fill_between(centres, lo, hi, color=colour, alpha=alpha, lw=0)
        for edge in (lo, hi):
            ax.plot(centres, edge, color=colour, lw=EDGE_LW * 0.8,
                    solid_capstyle="round")

    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    style(ax, panel, f"{label}  [{unit}]", "density" if show_ylabel else "")


def pdfs(d, hosts, sources, names):
    levels = np.array([5000.0, 10000.0]) / 1000.0
    threshold = 0.01e-3
    fig, axes = plt.subplots(len(levels), len(VARS), figsize=(10.0, 5.2))
    panels = iter("abcdefgh")
    for i, z_km in enumerate(levels):
        for j, v in enumerate(VARS):
            pdf_panel(axes[i, j], d, hosts, sources, v, z_km, next(panels),
                      j == 0, threshold)
        axes[i, 0].text(-0.32, 0.5, f"{z_km:.0f} km",
                        transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="center", color=INK, fontsize=9.5)
    fig.legend(handles=legend_handles(sources, names)[:2], loc="upper center",
               ncol=2, handlelength=1.6, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return fig


def save(fig, stem):
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGS / f"{stem}.png", dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA.name} not found -- run "
                         f"compute_rcemip_stats.py first")
    d = np.load(DATA)
    hosts = [str(h) for h in d["hosts"]]
    sets = [str(s) for s in d["sets"]]
    z = common_z(d, hosts)

    no_p = [h for h in hosts if h not in var_hosts(d, hosts, "p")]
    if no_p:
        print(f"pressure panel over {len(hosts) - len(no_p)} of "
              f"{len(hosts)} hosts; {no_p} archive no 3-D pressure and are "
              f"dropped from BOTH bands in that panel")

    for lscale in [str(s) for s in d["lscales"]]:
        sources = ("host", *[f"{s}_{lscale}" for s in sets])
        L_km = float(d[f"L_{lscale}"]) / 1000.0
        names = group_names(L_km)
        print(f"{lscale} (L = {L_km:g} km): {len(hosts)} hosts, "
              f"{len(sets)} STEAM amplitudes each pooled into one band; "
              f"envelopes on {z.size} common levels, "
              f"{z[0]:.0f}-{z[-1]:.0f} m")
        save(profiles(d, hosts, sources, names),
             f"rcemip_profiles_{lscale}")
        save(pdfs(d, hosts, sources, names), f"rcemip_pdfs_{lscale}")


if __name__ == "__main__":
    main()
