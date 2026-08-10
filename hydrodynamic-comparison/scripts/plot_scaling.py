#!/usr/bin/env python3
"""Plot the Haar fluctuation functions and local slopes from
compute_scaling.py.

One figure per case per outer scale -- twpice_scaling, and
rcemip_scaling_Llong / rcemip_scaling_Lshort -- each a 2 x 4 grid: rows are
the two levels, and the columns pair each variable's fluctuation function
with its local slope.

Colour is the flux amplitude and the outer scale separates the figures,
rather than both competing for the line. The channels' two outer scales are
what the fluctuation functions show most plainly: the short-L cascade is
five dyads deep against the long-L one's nine, so its curve runs out of
variance to accumulate near 384 km while the long-L curve climbs on.

The twpice figure carries both SAM cases, TWPICE and GATE, on every panel:
colour is the source (host, or STEAM at each amplitude) and line style is
which LES, the same convention as the profile and PDF figures.

Curves are at native resolution, so the host and STEAM lines begin at
different smallest lags -- that offset is the point, not an artefact.

ENVELOPES ON THE RCEMIP FIGURES (2026-08-08), following the profile and PDF
pair: instead of 27 thin lines per panel -- nine hosts x three amplitudes --
each panel carries two shaded regions, the min-to-max over the nine hosts and
the min-to-max over the 27 STEAM runs, and where they overlap is the answer
the figure is for. The three flux amplitudes are pooled into the one STEAM
band because amplitude is a STEAM knob the way model identity is a host knob;
the outer scale is not pooled, and separates the figures, exactly as in
plot_rcemip.py.

The twpice figure keeps one line per run. Its two hosts are SAM-TWPICE and
SAM-GATE, told apart by line style, and a min/max over two curves is not a
spread -- it is those same two curves with the gap between them shaded, which
would draw a population that does not exist.

The dotted guide is the model's design exponent H_h, drawn as a power law on
the fluctuation panels and as a level on the slope panels.

Usage: python plot_scaling.py
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from steam.constants import hurst_horizontal as H_H

from common import (UNITS, INK, LABEL, COLOR, STEAM_BAND, by_amplitude,
                    rcparams, style)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
DATA = OUTPUT / "scaling_stats.npz"

VARS = ("h", "qt")
NAME = {"gigales": {"host": "LES host"}, "rcemip": {"host": "RCEMIP hosts"}}
STEAM_NAME = {"c002": "STEAM  $c=0.02$", "c005": "STEAM  $c=0.05$",
              "c017": "STEAM  $c=0.17$"}
WIDTH = {"gigales": 1.3, "rcemip": 0.56}
ALPHA = {"gigales": 1.0, "rcemip": 0.65}
# The two SAM cases share every panel, told apart by line style as in
# plot_gigales.py. The channels are unstyled: nine thin lines, one colour.
HOST_NAME = {"twpice": "SAM-TWPICE", "gate": "SAM-GATE"}
HOST_STYLE = {"twpice": "-", "gate": (0, (4, 2))}

# Cases drawn as envelopes rather than one line per run. twpice is absent on
# purpose: two curves are not a population, and shading between them would
# claim a spread that was never sampled.
BANDED = ("rcemip",)
# group -> (edge colour, fill opacity), as in plot_rcemip.py so the two figure
# families read as one. Hosts are the reference and keep the ink.
GROUPS = {"host": (INK, 0.16), "steam": (STEAM_BAND, 0.28)}
# The flux amplitudes pooled into the single STEAM band, and the same refusal
# plot_rcemip.members() makes: a band is a claim about what varies inside it,
# so an unfamiliar tag stops the figure rather than joining the band.
STEAM_SETS = ("c002", "c005", "c017")
EDGE_LW = 0.9

rcparams()


def order(sources):
    """STEAM first, hosts last, so the reference lines sit on top."""
    return [s for s in sources if s != "host"] + ["host"]


def members(sources):
    """Which source tags feed each envelope, guarded as in plot_rcemip.py.

    Every STEAM tag reaching one figure must carry the same outer scale. A
    band spanning both would merge the distinction the axis exists to show --
    and on these panels it would do it to the one feature that IS the result,
    the short-L rollover near 384 km against the long-L climb to 6144 km.
    """
    steam = [s for s in sources if s != "host"]
    unknown = [s for s in steam if amplitude(s) not in STEAM_SETS]
    if unknown:
        raise SystemExit(
            f"scaling_stats.npz carries STEAM source tags {unknown} that this "
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


def curves(d, case, hosts, tags, tag, v, kind):
    """Every (tag, host) curve for one panel, on the lag axis they share.

    Unlike the profiles, no interpolation is needed: within a band every run
    has the same dx and the same domain, so the lag axes are identical and a
    min/max is defined pointwise. That is a property of how the runs are
    generated rather than a guarantee, so it is checked -- interpolating
    silently onto one of them would be a worse answer than stopping, since
    these are log-log curves where a shifted lag axis is a shifted slope.
    """
    lags, stack = None, []
    for t in tags:
        for host in hosts:
            key = f"{case}_{host}_{v}_{tag}_{t}"
            these = d[f"{key}_lags"]
            if lags is None:
                lags = these
            elif these.shape != lags.shape or not np.allclose(these, lags):
                raise SystemExit(
                    f"{case} {v} {tag}: source {t} on host {host} is on a "
                    f"different lag axis from the rest of its band; a "
                    f"min/max across curves is only defined once they share "
                    f"one")
            stack.append(d[f"{key}_{kind}"])
    return lags, np.stack(stack)


def band(ax, lags, stack, colour, alpha, positive):
    """Min-to-max across the stack, as a filled region with thin edges.

    Non-finite entries are dropped per lag rather than per curve: the local
    slope is undefined at the ends of the range, where the half-decade window
    holds fewer than three lags, and dropping the whole curve for that would
    throw away the interior it does define. A lag no curve defines leaves a
    gap in the band, which is the honest thing for it to do.
    """
    y = np.where(np.isfinite(stack), stack, np.nan)
    if positive:
        y = np.where(y > 0, y, np.nan)          # log axis; F <= 0 is no datum
    with warnings.catch_warnings():
        # All-NaN lags are expected, and mean exactly the gap described above.
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanmin(y, axis=0), np.nanmax(y, axis=0)
    ax.fill_between(lags / 1000.0, lo, hi, color=colour, alpha=alpha, lw=0)
    for edge in (lo, hi):
        ax.plot(lags / 1000.0, edge, color=colour, lw=EDGE_LW,
                solid_capstyle="round")


def amplitude(source):
    """The flux-amplitude tag inside a `<set>_<lscale>` STEAM source.

    Colour means amplitude on these figures. The outer scale is not encoded
    in the line at all -- it separates the figures instead, as it does for
    the profile and PDF pair -- so it is stripped here.
    """
    return "host" if source == "host" else source.rsplit("_", 1)[0]


def guide(ax, d, case, hosts, tag, v):
    """Dotted H_h power law, anchored on the first host's own curve.

    Dotted, not dashed: dashes mean GATE here, and a design exponent must
    never be mistakable for data.
    """
    key = f"{case}_{hosts[0]}_{v}_{tag}_host"
    lags, F = d[f"{key}_lags"], d[f"{key}_F"]
    ok = np.isfinite(F) & (F > 0)
    mid = np.flatnonzero(ok)[len(np.flatnonzero(ok)) // 2]
    span = np.array([lags[ok].min(), lags[ok].max()])
    ax.plot(span / 1000.0, F[mid] * (span / lags[mid]) ** H_H,
            color=LABEL, lw=0.9, ls=(0, (1, 2)), zorder=0)


def panel(ax, d, case, hosts, sources, tag, v, kind, letter, ylabel):
    if case in BANDED:
        for group, tags in members(sources).items():
            lags, stack = curves(d, case, hosts, tags, tag, v, kind)
            colour, alpha = GROUPS[group]
            band(ax, lags, stack, colour, alpha, kind == "F")
    else:
        for s in order(sources):
            for host in hosts:
                key = f"{case}_{host}_{v}_{tag}_{s}"
                y = d[f"{key}_{kind}"]
                ax.plot(d[f"{key}_lags"] / 1000.0, y,
                        color=COLOR[amplitude(s)], lw=WIDTH[case],
                        alpha=ALPHA[case], ls=HOST_STYLE.get(host, "-"),
                        solid_capstyle="round")
    ax.set_xscale("log")
    label = UNITS[v][0]
    if kind == "F":
        ax.set_yscale("log")
        guide(ax, d, case, hosts, tag, v)
        ytext = rf"$F_1${label}"
    else:
        ax.axhline(H_H, color=LABEL, lw=0.9, ls=(0, (1, 2)), zorder=0)
        ax.set_ylim(-0.1, 1.1)
        ytext = rf"local slope, {label}"
    style(ax, letter, "lag  [km]", ytext if ylabel else "")


def band_handles(case, L_km):
    """Legend for a banded figure: one patch per envelope.

    The outer scale is named in the STEAM entry rather than left to the
    filename, because a reader of the paper meets the figure and not the
    stem, and `Llong` means nothing to them.
    """
    names = {"host": NAME[case]["host"],
             "steam": rf"STEAM  (all $c$),  $L = {L_km:g}$ km"}
    return [Patch(facecolor=c, edgecolor=c, alpha=max(a, 0.35),
                  label=names[g]) for g, (c, a) in GROUPS.items()]


def figure(d, case, lscale):
    hosts = [str(h) for h in d[f"{case}_hosts"]]
    sources = ("host", *[f"{s}_{lscale}"
                         for s in by_amplitude([str(s) for s in d["sets"]])])
    levels = [f"{z / 1000:.0f}km" for z in d["levels"]]

    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.6))
    letters = iter("abcdefgh")
    for i, tag in enumerate(levels):
        for j, v in enumerate(VARS):
            panel(axes[i, 2 * j], d, case, hosts, sources, tag, v, "F",
                  next(letters), True)
            panel(axes[i, 2 * j + 1], d, case, hosts, sources, tag, v,
                  "slope", next(letters), True)
        axes[i, 0].text(-0.38, 0.5, tag.replace("km", " km"),
                        transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="center", color=INK, fontsize=9.5)

    if case in BANDED:
        handles = band_handles(case, float(d[f"{case}_L_{lscale}"]) / 1000.0)
    else:
        handles = [Line2D([0], [0], color=COLOR[amplitude(s)], lw=1.4,
                          label=NAME[case].get(amplitude(s),
                                               STEAM_NAME.get(amplitude(s))))
                   for s in sources]
        # Colour is the source, style is which LES -- but only where more than
        # one LES is styled, so a single-LES figure keeps its short legend.
        styled = [h for h in hosts if h in HOST_STYLE]
        if len(styled) > 1:
            handles += [Line2D([0], [0], color=LABEL, lw=1.4,
                               ls=HOST_STYLE[h], label=HOST_NAME[h])
                        for h in styled]
    handles.append(Line2D([0], [0], color=LABEL, lw=0.9, ls=(0, (1, 2)),
                          label=rf"$H_h = {H_H:g}$"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               handlelength=1.8, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    return fig


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA.name} not found -- run compute_scaling.py first")
    d = np.load(DATA)
    for case in [str(c) for c in d["cases"]]:
        lscales = [str(s) for s in d[f"{case}_lscales"]]
        for lscale in lscales:
            fig = figure(d, case, lscale)
            # Tagged only where there is something to tell apart. TWPICE and
            # GATE are square, so their one outer-scale case is not a case
            # and putting Llong in the filename would imply a missing sibling.
            stem = f"{case}_scaling"
            if len(lscales) > 1:
                stem += f"_{lscale}"
            fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight",
                        pad_inches=0.05)
            fig.savefig(FIGS / f"{stem}.png", dpi=200, bbox_inches="tight",
                        pad_inches=0.05, facecolor="white")
            plt.close(fig)
            print(f"wrote {stem}.pdf and {stem}.png "
                  f"(L = {float(d[f'{case}_L_{lscale}']) / 1000:g} km)")


if __name__ == "__main__":
    main()
