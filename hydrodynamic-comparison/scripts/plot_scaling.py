#!/usr/bin/env python3
"""Plot the Haar fluctuation functions and local slopes from
compute_scaling.py.

One figure per case (twpice_scaling, rcemip_scaling), each a 2 x 4 grid:
rows are the two levels, and the columns pair each variable's fluctuation
function with its local slope.

The twpice figure carries both SAM cases, TWPICE and GATE, on every panel:
colour is the source (host, or STEAM at each amplitude) and line style is
which LES, the same convention as the profile and PDF figures.

Curves are at native resolution, so the host and STEAM lines begin at
different smallest lags -- that offset is the point, not an artefact.

On the RCEMIP figure every run gets one thin line, hosts drawn last so they
read on top. The dotted guide is the model's design exponent H_h, drawn as a
power law on the fluctuation panels and as a level on the slope panels.

Usage: python plot_scaling.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from steam.constants import hurst_horizontal as H_H

from common import UNITS, INK, LABEL, COLOR, rcparams, style

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
DATA = OUTPUT / "scaling_stats.npz"

VARS = ("h", "qt")
NAME = {"twpice": {"host": "LES host"}, "rcemip": {"host": "RCEMIP hosts"}}
STEAM_NAME = {"c005": "STEAM  $c=0.05$", "c017": "STEAM  $c=0.17$"}
WIDTH = {"twpice": 1.3, "rcemip": 0.56}
ALPHA = {"twpice": 1.0, "rcemip": 0.65}
# The two SAM cases share every panel, told apart by line style as in
# plot_twpice.py. The channels are unstyled: nine thin lines, one colour.
HOST_NAME = {"twpice": "SAM-TWPICE", "gate": "SAM-GATE"}
HOST_STYLE = {"twpice": "-", "gate": (0, (4, 2))}

rcparams()


def order(sources):
    """STEAM first, hosts last, so the reference lines sit on top."""
    return [s for s in sources if s != "host"] + ["host"]


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
    for s in order(sources):
        for host in hosts:
            key = f"{case}_{host}_{v}_{tag}_{s}"
            y = d[f"{key}_{kind}"]
            ax.plot(d[f"{key}_lags"] / 1000.0, y, color=COLOR[s],
                    lw=WIDTH[case], alpha=ALPHA[case],
                    ls=HOST_STYLE.get(host, "-"), solid_capstyle="round")
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


def figure(d, case):
    hosts = [str(h) for h in d[f"{case}_hosts"]]
    sources = ("host", *[str(s) for s in d["sets"]])
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

    handles = [Line2D([0], [0], color=COLOR[s], lw=1.4,
                      label=NAME[case].get(s, STEAM_NAME.get(s)))
               for s in sources]
    # Colour is the source, style is which LES -- but only where more than one
    # LES is styled, so the channel figure keeps its three-entry legend.
    styled = [h for h in hosts if h in HOST_STYLE]
    if len(styled) > 1:
        handles += [Line2D([0], [0], color=LABEL, lw=1.4, ls=HOST_STYLE[h],
                           label=HOST_NAME[h]) for h in styled]
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
        fig = figure(d, case)
        stem = f"{case}_scaling"
        fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
        fig.savefig(FIGS / f"{stem}.png", dpi=200, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
        plt.close(fig)
        print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    main()
