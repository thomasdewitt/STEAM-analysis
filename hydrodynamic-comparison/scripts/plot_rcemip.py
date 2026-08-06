#!/usr/bin/env python3
"""Plot the matched RCEMIP-channel comparison from compute_rcemip_stats.py.

Same two figures as the TWPICE side, across all nine hosts at once:

  rcemip_profiles  per-level standard deviation of h, qt, qc and qi, plus
                   cloud fraction
  rcemip_pdfs      single-level distributions of the same four variables at
                   5 and 10 km

One thin line per run -- nine hosts in ink, and the eighteen STEAM runs
coloured by flux amplitude. No individual model is identifiable and none is
meant to be: the figure is about whether STEAM falls inside the spread the
hydrodynamic models already show among themselves.

Condensate panels are cut at the 0.01 g/kg cloud threshold, as on the TWPICE
side.

Usage: python plot_rcemip.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import VARS, UNITS, INK, COLOR, rcparams, style

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
DATA = OUTPUT / "rcemip_stats.npz"

NAME = {"host": "RCEMIP hosts", "c005": "STEAM  $c=0.05$",
        "c017": "STEAM  $c=0.17$"}
LW = 0.56
ALPHA = 0.65

rcparams()


def order(sources):
    """STEAM first, hosts last, so the reference lines sit on top."""
    return [s for s in sources if s != "host"] + ["host"]


def draw(ax, d, hosts, sources, key, x_of):
    """One thin line per host per source."""
    for s in order(sources):
        for host in hosts:
            ax.plot(x_of(d[f"{key}_{host}_{s}"]), d[f"z_{host}"] / 1000.0,
                    color=COLOR[s], lw=LW, alpha=ALPHA,
                    solid_capstyle="round")


def legend_handles(sources):
    return [Line2D([0], [0], color=COLOR[s], lw=1.4, label=NAME[s])
            for s in sources]


def profiles(d, hosts, sources):
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0), sharey=True)
    flat = axes.ravel()

    for ax, panel, v in zip(flat, "abcd", VARS):
        label, scale, unit = UNITS[v]
        draw(ax, d, hosts, sources, f"std_{v}", lambda a, s=scale: a * s)
        style(ax, panel, f"std({label})  [{unit}]",
              "height  [km]" if ax is flat[0] or ax is flat[3] else "")

    draw(flat[4], d, hosts, sources, "cf", lambda a: a)
    style(flat[4], "e", "cloud fraction", "")
    flat[4].set_xlim(left=0)

    flat[5].axis("off")
    flat[5].legend(handles=legend_handles(sources), loc="center",
                   handlelength=1.6)

    top = max(d[f"z_{h}"].max() for h in hosts) / 1000.0
    for ax in flat[:5]:
        ax.set_ylim(0, top)
    fig.tight_layout()
    return fig


def pdf_panel(ax, d, hosts, sources, v, z_km, panel, show_ylabel, threshold):
    """One variable at one level: every host and source overlaid.

    Condensate is shown only above the cloud threshold, for the same reason
    as on the TWPICE side -- the hosts carry a large trace population well
    below anything STEAM produces.
    """
    label, scale, unit = UNITS[v]
    fields = {(host, s): d[f"pdf_{v}_{z_km:.0f}km_{host}_{s}"].ravel() * scale
              for s in order(sources) for host in hosts}
    condensate = v in ("qc", "qi")

    if condensate:
        cut = threshold * scale
        fields = {k: f[f >= cut] for k, f in fields.items()}
        hi = max((f.max() for f in fields.values() if f.size), default=1.0)
        bins = np.logspace(np.log10(cut), np.log10(hi), 50)
        ax.set_xscale("log")
    else:
        allv = np.concatenate([f for f in fields.values() if f.size])
        # A few hosts carry far-outlying tails; clipping the bin range at the
        # 0.01-99.99 percentile keeps every line visible instead of squeezing
        # the bulk into two bins. Outliers still land in the end bins.
        lo, hi = np.percentile(allv, [0.01, 99.99])
        bins = np.linspace(lo, hi, 80)

    for (host, s), f in fields.items():
        if f.size == 0:
            continue
        counts, edges = np.histogram(f, bins=bins, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centres, counts, color=COLOR[s], lw=LW, alpha=ALPHA,
                solid_capstyle="round")

    ax.set_yscale("log")
    style(ax, panel, f"{label}  [{unit}]", "density" if show_ylabel else "")


def pdfs(d, hosts, sources):
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
    fig.legend(handles=legend_handles(sources), loc="upper center", ncol=3,
               handlelength=1.6, bbox_to_anchor=(0.5, 1.04))
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
    sources = ("host", *[str(s) for s in d["sets"]])
    print(f"{len(hosts)} hosts, {len(sources) - 1} STEAM amplitudes each")

    save(profiles(d, hosts, sources), "rcemip_profiles")
    save(pdfs(d, hosts, sources), "rcemip_pdfs")


if __name__ == "__main__":
    main()
