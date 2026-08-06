#!/usr/bin/env python3
"""Plot the matched LES comparison from compute_twpice_stats.py.

Two figures:

  twpice_profiles  per-level standard deviation of h, qt, qc and qi, plus
                   cloud fraction, STEAM at two flux amplitudes against each
                   host.
  twpice_pdfs      single-level distributions of the same four variables at
                   5 and 10 km.

Two matched cases share every panel: SAM-TWPICE and SAM-GATE, both 2048^2 at
100 m and so both driving the identical STEAM config. Colour is the source
(host, or STEAM at each amplitude) and line style is which LES, so the two
cases read apart without doubling the palette.

GATE's condensate is SAM's linear liquid/ice partition of the archived QN,
applied before coarsening; TWPICE archives QC and QI directly. Worth
remembering when reading panels c and d in particular.

Both figures are drawn on the matched grid built by the compute step: the
host block-averaged 2x2 to STEAM's 200 m, and whichever field is finer
vertically averaged per level to bring the two spacings closest.

The condensate panels are distributions of the CLOUDY cells only, on a
logarithmic axis -- most cells hold no condensate at all, and a density over
everything would be one spike at zero and no visible shape.

Styling follows paper/concept-figs (turblib.py), as fractal_analysis does.

Usage: python plot_twpice.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import VARS, UNITS, INK, LABEL, COLOR, rcparams, style

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
DATA = OUTPUT / "twpice_stats.npz"

NAME = {"host": "LES host",
        "c005": "STEAM  $c=0.05$", "c017": "STEAM  $c=0.17$"}
CASE_NAME = {"twpice": "SAM-TWPICE", "gate": "SAM-GATE"}
CASE_STYLE = {"twpice": "-", "gate": (0, (4, 2))}

rcparams()


def order(sources):
    """STEAM first, hosts last, so the reference lines sit on top."""
    return [s for s in sources if s != "host"] + ["host"]


def draw(ax, d, cases, sources, key, x_of):
    for s in order(sources):
        for case in cases:
            ax.plot(x_of(d[f"{key}_{case}_{s}"]), d[f"{case}_z"] / 1000.0,
                    color=COLOR[s], lw=1.3, ls=CASE_STYLE[case],
                    solid_capstyle="round")


def legend_handles(cases, sources):
    handles = [Line2D([0], [0], color=COLOR[s], lw=1.4, label=NAME[s])
               for s in sources]
    handles += [Line2D([0], [0], color=LABEL, lw=1.4, ls=CASE_STYLE[c],
                       label=CASE_NAME[c]) for c in cases]
    return handles


def profiles(d, cases, sources):
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0), sharey=True)
    flat = axes.ravel()

    for ax, panel, v in zip(flat, "abcd", VARS):
        label, scale, unit = UNITS[v]
        draw(ax, d, cases, sources, f"std_{v}", lambda a, s=scale: a * s)
        style(ax, panel, f"std({label})  [{unit}]",
              "height  [km]" if ax is flat[0] or ax is flat[3] else "")

    draw(flat[4], d, cases, sources, "cf", lambda a: a)
    style(flat[4], "e", "cloud fraction", "")
    flat[4].set_xlim(left=0)

    flat[5].axis("off")
    flat[5].legend(handles=legend_handles(cases, sources), loc="center",
                   handlelength=1.8)

    top = max(d[f"{c}_z"].max() for c in cases) / 1000.0
    for ax in flat[:5]:
        ax.set_ylim(0, top)
    fig.tight_layout()
    return fig


def pdf_panel(ax, d, cases, sources, v, z_km, panel, show_ylabel, threshold):
    label, scale, unit = UNITS[v]
    fields = {(case, s): d[f"pdf_{v}_{z_km:.0f}km_{case}_{s}"].ravel() * scale
              for s in order(sources) for case in cases}
    condensate = v in ("qc", "qi")

    if condensate:
        cut = threshold * scale
        fields = {k: f[f >= cut] for k, f in fields.items()}
        hi = max((f.max() for f in fields.values() if f.size), default=1.0)
        bins = np.logspace(np.log10(cut), np.log10(hi), 51)
        ax.set_xscale("log")
    else:
        allv = np.concatenate([f for f in fields.values() if f.size])
        bins = np.linspace(allv.min(), allv.max(), 81)

    for (case, s), f in fields.items():
        if f.size == 0:
            continue
        counts, edges = np.histogram(f, bins=bins, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centres, counts, color=COLOR[s], lw=1.1,
                ls=CASE_STYLE[case], solid_capstyle="round")

    ax.set_yscale("log")
    style(ax, panel, f"{label}  [{unit}]", "density" if show_ylabel else "")


def pdfs(d, cases, sources):
    levels = d["pdf_levels"] / 1000.0
    threshold = float(d["cloud_kgkg"])
    fig, axes = plt.subplots(len(levels), len(VARS), figsize=(10.0, 5.2))
    panels = iter("abcdefgh")
    for i, z_km in enumerate(levels):
        for j, v in enumerate(VARS):
            pdf_panel(axes[i, j], d, cases, sources, v, z_km, next(panels),
                      j == 0, threshold)
        axes[i, 0].text(-0.32, 0.5, f"{z_km:.0f} km",
                        transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="center", color=INK, fontsize=9.5)
    fig.legend(handles=legend_handles(cases, sources), loc="upper center",
               ncol=5, handlelength=1.8, bbox_to_anchor=(0.5, 1.06))
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
                         f"compute_twpice_stats.py first")
    d = np.load(DATA)
    cases = [str(c) for c in d["cases"]]
    sources = ("host", *[str(s) for s in d["sets"]])

    for case in cases:
        n_steam, n_host = d[f"{case}_n_steam"], d[f"{case}_n_host"]
        print(f"{case}: {d[f'{case}_z'].size} levels to "
              f"{d[f'{case}_z'].max() / 1000:.1f} km; host 2x2 coarsened to "
              f"200 m; STEAM vertically coarsened by {n_steam.min()}-"
              f"{n_steam.max()}, host by {n_host.min()}-{n_host.max()}")
    print(f"cloud fraction threshold {float(d['cloud_kgkg']) * 1e3:g} g/kg")

    save(profiles(d, cases, sources), "twpice_profiles")
    save(pdfs(d, cases, sources), "twpice_pdfs")


if __name__ == "__main__":
    main()
