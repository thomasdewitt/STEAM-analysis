#!/usr/bin/env python3
"""Plot the matched TWPICE comparison from compute_twpice_stats.py.

Two figures:

  twpice_profiles  per-level standard deviation of h, qt, qc and qi, plus
                   cloud fraction, STEAM at two flux amplitudes against the
                   SAM-TWPICE host.
  twpice_pdfs      single-level distributions of the same four variables at
                   5 and 10 km.

Both are drawn on the matched grid built by the compute step: the host
block-averaged 2x2 to STEAM's 200 m, and whichever field is finer vertically
averaged per level to bring the two spacings closest.

The condensate panels are distributions of the CLOUDY cells only, on a
logarithmic axis -- most cells hold no condensate at all, and a density over
everything would be one spike at zero and no visible shape. The fraction of
cells that are cloudy is what the cloud-fraction profile already shows, and
it is printed per panel.

Styling follows paper/concept-figs (turblib.py), as fractal_analysis does.

Usage: python plot_twpice.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import VARS, UNITS, INK, COLOR, rcparams, style

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
DATA = OUTPUT / "twpice_stats.npz"

NAME = {"host": "SAM-TWPICE",
        "c005": "STEAM  $c=0.05$", "c017": "STEAM  $c=0.17$"}

rcparams()


def profiles(d, sources):
    z = d["z"] / 1000.0
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.0), sharey=True)
    flat = axes.ravel()

    for ax, panel, v in zip(flat, "abcd", VARS):
        label, scale, unit = UNITS[v]
        for s in sources:
            ax.plot(d[f"std_{v}_{s}"] * scale, z, color=COLOR[s], lw=1.4,
                    solid_capstyle="round", label=NAME[s])
        style(ax, panel, f"std({label})  [{unit}]",
              "height  [km]" if ax is flat[0] or ax is flat[3] else "")

    ax = flat[4]
    for s in sources:
        ax.plot(d[f"cf_{s}"], z, color=COLOR[s], lw=1.4,
                solid_capstyle="round", label=NAME[s])
    style(ax, "e", "cloud fraction", "")
    ax.set_xlim(left=0)

    flat[5].axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    flat[5].legend(handles, labels, loc="center", handlelength=1.6)

    for ax in flat[:5]:
        ax.set_ylim(0, z.max())
    fig.tight_layout()
    return fig


def pdf_panel(ax, d, sources, v, z_km, panel, show_ylabel, threshold):
    """One variable at one level, all sources overlaid.

    Condensate is shown only above the cloud threshold. Below it SAM carries
    an enormous population of trace values -- microphysical residue, four
    decades below anything STEAM produces -- and including them makes the
    panel a picture of that residue rather than of cloud.
    """
    label, scale, unit = UNITS[v]
    fields = {s: d[f"pdf_{v}_{z_km:.0f}km_{s}"].ravel() * scale
              for s in sources}
    condensate = v in ("qc", "qi")

    if condensate:
        cut = threshold * scale
        fields = {s: f[f >= cut] for s, f in fields.items()}
        hi = max((f.max() for f in fields.values() if f.size), default=1.0)
        bins = np.logspace(np.log10(cut), np.log10(hi), 50)
        ax.set_xscale("log")
    else:
        allv = np.concatenate(list(fields.values()))
        bins = np.linspace(allv.min(), allv.max(), 80)

    for s in sources:
        f = fields[s]
        if f.size == 0:
            continue
        counts, edges = np.histogram(f, bins=bins, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centres, counts, color=COLOR[s], lw=1.2,
                solid_capstyle="round", label=NAME[s])

    ax.set_yscale("log")
    style(ax, panel, f"{label}  [{unit}]",
          "density" if show_ylabel else "")


def pdfs(d, sources):
    levels = d["pdf_levels"] / 1000.0
    threshold = float(d["cloud_kgkg"])
    fig, axes = plt.subplots(len(levels), len(VARS), figsize=(10.0, 5.2))
    panels = iter("abcdefgh")
    for i, z_km in enumerate(levels):
        for j, v in enumerate(VARS):
            pdf_panel(axes[i, j], d, sources, v, z_km, next(panels), j == 0,
                      threshold)
        axes[i, 0].text(-0.32, 0.5, f"{z_km:.0f} km", transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="center", color=INK,
                        fontsize=9.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, handlelength=1.6,
               bbox_to_anchor=(0.5, 1.04))
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
    sources = ("host", *[str(s) for s in d["sets"]])

    n_steam, n_host = d["n_steam"], d["n_host"]
    print(f"{d['z'].size} levels to {d['z'].max() / 1000:.1f} km; host 2x2 "
          f"coarsened to 200 m; STEAM vertically coarsened by "
          f"{n_steam.min()}-{n_steam.max()}, host by "
          f"{n_host.min()}-{n_host.max()}")
    print(f"cloud fraction threshold {float(d['cloud_kgkg']) * 1e3:g} g/kg")

    save(profiles(d, sources), "twpice_profiles")
    save(pdfs(d, sources), "twpice_pdfs")


if __name__ == "__main__":
    main()
