#!/usr/bin/env python3
"""Plot the matched gigaLES comparison from compute_gigales_stats.py.

Three figures:

  figs/gigales_profiles           per-level standard deviation of h, qt, T,
                                  qc and qi, plus cloud fraction, STEAM at
                                  the main-text flux amplitude.
  figs/appendix/gigales_profiles  the same figure with all three amplitudes
                                  drawn -- the version this script wrote
                                  until 2026-08-13.
  figs/gigales_pdfs               single-level distributions of h, qt, qc
                                  and qi at 5 and 10 km, all amplitudes.

ONE AMPLITUDE ON THE MAIN PROFILE FIGURE (2026-08-13). The paper's profile
figure carries the moderate amplitude alone and sends the other two to an
appendix, because the amplitude barely moves a standard deviation profile
and three near-coincident ladders spent the panel on nothing.

The appendix version is the SAME FIGURE AT THE SAME STEM in figs/appendix/,
not a differently-named one beside it. What distinguishes them is which
figure of the paper they are, and that is what the directory says; a name
like `_allc` would have put the distinction in the filename and then still
needed the reader to know which one the paper takes. Both are written every
run, so the appendix version is never a rerun with a flag flipped.

Which amplitudes get LINES is the only difference between them: the grey
backdrop is the same in both, min-to-max over all thirty runs, since it is
what bounds everything STEAM produced and that claim does not depend on
which lines are drawn on top of it.

The PDFs are not split this way. There the amplitude does separate the
curves, so all three stay on the one figure. Neither is the RCEMIP profile
figure, whose STEAM band pools the amplitudes and continues to -- narrowing
that band would narrow what it is a claim about (his ruling, 2026-08-13:
"that band should still be over the full suite").

T JOINED THE PROFILE FIGURE ON 2026-08-10, drawn exactly like the four that
were already there, and on the profiles only. A pressure panel was added
beside it the same day and dropped again -- see common.py.

BOTH LAYERS OF THE ENSEMBLE ARE ON THE PROFILE FIGURE, and they say
different things. Each coloured line is the POOLED statistic for one case
and amplitude -- the five members' cells at that level taken as one
population -- which is the model's answer with the sampling noise of a
single draw taken out of it. The grey backdrop is the min-to-max over the
thirty INDIVIDUAL runs, which is how far one draw moves. A line near a host
inside a wide grey band is a different claim from a line near a host inside
a narrow one.

The PDFs are pooled and have no such band: the histogram flattens the
member axis, so pooling is simply five times the samples at each level.

Two matched cases share every panel: SAM-TWPICE and SAM-GATE, both 2048^2 at
100 m and so both driving the identical STEAM config. Colour is the source
(host, or STEAM at each amplitude) and line style is which LES, so the two
cases read apart without doubling the palette.

GATE's condensate is SAM's linear liquid/ice partition of the archived QN,
applied before coarsening; TWPICE archives QC and QI directly. Worth
remembering when reading panels c and d in particular.

Both figures are drawn on the matched grid built by the compute step: the
host block-averaged onto STEAM's spacing, and whichever field is finer
vertically averaged per level to bring the two spacings closest. The factors
actually used are printed at run time rather than quoted here.

The condensate panels are distributions of the CLOUDY cells only, on a
logarithmic axis -- most cells hold no condensate at all, and a density over
everything would be one spike at zero and no visible shape.

Styling follows paper/concept-figs (turblib.py), as fractal-analysis does.

Usage: python plot_gigales.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import (VARS, STD_VARS, UNITS, INK, LABEL, COLOR, amplitude_value,
                    by_amplitude, rcparams, require_main_set, style)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
# Figures the paper carries in an appendix rather than the main text. The
# stem is the same as the main-text figure it varies, so the directory is
# what tells them apart -- see the note above on why.
APPENDIX = FIGS / "appendix"
DATA = OUTPUT / "gigales_stats.npz"

CASE_NAME = {"twpice": "SAM-TWPICE", "gate": "SAM-GATE"}
CASE_STYLE = {"twpice": "-", "gate": (0, (4, 2))}

# The backdrop on the profile panels: min-to-max over every individual STEAM
# run, both cases x three amplitudes x five members. One shading rather than
# one per amplitude, so it reads as the extent of everything STEAM produced
# rather than as competing claims -- and grey, so it never competes with the
# amplitude ladder drawn on top of it. No edge line: an outline would give a
# backdrop the weight of a measurement.
BAND = "#9a9a9a"
BAND_ALPHA = 0.30
GRID_TOL = 0.01            # m; the two SAM grids differ by ~0.5 mm

rcparams()


def name(source):
    """Legend label for one source, the amplitude read off its own tag."""
    if source == "host":
        return "LES host"
    return rf"STEAM  $c={amplitude_value(source):g}$"


def order(sources):
    """STEAM first, hosts last, so the reference lines sit on top."""
    return [s for s in sources if s != "host"] + ["host"]


def run_envelope(d, cases, sources, key, x_of):
    """Min-to-max over every individual STEAM run, on one shared grid.

    The per-member statistics, not the pooled one: this is the spread a
    single realization moves over, which is the thing the pooled line hides.
    The two cases sit on the same SAM grid to within half a millimetre, so
    the min/max is pointwise and needs no interpolation -- checked rather
    than assumed, since a silent interpolation would blur the very quantity
    the band is drawn to show.
    """
    z = d[f"{cases[0]}_z"]
    runs = []
    for case in cases:
        if not np.allclose(d[f"{case}_z"], z, rtol=0, atol=GRID_TOL):
            raise SystemExit(
                f"{case} is not on the same comparison grid as {cases[0]}; "
                f"one band across both would not be pointwise")
        for s in sources:
            if s == "host":
                continue
            runs.append(x_of(d[f"{key}_{case}_{s}_members"]))
    stack = np.concatenate(runs, axis=0)
    return z, stack.min(axis=0), stack.max(axis=0)


def draw(ax, d, cases, band_sources, line_sources, key, x_of):
    """The backdrop over band_sources, pooled lines for line_sources.

    The two are separate arguments because the main-text figure narrows the
    lines to one amplitude without narrowing what the backdrop bounds.
    """
    z, lo, hi = run_envelope(d, cases, band_sources, key, x_of)
    ax.fill_betweenx(z / 1000.0, lo, hi, color=BAND, alpha=BAND_ALPHA, lw=0,
                     zorder=0)
    for s in order(line_sources):
        for case in cases:
            ax.plot(x_of(d[f"{key}_{case}_{s}"]), d[f"{case}_z"] / 1000.0,
                    color=COLOR[s], lw=1.3, ls=CASE_STYLE[case],
                    solid_capstyle="round", zorder=2)


def legend_handles(cases, sources, n_runs=None):
    handles = [Line2D([0], [0], color=COLOR[s], lw=1.4, label=name(s))
               for s in sources]
    handles += [Line2D([0], [0], color=LABEL, lw=1.4, ls=CASE_STYLE[c],
                       label=CASE_NAME[c]) for c in cases]
    if n_runs is not None:
        handles.append(Patch(facecolor=BAND, edgecolor="none",
                             alpha=BAND_ALPHA,
                             label=f"all {n_runs} STEAM runs"))
    return handles


# Five std panels plus cloud fraction fill a 2 x 3 grid exactly, so the
# legend goes above the figure rather than into a spare axis. It used to sit
# in the sixth slot, which existed only because there were four std panels;
# a 2 x 4 grid to keep that habit would leave a dead quadrant.
NCOL = 3


def profiles(d, cases, sources, line_sources, n_runs):
    fig, axes = plt.subplots(2, NCOL, figsize=(9.0, 6.0), sharey=True)
    flat = axes.ravel()
    panels = iter("abcdef")

    for ax, v in zip(flat, STD_VARS):
        label, scale, unit = UNITS[v]
        draw(ax, d, cases, sources, line_sources, f"std_{v}",
             lambda a, s=scale: a * s)
        style(ax, next(panels), f"std({label})  [{unit}]",
              "height  [km]" if ax in (flat[0], flat[NCOL]) else "")

    cf_ax = flat[len(STD_VARS)]
    draw(cf_ax, d, cases, sources, line_sources, "cf", lambda a: a)
    style(cf_ax, next(panels), "cloud fraction", "")
    cf_ax.set_xlim(left=0)

    used = len(STD_VARS) + 1
    if used > flat.size:
        raise SystemExit(
            f"{used} panels do not fit a 2 x {NCOL} grid; widen it")
    for ax in flat[used:]:
        ax.axis("off")

    top = max(d[f"{c}_z"].max() for c in cases) / 1000.0
    for ax in flat[:used]:
        ax.set_ylim(0, top)
    fig.tight_layout()
    fig.legend(handles=legend_handles(cases, line_sources, n_runs),
               loc="upper center", ncol=4, handlelength=1.8,
               bbox_to_anchor=(0.5, 1.09),
               title=f"lines pooled over {int(d['n_members'])} realizations")
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


def save(fig, stem, into=FIGS):
    into.mkdir(parents=True, exist_ok=True)
    fig.savefig(into / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(into / f"{stem}.png", dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {into.relative_to(BASE)}/{stem}.pdf and .png")


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA.name} not found -- run "
                         f"compute_gigales_stats.py first")
    d = np.load(DATA)
    cases = [str(c) for c in d["cases"]]
    sets = by_amplitude([str(s) for s in d["sets"]])
    sources = ("host", *sets)
    main_sources = ("host", require_main_set(sets))

    for case in cases:
        n_steam, n_host = d[f"{case}_n_steam"], d[f"{case}_n_host"]
        f = int(d[f"{case}_xy_coarsen"])
        print(f"{case}: {d[f'{case}_z'].size} levels to "
              f"{d[f'{case}_z'].max() / 1000:.1f} km; host {f}x{f}x{f} "
              f"coarsened to {float(d[f'{case}_dx']):.0f} m; "
              f"STEAM vertically coarsened "
              f"by {n_steam.min()}-{n_steam.max()}, host by "
              f"{n_host.min()}-{n_host.max()}")
    print(f"cloud fraction threshold {float(d['cloud_kgkg']) * 1e3:g} g/kg")

    n_members = int(d["n_members"])
    n_runs = len(cases) * (len(sources) - 1) * n_members
    print(f"{len(cases)} cases x {len(sources) - 1} amplitudes x "
          f"{n_members} members = {n_runs} STEAM runs")

    print(f"profile lines: {main_sources[1]} in figs/, all {len(sets)} "
          f"amplitudes in figs/appendix/; the backdrop bounds all "
          f"{n_runs} runs on both")

    save(profiles(d, cases, sources, main_sources, n_runs),
         "gigales_profiles")
    save(profiles(d, cases, sources, sources, n_runs),
         "gigales_profiles", into=APPENDIX)
    save(pdfs(d, cases, sources), "gigales_pdfs")


if __name__ == "__main__":
    main()
