#!/usr/bin/env python3
"""Plot the four fractal metrics written by compute_fractal_metrics.py.

One four-panel figure, one panel per metric, each showing the scaling
function the exponent was fitted to alongside the fitted value:

  a  correlation integral C(r)         -> D_e
  b  individual perimeter vs. size     -> D_f
  c  cloud area distribution           -> tau_area
  d  nested-perimeter distribution     -> tau_per

Showing the scaling functions and not only the numbers is the point: an
exponent is only meaningful where its scaling function is straight in
log-log, so the panels are what lets a reader check that.

Each panel carries one curve per albedo threshold (R = 0.1, 0.2, 0.3),
darker with increasing R. Thresholding on albedo rather than optical depth
is what makes these directly comparable to the satellite retrievals of
DeWitt et al. (2026); see albedo.py.

Every panel plots the variables themselves on logarithmic axes (objscale
hands back log10 values, which are exponentiated here), so the axes read in
physical units. The dashed guide in each panel is the fitted power law,
anchored at the data's log-space centre of mass.

Sign conventions, from objscale: the correlation integral and the
individual perimeter scale as C(r) ~ r^{D_e} and P ~ l^{D_f}, while the
size distributions scale as n ~ s^{-tau}.

Styling follows paper/concept-figs (turblib.py): the same ink, rule and
label greys, the same earth-adjacent palette, hairline axes, no top or
right spine, PDF out.

Usage: python plot_fractal_metrics.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from albedo import threshold_tag

HERE = Path(__file__).resolve().parent
DATA = HERE / "fractal_metrics_C1large.npz"
OUT = HERE / "fractal_metrics_C1large"

# paper/concept-figs/turblib.py
INK = "#111111"
RULE = "#e3e3e3"
LABEL = "#7a7a7a"
# One tone per albedo threshold, darkening with R so the ordering reads
# without consulting the legend.
THRESHOLD_COLORS = ("#79B0AC", "#2E7E7A", "#12403E")

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
    "xtick.color": LABEL,
    "ytick.color": LABEL,
    "legend.frameon": False,
    "figure.facecolor": "white",
})


def style(ax, panel, xlabel, ylabel):
    """Log-log axes with a bare panel letter; descriptions go in the caption."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(panel, color=INK, pad=8, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(which="both", length=3, width=0.8, colors=LABEL)
    ax.tick_params(which="minor", length=1.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def annotate(ax, text, rising):
    """The fitted value, set quietly in the empty upper corner.

    Rising panels fill the upper right, falling panels the upper left, so the
    label goes to whichever corner the data leaves free.
    """
    x, ha = (0.03, "left") if rising else (0.97, "right")
    ax.text(x, 0.94, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=9.5, color=INK)


def positive(*arrays):
    """Keep only points that are finite and strictly positive in every array."""
    good = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        good &= np.isfinite(a) & (a > 0)
    return [a[good] for a in arrays]


def guide(ax, x, y, slope, color):
    """Dashed power law of the fitted slope, through the log centre of mass."""
    if len(x) == 0 or not np.isfinite(slope):
        return
    log_x0 = np.log10(x).mean()
    log_y0 = np.log10(y).mean()
    span = np.array([x.min(), x.max()])
    # Dotted, not dashed: the SAM figure uses dashes to mean GATE, and a
    # fit line must never be mistakable for data.
    ax.plot(span, 10.0 ** (log_y0 + slope * (np.log10(span) - log_x0)),
            color=color, lw=0.9, ls=(0, (1, 2)), alpha=0.8, zorder=1)


PANELS = (
    # letter, metric key, x key, y key, x label, y label, marker, rising
    ("a", "D_e", "C_bins", "C_l", "separation $r$  [km]",
     "correlation integral  $C(r)$", None, True),
    ("b", "D_f", "ind_log_length", "ind_log_perimeter",
     "length scale  [km]", "filled perimeter  [km]", 2.8, True),
    ("c", "tau_area", "area_log_bins", "area_log_counts",
     "area  [km$^2$]", "counts", 3.4, False),
    ("d", "tau_per", "per_log_bins", "per_log_counts",
     "nested perimeter  [km]", "counts", 3.4, False),
)
LOGGED = {"ind_log_length", "ind_log_perimeter", "area_log_bins",
          "area_log_counts", "per_log_bins", "per_log_counts"}


def series(d, tag, key):
    """objscale hands back log10 for some outputs; exponentiate those."""
    a = np.asarray(d[f"{tag}_{key}"], float)
    return 10.0 ** a if key in LOGGED else a


def draw_panel(ax, d, thresholds, letter, metric, xk, yk, xlabel, ylabel,
               marker, rising):
    labels = []
    for R, color in zip(thresholds, THRESHOLD_COLORS):
        tag = threshold_tag(R)
        x, y = positive(series(d, tag, xk), series(d, tag, yk))
        if x.size == 0:
            continue
        slope = float(d[f"{tag}_{metric}"])
        if marker is None:
            ax.plot(x, y, color=color, lw=1.5, solid_capstyle="round")
        else:
            ax.plot(x, y, lw=0, marker="o", ms=marker, mfc=color,
                    mec="none", alpha=0.85)
        guide(ax, x, y, slope if rising else -slope, color)
        labels.append(f"{slope:.2f}")
    style(ax, letter, xlabel, ylabel)
    if labels:
        annotate(ax, "  ".join(labels), rising=rising)


def write_table(path, header, rows, d, metrics):
    """Plain-text table of the fitted exponents, beside the figure.

    Written because the numbers are what gets quoted; reading them off a
    log or a panel annotation invites transcription errors.
    """
    label_w = max(len(label) for label, _ in rows)
    lines = [header, "",
             f"{'mask':<{label_w}}  {'cover':>7}"
             + "".join(f"  {m:>9}" for m in metrics)]
    for label, tag in rows:
        cells = "".join(f"  {float(d[f'{tag}_{m}']):>9.3f}"
                        if f"{tag}_{m}" in d else f"  {'--':>9}"
                        for m in metrics)
        lines.append(f"{label:<{label_w}}  "
                     f"{float(d[f'{tag}_cover']):>7.4f}{cells}")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text)
    print(text)
    print(f"wrote {Path(path).name}")


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA.name} not found -- run "
                         f"compute_fractal_metrics.py first")
    d = np.load(DATA, allow_pickle=False)
    thresholds = [float(R) for R in d["thresholds"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2))
    for ax, spec in zip(axes.ravel(), PANELS):
        draw_panel(ax, d, thresholds, *spec)

    handles = [Line2D([0], [0], color=c, lw=1.6, label=f"$R > {R:g}$")
               for R, c in zip(thresholds, THRESHOLD_COLORS)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               handlelength=1.6, bbox_to_anchor=(0.5, 1.04))

    # Provenance is printed rather than drawn -- it belongs in the caption.
    header = (f"STEAM square campaign, {int(d['n_members'])} members pooled "
              f"at dx = {float(d['dx_km']):g} km\n"
              f"pattern {str(d['pattern'])}, objscale "
              f"{str(d['objscale_version'])}")
    rows = [(f"R>{R:g}", threshold_tag(R)) for R in thresholds]
    write_table(f"{OUT}.txt", header, rows, d, ("D_e", "D_f", "tau_area",
                                                "tau_per"))

    fig.tight_layout()
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.name}.pdf and {OUT.name}.png")


if __name__ == "__main__":
    main()
