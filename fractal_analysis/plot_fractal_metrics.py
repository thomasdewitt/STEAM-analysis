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

HERE = Path(__file__).resolve().parent
DATA = HERE / "fractal_metrics.npz"
OUT = HERE / "fractal_metrics"

# paper/concept-figs/turblib.py
INK = "#111111"
RULE = "#e3e3e3"
LABEL = "#7a7a7a"
PALETTE = {
    "qt": "#1F6E6B",   # deep teal
    "h": "#C08A2D",    # ochre
    "T": "#B5502A",    # sienna
    "p": "#5B6B8C",    # slate blue
}
# one colour per panel, in the concept-figure order
COLORS = [PALETTE["qt"], PALETTE["T"], PALETTE["h"], PALETTE["p"]]

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
    ax.plot(span, 10.0 ** (log_y0 + slope * (np.log10(span) - log_x0)),
            color=color, lw=0.9, ls=(0, (5, 3)), alpha=0.8, zorder=1)


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA.name} not found -- run "
                         f"compute_fractal_metrics.py first")
    d = np.load(DATA, allow_pickle=False)
    n = int(d["n_members"])
    cover = float(d["cover"])
    D_e, D_f = float(d["D_e"]), float(d["D_f"])
    tau_area, tau_per = float(d["tau_area"]), float(d["tau_per"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # a. correlation integral -> ensemble fractal dimension.  C(r) ~ r^{D_e}
    r, C_l = positive(np.asarray(d["C_bins"], float),
                      np.asarray(d["C_l"], float))
    ax_a.plot(r, C_l, color=COLORS[0], lw=1.5, solid_capstyle="round")
    guide(ax_a, r, C_l, D_e, COLORS[0])
    style(ax_a, "a",
          "separation $r$  [km]", "correlation integral  $C(r)$")
    annotate(ax_a, f"$D_e = {D_e:.2f}$", rising=True)

    # b. individual perimeter vs length scale.  P ~ l^{D_f}
    length, perimeter = positive(10.0 ** np.asarray(d["ind_log_length"], float),
                                 10.0 ** np.asarray(d["ind_log_perimeter"], float))
    ax_b.plot(length, perimeter, lw=0, marker="o", ms=2.8,
              mfc=COLORS[1], mec="none", alpha=0.85)
    guide(ax_b, length, perimeter, D_f, COLORS[1])
    style(ax_b, "b",
          "length scale  [km]", "filled perimeter  [km]")
    annotate(ax_b, f"$D_f = {D_f:.2f}$", rising=True)

    # c. cloud area distribution.  n(A) ~ A^{-tau_area}
    area, area_counts = positive(10.0 ** np.asarray(d["area_log_bins"], float),
                                 10.0 ** np.asarray(d["area_log_counts"], float))
    ax_c.plot(area, area_counts, lw=0, marker="o", ms=3.4,
              mfc=COLORS[2], mec="none")
    guide(ax_c, area, area_counts, -tau_area, COLORS[2])
    style(ax_c, "c",
          "area  [km$^2$]", "counts")
    annotate(ax_c, rf"$\tau_\mathrm{{area}} = {tau_area:.2f}$", rising=False)

    # d. nested-perimeter distribution.  n(P) ~ P^{-tau_per}
    per, per_counts = positive(10.0 ** np.asarray(d["per_log_bins"], float),
                               10.0 ** np.asarray(d["per_log_counts"], float))
    ax_d.plot(per, per_counts, lw=0, marker="o", ms=3.4,
              mfc=COLORS[3], mec="none")
    guide(ax_d, per, per_counts, -tau_per, COLORS[3])
    style(ax_d, "d",
          "nested perimeter  [km]", "counts")
    annotate(ax_d, rf"$\tau_\mathrm{{per.}} = {tau_per:.2f}$", rising=False)

    # Provenance is printed rather than drawn -- it belongs in the caption.
    print(f"{n} member{'s' if n != 1 else ''} pooled, "
          f"tau > {float(d['tau_threshold']):g} cloud mask, "
          f"cloud cover {cover:.2f}")

    fig.tight_layout()
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT.name}.pdf and {OUT.name}.png")


if __name__ == "__main__":
    main()
