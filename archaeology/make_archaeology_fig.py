#!/usr/bin/env python3
"""Summary figure for the 2026-07-27 git archaeology (flat-xi(1) bug).

Four eras/variants, identical config (512^2 @ 1000 km, outer 500 km,
constant 30 m spheroscale, icon_lem snap0, seed 7), h at ~7 km:

  - EGU checkpoint ac16b35 (May 1)          — the known-good reference
  - 71fd792 (Jul 22, E[W] ensemble norm)    — "good"-looking but clip-shaped
  - HEAD (realized W norm)                  — the flat-xi(1) state
  - HEAD + joint product norm (candidate)   — normalize W*S_k jointly

Panel a: realized mean-abs turbulon amplitude <|A|> per class.
Panel b: final-field Haar fluctuation at the class lags.

Reads stats/ladder_compare_{label}.npz; writes
figs/archaeology/ladder_archaeology.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
H_H = 0.45

CURVES = (
    ("egu_checkpoint",     "May checkpoint ac16b35",        "#6b6b6b"),
    ("good_ensemble_norm", "Jul 22 E[W] norm (71fd792)",    "#e8a13c"),
    ("head_realized_norm", "HEAD, realized W norm",         "#c0392b"),
    ("head_joint_norm",    "HEAD + joint W·S$_k$ norm",     "#127a6d"),
)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})


def ref_line(ax, x_mid, y_mid, slope, span=10 ** 0.7, label="$H_h$"):
    xs = np.array([x_mid / span, x_mid * span])
    ax.loglog(xs, y_mid * (xs / x_mid) ** slope, color="0.35", lw=0.9, ls="-.")
    ax.annotate(label, (xs[1], y_mid * span ** slope), fontsize=6.5,
                color="0.35", xytext=(2, -2), textcoords="offset points")


def main():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 4))
    for label, name, color in CURVES:
        d = np.load(HERE / "stats" / f"ladder_compare_{label}.npz")
        k = d["k_values"] / 1000.0
        ax_a.loglog(k, d["amps"], "o-", color=color, lw=1.2, ms=3, label=name)
        ax_b.loglog(d["final_lags_m"] / 1000.0, d["final_F"], "o-",
                    color=color, lw=1.2, ms=3, label=name)

    d_ref = np.load(HERE / "stats" / "ladder_compare_egu_checkpoint.npz")
    mid = len(d_ref["k_values"]) // 2
    ref_line(ax_a, d_ref["k_values"][mid] / 1000, d_ref["amps"][mid], H_H)
    lags_km = d_ref["final_lags_m"] / 1000.0
    ref_line(ax_b, lags_km[mid], d_ref["final_F"][mid], H_H)

    ax_a.set(xlabel="class scale $k$ [km]",
             ylabel=r"$\langle |A| \rangle$ at 7 km  [J kg$^{-1}$]",
             title="a) realized turbulon amplitude ladder")
    ax_b.set(xlabel="lag [km]", ylabel=r"$\hat{M}_1(\ell)$  [J kg$^{-1}$]",
             title="b) final-field Haar fluctuation")
    for ax in (ax_a, ax_b):
        ax.grid(True, which="both", alpha=0.5)
    ax_a.legend(fontsize=7)
    fig.suptitle("Ladder archaeology — h at 7 km, 512$^2$ @ 1000 km, "
                 "outer 500 km, seed 7 (design $H_h$ = 0.45)", fontsize=9.5)
    fig.tight_layout()
    out = HERE / "figs" / "archaeology" / "ladder_archaeology.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
