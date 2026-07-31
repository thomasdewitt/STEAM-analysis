#!/usr/bin/env python3
"""Matched-geometry profile figures: each host against its own STEAM run.

Unlike make_profiles_ensemble.py there is no ensemble to shade here -- one
STEAM member per host -- so the comparison is line against line, host solid
and STEAM dashed in one colour per model (make_pdfs.py's convention).

TWPICE is kept on its own axes: it is an observed-case SAM run of the TWP-ICE
campaign, not RCE, and its profiles sit well away from the RCEMIP
RCE_small_les300 LES.

  figs/matched/profiles_twpice.png            std h', std qt', cloud fraction
  figs/matched/profiles_twpice_diagnostic.png mean and std of qc, qi, T
  figs/matched/profiles_les.png               (same, four LES together)
  figs/matched/profiles_les_diagnostic.png

Pressure is omitted throughout, as in make_profiles_ensemble.py: the host
adapters carry no 3D pressure field to compare against.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
STATS = HERE / "stats"
FIGS = HERE / "figs" / "matched"
Z = np.arange(0.0, 20000.0 + 100.0, 100.0)
ZKM = Z / 1000.0

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
    "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})
TWPICE = ["twpice"]
LES = ["les_cm1", "les_dales", "les_icon_lem", "les_sam"]
COLORS = {"twpice": "#B0532E", "les_cm1": "#1F6F8B", "les_dales": "#C9772F",
          "les_icon_lem": "#6B4C9A", "les_sam": "#3E8E5A"}

# (key, axis label, unit scale). Water variables are plotted in g/kg: in
# kg/kg the tick labels are 0.00025-wide and collide.
PROGNOSTIC = [("h_var", r"std $h'$ [J kg$^{-1}$]", 1.0),
              ("qt_var", r"std $q_t'$ [g kg$^{-1}$]", 1000.0),
              ("cloud_fraction", "cloud fraction", 1.0)]
DIAGNOSTIC = [("qc_mean", r"mean $q_c$ [g kg$^{-1}$]", 1000.0),
              ("qi_mean", r"mean $q_i$ [g kg$^{-1}$]", 1000.0),
              ("T_mean", "mean T [K]", 1.0),
              ("qc_std", r"std $q_c$ [g kg$^{-1}$]", 1000.0),
              ("qi_std", r"std $q_i$ [g kg$^{-1}$]", 1000.0),
              ("T_std", "std T [K]", 1.0)]


def profile(name, key, scale):
    """Interpolated to the common Z grid; _var keys come back as std."""
    d = np.load(STATS / f"{name}.npz")
    v = np.sqrt(d[key]) if key.endswith("_var") else d[key]
    good = np.isfinite(v)
    return scale * np.interp(Z, d["z"].astype(float)[good], v[good],
                             left=np.nan, right=np.nan)


def sheet(models, specs, diag, shape, figsize, legend_ax, out_name, title):
    prefix = "diag_" if diag else ""
    fig, axes = plt.subplots(*shape, figsize=figsize, sharey=True)
    for ax, (key, lab, scale) in zip(np.atleast_1d(axes).ravel(), specs):
        for m in models:
            ax.plot(profile(f"{prefix}{m}_snap0", key, scale), ZKM, lw=1.1,
                    color=COLORS[m], label=m)
            ax.plot(profile(f"{prefix}steam_matched_{m}", key, scale), ZKM,
                    lw=1.1, ls="--", color=COLORS[m])
        ax.set_xlabel(lab)
        ax.set_ylim(0, 20)
        ax.grid(True)
    for row in np.atleast_2d(axes):
        row[0].set_ylabel("z [km]")
    # legend_ax picks the panel whose upper-right corner is empty: the cloud
    # fraction panel on the prognostic sheets, mean q_c on the diagnostic ones.
    np.atleast_1d(axes).ravel()[legend_ax].legend(
        fontsize=7, loc="upper right", title="solid host / dash STEAM",
        title_fontsize=7)
    fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/matched/{out_name}", flush=True)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    for models, tag, title in (
            (TWPICE, "twpice", "matched STEAM vs SAM-TWPICE"),
            (LES, "les", "matched STEAM vs RCEMIP RCE_small_les300")):
        sheet(models, PROGNOSTIC, False, (1, 3), (9.0, 3.6), -1,
              f"profiles_{tag}.png", title + ", prognostic")
        sheet(models, DIAGNOSTIC, True, (2, 3), (9.0, 6.0), 0,
              f"profiles_{tag}_diagnostic.png", title + ", diagnostic")


if __name__ == "__main__":
    main()
