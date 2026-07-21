#!/usr/bin/env python3
"""Correlation dimension and size distributions of tau > 1 cloud masks.

Per model (host and STEAM), the three snapshots' vertically-integrated
optical depth fields (liquid + ice, cloudyview SAM relationships) are
thresholded at tau = 1 into binary masks. objscale then gives, per case
(all three masks passed at once):
  - ensemble correlation dimension (with C_l for the scaling plot),
  - finite-domain size-distribution exponent for area (with counts).

Channel geometry: ~2000 x ~130 cells at dx = 3 km. Domains are periodic
but the finite_* truncation correction treats edges as truncating --
conservative for the hosts (noted in the figure caption). Correlation
lags run ~24-127 km (defaults: 8x pixel to 0.33x the 384-432 km width).

Writes stats/fractal.npz and figs/fractal.png.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import objscale

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
TAU_THRESHOLD = 1.0
DX = 3000.0

spec = importlib.util.spec_from_file_location(
    "cv_optical_depth",
    Path.home() / "code-and-data/cloudyview/cloudyview/optical_depth.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def host_masks(model):
    from extract_stats import ADAPTERS
    masks = []
    for i in range(3):
        z, T, qv, qc, qi, _ = ADAPTERS[model](i)
        # adapter fields are (nz, ny, nx) mixing ratio kg/kg; cloudyview wants
        # (nx, ny, nz) g/kg.
        lwc = np.moveaxis(qc, 0, -1) * 1000.0
        iwc = np.moveaxis(qi, 0, -1) * 1000.0
        tau = cv.vertically_integrated_optical_depth(
            lwc, np.asarray(z, dtype=np.float64), iwc=iwc)
        masks.append((tau > TAU_THRESHOLD).astype(np.float32))
    return masks


def steam_masks(model):
    import netCDF4
    masks = []
    for i in range(3):
        ds = netCDF4.Dataset(RUNS / f"steam_{model}_snap{i}.nc")
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(np.float64)
        lwc = ds.variables["qc"][:] * 1000.0   # (x, y, z) already
        iwc = ds.variables["qi"][:] * 1000.0
        ds.close()
        tau = cv.vertically_integrated_optical_depth(lwc, z, iwc=iwc)
        masks.append((tau > TAU_THRESHOLD).astype(np.float32))
    return masks


def analyze(masks):
    sizes = [np.full(m.shape, DX) for m in masks]
    # objscale accepts one x_sizes/y_sizes pair; shapes differ per host model
    # only between models, not within one case, so this is safe.
    # Default minlength (8 px = 24 km) leaves ratio < 10 against the ~127 km
    # channel-width cap; drop to 2 px (6 km), accepting pixel-scale bias at
    # the small end (flagged in the caption).
    dim, bins, C_l = objscale.ensemble_correlation_dimension(
        masks, x_sizes=sizes[0], y_sizes=sizes[0], minlength=2 * DX,
        point_reduction_factor=10, return_C_l=True)
    exponent, (log_bins, log_counts) = objscale.finite_array_powerlaw_exponent(
        masks, "area", x_sizes=sizes[0], y_sizes=sizes[0],
        return_counts=True)
    cf = float(np.mean([m.mean() for m in masks]))
    return dict(dim=dim, C_bins=bins, C_l=C_l, exponent=exponent,
                sd_log_bins=log_bins, sd_log_counts=log_counts, cover=cf)


def compute():
    models = sorted({p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
                     if not p.name.startswith("steam_")})
    out = {"models": np.array(models)}
    for model in models:
        for prefix, masks in ((model, host_masks(model)),
                              (f"steam_{model}", steam_masks(model))):
            r = analyze(masks)
            for k, v in r.items():
                out[f"{prefix}_{k}"] = v
            print(f"{prefix}: D2={r['dim']:.2f}, "
                  f"area exp={r['exponent']:.2f}, cover={r['cover']:.2f}")
    np.savez(STATS / "fractal.npz", **out)
    print("wrote stats/fractal.npz")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
        "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })
    d = np.load(STATS / "fractal.npz", allow_pickle=True)
    models = list(d["models"])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for m, c in zip(models, colors):
        for prefix, ls in ((m, "-"), (f"steam_{m}", "--")):
            label = (f"{m} {d[f'{m}_dim']:.2f}/"
                     f"{d[f'steam_{m}_dim']:.2f}") if ls == "-" else None
            axes[0].loglog(d[f"{prefix}_C_bins"] / 1000.0, d[f"{prefix}_C_l"],
                           color=c, lw=1.1, ls=ls, label=label)
            sd_label = (f"{m} {d[f'{m}_exponent']:.2f}/"
                        f"{d[f'steam_{m}_exponent']:.2f}") if ls == "-" else None
            axes[1].plot(d[f"{prefix}_sd_log_bins"] - 6.0,  # m^2 -> km^2
                         d[f"{prefix}_sd_log_counts"],
                         color=c, lw=1.1, ls=ls, label=sd_label)
    axes[0].set(xlabel="r [km]", ylabel="correlation integral $C(r)$",
                title="$\\tau>1$ mask correlation integral (host/STEAM $D_2$)")
    axes[1].set(xlabel="log$_{10}$ area [km$^2$]", ylabel="log$_{10}$ counts",
                title="area distribution, truncation-corrected (host/STEAM slope)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "figs" / "fractal.png")
    print("wrote figs/fractal.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["compute", "figure"]
    if "compute" in wanted:
        compute()
    if "figure" in wanted:
        figure()
