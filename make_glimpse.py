#!/usr/bin/env python3
"""Top-down glimpse renders, one seed per case (cloudyview glimpse tier).

cv.glimpse is the two-stream visual albedo A = tau/(tau + 2/(1-g)) of the
vertically integrated condensate (qc + qi), which unlike beam opacity keeps
contrast between cirrus and deep cores. Cheap enough to run on every case,
so every paper-bound simulation gets one look: the m00 member of each
square profile and all five matched sims.

Run with cloudyview's venv (has its own deps):
  ~/code-and-data/cloudyview/.venv/bin/python make_glimpse.py

Writes figs/square/glimpse_<profile>.png and figs/matched/glimpse_<model>.png.
Skips a case whose PNG already exists.
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / "code-and-data/cloudyview"))
import cloudyview as cv
from cloudyview.basic_render import cloud_colors

HERE = Path(__file__).parent
RUNS = HERE / "runs"

CASES = (
    [(RUNS / f"steam_sq10_{m}_m00.nc", HERE / "figs" / "square",
      f"glimpse_{m}.png", f"square, {m}, m00") for m in
     ("ukmo_ra1t", "icon_nwp")]
    + [(RUNS / f"steam_matched_{m}.nc", HERE / "figs" / "matched",
        f"glimpse_{m}.png", f"matched, {m}") for m in
       ("twpice", "les_cm1", "les_sam", "les_dales", "les_icon_lem")]
)


def glimpse_one(path, outdir, name, title):
    out = outdir / name
    if out.exists():
        print(f"{out.relative_to(HERE)} exists, skipping", flush=True)
        return
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    field = cv.CloudField(
        lwc=ds.variables["qc"][:].astype(np.float32) * 1000.0,   # -> g/kg
        iwc=ds.variables["qi"][:].astype(np.float32) * 1000.0,
        x=ds.variables["x"][:].astype(np.float64),
        y=ds.variables["y"][:].astype(np.float64),
        z=ds.variables["z"][:].astype(np.float64))
    ds.close()
    albedo = cv.glimpse(field)
    extent = [field.x[0] / 1000.0, field.x[-1] / 1000.0,
              field.y[0] / 1000.0, field.y[-1] / 1000.0]

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
        "figure.dpi": 200,
    })
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    im = ax.imshow(albedo, origin="lower", extent=extent, cmap=cloud_colors,
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set(xlabel="x [km]", ylabel="y [km]",
           title=f"{title}\ntwo-stream visual albedo, $q_c + q_i$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("albedo")
    cb.outline.set_visible(False)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}  "
          f"(mean albedo {float(albedo.mean()):.3f}, "
          f"albedo>0.5 fraction {float((albedo > 0.5).mean()):.3f})",
          flush=True)


if __name__ == "__main__":
    for path, outdir, name, title in CASES:
        glimpse_one(path, outdir, name, title)
