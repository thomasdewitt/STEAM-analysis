#!/usr/bin/env python3
"""Cloud renders of the nested domains (cloudyview witness tier).

Renders the two render nests written by run_render_nests.py into the
icon_lem m00 production square: nest A (24 km @ 46.9 m, full depth) and
nest B (6 km @ ~2.9 m, lowest 2 km). Witness = ray-marching tier (the
Monte Carlo tier does not converge for thick clouds; main.tex L253).
A few azimuths + one near-nadir overview each; all saved to
figs/renders/ for Thomas to pick from.

Run with cloudyview's venv (has xarray/numba):
  ~/code-and-data/cloudyview/.venv/bin/python make_renders.py
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4

sys.path.insert(0, str(Path.home() / "code-and-data/cloudyview"))
import cloudyview as cv

HERE = Path(__file__).parent
PARENT = HERE / "runs" / "steam_sq10_icon_lem_m00.nc"
OUT = HERE / "figs" / "renders"


def load_group(group):
    ds = netCDF4.Dataset(PARENT)
    ds.set_auto_mask(False)
    g = ds
    for part in group.split("/"):
        g = g.groups[part]
    z = g.variables["z"][:].astype(np.float64)
    lwc = g.variables["qc"][:].astype(np.float32) * 1000.0   # kg/kg -> g/kg
    iwc = g.variables["qi"][:].astype(np.float32) * 1000.0
    dx = float(g.dx) if hasattr(g, "dx") else float(
        np.diff(g.variables["x"][:2])[0]) if "x" in g.variables else None
    if "x" in g.variables:
        x = g.variables["x"][:].astype(np.float64)
        y = g.variables["y"][:].astype(np.float64)
    else:
        x = np.arange(lwc.shape[0]) * dx
        y = np.arange(lwc.shape[1]) * dx
    ds.close()
    return cv.CloudField(lwc=lwc, x=x, y=y, z=z, iwc=iwc)


def render_set(field, name, size=(1200, 800)):
    OUT.mkdir(parents=True, exist_ok=True)
    shots = [
        ("oblique_n", cv.Camera(position=(0, -0.85, -0.9), azimuth=0,
                                elevation=25, fov=90)),
        ("oblique_e", cv.Camera(position=(-0.85, 0, -0.9), azimuth=90,
                                elevation=25, fov=90)),
        ("low_horizon", cv.Camera(position=(0, -0.98, -0.995), azimuth=0,
                                  elevation=12, fov=75)),
    ]
    for tag, cam in shots:
        img = cv.witness(field, camera=cam, size=size, verbose=True)
        path = OUT / f"{name}_{tag}.png"
        cv.save_image(img, str(path))
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    for group, name in [("refinements/render_a", "nest24km"),
                        ("refinements/render_b", "nest6km")]:
        print(f"loading {group}", flush=True)
        field = load_group(group)
        print(field, flush=True)
        render_set(field, name)
