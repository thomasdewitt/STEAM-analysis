#!/usr/bin/env python3
"""A small, finely resolved STEAM domain for visualization.

One run per flux amplitude, on the same ukmo_ra1t profile the square campaign
uses: a 40.96 x 20.48 km parent at dx = 20 m from the surface to 5 km, outer
scale L = 40 960 m (the full long dimension), with a centered 2.56 x 2.56 km
nest at dx = 5 m carried to full depth. Constant 10 m spheroscale, anchored
bounds, H_h and lambda at package defaults.

This is a rendering target, not an analysis product -- the point is a domain
fine enough to look at closely. The nest is where that happens: its finest
cascade class is 2 dx = 10 m, exactly the spheroscale, so it resolves the
cascade down to the isotropic floor and no further.

GRID. The vertical spacing is not free: dz = k_z(2 dx) / 2, so it follows dx
through the aspect-ratio scaling. The parent is (2048, 1024, 463) at
dz = 10.80 m -- a 3.62 GiB field, ~38 GiB at the cascade's peak. Doubling dx
is what makes the wider 2048 x 1024 footprint fit: it halves the level count,
and the field comes out slightly SMALLER than the 2048 x 768 run at dx = 10 m
it replaces (3.99 GiB), which is why the preflight that refused
2048 x 1024 x 681 accepts this. The nest is (512, 512, 1001) at dz = 5.00 m
-- 0.98 GiB, ~10 GiB at its peak; dz stops falling with dx there because
2 dx has reached the spheroscale and the finest class is isotropic.

Runs are serial and the parent's working set is close enough to the machine's
limit that they should not be run alongside anything large.

Output: runs/small-domain/small_<set>.nc -- full fields plus diagnostics for
the parent, and the nest under refinements/r0 with its own diagnostics.

Usage: python run_steam_simulations.py [SET]      (default: all)
"""

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4

import importlib
_steam_simulate = importlib.import_module("steam.simulate")

from steam.simulate import simulate, refine
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # small-domain/
REPO = BASE.parent
RUNS = REPO / "runs" / "small-domain"
PROFILES = REPO / "runs" / "input_profiles"

PROFILE_HOST = "ukmo_ra1t"          # the square campaign's profile
SETS = {"c002": 0.02, "c005": 0.05, "c017": 0.17}

NX, NY = 2048, 1024
DX = 20.0
OUTER_SCALE = 40960.0               # the full long dimension
SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 5000.0
PROFILE_DZ = 50.0
DEVICE = "cpu"

# Centered nest: 512 x 512 cells at dx = 5 m (2.56 x 2.56 km), full depth.
# 2560 m spans 128 parent cells at dx = 20 m, so the window is the middle 128
# of 2048 in x and of 1024 in y.
NEST_NX = 512
NEST_DX = 5.0
_NEST_PARENT_CELLS = int(NEST_NX * NEST_DX / DX)        # 128
NEST = dict(x_start=(NX - _NEST_PARENT_CELLS) // 2,
            x_stop=(NX + _NEST_PARENT_CELLS) // 2,
            y_start=(NY - _NEST_PARENT_CELLS) // 2,
            y_stop=(NY + _NEST_PARENT_CELLS) // 2,
            dx=NEST_DX, dy=NEST_DX)
NEST_GROUP = "refinements/r0"


def run_nest(out_nc):
    """Cut the centered nest, if the file does not already carry it."""
    with netCDF4.Dataset(out_nc) as ds:
        exists = ("refinements" in ds.groups
                  and "r0" in ds.groups["refinements"].groups)
    if exists:
        print(f"  nest exists in {out_nc.name}, skipping", flush=True)
        return
    print(f"  nest {NEST_NX} x {NEST_NX} at dx = {NEST_DX:.0f} m "
          f"({NEST_NX * NEST_DX / 1000:.2f} km square), parent cells "
          f"x {NEST['x_start']}:{NEST['x_stop']}, "
          f"y {NEST['y_start']}:{NEST['y_stop']}", flush=True)
    t0 = time.perf_counter()
    refine(str(out_nc), parent_group="/", output_group=NEST_GROUP,
           device=DEVICE, compress=True, **NEST)
    compute_diagnostics(str(out_nc), group=NEST_GROUP, compress=True)
    print(f"  nest done in {time.perf_counter() - t0:.0f} s", flush=True)


def run_parent(set_tag, out_nc):
    if out_nc.exists():
        print(f"{out_nc.name} exists, skipping the parent", flush=True)
        return

    src = np.load(PROFILES / f"{PROFILE_HOST}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    spheroscale = np.full(src["z_profile"].size, SPHEROSCALE_CONSTANT)
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    # Anchored bounds, as the square campaign sets them.
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                  float(h_profile.max()))
    h_lower = float(h_profile.min()) - 10.0 * cp

    _steam_simulate.FLUX_SCALE = SETS[set_tag]
    print(f"=== {set_tag} === {NX} x {NY} at dx = {DX:.0f} m "
          f"({NX * DX / 1000:.2f} x {NY * DX / 1000:.2f} km), "
          f"top {DOMAIN_HEIGHT / 1000:.0f} km, L = {OUTER_SCALE / 1000:.2f} km, "
          f"c = {SETS[set_tag]}", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    simulate(
        h_profile, qt_profile,
        nx=NX, ny=NY, dx=DX, dy=DX,
        outer_scale=OUTER_SCALE,
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=surface_pressure,
        seed=7000 + list(SETS).index(set_tag),
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device=DEVICE,
        save_for_refinement=True,
    )
    compute_diagnostics(str(out_nc), compress=True)
    print(f"{out_nc.name} parent done in {time.perf_counter() - t0:.0f} s "
          f"({out_nc.stat().st_size / 1e9:.1f} GB)", flush=True)


def main():
    tags = sys.argv[1:] or list(SETS)
    for tag in tags:
        if tag not in SETS:
            raise SystemExit(f"unknown set {tag!r} (have {list(SETS)})")
        out_nc = RUNS / f"small_{tag}.nc"
        run_parent(tag, out_nc)
        run_nest(out_nc)
    print("small-domain generation complete", flush=True)


if __name__ == "__main__":
    main()
