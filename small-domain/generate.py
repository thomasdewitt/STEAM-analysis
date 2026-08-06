#!/usr/bin/env python3
"""A small, finely resolved STEAM domain for visualization.

Two runs, one per flux amplitude, on the same ukmo_ra1t profile the square
campaign uses: 20.48 x 7.68 km horizontally at dx = 10 m, from the surface to
5 km, outer scale L = 20 480 m (the full long dimension). Constant 10 m
spheroscale, anchored bounds, H_h and lambda at package defaults.

This is a rendering target, not an analysis product -- the point is a domain
fine enough to look at closely, with the finest cascade class at 2 dx = 20 m
sitting only one octave above the spheroscale.

GRID. The vertical spacing is not free: it follows from dx through the
aspect-ratio scaling, so dz = 7.34 m and the finest grid is
(2048, 768, 681). That is a 3.99 GiB field and ~46.5 GiB at the cascade's
peak. The requested 2048 x 1024 would have been (2048, 1024, 681), a
5.32 GiB field needing ~62 GiB, which STEAM's preflight refuses on this box
-- hence 768. Both runs are serial and the working set is close enough to
the machine's limit that they should not be run alongside anything large.

Output: runs/small-domain/small_<set>.nc, full fields plus diagnostics.

Usage: python generate.py [SET]      (default: both)
"""

import sys
import time
from pathlib import Path

import numpy as np

import importlib
_steam_simulate = importlib.import_module("steam.simulate")

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "runs" / "small-domain"
PROFILES = REPO / "runs" / "input_profiles"

PROFILE_HOST = "ukmo_ra1t"          # the square campaign's profile
SETS = {"c002": 0.02, "c005": 0.05, "c017": 0.17}

NX, NY = 2048, 768
DX = 10.0
OUTER_SCALE = 20480.0               # the full long dimension
SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 5000.0
PROFILE_DZ = 50.0
DEVICE = "cpu"


def run_one(set_tag):
    out_nc = RUNS / f"small_{set_tag}.nc"
    if out_nc.exists():
        print(f"{out_nc.name} exists, skipping", flush=True)
        return

    src = np.load(PROFILES / f"{PROFILE_HOST}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    spheroscale = np.full(src["z_profile"].size, SPHEROSCALE_CONSTANT)
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    # Anchored bounds, as the square campaign sets them.
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                  float(h_profile.max()) + 1.0)
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
    )
    compute_diagnostics(str(out_nc), compress=True)
    print(f"{out_nc.name} done in {time.perf_counter() - t0:.0f} s "
          f"({out_nc.stat().st_size / 1e9:.1f} GB)", flush=True)


def main():
    tags = sys.argv[1:] or list(SETS)
    for tag in tags:
        if tag not in SETS:
            raise SystemExit(f"unknown set {tag!r} (have {list(SETS)})")
        run_one(tag)
    print("small-domain generation complete", flush=True)


if __name__ == "__main__":
    main()
