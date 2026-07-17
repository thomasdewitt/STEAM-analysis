#!/usr/bin/env python3
"""Constant-spheroscale (10 m) variant of the canonical TWPICE STEAM run.

Identical to run_twpice.py in every respect (grid, outer scale, seed, flux
noise, 20 km domain top) except the spheroscale profile: constant 10 m at all
heights instead of log-linear 1000 m -> 10 m. Isolates the effect of the
spheroscale profile on variance, PDFs, and the upper-level cloud fraction.

The constant 10 m spheroscale keeps dz fine over the whole column: the
finest grid has 758 levels (vs 342 in the varying case). At the canonical
2048 x 1024 that peaks ~47 GiB resident plus interpolation temporaries and
OOMs the 60 GiB host even at MemoryMax=56G (tried 2026-07-17, killed twice
at the finest step). So this case runs at 1024 x 1024 -- 102.4 km square,
exactly one outer-scale tile in each direction; finest grid 1024 x 1024 x
758, ~24 GiB peak.

RUN SANDBOXED (see run_twpice.py):

    systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=2G \
        /usr/bin/time -v uv run python -u run_twpice_ls10.py
"""

import time

import netCDF4
import numpy as np

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

PROFILE_DZ = 50.0
DOMAIN_HEIGHT = 20000.0
SPHEROSCALE = 10.0
SEED = 20260714
OUTPUT = "steam_twpice_ls10.nc"

src = netCDF4.Dataset("../profiles/twpice_mean_profiles.nc")
z_sam = src.variables["z"][:].astype(np.float64)
h_sam = src.variables["h"][:].astype(np.float64)
qt_sam = src.variables["qt"][:].astype(np.float64)
surface_pressure = float(src.surface_pressure)
src.close()

z_uniform = np.arange(0.0, DOMAIN_HEIGHT + PROFILE_DZ, PROFILE_DZ)
h_profile = np.interp(z_uniform, z_sam, h_sam)
qt_profile = np.interp(z_uniform, z_sam, qt_sam)

spheroscale = np.full_like(z_uniform, SPHEROSCALE)

h_min = h_profile.min() - 10 * cp
h_max = h_profile.max() + 10 * cp

started = time.perf_counter()
simulate(
    h_profile, qt_profile,
    nx=1024, ny=1024, dx=100.0, dy=100.0,
    outer_scale=102400.0,
    spheroscale=spheroscale,
    anisotropy="piecewise_isotropic_below_spheroscale",
    domain_height=DOMAIN_HEIGHT,
    profile_dz=PROFILE_DZ,
    output_path=OUTPUT,
    surface_pressure=surface_pressure,
    seed=SEED,
    h_min=h_min, h_max=h_max,
    qt_min=0.0, qt_max=0.03,
    compress=True,
    device="cuda",
)
print(f"simulate: {time.perf_counter() - started:.0f} s")

started = time.perf_counter()
compute_diagnostics(OUTPUT, compress=True)
print(f"diagnostics: {time.perf_counter() - started:.0f} s")
