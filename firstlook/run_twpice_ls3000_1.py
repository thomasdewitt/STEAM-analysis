#!/usr/bin/env python3
"""Steep-spheroscale variant of the canonical TWPICE STEAM run.

Identical to run_twpice.py except the spheroscale profile: log-linear from
3000 m at the surface to 1 m at the 20 km top (one decade per ~5.7 km;
~440 m at 5 km, ~65 m at 10 km, ~9.7 m at 15 km). Steeper decay than the
canonical 1000 -> 10 m case, with a larger boundary-layer spheroscale.

Grid follows the ls10 lesson: the fine spheroscale aloft deepens the finest
z-grid, so run at 1024 x 1024 (one outer-scale tile per direction) to stay
inside host memory.

RUN SANDBOXED (see run_twpice.py):

    systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=2G \
        /usr/bin/time -v uv run python -u run_twpice_ls3000_1.py
"""

import time

import netCDF4
import numpy as np

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

PROFILE_DZ = 50.0
DOMAIN_HEIGHT = 20000.0
SPHEROSCALE_SURFACE = 3000.0
SPHEROSCALE_TOP = 1.0
SEED = 20260714
OUTPUT = "steam_twpice_ls3000_1.nc"

src = netCDF4.Dataset("../profiles/twpice_mean_profiles.nc")
z_sam = src.variables["z"][:].astype(np.float64)
h_sam = src.variables["h"][:].astype(np.float64)
qt_sam = src.variables["qt"][:].astype(np.float64)
surface_pressure = float(src.surface_pressure)
src.close()

z_uniform = np.arange(0.0, DOMAIN_HEIGHT + PROFILE_DZ, PROFILE_DZ)
h_profile = np.interp(z_uniform, z_sam, h_sam)
qt_profile = np.interp(z_uniform, z_sam, qt_sam)

spheroscale = SPHEROSCALE_SURFACE * (
    SPHEROSCALE_TOP / SPHEROSCALE_SURFACE) ** (z_uniform / DOMAIN_HEIGHT)

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
