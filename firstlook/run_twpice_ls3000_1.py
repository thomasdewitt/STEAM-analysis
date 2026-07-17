#!/usr/bin/env python3
"""Large-low-level-spheroscale variant of the canonical TWPICE STEAM run.

Identical to run_twpice.py except the spheroscale profile: constant 3000 m
from the surface to 4 km, then log-linear down to 1 m at the 20 km top
(one decade per ~2 km above the break: ~1090 m at 6 km, ~130 m at 10 km,
~4 m at 17 km). Motivated by cloudyview renders wanting much larger
low-level spheroscale.

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
SPHEROSCALE_LOW = 3000.0     # constant below the breakpoint
BREAK_HEIGHT = 4000.0        # top of the constant layer
SPHEROSCALE_TOP = 1.0        # at the domain top (log-linear above the break)
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

# Constant SPHEROSCALE_LOW below BREAK_HEIGHT; log-linear down to
# SPHEROSCALE_TOP at the domain top above it.
above = np.clip(
    (z_uniform - BREAK_HEIGHT) / (DOMAIN_HEIGHT - BREAK_HEIGHT), 0.0, 1.0)
spheroscale = SPHEROSCALE_LOW * (SPHEROSCALE_TOP / SPHEROSCALE_LOW) ** above

h_min = h_profile.min() - 10 * cp
h_max = h_profile.max() + 10 * cp

started = time.perf_counter()
simulate(
    h_profile, qt_profile,
    nx=2048, ny=512, dx=100.0, dy=100.0,
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
