#!/usr/bin/env python3
"""Canonical STEAM run initialized from the SAM-TWPICE mean profiles.

Matches SAM's horizontal grid exactly (2048 x 2048, dx = 100 m, 204.8 km) and
its vertical extent (26.65 km). Outer scale = the domain (204.8 km), canonical
anisotropy with constant spheroscale 100 m, flux noise c = 0.21 (the TWPICE
C1 = 0.1 peg). Appends T/qv/qc/qi/p diagnostics for cloud statistics.
"""

import time

import netCDF4
import numpy as np

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

PROFILE_DZ = 50.0
DOMAIN_HEIGHT = 26650.0
SEED = 20260714
OUTPUT = "steam_twpice.nc"

src = netCDF4.Dataset("../profiles/twpice_mean_profiles.nc")
z_sam = src.variables["z"][:].astype(np.float64)
h_sam = src.variables["h"][:].astype(np.float64)
qt_sam = src.variables["qt"][:].astype(np.float64)
surface_pressure = float(src.surface_pressure)
src.close()

# SAM's grid is stretched; STEAM wants the profile at uniform spacing.
z_uniform = np.arange(0.0, DOMAIN_HEIGHT + PROFILE_DZ, PROFILE_DZ)
h_profile = np.interp(z_uniform, z_sam, h_sam)
qt_profile = np.interp(z_uniform, z_sam, qt_sam)

# Soft-clamp bounds: 10 K beyond the profile range for h; qt bounded by zero
# and a ceiling above the moistest SAM air (~22 g/kg).
h_min = h_profile.min() - 10 * cp
h_max = h_profile.max() + 10 * cp

started = time.perf_counter()
simulate(
    h_profile, qt_profile,
    nx=2048, ny=2048, dx=100.0, dy=100.0,
    outer_scale=204800.0,
    spheroscale=100.0,
    domain_height=DOMAIN_HEIGHT,
    profile_dz=PROFILE_DZ,
    output_path=OUTPUT,
    surface_pressure=surface_pressure,
    seed=SEED,
    h_min=h_min, h_max=h_max,
    qt_min=0.0, qt_max=0.03,
    compress=True,
)
print(f"simulate: {time.perf_counter() - started:.0f} s")

started = time.perf_counter()
compute_diagnostics(OUTPUT, compress=True)
print(f"diagnostics: {time.perf_counter() - started:.0f} s")
