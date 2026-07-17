#!/usr/bin/env python3
"""Canonical STEAM run initialized from the SAM-TWPICE mean profiles.

Matches SAM's grid in x (2048, dx = 100 m, 204.8 km) at half its y extent
(1024, 102.4 km). The simulated domain is the lowest 20 km. Outer scale =
102.4 km (each horizontal extent must be an integer multiple of it: x holds
two tiles, y one); spheroscale profile log-linear from 1000 m at the surface
to 10 m at 20 km, with turbulons isotropic below the spheroscale. Flux noise
c = 0.21 (the TWPICE C1 = 0.1 peg). Appends T/qv/qc/qi/p diagnostics.

RUN SANDBOXED — a host OOM here once killed the whole login session. Launch as

    systemd-run --user --scope -p MemoryMax=28G -p MemorySwapMax=2G \
        /usr/bin/time -v uv run python -u run_twpice.py

so the kernel kills only this run, never the session. 28G covers the guarded
8-field peak (21.4 GiB at 2048 x 1024 x 342) plus torch/CUDA overhead. The
full-scale option (ny=2048, outer_scale=102400.0, peak 42.8 GiB, MemoryMax=48G)
fits only when nothing else heavy runs on the 60 GiB host.
"""

import time

import netCDF4
import numpy as np

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

PROFILE_DZ = 50.0
DOMAIN_HEIGHT = 20000.0
SPHEROSCALE_SURFACE = 1000.0
SPHEROSCALE_TOP = 10.0
SEED = 20260714
OUTPUT = "steam_twpice.nc"

src = netCDF4.Dataset("../profiles/twpice_mean_profiles.nc")
z_sam = src.variables["z"][:].astype(np.float64)
h_sam = src.variables["h"][:].astype(np.float64)
qt_sam = src.variables["qt"][:].astype(np.float64)
surface_pressure = float(src.surface_pressure)
src.close()

# SAM's grid is stretched; STEAM wants the profile at uniform spacing,
# truncated to the simulated 20 km.
z_uniform = np.arange(0.0, DOMAIN_HEIGHT + PROFILE_DZ, PROFILE_DZ)
h_profile = np.interp(z_uniform, z_sam, h_sam)
qt_profile = np.interp(z_uniform, z_sam, qt_sam)

# Spheroscale: log-linear in z, 1000 m at the surface -> 10 m at 20 km.
spheroscale = SPHEROSCALE_SURFACE * (
    SPHEROSCALE_TOP / SPHEROSCALE_SURFACE) ** (z_uniform / DOMAIN_HEIGHT)

# Soft-clamp bounds: 10 K beyond the profile range for h; qt bounded by zero
# and a ceiling above the moistest SAM air (~22 g/kg).
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
    device="cuda",  # keep the finest-grid FFT working set off the 60 GiB host
)
print(f"simulate: {time.perf_counter() - started:.0f} s")

started = time.perf_counter()
compute_diagnostics(OUTPUT, compress=True)
print(f"diagnostics: {time.perf_counter() - started:.0f} s")
