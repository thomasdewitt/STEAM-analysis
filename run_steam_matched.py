#!/usr/bin/env python3
"""Matched STEAM runs: one sim per comparison host, on that host's geometry.

Five hosts, one snapshot each: SAM-TWPICE (an observed-case simulation, not
RCE) and the four RCEMIP RCE_small_les300 LES. One STEAM member per host, at
the committed 2026-07-31 config -- C1 = 0.05, constant 10 m spheroscale,
H_h = 0.45 (constants at HEAD), anchored bounds, domain top 20 km -- so the
only thing that varies across the five is the host geometry and profile.

Outer scale follows the committed convention L = (longest horizontal domain
dimension) / 4, the same L_x/4 used by the square campaign and C17 arm 3.
The domains are square, so L = nx*dx/4 for every case: 51.2 km for TWPICE,
24-27 km for the LES.

GEOMETRY CAVEATS (both reported 2026-07-31, neither is a code fallback --
they are what simulate() does with these numbers):

  1. TWPICE is run at 1024^2 x dx = 200 m, NOT the host's 2048^2 x 100 m.
     The full match puts the finest class at 2*dx = 200 m, whose vertical
     counterpart k_z = 53 m gives a finest grid of (2048, 2048, 758) -- an
     11.8 GiB field, so ~89 GiB at the cascade's measured 7.5x peak, against
     56 GiB available on this box. simulate()'s preflight refuses it. Of the
     two ways to shrink it, halving the resolution is the right one: it keeps
     the domain (204.8 km) and therefore the L = L_x/4 outer scale (51.2 km)
     exact, and costs STEAM only the 200-400 m octave of variance. Halving
     the domain instead would have cut L to 25.6 km, and for H_h > 0 the
     largest scales carry most of the variance. It also puts TWPICE on the
     same 200 m footing as the four LES.

  2. The LES output grids are 512^2, not the hosts' 540/480/504/500^2.
     simulate() takes nx*dx as the domain and lays its own dyadic class
     ladder inside it, so the horizontal EXTENT is matched exactly while the
     grid is the cascade's natural power of two (dx_out = 188-211 m against
     the hosts' 200 m). Nothing is lost: the comparison statistics are
     per-level moments and PDFs, not cell-by-cell.

Writes runs/steam_matched_<model>.nc, stats/steam_matched_<model>.npz,
stats/diag_steam_matched_<model>.npz and stats/cover_matched_<model>.npz
(tau > 1 cloud cover for STEAM and its host). Restartable: a model whose
stats .npz exists is skipped. The .nc are deleted once the figures are made.

Usage: python run_steam_matched.py [model ...]   (default: all five)
"""

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4

import importlib
_steam_simulate = importlib.import_module("steam.simulate")
C1_TARGET = 0.05
_steam_simulate.FLUX_SCALE = (C1_TARGET / 1.681) ** (1 / 1.8)  # re-fit 2026-07-28

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv

from run_steam import steam_stats
from make_diag_stats import do_steam
from make_fractal import cv, TAU_THRESHOLD

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
# model -> (nx = ny, dx = dy [m]). See the TWPICE caveat above for why it is
# 1024 x 200 m rather than the host's 2048 x 100 m.
MATCHED = {
    "twpice": (1024, 200.0),
    "les_cm1": (540, 200.0),
    "les_sam": (480, 200.0),
    "les_dales": (504, 200.0),
    "les_icon_lem": (500, 200.0),
}
SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0


def covers(model, out_nc):
    """tau > 1 cloud cover, STEAM against its host."""
    ds = netCDF4.Dataset(out_nc)
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    lwc = ds.variables["qc"][:] * 1000.0      # (x, y, z) already, -> g/kg
    iwc = ds.variables["qi"][:] * 1000.0
    ds.close()
    steam_cover = float(
        (cv.vertically_integrated_optical_depth(lwc, z, iwc=iwc)
         > TAU_THRESHOLD).mean())
    del lwc, iwc

    from extract_stats import ADAPTERS
    z, T, qv, qc, qi, _ = ADAPTERS[model](0)
    del T, qv                                  # TWPICE: 4.3 GB each
    # adapter fields are (nz, ny, nx) kg/kg; cloudyview wants (nx, ny, nz) g/kg
    lwc = np.moveaxis(qc, 0, -1) * 1000.0
    iwc = np.moveaxis(qi, 0, -1) * 1000.0
    host_cover = float(
        (cv.vertically_integrated_optical_depth(
            lwc, np.asarray(z, dtype=np.float64), iwc=iwc)
         > TAU_THRESHOLD).mean())
    np.savez(STATS / f"cover_matched_{model}.npz",
             steam_cover=steam_cover, host_cover=host_cover)
    print(f"{model}: tau>1 cover STEAM {steam_cover:.3f} host {host_cover:.3f}",
          flush=True)


def run_one(model):
    name = f"steam_matched_{model}"
    if (STATS / f"{name}.npz").exists():
        print(f"{name} exists, skipping", flush=True)
        return
    nx, dx = MATCHED[model]
    src = np.load(STATS / f"{model}_snap0.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    z = src["z_profile"]
    spheroscale = np.full(z.size, SPHEROSCALE_CONSTANT)
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    # Anchored bounds (supp S2 as amended 2026-07-27): the gz term puts
    # stratospheric <h> above any surface value, so the upper bound is the
    # LARGER of surface-saturation MSE and the profile max.
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                  float(h_profile.max()) + 1.0)
    h_lower = float(h_profile.min()) - 10.0 * cp
    out_nc = RUNS / f"{name}.nc"
    RUNS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    simulate(
        h_profile, qt_profile,
        nx=nx, ny=nx, dx=dx, dy=dx,
        outer_scale=nx * dx / 4,   # committed L = L_x/4 convention
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=surface_pressure,
        seed=3000 + list(MATCHED).index(model),
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device="cuda",
    )
    compute_diagnostics(str(out_nc), compress=True)
    steam_stats(out_nc, STATS / f"{name}.npz")
    do_steam(out_nc)
    covers(model, out_nc)
    print(f"{name} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    for model in sys.argv[1:] or list(MATCHED):
        run_one(model)
