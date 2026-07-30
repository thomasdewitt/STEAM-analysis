#!/usr/bin/env python3
"""Production square-domain ensembles + strip nests (Thomas, 2026-07-27).

Two cases, 10 members each:
  - icon_lem  (the existing case)
  - ukmo_ra1t (the lower-CF case: smallest STEAM integrated cloud
    fraction across host profiles in the constant-10m comparison set)

Each member: 2048 x 2048 at dx = 3 km (6144 km square), outer scale
L/4 = 1536 km, constant 10 m spheroscale, H_h = 0.45, c for C1 = 0.05
under the 2026-07-27 calibration (C1 = 3.097 c^1.8), anchored bounds
(supp S2 as amended), snap0 profile, seeds 2000 + member. After each
square finishes, its strip nest (spanning x, 48 km wide, dx = 375 m)
runs on the CPU while the next square uses the GPU.

Restartable: members whose .nc exists are skipped; nests whose group
exists are skipped.

Usage: python run_production_squares.py [model ...]
"""

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import importlib
_steam_simulate = importlib.import_module("steam.simulate")
C1_TARGET = 0.05
_steam_simulate.FLUX_SCALE = (C1_TARGET / 1.681) ** (1 / 1.8)  # re-fit 2026-07-28 (flux compensation)

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
MODELS = ("icon_lem", "ukmo_ra1t")
N_MEMBERS = 3   # trimmed from 10 (Thomas, 2026-07-28: runtime)
SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
OUTER_SCALE = 1536000.0
NEW_DX = 375.0          # strip-nest resolution
Y_HALF_CELLS = 8        # strip half-width in parent cells (48 km total)


def run_square(model, member):
    out_nc = RUNS / f"steam_sq10_{model}_m{member:02d}.nc"
    if out_nc.exists():
        print(f"square {model} m{member} exists, skipping", flush=True)
        return out_nc
    src = np.load(STATS / f"{model}_snap0.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    z = src["z_profile"]
    spheroscale = np.full(z.size, SPHEROSCALE_CONSTANT)
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                  float(h_profile.max()) + 1.0)
    h_lower = float(h_profile.min()) - 10.0 * cp
    RUNS.mkdir(exist_ok=True)
    simulate(
        h_profile, qt_profile,
        nx=2048, ny=2048, dx=3000.0, dy=3000.0,
        outer_scale=OUTER_SCALE,
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=surface_pressure,
        seed=2000 + member,
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device="cuda",
        # Store per-class increments: the strips are refined from these
        # parents, and refine() needs them to re-scale inherited content
        # to the nest's interpolation-compensation reference
        # (Thomas's ruling, 2026-07-28). ~6 GB extra per member.
        save_class_increments=True,
    )
    compute_diagnostics(str(out_nc), compress=True)
    print(f"square {model} m{member} done", flush=True)
    return out_nc


def run_nest(path_str):
    """Strip nest on the CPU; runs in a worker process."""
    import netCDF4
    from steam.simulate import refine
    with netCDF4.Dataset(path_str) as ds:
        if "refinements" in ds.groups and "r0" in ds.groups["refinements"].groups:
            print(f"nest exists in {Path(path_str).name}, skipping", flush=True)
            return
    t0 = time.perf_counter()
    # compress=True like the square: with the blosc_zstd c1 filter the nest
    # costs ~6 s of its ~660 s to save ~4.15 GB per member, and the nest is
    # the larger half of the file. It was uncompressed only because refine()
    # falls back to constants.output_compress when compress= is not passed.
    refine(path_str, 0, 2048, 1024 - Y_HALF_CELLS, 1024 + Y_HALF_CELLS,
           NEW_DX, NEW_DX, device="cpu", compress=True)
    print(f"nest done in {Path(path_str).name} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)


def main():
    models = sys.argv[1:] or list(MODELS)
    # spawn, not fork: the parent holds a warm CUDA context and torch
    # thread pool after the first square; a forked worker inherits a
    # locked intra-op pool and deadlocks on its first big CPU op
    # (observed 2026-07-27: futex hang inside F.interpolate).
    nest_pool = ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    pending = []
    for model in models:
        for member in range(N_MEMBERS):
            path = run_square(model, member)
            pending.append(nest_pool.submit(run_nest, str(path)))
    for future in pending:
        future.result()
    nest_pool.shutdown()
    print("all squares and nests complete", flush=True)


if __name__ == "__main__":
    main()
