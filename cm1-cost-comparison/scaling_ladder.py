#!/usr/bin/env python3
"""CPU-only STEAM timing ladder on the paper's channel geometry.

Fixed physical domain (6144 x 384 km, the frozen channel), dx halving per
rung: 24, 12, 6 (the paper's matched run), 3 (host-native CM1 grid), 1.5 km.
nz follows dx through the aspect-ratio scaling exactly as in the reference
script (run_steam_simulations.py, hydrodynamic-comparison) -- nothing here
coarsens or pins the vertical. cm1 host profile, c=0.05, same seed as the
paper's cm1_c005. Outputs go to this scratch dir and are deleted after their
size is recorded (only timings are the product).
"""
import sys, time, json, tempfile
from pathlib import Path

import numpy as np
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hydrodynamic-comparison/scripts"))

_steam_simulate = importlib.import_module("steam.simulate")
from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv
import netCDF4

REPO = Path(__file__).resolve().parent.parent
PROFILES = REPO / "runs" / "input_profiles"
SCRATCH = Path(tempfile.mkdtemp(prefix="ladder_", dir=Path(__file__).resolve().parent))

SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
FLUX_C = 0.05

src = np.load(PROFILES / "cm1.npz")
h_profile = src["h_profile"]
qt_profile = src["qt_profile"]
spheroscale = np.full(src["z_profile"].size, SPHEROSCALE_CONSTANT)
surface_pressure = float(src["surface_pressure"])
qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
h_upper = max(cp * 300.0 + Lv * qt_sat_surface, float(h_profile.max()))
h_lower = float(h_profile.min()) - 10.0 * cp

# (nx, ny, dx): fixed 6144 x 384 km footprint, L = 1536 km throughout
LADDER = [
    (256,   16, 24000.0),
    (512,   32, 12000.0),
    (1024,  64,  6000.0),   # the paper's matched channel run
    (2048, 128,  3000.0),   # host-native CM1 grid spacing
    (4096, 256,  1500.0),
]

def one_run(nx, ny, dx, compress):
    outer_scale = nx * dx / 4
    out_nc = SCRATCH / f"bench_{nx}x{ny}_{'z' if compress else 'r'}.nc"
    out_nc.unlink(missing_ok=True)
    _steam_simulate.FLUX_SCALE = FLUX_C
    try:
        t0 = time.perf_counter()
        simulate(
            h_profile, qt_profile,
            nx=nx, ny=ny, dx=dx, dy=dx,
            outer_scale=outer_scale,
            spheroscale=spheroscale,
            anisotropy="piecewise_isotropic_below_spheroscale",
            domain_height=DOMAIN_HEIGHT,
            profile_dz=PROFILE_DZ,
            output_path=str(out_nc),
            surface_pressure=surface_pressure,
            seed=5010,   # matches the paper's cm1_c005 seed formula
            h_min=h_lower, h_max=h_upper,
            qt_min=0.0, qt_max=qt_sat_surface,
            compress=compress,
            device="cpu",
        )
        t_sim = time.perf_counter() - t0
        t_diag = None
        if compress:   # diagnostics only on the paper-config pass
            t0 = time.perf_counter()
            compute_diagnostics(str(out_nc), compress=True)
            t_diag = round(time.perf_counter() - t0, 2)
        with netCDF4.Dataset(out_nc) as ds:
            nz = ds.dimensions["z"].size
        size_gb = out_nc.stat().st_size / 1e9
        return dict(nx=nx, ny=ny, dx=dx, nz=nz, N=nx * ny * nz,
                    compress=compress, t_simulate=round(t_sim, 2),
                    t_diagnostics=t_diag, size_gb=round(size_gb, 3))
    finally:
        out_nc.unlink(missing_ok=True)

results = []
for nx, ny, dx in LADDER:
    print(f"=== {nx}x{ny} dx={dx/1000:g} km ===", flush=True)
    try:
        for compress in ([True, False] if nx <= 2048 else [True]):
            row = one_run(nx, ny, dx, compress)
            print(json.dumps(row), flush=True)
            results.append(row)
    except Exception as e:
        print(f"FAILED at {nx}x{ny}: {type(e).__name__}: {e}", flush=True)
        break

(SCRATCH / "bench_results.json").write_text(json.dumps(results, indent=1))
print("ladder complete", flush=True)
