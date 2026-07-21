#!/usr/bin/env python3
"""Large-domain STEAM ensemble for size distributions (no host comparison).

One 2048 x 1024 run per host profile (snap0), same config as run_steam.py
otherwise: H_h = 0.45, spheroscale 100 m -> 1 m linear, outer scale 96 km,
dx = 3 km (domain 6144 x 3072 km). The channel geometry is too narrow for
size-distribution estimation (DeWitt 2024b), so these wider domains supply
the paper's cloud size distributions and large-domain fractal dimensions.

Writes runs/steam_large_<model>.nc.

Usage: python run_steam_large.py [model ...]   (default: all with extracted stats)
"""

import sys
from pathlib import Path

import numpy as np

# Same H_h override as run_steam.py: steam.simulate reads its module-level
# H_h at call time; steam/__init__ rebinds the submodule name, so fetch the
# real module object.
import importlib
_steam_simulate = importlib.import_module("steam.simulate")
_steam_simulate.H_h = 0.45
# Same C1 override as run_steam.py (calibration C1 = 1.674 c^1.8).
C1_TARGET = 0.05
_steam_simulate.FLUX_SCALE = (C1_TARGET / 1.674) ** (1 / 1.8)

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
SPHEROSCALE_SURFACE = 100.0
SPHEROSCALE_TOP = 1.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0


def run_one(model, seed):
    src = np.load(STATS / f"{model}_snap0.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    z = src["z_profile"]
    spheroscale = SPHEROSCALE_SURFACE + (SPHEROSCALE_TOP - SPHEROSCALE_SURFACE) * z / DOMAIN_HEIGHT
    RUNS.mkdir(exist_ok=True)
    out_nc = RUNS / f"steam_large_{model}.nc"
    simulate(
        h_profile, qt_profile,
        nx=2048, ny=1024, dx=3000.0, dy=3000.0,
        outer_scale=96000.0,
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=float(src["surface_pressure"]),
        seed=seed,
        h_min=h_profile.min() - 10 * cp,
        h_max=h_profile.max() + 10 * cp,
        qt_min=0.0, qt_max=max(0.03, 1.5 * qt_profile.max()),
        compress=True,
        device="cuda",
    )
    compute_diagnostics(str(out_nc), compress=True)
    print(f"steam large {model} done")


if __name__ == "__main__":
    models = sys.argv[1:] or sorted({
        p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
        if not p.name.startswith("steam_")})
    for j, model in enumerate(models):
        run_one(model, seed=2000 + j)
