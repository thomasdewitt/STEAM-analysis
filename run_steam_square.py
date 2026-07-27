#!/usr/bin/env python3
"""Large SQUARE domain STEAM runs: 6144 x 6144 km, outer scale L/4 = 1536 km.

Campaign-2 opener (2026-07-27): 2-member ensemble on ICON-LEM profiles
(snap0, snap1), constant 10 m spheroscale (Thomas's lean from today's
comparison), dx = 3 km as in the RCEMIP channel runs, frozen config
otherwise (H_h = 0.45, C1 = 0.05). For cloudyview glimpses and cloud
fractal metrics; nesting experiments hang off these parents later.

Writes runs/steam_square_icon_lem_snap<i>.nc. Skips existing outputs.

Usage: python run_steam_square.py [snap_index ...]   (default: 0 1)
"""

import sys
from pathlib import Path

import numpy as np

import importlib
_steam_simulate = importlib.import_module("steam.simulate")
_steam_simulate.H_h = 0.45
C1_TARGET = 0.05
_steam_simulate.FLUX_SCALE = (C1_TARGET / 3.097) ** (1 / 1.8)

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
MODEL = "icon_lem"
SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
OUTER_SCALE = 1536000.0  # 6144 km / 4


def run_one(i):
    out_nc = RUNS / f"steam_square_{MODEL}_snap{i}.nc"
    if out_nc.exists():
        print(f"steam_square {MODEL} snap{i} already exists, skipping")
        return
    src = np.load(STATS / f"{MODEL}_snap{i}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    z = src["z_profile"]
    spheroscale = np.full(z.size, SPHEROSCALE_CONSTANT)
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
        surface_pressure=float(src["surface_pressure"]),
        seed=1000 + i,
        h_min=h_profile.min() - 10 * cp,
        h_max=h_profile.max() + 10 * cp,
        qt_min=0.0, qt_max=max(0.03, 1.5 * qt_profile.max()),
        compress=True,
        device="cuda",
    )
    compute_diagnostics(str(out_nc), compress=True)
    print(f"steam_square {MODEL} snap{i} done", flush=True)


if __name__ == "__main__":
    snaps = [int(a) for a in sys.argv[1:]] or [0, 1]
    for i in snaps:
        run_one(i)
