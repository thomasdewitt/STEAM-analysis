#!/usr/bin/env python3
"""Run the frozen STEAM config against every RCEMIP channel profile.

One config, deliberately untuned: H_h = 0.45, spheroscale decreasing
linearly from 100 m at the surface to 1 m at the domain top, outer scale
96 km, channel strip 2048 x 128 at dx = 3 km (6144 x 384 km ~ the RCEMIP
large-domain geometry; y is a strip axis at 4 outer-scale tiles... x is 64
tiles, y is 4). Domain top 20 km. One run per model snapshot (seed = snapshot
index + 1), initialized from that snapshot's mean h/qt profile extracted by
extract_stats.py. Writes runs/steam_<model>_snap<i>.nc (compressed) and
stats/steam_<model>_snap<i>.npz with the same per-level stats as the host.

Usage: python run_steam.py [model ...]   (default: all with extracted stats)
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

# Override the horizontal Hurst exponent for this production run set.
# steam.simulate reads its module-level H_h at call time; steam/__init__
# rebinds the submodule name, so fetch the real module object.
import importlib
_steam_simulate = importlib.import_module("steam.simulate")
_steam_simulate.H_h = 0.45
# Halve the realized flux C1 to 0.05 for the RCEMIP case (default FLUX_SCALE
# targets C1 = 0.1); calibration C1 = 1.674 c^1.8, so c = (C1/1.674)^(1/1.8).
_steam_simulate.FLUX_SCALE = (0.05 / 1.674) ** (1 / 1.8)

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
CLOUD_KGKG = 0.01e-3
SPHEROSCALE_SURFACE = 100.0
SPHEROSCALE_TOP = 1.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0


def steam_stats(path, out_path):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    h = ds.variables["h"][:]
    qt = ds.variables["qt"][:]
    cond = ds.variables["qc"][:] + ds.variables["qi"][:]
    ds.close()
    nz = z.size
    h_mean = np.empty(nz)
    h_var = np.empty(nz)
    qt_mean = np.empty(nz)
    qt_var = np.empty(nz)
    cloud_fraction = np.empty(nz)
    for k in range(nz):
        hk = h[:, :, k].astype(np.float64)
        qtk = qt[:, :, k].astype(np.float64)
        h_mean[k] = hk.mean()
        h_var[k] = hk.var()
        qt_mean[k] = qtk.mean()
        qt_var[k] = qtk.var()
        cloud_fraction[k] = np.mean(cond[:, :, k] > CLOUD_KGKG)
    np.savez(out_path, z=z, h_mean=h_mean, h_var=h_var,
             qt_mean=qt_mean, qt_var=qt_var, cloud_fraction=cloud_fraction)


def run_one(model, i):
    src = np.load(STATS / f"{model}_snap{i}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    RUNS.mkdir(exist_ok=True)
    out_nc = RUNS / f"steam_{model}_snap{i}.nc"

    z = src["z_profile"]
    spheroscale = SPHEROSCALE_SURFACE + (SPHEROSCALE_TOP - SPHEROSCALE_SURFACE) * z / DOMAIN_HEIGHT
    simulate(
        h_profile, qt_profile,
        nx=2048, ny=128, dx=3000.0, dy=3000.0,
        outer_scale=96000.0,
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
    steam_stats(out_nc, STATS / f"steam_{model}_snap{i}.npz")
    print(f"steam {model} snap{i} done")


if __name__ == "__main__":
    models = sys.argv[1:] or sorted({
        p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
        if not p.name.startswith("steam_")})
    for model in models:
        for i in range(3):
            run_one(model, i)
