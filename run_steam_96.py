#!/usr/bin/env python3
"""run_steam.py variant: outer scale = longer domain dimension / 4.

Identical frozen config to run_steam.py (H_h = 0.45, C1 = 0.05, linear
100 m -> 1 m spheroscale, same seeds) except outer_scale = 3072 km
(the production value, rerun under current code). For the C17
outer-scale ruling: profile figs comparing the two. Writes
runs/steam_96_<model>_snap<i>.nc and stats/steam_96_<model>_snap<i>.npz.
Skips snapshots whose stats file already exists (restartable).

Usage: python run_steam_96.py [model ...]   (default: all with extracted stats)
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp

import importlib
_steam_simulate = importlib.import_module("steam.simulate")
_steam_simulate.H_h = 0.45
C1_TARGET = 0.05
_steam_simulate.FLUX_SCALE = (C1_TARGET / 1.674) ** (1 / 1.8)

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
CLOUD_KGKG = 0.01e-3
SPHEROSCALE_SURFACE = 100.0
SPHEROSCALE_TOP = 1.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
OUTER_SCALE = 96000.0  # 6144 km (longer domain dimension) / 64 = production


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
    out_stats = STATS / f"steam_96_{model}_snap{i}.npz"
    if out_stats.exists():
        print(f"steam_96 {model} snap{i} already done, skipping")
        return
    src = np.load(STATS / f"{model}_snap{i}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    RUNS.mkdir(exist_ok=True)
    out_nc = RUNS / f"steam_96_{model}_snap{i}.nc"

    z = src["z_profile"]
    spheroscale = SPHEROSCALE_SURFACE + (SPHEROSCALE_TOP - SPHEROSCALE_SURFACE) * z / DOMAIN_HEIGHT
    simulate(
        h_profile, qt_profile,
        nx=2048, ny=128, dx=3000.0, dy=3000.0,
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
    steam_stats(out_nc, out_stats)
    print(f"steam_96 {model} snap{i} done", flush=True)


if __name__ == "__main__":
    models = sys.argv[1:] or sorted({
        p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
        if not p.name.startswith("steam")})
    for model in models:
        for i in range(3):
            run_one(model, i)
