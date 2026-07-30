#!/usr/bin/env python3
"""Run the STEAM RCEMIP-channel ensemble against every host profile.

2x2 config grid (C1 in {0.03, 0.1} x constant spheroscale in {3, 10} m),
3 snapshots each -> 12 STEAM members per host model. Files carry a
_C1xxx_lsxx tag. Otherwise the frozen setup, deliberately untuned: H_h = 0.45, spheroscale decreasing
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

# steam/__init__ rebinds the submodule name, so fetch the real module
# object to override FLUX_SCALE below.
import importlib
_steam_simulate = importlib.import_module("steam.simulate")
# RCEMIP ensemble (Thomas, 2026-07-28 evening): 2x2 config grid over
# intermittency C1 in {0.03, 0.1} and constant spheroscale in {3 m, 10 m}.
# c = (C1/1.681)^(1/1.8) per the 2026-07-28 re-fit (flux compensation).
CONFIGS = [
    # Effectively constant flux (c ~ 0.016), Thomas 2026-07-29: replaces
    # the short-lived C1=0.01/ls=1 config. Tag reads "C1 = 0.001" ("p"
    # for the decimal point, breaking the 100x-C1 pattern of the others).
    {"C1": 0.001, "ls": 10.0, "tag": "C1p001_ls10"},
    {"C1": 0.03, "ls": 3.0, "tag": "C1003_ls03"},
    {"C1": 0.03, "ls": 10.0, "tag": "C1003_ls10"},
    {"C1": 0.10, "ls": 3.0, "tag": "C1010_ls03"},
    {"C1": 0.10, "ls": 10.0, "tag": "C1010_ls10"},
]

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


def run_one(model, i, cfg):
    tag = cfg["tag"]
    name = f"steam_{model}_snap{i}_{tag}"
    if (STATS / f"{name}.npz").exists():
        print(f"{name} exists, skipping", flush=True)
        return
    _steam_simulate.FLUX_SCALE = (cfg["C1"] / 1.681) ** (1 / 1.8)
    src = np.load(STATS / f"{model}_snap{i}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    RUNS.mkdir(exist_ok=True)
    from steam.thermodynamics import _saturation_mixing_ratio
    from steam.constants import latent_heat_vaporization as Lv
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    # The gz term puts stratospheric <h> above any surface value, so the
    # upper bound is the LARGER of surface-saturation MSE and the profile
    # max (S2 prescription as amended 2026-07-27; the 07-22 wording used
    # surface-saturation MSE alone and fails validation).
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface, float(h_profile.max()) + 1.0)
    h_lower = float(h_profile.min()) - 10.0 * cp
    out_nc = RUNS / f"{name}.nc"

    z = src["z_profile"]
    # Constant spheroscale per config (2026-07-27 lean; ensemble values 3/10 m).
    spheroscale = np.full(z.size, cfg["ls"])
    simulate(
        h_profile, qt_profile,
        nx=2048, ny=128, dx=3000.0, dy=3000.0,
        outer_scale=96000.0,
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=surface_pressure,
        seed=1000 + i + 100 * CONFIGS.index(cfg),
        # Anchored bounds (supp S2, agreed 2026-07-22): every realization
        # sees the prescribed 300 K SST, so qt is capped at surface
        # saturation and h at surface saturation MSE; the lower h bound
        # allows a 10 K deficit below the coldest point of the profile.
        h_min=h_lower,
        h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device="cuda",
    )
    compute_diagnostics(str(out_nc), compress=True)
    steam_stats(out_nc, STATS / f"{name}.npz")
    print(f"{name} done", flush=True)


if __name__ == "__main__":
    models = sys.argv[1:] or sorted({
        p.name.split("_snap")[0] for p in STATS.glob("*_snap*.npz")
        if not p.name.startswith(("steam_", "diag_"))})
    for model in models:
        for cfg in CONFIGS:
            for i in range(3):
                run_one(model, i, cfg)
