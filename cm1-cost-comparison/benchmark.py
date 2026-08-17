#!/usr/bin/env python3
"""Wall-clock cost of RCEMIP-configured CM1 versus STEAM, on one machine.

CM1 leg: one model hour of the RCE_large300 channel (2016 x 134 x 74, 3 km),
fixed dt = 18 s (the production-average step reported in the CM1 model
documentation form), 200 steps, cold start from the RCEMIP analytic sounding.
The run is quiescent, which flatters CM1 slightly: convecting-state
microphysics would add a few percent. CM1's own "Total time" statistic is
used (solver wall time, excluding initialization).

STEAM leg: two single realizations on the channel footprint, CPU only.
  matched      1024 x  64 x  78 at 6 km  (the configuration the paper uses)
  host-native  2048 x 128 x 115 at 3 km  (CM1's own grid spacing; more
                                          points than CM1's grid)

Run from the turbulon-analysis venv:
    .venv/bin/python cm1-cost-comparison/benchmark.py [--ranks 16]
Run get_cm1.sh first. Results land in results.json and stdout.
"""
import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Independent-sample accounting for the derived numbers (see README):
# the paper's analysis uses three independent snapshots per 100-day
# channel run (supplement, comparison-data subsection).
PRODUCTION_DAYS = 100.0
INDEPENDENT_FIELDS_PER_RUN = 3.0


def run_cm1(ranks):
    exe = HERE / "cm1r19/run/cm1.exe"
    if not exe.exists():
        sys.exit("cm1r19/run/cm1.exe not found; run get_cm1.sh first")
    with tempfile.TemporaryDirectory(dir=HERE) as td:
        td = Path(td)
        for f in ["namelist.input", "input_grid_z"]:
            shutil.copy(HERE / "run" / f, td)
        for f in ["RRTMG_LW_DATA", "RRTMG_SW_DATA", "LANDUSE.TBL"]:
            shutil.copy(HERE / "cm1r19/run" / f, td)
        shutil.copy(exe, td)
        print(f"CM1: 1 model hour, dt=18 s, {ranks} MPI ranks ...", flush=True)
        log = subprocess.run(
            ["mpirun", "-np", str(ranks), "./cm1.exe"],
            cwd=td, capture_output=True, text=True, check=True,
        ).stdout
        (HERE / "cm1_benchmark.log").write_text(log)

    total = float(re.search(r"Total time:\s+([\d.]+)", log).group(1))
    comp = dict(re.findall(r"^\s+(\w+)\s+:\s+([\d.]+)\s+[\d.]+%", log, re.M))
    return dict(
        wall_s_per_model_hour=round(total, 1),
        wall_h_per_model_day=round(total * 24 / 3600, 2),
        wall_s_per_100_day_run=round(total * 24 * PRODUCTION_DAYS),
        radiation_s=float(comp.get("radiatio", "nan")),
        microphysics_s=float(comp.get("microphy", "nan")),
        ranks=ranks,
    )


def run_steam(nx, ny, dx):
    sys.path.insert(0, str(REPO / "hydrodynamic-comparison/scripts"))
    _steam_simulate = importlib.import_module("steam.simulate")
    from steam.simulate import simulate
    from steam.thermodynamics import _saturation_mixing_ratio
    from steam.constants import specific_heat_dry_air as cp
    from steam.constants import latent_heat_vaporization as Lv

    src = np.load(REPO / "runs/input_profiles/cm1.npz")
    qt_sat = float(_saturation_mixing_ratio(300.0, float(src["surface_pressure"])))
    _steam_simulate.FLUX_SCALE = 0.05
    with tempfile.TemporaryDirectory(dir=HERE) as td:
        out = Path(td) / "field.nc"
        print(f"STEAM: {nx} x {ny}, dx={dx/1000:g} km, cpu ...", flush=True)
        t0 = time.perf_counter()
        simulate(
            src["h_profile"], src["qt_profile"],
            nx=nx, ny=ny, dx=dx, dy=dx,
            outer_scale=nx * dx / 4,
            spheroscale=np.full(src["z_profile"].size, 10.0),
            anisotropy="piecewise_isotropic_below_spheroscale",
            domain_height=20000.0, profile_dz=50.0,
            output_path=str(out),
            surface_pressure=float(src["surface_pressure"]),
            seed=5010,
            h_min=float(src["h_profile"].min()) - 10.0 * cp,
            h_max=max(cp * 300.0 + Lv * qt_sat, float(src["h_profile"].max())),
            qt_min=0.0, qt_max=qt_sat,
            compress=True, device="cpu",
        )
        wall = time.perf_counter() - t0
        import netCDF4
        with netCDF4.Dataset(out) as ds:
            nz = ds.dimensions["z"].size
    return dict(nx=nx, ny=ny, dx=dx, nz=nz, n_points=nx * ny * nz,
                wall_s=round(wall, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=16)
    args = ap.parse_args()

    cm1 = run_cm1(args.ranks)
    steam_matched = run_steam(1024, 64, 6000.0)
    steam_native = run_steam(2048, 128, 3000.0)

    cm1_per_sample = cm1["wall_s_per_100_day_run"] / INDEPENDENT_FIELDS_PER_RUN
    results = dict(
        cm1=cm1, steam_matched=steam_matched, steam_native=steam_native,
        cm1_wall_s_per_independent_field=round(cm1_per_sample),
        ratio_per_independent_field_matched=round(
            cm1_per_sample / steam_matched["wall_s"]),
        ratio_per_independent_field_native=round(
            cm1_per_sample / steam_native["wall_s"]),
    )
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
