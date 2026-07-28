#!/usr/bin/env python3
"""May checkpoint (ac16b35) at the EXACT production square config.

Thomas's overnight benchmark (2026-07-27): if the May code is straight
at the production config, the residual curvature in tonight's squares
is July machinery; if May bends too, part of it is the config itself
(outer-scale rollover + deep vertical grids).

Runs 3 members (seeds 2000-2002, matching production m00-m02):
2048^2 at dx = 3 km (6144 km domain), outer 1536 km, constant 10 m
spheroscale, piecewise anisotropy, icon_lem snap0, old-code soft-clamp
bounds set to the production anchored values. Then an old-refine()
strip nest on member 0 (y cells 1016:1032 = 48 km, full x span,
dx = 375 m -> group refinements/r0), mirroring the production strips.

Restartable: existing members/nest are skipped.

MUST run with the EGU worktree venv:
    ~/code-and-data/turbulon-egu/.venv/bin/python archaeology/may_production_benchmark.py
"""

import time
from pathlib import Path

import numpy as np

import steam
from steam.simulate import simulate, refine

HERE = Path(__file__).resolve().parent.parent
EGU_REPO = Path.home() / "code-and-data" / "turbulon-egu"
RUNS = HERE / "runs" / "archaeology"
CP = 1004.0
LV = 2.5e6
N_MEMBERS = 3


def qsat_300(surface_pressure):
    """Saturation mixing ratio at 300 K — old thermodynamics module."""
    from steam.thermodynamics import _saturation_mixing_ratio
    return float(_saturation_mixing_ratio(300.0, surface_pressure))


def main():
    src_file = Path(steam.__file__).resolve()
    assert EGU_REPO in src_file.parents, f"steam from {src_file}"
    import subprocess
    head = subprocess.run(["git", "-C", str(EGU_REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert head == "ac16b35", f"EGU worktree at {head}, expected ac16b35"

    src = np.load(HERE / "stats" / "icon_lem_snap0.npz")
    z = src["z_profile"]
    sp = float(src["surface_pressure"])
    qts = qsat_300(sp)
    h_upper = max(CP * 300.0 + LV * qts, float(src["h_profile"].max()) + 1.0)
    h_lower = float(src["h_profile"].min()) - 10.0 * CP
    RUNS.mkdir(parents=True, exist_ok=True)

    paths = []
    for m in range(N_MEMBERS):
        out = RUNS / f"may_sq_m{m:02d}.nc"
        paths.append(out)
        if out.exists():
            print(f"member {m} exists, skipping", flush=True)
            continue
        t0 = time.monotonic()
        simulate(
            h_profile=src["h_profile"], qt_profile=src["qt_profile"],
            nx=2048, ny=2048, dx=3000.0, dy=3000.0,
            outer_scale=1_536_000.0,
            spheroscale=np.full(z.size, 10.0),
            domain_height=20_000.0,
            profile_dz=50.0,
            output_path=out,
            sparsity_factors=(1, 1, 1),
            n_scale_classes_per_dyad=1,
            surface_pressure=sp,
            seed=2000 + m,
            h_min=h_lower, h_max=h_upper,
            qt_min=0.0, qt_max=qts,
            anisotropy="piecewise_isotropic_below_spheroscale",
            compress=False,
        )
        print(f"member {m}: {time.monotonic() - t0:.0f} s", flush=True)

    # Strip nest on member 0, mirroring the production strips.
    import netCDF4
    with netCDF4.Dataset(paths[0]) as ds:
        has_nest = ("refinements" in ds.groups
                    and "r0" in ds.groups["refinements"].groups)
    if has_nest:
        print("nest exists, skipping", flush=True)
    else:
        t0 = time.monotonic()
        refine(
            paths[0], 0, 2048, 1016, 1032, 375.0, 375.0,
            parent_group="/", output_group="refinements/r0",
            seed=3000,
            anisotropy="piecewise_isotropic_below_spheroscale",
            compress=False,
        )
        print(f"nest: {time.monotonic() - t0:.0f} s", flush=True)
    print("may benchmark sims complete", flush=True)


if __name__ == "__main__":
    main()
