#!/usr/bin/env python3
"""git-bisect driver for the flat-xi(1) bug (2026-07-27).

Runs a small fixed-seed sim against whatever steam code is checked out
in ~/code-and-data/turbulon-egu (the bisect worktree), computes the
single-level horizontal Haar slope of h at ~7 km, and exits:

    0   good  (slope >= 0.30 -- EGU-era behavior, ~0.5)
    1   bad   (slope <  0.30 -- flat-xi(1) behavior, ~0.09)
    125 skip  (commit does not run: import/signature/runtime error)

Config: 512^2 @ 1000 km domain, outer = 500 km, constant 30 m
spheroscale, canonical anisotropy, s = 1, icon_lem snap0 profile,
seed 7. Kwargs are filtered against the era's simulate() signature so
the same driver spans the API drift (flux cascade, device kwarg, ...).

Usage:  ~/code-and-data/turbulon-egu/.venv/bin/python archaeology/bisect_haar.py
        (or as `git bisect run` inside the turbulon-egu worktree)
"""

import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
EGU_REPO = Path.home() / "code-and-data" / "turbulon-egu"
CP = 1004.0
SLOPE_THRESHOLD = 0.30


def run():
    import steam
    src = Path(steam.__file__).resolve()
    repo = next(p for p in src.parents if (p / ".git").exists())
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet"]).returncode:
        head += "+dirty"
    print(f"steam from {repo} @ {head}")
    from steam.simulate import simulate

    src_prof = np.load(HERE / "stats" / "icon_lem_snap0.npz")
    z = src_prof["z_profile"]

    out = Path(tempfile.gettempdir()) / "bisect_haar.nc"
    if out.exists():
        out.unlink()

    kwargs = dict(
        h_profile=src_prof["h_profile"], qt_profile=src_prof["qt_profile"],
        nx=512, ny=512,
        dx=1_000_000.0 / 512, dy=1_000_000.0 / 512,
        outer_scale=500_000.0,
        spheroscale=np.full(z.size, 30.0),
        domain_height=20_000.0,
        profile_dz=50.0,
        output_path=str(out),
        sparsity_factors=(1, 1, 1),
        surface_pressure=float(src_prof["surface_pressure"]),
        seed=7,
        h_min=250.0 * CP, h_max=420.0 * CP,
        qt_min=0.0, qt_max=0.03,
        anisotropy="canonical",
        compress=False,
        device="cpu",
        n_scale_classes_per_dyad=1,
        turbulon_shape="mexican_hat",
    )
    accepted = set(inspect.signature(simulate).parameters)
    dropped = sorted(set(kwargs) - accepted)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    if dropped:
        print(f"[{head}] dropped kwargs not in this era: {dropped}")

    simulate(**kwargs)

    import netCDF4
    import scaleinvariance as si
    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        zf = ds.variables["z"][:]
        iz = int(np.argmin(np.abs(zf - 7000.0)))
        h_level = np.asarray(ds.variables["h"][:, :, iz], dtype=np.float32)
        dx_out = float(ds.dx)

    # Fit over 4..64 cells (~8..125 km): inside the scaling range, well
    # below the 500 km outer-scale rollover.
    lags = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64])
    lag_arr, F = si.haar_fluctuation(h_level, order=1.0, axis=0,
                                     lags=lags, periodic=True)
    slope = float(np.polyfit(np.log(lag_arr), np.log(F), 1)[0])
    print(f"[{head}] h Haar slope @ z={zf[iz]:.0f} m: {slope:+.3f} "
          f"({'GOOD' if slope >= SLOPE_THRESHOLD else 'BAD'}, "
          f"threshold {SLOPE_THRESHOLD})")
    return 0 if slope >= SLOPE_THRESHOLD else 1


if __name__ == "__main__":
    try:
        sys.exit(run())
    except SystemExit:
        raise
    except Exception as e:
        print(f"SKIP: {type(e).__name__}: {e}")
        sys.exit(125)
