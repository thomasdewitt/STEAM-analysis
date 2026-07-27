#!/usr/bin/env python3
"""Dissect 544fbb0: which of its three mechanisms kills the Haar slope?

Monkeypatches steam.simulate before running the bisect-driver config
(512^2 @ 1000 km, outer 500 km, seed 7), one toggle set per invocation:

    python toggle_experiment.py [no_taper] [no_projection] [no_realized_norm]

no_taper          -> _bound_taper returns 1 (taper disabled)
no_projection     -> _project_onto_bounds is a no-op
no_realized_norm  -> W normalization uses the DOMAIN-margin... (not
                     implemented; norm is inline -- by elimination)

Run with the turbulon-egu venv against whatever is checked out there.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
EGU_REPO = Path.home() / "code-and-data" / "turbulon-egu"
CP = 1004.0


def main():
    toggles = set(sys.argv[1:])
    import importlib
    sm = importlib.import_module("steam.simulate")
    src = Path(sm.__file__).resolve()
    assert EGU_REPO in src.parents, f"steam imported from {src}"
    head = subprocess.run(["git", "-C", str(EGU_REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    if "no_taper" in toggles:
        sm._bound_taper = lambda running_sum, b, mn, mx: np.float32(1.0)
    if "no_projection" in toggles:
        sm._project_onto_bounds = lambda *a, **k: None

    prof = np.load(HERE / "stats" / "icon_lem_snap0.npz")
    z = prof["z_profile"]
    out = Path(tempfile.gettempdir()) / f"toggle_{'_'.join(sorted(toggles)) or 'none'}.nc"
    if out.exists():
        out.unlink()

    import inspect
    kwargs = dict(
        h_profile=prof["h_profile"], qt_profile=prof["qt_profile"],
        nx=512, ny=512, dx=1_000_000.0 / 512, dy=1_000_000.0 / 512,
        outer_scale=500_000.0,
        spheroscale=np.full(z.size, 30.0),
        domain_height=20_000.0, profile_dz=50.0,
        output_path=str(out),
        sparsity_factors=(1, 1, 1),
        surface_pressure=float(prof["surface_pressure"]),
        seed=7,
        h_min=250.0 * CP, h_max=420.0 * CP, qt_min=0.0, qt_max=0.03,
        anisotropy="canonical", compress=False, device="cpu",
    )
    accepted = set(inspect.signature(sm.simulate).parameters)
    sm.simulate(**{k: v for k, v in kwargs.items() if k in accepted})

    import netCDF4
    import scaleinvariance as si
    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        zf = ds.variables["z"][:]
        iz = int(np.argmin(np.abs(zf - 7000.0)))
        h_level = np.asarray(ds.variables["h"][:, :, iz], dtype=np.float32)
    lags = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64])
    lag_arr, F = si.haar_fluctuation(h_level, order=1.0, axis=0,
                                     lags=lags, periodic=True)
    slope = float(np.polyfit(np.log(lag_arr), np.log(F), 1)[0])
    print(f"RESULT [{head}] toggles={sorted(toggles) or ['none']}: "
          f"h Haar slope = {slope:+.3f}")
    out.unlink()


if __name__ == "__main__":
    main()
