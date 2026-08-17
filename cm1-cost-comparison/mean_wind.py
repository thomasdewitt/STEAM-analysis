#!/usr/bin/env python3
"""Mean wind at 10 km in the analyzed CM1 channel snapshots.

Supports the decorrelation-timescale estimate in the supplement's
computational cost section: the advective flushing time is the channel
length divided by this wind speed. Uses the same three snapshots the
paper analyzes (days 80, 90, 100) and averages u and v over the full
domain at the model level nearest 10 km.
"""
from pathlib import Path

import netCDF4
import numpy as np

FILES = sorted((Path(__file__).resolve().parent.parent / "data/cm1").glob(
    "CM1_RCE_large300_3D_allvars_hour*.nc"))
TARGET_Z = 10000.0

u_means, v_means = [], []
for f in FILES:
    with netCDF4.Dataset(f) as ds:
        z = ds["z"][:]
        k = int(np.argmin(np.abs(z - TARGET_Z)))
        u = float(np.mean(np.asarray(ds["ua"][0, k], dtype=np.float64)))
        v = float(np.mean(np.asarray(ds["va"][0, k], dtype=np.float64)))
    u_means.append(u)
    v_means.append(v)
    print(f"{f.name}: z={z[k]:.0f} m  <u>={u:+.2f}  <v>={v:+.2f} m/s")

u, v = np.mean(u_means), np.mean(v_means)
speed = np.hypot(u, v)
print(f"\nacross the three snapshots: <u>={u:+.2f} m/s, <v>={v:+.2f} m/s, "
      f"|(<u>,<v>)|={speed:.2f} m/s")
print(f"channel flushing time 6048 km / {speed:.2f} m/s = "
      f"{6048e3 / speed / 86400:.1f} days")
