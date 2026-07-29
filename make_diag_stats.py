#!/usr/bin/env python3
"""Cache per-case diagnostic profiles: mean/std of qc, qi, T per level.

Hosts: via extract_stats.ADAPTERS (3 snapshots each). STEAM: every
runs/steam_{model}_snap{i}_{tag}.nc from the 2026-07-29 ensemble.
Writes stats/diag_{case}.npz with z, {qc,qi,T}_{mean,std}. Pressure is
omitted: the host adapters carry no 3D pressure field to compare against.
Restartable (existing outputs skipped).
"""

from pathlib import Path

import numpy as np
import netCDF4

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"


def profile_stats(fields, z):
    out = {"z": z}
    for name, (f, level_axis) in fields.items():
        f = np.moveaxis(f, level_axis, 0)
        nz = f.shape[0]
        mean = np.empty(nz)
        std = np.empty(nz)
        for k in range(nz):
            fk = f[k].astype(np.float64)
            mean[k] = fk.mean()
            std[k] = fk.std()
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = std
    return out


def do_host(model, i):
    out_path = STATS / f"diag_{model}_snap{i}.npz"
    if out_path.exists():
        return
    from extract_stats import ADAPTERS
    z, T, qv, qc, qi, _ = ADAPTERS[model](i)   # fields are (nz, ny, nx)
    out = profile_stats(
        {"qc": (qc, 0), "qi": (qi, 0), "T": (T, 0)}, np.asarray(z, float))
    np.savez(out_path, **out)
    print(f"diag {model} snap{i}", flush=True)


def do_steam(nc_path):
    out_path = STATS / f"diag_{nc_path.stem}.npz"
    if out_path.exists():
        return
    ds = netCDF4.Dataset(nc_path)
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    fields = {name: (ds.variables[name][:], 2) for name in ("qc", "qi", "T")}
    out = profile_stats(fields, z)
    ds.close()
    np.savez(out_path, **out)
    print(f"diag {nc_path.stem}", flush=True)


if __name__ == "__main__":
    from extract_stats import ADAPTERS
    for model in ADAPTERS:
        for i in range(3):
            if (STATS / f"{model}_snap{i}.npz").exists():
                do_host(model, i)
    for p in sorted(RUNS.glob("steam_*_snap*_C1*.nc")):
        do_steam(p)
