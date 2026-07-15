#!/usr/bin/env python3
"""Horizontal-mean h and qt profiles from SAM LES, for STEAM initialization.

Writes twpice_mean_profiles.nc and gate_mean_profiles.nc: the ensemble-mean
moist static energy h [J/kg] and total water qt [kg/kg] profiles that STEAM
takes as its <Phi>_t(z) inputs, plus the reference pressure/temperature.

SAM 3D netCDFs are float32; every horizontal reduction of a K-scale field (MSE,
TABS) uses a float64 accumulator, else the partial sums lose O(10) ULP and the
profile comes out jagged.

Sources are read-only. h = cp*T + g*z + Lv*qv; qt = vapor + non-precipitating
condensate. TWPICE stores MSE directly (K); GATE stores TABS, so h is built.
"""

import glob
import os

import netCDF4
import numpy as np

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

EXPANSION = "/run/media/thomas/Expansion/hydrodynamic-model-output"


def twpice_profiles(paths):
    """Mean (h [J/kg], qt [kg/kg]) over the given TWPICE snapshots.

    Fields are (x, y, z); qt = QV + QC + QI (vapor + cloud water + cloud ice).
    MSE is already h/cp in K. Reduces one field at a time to bound memory.
    """
    base = f"{EXPANSION}/SAM-TWPICE"
    h = qt = None
    for stamp in paths:
        mse = netCDF4.Dataset(f"{base}/OUT_3D.MSE/TWPICE_LPT_3D_MSE_{stamp}.nc")
        z = mse.variables["z"][:].astype(np.float64)
        h_snap = cp * mse.variables["MSE"][0].mean(axis=(0, 1), dtype=np.float64)
        mse.close()
        qt_snap = np.zeros_like(z)
        for var in ("QV", "QC", "QI"):
            ds = netCDF4.Dataset(f"{base}/OUT_3D.{var}/TWPICE_LPT_3D_{var}_{stamp}.nc")
            qt_snap += ds.variables[var][0].mean(axis=(0, 1), dtype=np.float64) / 1000.0
            ds.close()
        h = h_snap if h is None else h + h_snap
        qt = qt_snap if qt is None else qt + qt_snap
    pp = netCDF4.Dataset(f"{base}/OUT_3D.PP/TWPICE_LPT_3D_PP_{paths[0]}.nc")
    pres = pp.variables["pres"][:].astype(np.float64) * 100.0  # mb -> Pa
    pp.close()
    n = len(paths)
    return z, h / n, qt / n, pres


def gate_profiles(paths):
    """Mean (h [J/kg], qt [kg/kg]) over the given GATE snapshots.

    Fields are (z, y, x); h is built from TABS, QV and geopotential;
    qt = QV + QN (vapor + non-precipitating condensate).
    """
    z = h = qt = pres = None
    for path in paths:
        ds = netCDF4.Dataset(path)
        z = ds.variables["z"][:].astype(np.float64)
        pres = ds.variables["p"][:].astype(np.float64) * 100.0  # mb -> Pa
        tabs = ds.variables["TABS"][0].mean(axis=(1, 2), dtype=np.float64)
        qv = ds.variables["QV"][0].mean(axis=(1, 2), dtype=np.float64) / 1000.0
        qn = ds.variables["QN"][0].mean(axis=(1, 2), dtype=np.float64) / 1000.0
        ds.close()
        h_snap = cp * tabs + g * z + Lv * qv
        qt_snap = qv + qn
        h = h_snap if h is None else h + h_snap
        qt = qt_snap if qt is None else qt + qt_snap
    n = len(paths)
    return z, h / n, qt / n, pres


def write_profiles(path, z, h, qt, pres, source):
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("z", z.size)
    for name, data, units in [
        ("z", z, "m"), ("h", h, "J/kg"), ("qt", qt, "kg/kg"), ("pres", pres, "Pa"),
    ]:
        var = ds.createVariable(name, "f8", ("z",))
        var[:] = data
        var.units = units
    ds.surface_pressure = float(pres[0])
    ds.source = source
    ds.close()
    print(f"wrote {path}: nz={z.size}, h {h[0]:.0f}->{h[-1]:.0f} J/kg, "
          f"qt {qt[0]*1e3:.1f}->{qt[-1]*1e3:.2f} g/kg, p_sfc {pres[0]:.0f} Pa")


def main():
    # File stamps are timestep numbers at dt = 2 s.
    # TWPICE: 5-min cadence over a ~1.9 h window at day 20; snapshots are
    # highly correlated, so three spread across the window suffice.
    twpice_stamps = ["0000000150", "0000001800", "0000003450"]
    if not os.path.exists("twpice_mean_profiles.nc"):
        write_profiles(
            "twpice_mean_profiles.nc", *twpice_profiles(twpice_stamps),
            source=f"SAM-TWPICE OUT_3D, snapshots {twpice_stamps}; horizontal mean, float64",
        )

    # GATE: hourly snapshots 1-23 h; use the final ~12 h (developed convection).
    gate_all = sorted(glob.glob(f"{EXPANSION}/SAM-GATE/com3D/GATE_IDEAL_S_*.nc"))
    gate_paths = [p for p in gate_all if int(p.split("_")[-1][:-3]) * 2 >= 11 * 3600]
    if not os.path.exists("gate_mean_profiles.nc"):
        write_profiles(
            "gate_mean_profiles.nc", *gate_profiles(gate_paths),
            source=("SAM-GATE com3D, snapshots "
                    f"{[p.split('_')[-1][:-3] for p in gate_paths]} (11-23 h); "
                    "horizontal mean, float64"),
        )


if __name__ == "__main__":
    main()
