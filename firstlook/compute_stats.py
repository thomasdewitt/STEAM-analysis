#!/usr/bin/env python3
"""Reduce SAM-TWPICE (final snapshot) and the STEAM run to small stat caches.

Writes sam_stats.npz and steam_stats.npz: per-level mean/variance/skewness of
h and qt, cloud fraction (condensate > 0.01 g/kg), Haar fluctuation functions
(orders 1 and 2) at three representative levels, one mid-troposphere qt' slice,
and the column-integrated condensate.

SAM 3D fields are float32 (MSE float64 on disk); every K-scale reduction uses
a float64 accumulator (see CLAUDE.md gotcha).
"""

import netCDF4
import numpy as np
import scaleinvariance

from steam.constants import specific_heat_dry_air as cp

SAM = "/run/media/thomas/Expansion/hydrodynamic-model-output/SAM-TWPICE"
STAMP = "0000003450"  # final snapshot (timestep*2s; day 20 + 1.9 h)
LEVELS_M = (1500.0, 5000.0, 9000.0)  # boundary layer / mid / upper troposphere
CLOUD_KGKG = 0.01e-3  # condensate threshold, 0.01 g/kg
XSECT_M = 5000.0


def level_moments(field, iz):
    """(mean, var, skew) of one level with float64 accumulation."""
    col = field[:, :, iz].astype(np.float64, copy=False)
    m = col.mean(dtype=np.float64)
    d = col - m
    m2 = np.mean(d * d, dtype=np.float64)
    m3 = np.mean(d * d * d, dtype=np.float64)
    return m, m2, m3 / m2**1.5 if m2 > 0 else 0.0


def profile_stats(field):
    nz = field.shape[2]
    mean = np.empty(nz)
    var = np.empty(nz)
    skew = np.empty(nz)
    for iz in range(nz):
        mean[iz], var[iz], skew[iz] = level_moments(field, iz)
    return mean, var, skew


def fluctuations(field, z, mean):
    """Haar fluctuation functions F_1, F_2 along x at LEVELS_M."""
    out = {}
    for target in LEVELS_M:
        iz = int(np.argmin(np.abs(z - target)))
        anom = field[:, :, iz].astype(np.float64) - mean[iz]
        lags, F = scaleinvariance.haar_fluctuation(anom, order=[1, 2], axis=0)
        out[target] = (z[iz], lags, F)
    return out


def reduce_fields(h, qt, cond, z, dz, tag):
    """All stats for one model; h, qt, cond in (x, y, z), SI units."""
    h_mean, h_var, h_skew = profile_stats(h)
    qt_mean, qt_var, qt_skew = profile_stats(qt)
    cloud_fraction = np.array([
        np.mean(cond[:, :, iz] > CLOUD_KGKG, dtype=np.float64)
        for iz in range(z.size)
    ])
    column = np.zeros(cond.shape[:2])
    for iz in range(z.size):
        column += cond[:, :, iz].astype(np.float64) * dz[iz]

    iz_x = int(np.argmin(np.abs(z - XSECT_M)))
    qt_slice = qt[:, :, iz_x].astype(np.float64) - qt_mean[iz_x]

    fluct = {}
    for name, field, mean in [("h", h, h_mean), ("qt", qt, qt_mean)]:
        for target, (z_used, lags, F) in fluctuations(field, z, mean).items():
            fluct[f"fluct_{name}_{target:.0f}_z"] = z_used
            fluct[f"fluct_{name}_{target:.0f}_lags"] = lags
            fluct[f"fluct_{name}_{target:.0f}_F"] = F

    np.savez(
        f"{tag}_stats.npz",
        z=z, h_mean=h_mean, h_var=h_var, h_skew=h_skew,
        qt_mean=qt_mean, qt_var=qt_var, qt_skew=qt_skew,
        cloud_fraction=cloud_fraction, column_condensate=column,
        qt_slice=qt_slice, qt_slice_z=z[iz_x], levels=np.array(LEVELS_M),
        **fluct,
    )
    print(f"wrote {tag}_stats.npz")


def sam():
    ds = netCDF4.Dataset(f"{SAM}/OUT_3D.MSE/TWPICE_LPT_3D_MSE_{STAMP}.nc")
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    h = ds.variables["MSE"][0].astype(np.float32) * np.float32(cp)  # K -> J/kg
    ds.close()

    def read(var):
        ds = netCDF4.Dataset(f"{SAM}/OUT_3D.{var}/TWPICE_LPT_3D_{var}_{STAMP}.nc")
        ds.set_auto_mask(False)
        field = ds.variables[var][0].astype(np.float32) / np.float32(1000.0)
        ds.close()
        return field  # g/kg -> kg/kg

    cond = read("QC")
    cond += read("QI")
    qt = read("QV")
    qt += cond

    dz = np.gradient(z)
    reduce_fields(h, qt, cond, z, dz, "sam")


def steam():
    ds = netCDF4.Dataset("steam_twpice.nc")
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    dz = ds.variables["dz"][:].astype(np.float64)
    h = ds.variables["h"][:]
    qt = ds.variables["qt"][:]
    cond = ds.variables["qc"][:] + ds.variables["qi"][:]
    ds.close()
    reduce_fields(h, qt, cond, z, dz, "steam")


if __name__ == "__main__":
    sam()
    steam()
