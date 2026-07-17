#!/usr/bin/env python3
"""Reduce SAM-TWPICE (final snapshot) and the STEAM runs to small stat caches.

Writes <tag>_stats.npz: per-level mean/variance/skewness of h and qt, cloud
fraction (condensate > 0.01 g/kg), separate qc/qi mean profiles, Haar
fluctuation functions (orders 1 and 2) at three representative levels, 1D PDFs
of h and qt at 4 km and 10 km (full-data-range bins -- no percentile capping,
which fakes a tail cliff), one mid-troposphere qt' slice, and the
column-integrated condensate.

SAM 3D fields are float32 (MSE float64 on disk); every K-scale reduction uses
a float64 accumulator (see CLAUDE.md gotcha).

Usage: python compute_stats.py [sam] [steam] [steam_ls10]
       (no args = all three)
"""

import sys

import netCDF4
import numpy as np
import scaleinvariance

from steam.constants import specific_heat_dry_air as cp

SAM = "/run/media/thomas/Expansion/hydrodynamic-model-output/SAM-TWPICE"
STAMP = "0000003450"  # final snapshot (timestep*2s; day 20 + 1.9 h)
LEVELS_M = (1500.0, 5000.0, 9000.0)  # boundary layer / mid / upper troposphere
PDF_LEVELS_M = (4000.0, 10000.0)
PDF_BINS = 200
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


def level_pdfs(field, z):
    """Histogram (density) over the full data range at PDF_LEVELS_M."""
    out = {}
    for target in PDF_LEVELS_M:
        iz = int(np.argmin(np.abs(z - target)))
        col = field[:, :, iz].astype(np.float64, copy=False).ravel()
        density, edges = np.histogram(col, bins=PDF_BINS, density=True)
        out[target] = (z[iz], edges, density)
    return out


def reduce_fields(h, qt, qc, qi, z, dz, tag):
    """All stats for one model; h, qt, qc, qi in (x, y, z), SI units."""
    cond = qc + qi
    h_mean, h_var, h_skew = profile_stats(h)
    qt_mean, qt_var, qt_skew = profile_stats(qt)
    qc_mean = np.array([qc[:, :, iz].mean(dtype=np.float64)
                        for iz in range(z.size)])
    qi_mean = np.array([qi[:, :, iz].mean(dtype=np.float64)
                        for iz in range(z.size)])
    cloud_fraction = np.array([
        np.mean(cond[:, :, iz] > CLOUD_KGKG, dtype=np.float64)
        for iz in range(z.size)
    ])
    column = np.zeros(cond.shape[:2])
    for iz in range(z.size):
        column += cond[:, :, iz].astype(np.float64) * dz[iz]

    iz_x = int(np.argmin(np.abs(z - XSECT_M)))
    qt_mean_x = qt[:, :, iz_x].astype(np.float64).mean(dtype=np.float64)
    qt_slice = qt[:, :, iz_x].astype(np.float64) - qt_mean_x

    extras = {}
    for name, field, mean in [("h", h, h_mean), ("qt", qt, qt_mean)]:
        for target, (z_used, lags, F) in fluctuations(field, z, mean).items():
            extras[f"fluct_{name}_{target:.0f}_z"] = z_used
            extras[f"fluct_{name}_{target:.0f}_lags"] = lags
            extras[f"fluct_{name}_{target:.0f}_F"] = F
        for target, (z_used, edges, density) in level_pdfs(field, z).items():
            extras[f"pdf_{name}_{target:.0f}_z"] = z_used
            extras[f"pdf_{name}_{target:.0f}_edges"] = edges
            extras[f"pdf_{name}_{target:.0f}_density"] = density

    np.savez(
        f"{tag}_stats.npz",
        z=z, h_mean=h_mean, h_var=h_var, h_skew=h_skew,
        qt_mean=qt_mean, qt_var=qt_var, qt_skew=qt_skew,
        qc_mean=qc_mean, qi_mean=qi_mean,
        cloud_fraction=cloud_fraction, column_condensate=column,
        qt_slice=qt_slice, qt_slice_z=z[iz_x], levels=np.array(LEVELS_M),
        pdf_levels=np.array(PDF_LEVELS_M),
        **extras,
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

    qc = read("QC")
    qi = read("QI")
    qt = read("QV")
    qt += qc + qi

    dz = np.gradient(z)
    reduce_fields(h, qt, qc, qi, z, dz, "sam")


def steam(path="steam_twpice.nc", tag="steam"):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    dz = ds.variables["dz"][:].astype(np.float64)
    h = ds.variables["h"][:]
    qt = ds.variables["qt"][:]
    qc = ds.variables["qc"][:]
    qi = ds.variables["qi"][:]
    ds.close()
    reduce_fields(h, qt, qc, qi, z, dz, tag)


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["sam", "steam", "steam_ls10"]
    if "sam" in wanted:
        sam()
    if "steam" in wanted:
        steam()
    if "steam_ls10" in wanted:
        steam("steam_twpice_ls10.nc", "steam_ls10")
