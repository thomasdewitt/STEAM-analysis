#!/usr/bin/env python3
"""Vertical Mexican-hat fluctuation functions: 3D field vs mean profile.

For z-bands 0-5, 5-10, 10-15 km, computes the vertical (along-z) first-order
Mexican-hat fluctuation function of (1) the full 3D field and (2) the 1D
horizontally-averaged profile, for qt and h, all cases. The idea: the lag
where the 3D-field curve descends to the profile curve marks the vertical
outer scale of the turbulent fluctuations (below it the 3D fluctuations are
dominated by turbulence; above it by the mean profile's vertical structure).

Grids are nonuniform in z, so each band is interpolated to a uniform grid at
the band's median dz first. Mexican-hat kernels span ~5.8x the lag, so the
largest reachable lag in a 5 km band is ~860 m -- curves that haven't met by
then have a vertical outer scale beyond the band's reach.

Writes vertical_fluct_<tag>.npz per case (run with case args like
compute_stats.py) and figs/vertical_fluctuations.png (run with "figure").
"""

import sys

import netCDF4
import numpy as np
import scaleinvariance

from steam.constants import specific_heat_dry_air as cp

SAM = "/run/media/thomas/Expansion/hydrodynamic-model-output/SAM-TWPICE"
STAMP = "0000003450"
BANDS_M = ((0.0, 5000.0), (5000.0, 10000.0), (10000.0, 15000.0),
           (0.0, 15000.0))

scaleinvariance.set_backend("torch")
scaleinvariance.set_device("cuda")
scaleinvariance.set_numerical_precision("float64")


def interpolate_band(field, z, z_lo, z_hi):
    """Field and mean profile on a uniform z grid (median-dz) inside a band."""
    inside = (z >= z_lo) & (z <= z_hi)
    dz_uniform = float(np.median(np.diff(z[inside])))
    z_uniform = np.arange(z_lo, z_hi + 0.5 * dz_uniform, dz_uniform)
    z_uniform = z_uniform[(z_uniform >= z[0]) & (z_uniform <= z[-1])]

    indices = np.searchsorted(z, z_uniform, side="right") - 1
    indices = np.clip(indices, 0, z.size - 2)
    weight = (z_uniform - z[indices]) / (z[indices + 1] - z[indices])

    band = np.empty(field.shape[:2] + (z_uniform.size,), dtype=np.float64)
    for j, (i, w) in enumerate(zip(indices, weight)):
        band[:, :, j] = ((1.0 - w) * field[:, :, i].astype(np.float64)
                         + w * field[:, :, i + 1].astype(np.float64))
    profile = band.mean(axis=(0, 1), dtype=np.float64)
    return band, profile, dz_uniform


N_CHUNKS = 8  # x-chunks for the 3D convolution (GPU memory). Averaging
# F_1 over equal-size NaN-free chunks is exact: order 1 is a mean over
# positions and every chunk contributes the same count per lag.


def band_fluctuations(field, z, tag_prefix, out):
    for z_lo, z_hi in BANDS_M:
        band, profile, dz_uniform = interpolate_band(field, z, z_lo, z_hi)
        chunks = np.array_split(band, N_CHUNKS, axis=0)
        assert len({c.shape for c in chunks}) == 1
        F_sum = None
        for chunk in chunks:
            lags_field, F_chunk = scaleinvariance.wavelet_fluctuation(
                chunk, wavelet="mexican_hat", order=1, axis=2)
            F_sum = F_chunk if F_sum is None else F_sum + F_chunk
        F_field = F_sum / N_CHUNKS
        lags_profile, F_profile = scaleinvariance.wavelet_fluctuation(
            profile, wavelet="mexican_hat", order=1, axis=0)
        key = f"{tag_prefix}_{z_lo / 1000:.0f}-{z_hi / 1000:.0f}"
        out[f"{key}_dz"] = dz_uniform
        out[f"{key}_lags_field"] = lags_field * dz_uniform
        out[f"{key}_F_field"] = F_field
        out[f"{key}_lags_profile"] = lags_profile * dz_uniform
        out[f"{key}_F_profile"] = F_profile
        print(f"  {key}: dz={dz_uniform:.0f} m, "
              f"max field lag={np.nanmax(lags_field) * dz_uniform:.0f} m")


def reduce_case(h, qt, z, tag):
    out = {}
    band_fluctuations(qt, z, "qt", out)
    band_fluctuations(h, z, "h", out)
    np.savez(f"vertical_fluct_{tag}.npz", **out)
    print(f"wrote vertical_fluct_{tag}.npz")


def sam():
    ds = netCDF4.Dataset(f"{SAM}/OUT_3D.MSE/TWPICE_LPT_3D_MSE_{STAMP}.nc")
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    h = ds.variables["MSE"][0].astype(np.float32) * np.float32(cp)
    ds.close()

    def read(var):
        ds = netCDF4.Dataset(f"{SAM}/OUT_3D.{var}/TWPICE_LPT_3D_{var}_{STAMP}.nc")
        ds.set_auto_mask(False)
        field = ds.variables[var][0].astype(np.float32) / np.float32(1000.0)
        ds.close()
        return field

    qt = read("QV")
    qt += read("QC")
    qt += read("QI")
    reduce_case(h, qt, z, "sam")


def steam(path, tag):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    h = ds.variables["h"][:]
    qt = ds.variables["qt"][:]
    ds.close()
    reduce_case(h, qt, z, tag)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
        "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })
    cases = [
        ("sam", "SAM", "#C2410C"),
        ("steam", "STEAM ls 1000$\\to$10 m", "#1268A3"),
        ("steam_ls10", "STEAM ls 10 m", "#3FA34D"),
        ("steam_ls3000_1", "STEAM ls 3000 m (<4 km) $\\to$ 1 m", "#7C3AAD"),
    ]
    data = [(np.load(f"vertical_fluct_{tag}.npz"), label, color)
            for tag, label, color in cases]

    fig, axes = plt.subplots(2, len(BANDS_M), figsize=(3.5 * len(BANDS_M), 6.4))
    for row, name, symbol in [(0, "qt", "q_t"), (1, "h", "h")]:
        for col, (z_lo, z_hi) in enumerate(BANDS_M):
            ax = axes[row, col]
            key = f"{name}_{z_lo / 1000:.0f}-{z_hi / 1000:.0f}"
            for d, label, color in data:
                ax.loglog(d[f"{key}_lags_field"], d[f"{key}_F_field"],
                          color=color, lw=1.5, label=f"{label} 3D")
                ax.loglog(d[f"{key}_lags_profile"], d[f"{key}_F_profile"],
                          color=color, lw=1.0, ls="--",
                          label=f"{label} profile")
            ax.set_title(f"${symbol}$,  {z_lo / 1000:.0f}$-${z_hi / 1000:.0f} km")
            ax.grid(True, which="both", alpha=0.5)
            if row == 1:
                ax.set_xlabel("vertical lag [m]")
            if col == 0:
                ax.set_ylabel(f"$F_1$ (Mexican hat, z) of ${symbol}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("figs/vertical_fluctuations.png")
    print("wrote figs/vertical_fluctuations.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or [
        "sam", "steam", "steam_ls10", "steam_ls3000_1", "figure"]
    if "sam" in wanted:
        sam()
    if "steam" in wanted:
        steam("steam_twpice.nc", "steam")
    if "steam_ls10" in wanted:
        steam("steam_twpice_ls10.nc", "steam_ls10")
    if "steam_ls3000_1" in wanted:
        steam("steam_twpice_ls3000_1.nc", "steam_ls3000_1")
    if "figure" in wanted:
        figure()
