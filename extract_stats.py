#!/usr/bin/env python3
"""Extract per-level stats + STEAM input profiles from RCEMIP channel snapshots.

Nine RCE_large300 models, three well-separated snapshots each (see inventory in
the 2026-07-17 session notes). MESONH is excluded: its archived 3D hus is a
documented RCEMIP data error (Known RCEMIP Bugs, Sec. 17 -- "much too small in
the lower troposphere"; confirmed empirically 2026-07-21). For each model snapshot this writes
stats/<model>_snap<i>.npz with: native z [m], mean/variance of h and qt per
level, cloud fraction (qc+qi > 0.01 g/kg), and the STEAM input profile
(h, qt interpolated to uniform 50 m spacing, surface pressure).

Conventions: all humidities converted to MIXING RATIO (r = q/(1-q) from
specific humidity) to match STEAM/SAM thermodynamics. h = cp*T + g*z + Lv*r_v
with steam's constants. qt = r_v + r_c + r_i (no precip water -- not all
models provide it). Every reduction uses float64 accumulators (float32 source;
see CLAUDE.md gotcha). ICON's z comes from inverting its frozen-MSE file:
zf = (fmse - 1004.64*ta - 2500800*hus + 333700*cli) / 9.80665 (its constants,
not steam's -- used only for z recovery).

Five single-snapshot comparison datasets are handled by the same machinery:
twpice (SAM-TWPICE, 100 m, 2048^2 x 255) and les_{cm1,sam,dales,icon_lem}
(RCEMIP RCE_small_les300, 200 m).

Usage: python extract_stats.py [model ...]   (default: all)
"""

import sys
from pathlib import Path

import netCDF4
import numpy as np

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

DATA = Path(__file__).parent / "data"
STATS = Path(__file__).parent / "stats"
CLOUD_KGKG = 0.01e-3
PROFILE_DZ = 50.0
DOMAIN_HEIGHT = 20000.0
P0_DEFAULT = 101480.0  # RCEMIP analytic sounding surface pressure [Pa]


def spec_to_mr(q):
    """Specific humidity/mass fraction -> mixing ratio (vapor-only denom)."""
    return q / (1.0 - q)


def read_var(path, name, index=None):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    v = ds.variables[name]
    data = v[index] if index is not None else v[:]
    ds.close()
    return np.asarray(data)


# ── adapters: yield (z_1d_m, T, qv_mr, qc_mr, qi_mr, p_surface) bottom-up,
#    fields shaped (nz, ny, nx), one call per snapshot index ────────────────


def sam(i):
    files = sorted((DATA / "sam").glob("SAM_CRM_RCE_large300_3D_*.nc"))
    path = files[i]
    z = read_var(path, "z")
    T = read_var(path, "ta", 0)
    qv = read_var(path, "QV", 0)          # g/g mixing ratio already
    qc = read_var(path, "clw")            # (z,y,x), no time dim; g/g specific
    qi = read_var(path, "cli")
    pa = read_var(path, "pa", 0)
    return z, T, qv, spec_to_mr(qc), spec_to_mr(qi), float(
        pa[0].mean(dtype=np.float64))


def _cm1(subdir, i):
    files = sorted((DATA / subdir).glob("CM1_RCE_*300_3D_allvars_hour*.nc"))
    path = files[i]
    z = read_var(path, "z")
    T = read_var(path, "ta", 0)
    qv = spec_to_mr(read_var(path, "hus", 0))
    qc = spec_to_mr(read_var(path, "clw", 0))
    qi = spec_to_mr(read_var(path, "cli", 0))
    pa = read_var(path, "pa", 0)
    return z, T, qv, qc, qi, float(pa[0].mean(dtype=np.float64))


def cm1(i):
    return _cm1("cm1", i)


def les_cm1(i):
    return _cm1("les_cm1", i)


def _ukmo(subdir, i):
    files = sorted((DATA / subdir).glob("*_RCE_large300_3D_*.nc"))
    path = files[i]
    z = read_var(path, "rholev_zsea_rho")
    T = read_var(path, "ta", 0)
    qv = spec_to_mr(read_var(path, "hus", 0))
    qc = spec_to_mr(read_var(path, "clw", 0))
    qi = spec_to_mr(read_var(path, "cli", 0))
    pa = read_var(path, "pa", 0)
    return z, T, qv, qc, qi, float(pa[0].mean(dtype=np.float64))


def ukmo_casim(i):
    return _ukmo("ukmo_casim", i)


def ukmo_ra1t(i):
    return _ukmo("ukmo_ra1t", i)


def ukmo_ra1t_nocloud(i):
    return _ukmo("ukmo_ra1t_nocloud", i)


def scale(i):
    d = DATA / "scale"
    z = read_var(d / "SCALE_RCE_large300_3D_ta_last25d.nc", "lev")
    T = read_var(d / "SCALE_RCE_large300_3D_ta_last25d.nc", "ta", i)
    qv = spec_to_mr(read_var(d / "SCALE_RCE_large300_3D_hus_last25d.nc", "hus", i))
    qc = spec_to_mr(read_var(d / "SCALE_RCE_large300_3D_clw_last25d.nc", "clw", i))
    qi = spec_to_mr(read_var(d / "SCALE_RCE_large300_3D_cli_last25d.nc", "cli", i))
    return z, T, qv, qc, qi, P0_DEFAULT


def ucla(i):
    d = DATA / "ucla"
    # Dims (time, yt, xt, zt), z LAST; zt[0] = -37 m ghost level -> drop.
    z = read_var(d / "UCLA-CRM_RCE_large300_3D_ta.nc", "zt")[1:]

    def get(fname, var):
        a = read_var(d / fname, var, i)          # (yt, xt, zt)
        return np.moveaxis(a, -1, 0)[1:]         # -> (zt-1, yt, xt)

    T = get("UCLA-CRM_RCE_large300_3D_ta.nc", "ta")
    qv = spec_to_mr(get("UCLA-CRM_RCE_large300_3D_hus.nc", "hus"))
    qc = spec_to_mr(get("UCLA-CRM_RCE_large300_3D_clw.nc", "clw"))
    qi = spec_to_mr(get("UCLA-CRM_RCE_large300_3D_ice.nc", "ice"))
    return z, T, qv, qc, qi, P0_DEFAULT


def _icon(main_file, fmse_file, fmse_var, i):
    """ICON: top-down index z; recover physical z from the frozen-MSE file."""
    T = read_var(main_file, "ta", i)
    hus = read_var(main_file, "hus", i)
    clw = read_var(main_file, "clw", i)
    cli = read_var(main_file, "cli", i)
    pa = read_var(main_file, "pa", i)
    fmse = read_var(fmse_file, fmse_var, 0)
    T0 = read_var(main_file, "ta", 0)
    hus0 = read_var(main_file, "hus", 0)
    cli0 = read_var(main_file, "cli", 0)
    # zf per column at snapshot 0 (z is time-independent); ICON's constants.
    zf = (fmse - 1004.64 * T0 - 2500800.0 * hus0 + 333700.0 * cli0) / 9.80665
    z = zf.mean(axis=(1, 2), dtype=np.float64)
    # Guard: z must be near-constant per level.
    assert np.all(zf.std(axis=(1, 2), dtype=np.float64) < 50.0), "zf not flat"

    def flip(a):
        return a[::-1]  # top-down -> bottom-up

    p_surf = float(pa[-1].mean(dtype=np.float64))
    return (flip(z), flip(T), flip(spec_to_mr(hus)), flip(spec_to_mr(clw)),
            flip(spec_to_mr(cli)), p_surf)


def icon_lem(i):
    d = DATA / "icon_lem"
    return _icon(d / "ICON_LEM_CRM-RCE_large_300-3D_last25d.nc",
                 d / "ICON_LEM_CRM-RCE_large_300-3D_FMSE_6h_last25d.nc",
                 "fmse_3d", i)


def icon_nwp(i):
    d = DATA / "icon_nwp"
    return _icon(d / "ICON_NWP_CRM-RCE_large_300-3D_last25d.nc",
                 d / "ICON_NWP_CRM-RCE_large_300-3D_FMSE_last25d.nc",
                 "fmse_3d", i)


def les_icon_lem(i):
    d = DATA / "les_icon_lem"
    return _icon(d / "ICON_LEM_CRM-RCE_small_les_300-3D_t3.nc",
                 d / "ICON_LEM_CRM-RCE_small_les_300-3D_FMSE_t3.nc",
                 "fmse_3d", i)


def les_sam(i):
    path = (DATA / "les_sam"
            / "RCEMIP_SST300_480x480x146-200m-2s_480_0002160000.nc")
    # Native SAM names, not the RCEMIP ones (Known RCEMIP Bugs, Sec. 21).
    z = read_var(path, "z")
    T = read_var(path, "TABS", 0)
    qv = read_var(path, "QV", 0) / 1000.0     # g/kg mixing ratio already
    qc = read_var(path, "QN", 0) / 1000.0     # QN = cloud water + cloud ice
    qi = np.zeros_like(qc)
    p = read_var(path, "p")                   # 1-D, mb
    return z, T, qv, qc, qi, float(p[0]) * 100.0


def les_dales(i):
    d = DATA / "les_dales"
    # Known RCEMIP Bugs: DALES 3-D ta is unreliable above the tropopause
    # (ozone misconfiguration); the troposphere used here is unaffected.
    z = read_var(d / "DALES_RCE_small_les300_3D_ta_t30.nc", "zt")
    T = read_var(d / "DALES_RCE_small_les300_3D_ta_t30.nc", "ta", 0)
    qv = spec_to_mr(read_var(d / "DALES_RCE_small_les300_3D_hus_t30.nc", "hus", 0))
    qc = spec_to_mr(read_var(d / "DALES_RCE_small_les300_3D_clw_t30.nc", "clw", 0))
    qi = spec_to_mr(read_var(d / "DALES_RCE_small_les300_3D_cli_t30.nc", "cli", 0))
    return z, T, qv, qc, qi, P0_DEFAULT       # no pa in the DALES output


def _twpice_field(path, name, y_first):
    """Read a TWPICE (time, ., ., z) field into a contiguous (z, y, x) float32
    array, one horizontal slab at a time. A single 2048^2 x 255 field is 4.3 GB
    (8.6 GB for the float64 MSE), so the whole file is never held at once, and
    the z-last file layout would make per-level reductions cache-hostile."""
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    v = ds.variables[name]
    nz = v.shape[3]
    ny, nx = (v.shape[1], v.shape[2]) if y_first else (v.shape[2], v.shape[1])
    field = np.empty((nz, ny, nx), dtype=np.float32)
    for a in range(0, v.shape[1], 128):
        slab = np.asarray(v[0, a:a + 128])
        if y_first:
            field[:, a:a + 128, :] = slab.transpose(2, 0, 1)
        else:
            field[:, :, a:a + 128] = slab.transpose(2, 1, 0)
    ds.close()
    return field


def twpice(i):
    d = DATA / "twpice"
    qv_file = d / "TWPICE_LPT_3D_QV_0000003450.nc"
    z = read_var(qv_file, "z")
    pres = read_var(qv_file, "pres")                       # 1-D, mb
    # g/kg SAM mixing ratios already; axis order is (y, x, z) here but
    # (x, y, z) in the MSE file.
    qv = _twpice_field(qv_file, "QV", True)
    qc = _twpice_field(d / "TWPICE_LPT_3D_QC_0000003450.nc", "QC", True)
    qi = _twpice_field(d / "TWPICE_LPT_3D_QI_0000003450.nc", "QI", True)
    qv /= 1000.0
    qc /= 1000.0
    qi /= 1000.0
    # No temperature field is archived: MSE is in K, = T + (g*z + Lv*qv)/cp
    # with steam's constants. Invert in place so T never costs a second array.
    T = _twpice_field(d / "TWPICE_LPT_3D_MSE_0000003450.nc", "MSE", False)
    for k in range(z.size):
        T[k] -= (g * z[k] + Lv * qv[k]) / cp
    return z, T, qv, qc, qi, float(pres[0]) * 100.0


ADAPTERS = {
    "sam": sam, "cm1": cm1,
    "ukmo_casim": ukmo_casim, "ukmo_ra1t": ukmo_ra1t,
    "ukmo_ra1t_nocloud": ukmo_ra1t_nocloud,
    "scale": scale, "ucla": ucla,
    "icon_lem": icon_lem, "icon_nwp": icon_nwp,
    "twpice": twpice, "les_cm1": les_cm1, "les_sam": les_sam,
    "les_dales": les_dales, "les_icon_lem": les_icon_lem,
}

# The comparison datasets below the RCE_large300 set are single snapshots.
SINGLE_SNAPSHOT = {"twpice", "les_cm1", "les_sam", "les_dales", "les_icon_lem"}

# The channel ensemble is exactly the RCE_large300 hosts: everything the
# STEAM channel runs and their figures iterate over. The single-snapshot
# comparison datasets belong to the matched-LES comparison, not here.
CHANNEL_HOSTS = [m for m in ADAPTERS if m not in SINGLE_SNAPSHOT]


def reduce_snapshot(model, i):
    z, T, qv, qc, qi, p_surf = ADAPTERS[model](i)
    z = np.asarray(z, dtype=np.float64)
    assert np.all(np.diff(z) > 0), f"{model}: z not increasing"
    nz = z.size
    assert T.shape[0] == nz, f"{model}: shape {T.shape} vs nz {nz}"

    h_mean = np.empty(nz)
    h_var = np.empty(nz)
    qt_mean = np.empty(nz)
    qt_var = np.empty(nz)
    cloud_fraction = np.empty(nz)
    # Level by level: TWPICE is 2048^2 x 255, so whole-field temporaries (even
    # a validity mask) are gigabytes.
    for k in range(nz):
        Tk = T[k].astype(np.float64)
        qvk = qv[k].astype(np.float64)
        qck = qc[k].astype(np.float64)
        qik = qi[k].astype(np.float64)
        for name, a in (("T", Tk), ("qv", qvk), ("qc", qck), ("qi", qik)):
            bad = ~np.isfinite(a) | (np.abs(a) > 1e8)
            assert not bad.any(), (
                f"{model} snap{i}: {name} has {bad.sum()} bad at level {k}")
        hk = cp * Tk + g * z[k] + Lv * qvk
        qtk = qvk + qck + qik
        condk = qck + qik
        h_mean[k] = hk.mean()
        h_var[k] = hk.var()
        qt_mean[k] = qtk.mean()
        qt_var[k] = qtk.var()
        cloud_fraction[k] = np.mean(condk > CLOUD_KGKG)

    z_uniform = np.arange(0.0, DOMAIN_HEIGHT + PROFILE_DZ, PROFILE_DZ)
    h_profile = np.interp(z_uniform, z, h_mean)
    qt_profile = np.interp(z_uniform, z, qt_mean)

    STATS.mkdir(exist_ok=True)
    np.savez(
        STATS / f"{model}_snap{i}.npz",
        z=z, h_mean=h_mean, h_var=h_var, qt_mean=qt_mean, qt_var=qt_var,
        cloud_fraction=cloud_fraction,
        z_profile=z_uniform, h_profile=h_profile, qt_profile=qt_profile,
        surface_pressure=p_surf, grid_shape=np.array(T.shape),
    )
    print(f"{model} snap{i}: nz={nz}, grid={T.shape[1:]}, "
          f"CF_max={cloud_fraction.max():.2f}, p0={p_surf:.0f}")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(ADAPTERS)
    for model in wanted:
        for i in range(1 if model in SINGLE_SNAPSHOT else 3):
            reduce_snapshot(model, i)
