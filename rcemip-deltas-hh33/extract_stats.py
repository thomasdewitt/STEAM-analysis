#!/usr/bin/env python3
"""Extract per-level stats + STEAM input profiles from RCEMIP channel snapshots.

Ten RCE_large300 models, three well-separated snapshots each (see inventory in
the 2026-07-17 session notes). For each model snapshot this writes
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
#    fields shaped (nz, ny, nx), one call per snapshot index 0..2 ─────────────


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


def cm1(i):
    files = sorted((DATA / "cm1").glob("CM1_RCE_large300_3D_allvars_hour*.nc"))
    path = files[i]
    z = read_var(path, "z")
    T = read_var(path, "ta", 0)
    qv = spec_to_mr(read_var(path, "hus", 0))
    qc = spec_to_mr(read_var(path, "clw", 0))
    qi = spec_to_mr(read_var(path, "cli", 0))
    pa = read_var(path, "pa", 0)
    return z, T, qv, qc, qi, float(pa[0].mean(dtype=np.float64))


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


def mesonh(i):
    d = DATA / "mesonh"
    # MESONH files carry no coordinates; the RCEMIP standard 74-level grid is
    # identical to SCALE's lev (verified in the inventory).
    z = read_var(DATA / "scale" / "SCALE_RCE_large300_3D_ta_last25d.nc", "lev")
    T = read_var(d / "MESONH_RCE_large300_3D_ta.nc", "ta", i)
    qv = spec_to_mr(read_var(d / "MESONH_RCE_large300_3D_hus.nc", "hus", i))
    qc = spec_to_mr(read_var(d / "MESONH_RCE_large300_3D_clw.nc", "clw", i))
    qi = spec_to_mr(read_var(d / "MESONH_RCE_large300_3D_cli.nc", "cli", i))
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


ADAPTERS = {
    "sam": sam, "cm1": cm1,
    "ukmo_casim": ukmo_casim, "ukmo_ra1t": ukmo_ra1t,
    "ukmo_ra1t_nocloud": ukmo_ra1t_nocloud,
    "mesonh": mesonh, "scale": scale, "ucla": ucla,
    "icon_lem": icon_lem, "icon_nwp": icon_nwp,
}


def reduce_snapshot(model, i):
    z, T, qv, qc, qi, p_surf = ADAPTERS[model](i)
    z = np.asarray(z, dtype=np.float64)
    assert np.all(np.diff(z) > 0), f"{model}: z not increasing"
    nz = z.size
    assert T.shape[0] == nz, f"{model}: shape {T.shape} vs nz {nz}"
    for name, a in (("T", T), ("qv", qv), ("qc", qc), ("qi", qi)):
        bad = ~np.isfinite(a) | (np.abs(a) > 1e8)
        assert not bad.any(), f"{model} snap{i}: {name} has {bad.sum()} bad"

    h_mean = np.empty(nz)
    h_var = np.empty(nz)
    qt_mean = np.empty(nz)
    qt_var = np.empty(nz)
    cloud_fraction = np.empty(nz)
    for k in range(nz):
        hk = (cp * T[k].astype(np.float64) + g * z[k]
              + Lv * qv[k].astype(np.float64))
        qtk = (qv[k].astype(np.float64) + qc[k].astype(np.float64)
               + qi[k].astype(np.float64))
        condk = qc[k].astype(np.float64) + qi[k].astype(np.float64)
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
        for i in range(3):
            reduce_snapshot(model, i)
