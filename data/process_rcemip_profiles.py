#!/usr/bin/env python3
"""
Process RCEMIP 1D profile data into a standardized dataset.

Datasets (excluding DALES):
    CM1_small   : CM1  RCE_small_les300
    ICON_small  : ICON_LEM_CRM  RCE_small_les_300
    SAM_large   : SAM_CRM  RCE_large300
    SAM_small   : SAM_CRM  RCE_small_les300
    Dropsonde   : BEACH ORCESTRA Level-3 dropsondes (HALO, 1115 sondes)

Output: rcemip_profiles.nc  (same directory as this script)
    Dimensions : (dataset=5, height=330)
    Variables  : mse  [J kg-1]   = cp*T + g*z + Lv*qv
                 qt   [kg kg-1]  = qv + qi + qc  (non-precipitating)

Averages are taken over the last half of each simulation's timesteps.
All profiles are interpolated to a common height grid:
    50 m to 33 000 m, 100 m spacing (330 levels).

Physical constants
    cp = 1004 J/(kg·K)
    g  = 9.81 m/s²
    Lv = 2.501e6 J/kg

Notes on per-dataset quirks
    CM1:       variables have extra singleton ni, nj dims – squeezed out.
    ICON:      height dim is a level index; actual heights live in height_avg
               which has shape (height, lat, lon) and no time dim.
    SAM large: hus_avg is all-NaN fill; water vapour comes from QV_avg (kg/kg).
               SAM uses -9999 as fill value (masked by xarray via _FillValue).
    SAM small:  merged file SAM_CRM_RCE_small_les300_1D.nc used.
    Dropsonde:  mse pre-computed in file; qt = q (condensate negligible).
                NaN above 14 590 m (dropsonde ceiling).
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

# ── Physical constants ────────────────────────────────────────────────────────
CP = 1004.0    # J / (kg·K)
G  = 9.81      # m / s²
LV = 2.501e6   # J / kg

# ── Common output height grid ─────────────────────────────────────────────────
Z_COMMON = np.arange(50.0, 33001.0, 100.0)   # 330 levels, 50–33 000 m

BASE = '/Volumes/BLUE/RCEMIP'


# ── Helpers ───────────────────────────────────────────────────────────────────
def last_half_mean(da, time_dim='time'):
    """Mean over the last half of a time dimension."""
    nt = da.sizes[time_dim]
    return da.isel({time_dim: slice(nt // 2, None)}).mean(time_dim)


def interp_profile(z_src, profile):
    """
    Linearly interpolate a 1-D profile to Z_COMMON.
    Values outside [z_src.min(), z_src.max()] become NaN.
    """
    # Ensure monotonically increasing (just in case)
    order = np.argsort(z_src)
    z_src = np.asarray(z_src)[order]
    profile = np.asarray(profile)[order]

    # Replace any residual fill / bad values with NaN before interpolating
    bad = ~np.isfinite(profile)
    if bad.any():
        print('FILLING BAD DATA W NAN')
        # simple linear fill over gaps so interp1d doesn't choke
        profile[bad] = np.interp(
            z_src[bad], z_src[~bad], profile[~bad]
        ) if (~bad).sum() > 1 else 0.0

    f = interp1d(z_src, profile, bounds_error=False, fill_value=np.nan)
    return f(Z_COMMON)


# ── Per-dataset loaders ───────────────────────────────────────────────────────

def load_cm1_small():
    """CM1 RCE_small_les300 – single merged file, dims (time, nk, nj=1, ni=1)."""
    path = f'{BASE}/CM1/RCE_small_les300/1D/CM1_RCE_small_les300_1D_allvars_alltimes.nc'
    ds = xr.open_dataset(path)

    z  = ds['z'].values          # (nk,) in metres

    ta = last_half_mean(ds['ta_avg']).values.squeeze()   # K
    qv = last_half_mean(ds['hus_avg']).values.squeeze()  # kg/kg
    qc = last_half_mean(ds['clw_avg']).values.squeeze()  # kg/kg
    qi = last_half_mean(ds['cli_avg']).values.squeeze()  # kg/kg
    ds.close()

    mse = CP * ta + G * z + LV * qv
    qt  = qv + qi + qc
    return interp_profile(z, mse), interp_profile(z, qt)


def load_icon_small():
    """ICON_LEM_CRM RCE_small_les_300 – main file; actual heights in height_avg."""
    path = f'{BASE}/ICON_LEM_CRM/RCE_small_les_300/1D/ICON_LEM_CRM-RCE_small_les_300-1D.nc'
    ds = xr.open_dataset(path)

    # height_avg: shape (height, lat, lon), no time dim
    z  = ds['height_avg'].values.squeeze()   # (156,) in metres

    ta = last_half_mean(ds['ta_avg']).values.squeeze()   # K
    qv = last_half_mean(ds['hus_avg']).values.squeeze()  # kg/kg
    qc = last_half_mean(ds['clw_avg']).values.squeeze()  # kg/kg
    qi = last_half_mean(ds['cli_avg']).values.squeeze()  # kg/kg
    ds.close()

    mse = CP * ta + G * z + LV * qv
    qt  = qv + qi + qc
    return interp_profile(z, mse), interp_profile(z, qt)


def load_sam_large():
    """SAM_CRM RCE_large300 – one file per variable; qv from QV_avg (hus is NaN)."""
    d = f'{BASE}/SAM_CRM/RCE_large300/1D'

    ds_ta = xr.open_dataset(f'{d}/SAM_CRM_RCE_large300_1D_ta_avg.nc')
    ds_qv = xr.open_dataset(f'{d}/SAM_CRM_RCE_large300_1D_QV_avg.nc')
    ds_qc = xr.open_dataset(f'{d}/SAM_CRM_RCE_large300_1D_clw_avg.nc')
    ds_qi = xr.open_dataset(f'{d}/SAM_CRM_RCE_large300_1D_cli_avg.nc')

    z  = ds_ta['z'].values   # (74,) in metres

    ta = last_half_mean(ds_ta['ta_avg']).values   # K
    qv = last_half_mean(ds_qv['QV_avg']).values   # kg/kg
    qc = last_half_mean(ds_qc['clw_avg']).values  # kg/kg
    qi = last_half_mean(ds_qi['cli_avg']).values  # kg/kg

    for ds in (ds_ta, ds_qv, ds_qc, ds_qi):
        ds.close()

    mse = CP * ta + G * z + LV * qv
    qt  = qv + qi + qc
    return interp_profile(z, mse), interp_profile(z, qt)


def load_sam_small():
    """SAM_CRM RCE_small_les300 – merged all-vars file."""
    path = f'{BASE}/SAM_CRM/RCE_small_les300/1D/SAM_CRM_RCE_small_les300_1D.nc'
    ds = xr.open_dataset(path)

    z  = ds['z'].values   # (146,) in metres

    ta = last_half_mean(ds['ta_avg']).values   # K
    qv = last_half_mean(ds['hus_avg']).values  # kg/kg
    qc = last_half_mean(ds['clw_avg']).values  # kg/kg
    qi = last_half_mean(ds['cli_avg']).values  # kg/kg
    ds.close()

    mse = CP * ta + G * z + LV * qv
    qt  = qv + qi + qc
    return interp_profile(z, mse), interp_profile(z, qt)


def load_dropsonde():
    """BEACH ORCESTRA dropsondes – mean over all 1115 sondes; qt = q (condensate negligible).

    mse is pre-computed in the file (J/kg).
    qt = q (specific humidity, kg/kg).
    Altitude: 0–14 590 m at 10 m spacing; output NaN above 14 590 m.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dropsonde_data.nc')
    ds = xr.open_dataset(path)

    z   = ds['altitude'].values                   # (1460,) in metres, 0–14 590 m
    mse = np.nanmean(ds['mse'].values, axis=0)    # mean over sondes → (1460,)
    qt  = np.nanmean(ds['q'].values,   axis=0)    # mean over sondes → (1460,)
    ds.close()

    return interp_profile(z, mse), interp_profile(z, qt)


def load_dropsonde_extrapolated():
    """Dropsonde mean profile extrapolated from 12 km to 20 km.

    Data above 12 km (sparse coverage) is discarded before extrapolating.
    qt  : exponential decay above 12 km, e-folding length 4 km.
    mse : quadratic through (12 km, dropsonde), (15 km, 350 kJ/kg), (20 km, 400 kJ/kg).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dropsonde_data.nc')
    ds = xr.open_dataset(path)

    z_raw   = ds['altitude'].values                   # (1460,) in metres, 0–14 590 m
    mse_raw = np.nanmean(ds['mse'].values, axis=0)    # mean over sondes → (1460,)
    qt_raw  = np.nanmean(ds['q'].values,   axis=0)    # mean over sondes → (1460,)
    ds.close()

    # ── Truncate to 0–12 km (discard sparse data above) ──────────────────────
    mask_base = z_raw <= 12000.0
    z_base    = z_raw[mask_base]
    mse_base  = mse_raw[mask_base]
    qt_base   = qt_raw[mask_base]

    # ── Extrapolation grid: just above 12 km up to 20 km (same 10 m spacing) ─
    z_extrap = np.arange(z_base[-1] + 10.0, 20001.0, 10.0)

    # qt: exponential decay from last value at ~12 km, e-folding = 4 km
    qt_extrap = qt_base[-1] * np.exp(-(z_extrap - z_base[-1]) / 4000.0)

    # mse: quadratic through (12 km, dropsonde value), (15 km, 350 kJ/kg), (20 km, 400 kJ/kg)
    # Upper anchor values (350, 400 kJ/kg) taken from SAM TWPICE mean profile.
    mse_anchors_z   = np.array([z_base[-1], 15000.0, 20000.0])
    mse_anchors_val = np.array([mse_base[-1], 350e3, 400e3])
    coeffs = np.polyfit(mse_anchors_z, mse_anchors_val, 2)
    mse_extrap = np.polyval(coeffs, z_extrap)

    # ── Combine base + extrapolated and interpolate to Z_COMMON ──────────────
    z_full   = np.concatenate([z_base,   z_extrap])
    mse_full = np.concatenate([mse_base, mse_extrap])
    qt_full  = np.concatenate([qt_base,  qt_extrap])

    return interp_profile(z_full, mse_full), interp_profile(z_full, qt_full)


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = [
    ('CM1_small',       load_cm1_small),
    ('ICON_small',      load_icon_small),
    ('SAM_large',       load_sam_large),
    ('SAM_small',       load_sam_small),
    ('Dropsonde',       load_dropsonde),
    ('Dropsonde_extrap', load_dropsonde_extrapolated),
]

mse_list, qt_list = [], []
for name, loader in DATASETS:
    print(f'Processing {name} ...', flush=True)
    mse, qt = loader()
    mse_list.append(mse)
    qt_list.append(qt)
    print(f'  mse range: {np.nanmin(mse):.1f} – {np.nanmax(mse):.1f} J/kg')
    print(f'  qt  range: {np.nanmin(qt)*1e3:.3f} – {np.nanmax(qt)*1e3:.3f} g/kg')

mse_arr = np.array(mse_list, dtype=np.float64)   # (4, 330)
qt_arr  = np.array(qt_list,  dtype=np.float64)

dataset_names = [name for name, _ in DATASETS]

ds_out = xr.Dataset(
    {
        'mse': xr.DataArray(
            mse_arr, dims=['dataset', 'height'],
            attrs={
                'units':     'J kg-1',
                'long_name': 'Moist static energy',
                'formula':   'cp*T + g*z + Lv*qv',
                'cp':        f'{CP} J/(kg·K)',
                'g':         f'{G} m/s²',
                'Lv':        f'{LV} J/kg',
            },
        ),
        'qt': xr.DataArray(
            qt_arr, dims=['dataset', 'height'],
            attrs={
                'units':     'kg kg-1',
                'long_name': 'Total non-precipitating water mixing ratio',
                'formula':   'qv + qi + qc',
                'note':      'Excludes precipitating hydrometeors (plw, pli)',
            },
        ),
    },
    coords={
        'height': xr.DataArray(
            Z_COMMON, dims=['height'],
            attrs={'units': 'm', 'long_name': 'Height above surface'},
        ),
        'dataset': xr.DataArray(dataset_names, dims=['dataset']),
    },
    attrs={
        'title':        'RCEMIP 1D profile averages – mse and qt',
        'source_dir':   BASE,
        'averaging':    'Last half of each simulation timesteps',
        'height_grid':  '100 m regular spacing, 50–33000 m (330 levels)',
        'datasets': (
            'CM1_small  = CM1 RCE_small_les300 (1201 steps, 50 days); '
            'ICON_small = ICON_LEM_CRM RCE_small_les_300 (1201 steps, 50 days); '
            'SAM_large  = SAM_CRM RCE_large300 (2400 steps, ~100 days); '
            'SAM_small  = SAM_CRM RCE_small_les300 (1200 steps, 50 days); '
            'Dropsonde  = BEACH ORCESTRA Level-3 dropsondes, 1115 sondes, '
            'mean over all sondes; qt=q (condensate negligible); NaN above 14590 m; '
            'Dropsonde_extrap = Dropsonde mean truncated at 12 km, extrapolated to 20 km: '
            'qt via exponential decay (e-fold 4 km), mse via linear slope from 10–12 km'
        ),
        'created': '2026-02-18',
    },
)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rcemip_profiles.nc')
ds_out.to_netcdf(out_path)
print(f'\nSaved → {out_path}')
print(ds_out)
