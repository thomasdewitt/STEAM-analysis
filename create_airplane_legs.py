"""
create_airplane_legs.py — one-time script to build data/airplane_legs.nc

Reads /Volumes/BLUE/maestro_airplane_analysis/maestro_atr42.nc,
identifies level straight legs from equatorial flights 6-27,
regrids each leg onto a uniform *spatial* grid (using actual TAS),
and writes data/airplane_legs.nc.

Run once:
    python create_airplane_legs.py
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────
SOURCE_FILE = Path('/Volumes/BLUE/maestro_airplane_analysis/maestro_atr42.nc')
OUTPUT_FILE = Path(__file__).resolve().parent / 'data' / 'airplane_legs.nc'

# ── Leg-finding thresholds ─────────────────────────────────────────────
ALT_DEVIATION_MAX = 20       # m, max altitude deviation from 60s running mean
STRAIGHTNESS_MIN  = 0.9      # end-to-end / total path length ratio
MIN_LEG_LENGTH    = 500      # minimum number of raw data points per leg
MAX_NAN_FRACTION  = 0.20     # drop legs with more NaN than this across analysis vars
GAP_THRESHOLD_M   = 150.0    # spatial gap (m) beyond which interpolated values → NaN

# ── Physical constants ──────────────────────────────────────────────────
CP = 1005.0    # J/(kg·K)
G  = 9.81      # m/s²
LV = 2.501e6   # J/kg


# ── Flight / leg identification ────────────────────────────────────────

def identify_flights(time):
    """Split dataset into individual flights based on >1 hr time gaps.

    Parameters
    ----------
    time : array of datetime64

    Returns
    -------
    starts, ends : arrays of int indices
    """
    dt = np.diff(time).astype(float) / 1e9  # nanoseconds → seconds
    gap_indices = np.where(dt > 3600)[0]
    starts = np.concatenate([[0], gap_indices + 1])
    ends   = np.concatenate([gap_indices + 1, [len(time)]])
    return starts, ends


def find_level_straight_legs(alt, lat, lon,
                              min_length=MIN_LEG_LENGTH,
                              alt_dev_max=ALT_DEVIATION_MAX,
                              straightness_min=STRAIGHTNESS_MIN):
    """Find contiguous segments that are level and straight.

    Returns
    -------
    list of (start, end) index pairs (relative to the passed arrays)
    """
    alt_smooth = uniform_filter1d(alt, size=60)
    alt_dev    = np.abs(alt - alt_smooth)
    level_mask = alt_dev < alt_dev_max

    changes    = np.diff(level_mask.astype(int))
    seg_starts = np.where(changes == 1)[0] + 1
    seg_ends   = np.where(changes == -1)[0] + 1
    if level_mask[0]:
        seg_starts = np.concatenate([[0], seg_starts])
    if level_mask[-1]:
        seg_ends = np.concatenate([seg_ends, [len(level_mask)]])

    legs = []
    for ss, se in zip(seg_starts, seg_ends):
        if se - ss < min_length:
            continue

        lat_leg  = lat[ss:se]
        lon_leg  = lon[ss:se]
        cos_lat  = np.cos(np.radians(np.mean(lat_leg)))

        e2e   = np.sqrt(((lat_leg[-1] - lat_leg[0]) * 111320.0) ** 2
                        + ((lon_leg[-1] - lon_leg[0]) * 111320.0 * cos_lat) ** 2)
        dlat  = np.diff(lat_leg) * 111320.0
        dlon  = np.diff(lon_leg) * 111320.0 * cos_lat
        total = np.sum(np.sqrt(dlat ** 2 + dlon ** 2))

        straightness = e2e / total if total > 0 else 0.0
        if straightness >= straightness_min:
            legs.append((ss, se))

    return legs


# ── Spatial regridding ─────────────────────────────────────────────────

def regrid_spatial(time_leg, tas_leg, data_dict, gap_threshold_m=GAP_THRESHOLD_M):
    """Interpolate data onto a uniform *spatial* grid using actual TAS.

    Cumulative spatial position:
        x[0] = 0
        x[i] = x[i-1] + ((TAS[i-1] + TAS[i]) / 2) * dt_i

    Parameters
    ----------
    time_leg : array of datetime64
        Raw timestamps.
    tas_leg : 1-D array of float
        True airspeed in m/s for each raw sample.
    data_dict : dict {str: 1-D array}
        Variables to regrid.
    gap_threshold_m : float
        Spatial gap (m) beyond which interpolated point → NaN.

    Returns
    -------
    x_uniform : 1-D array (m), uniformly spaced
    dx_uniform : float (m)
    regridded  : dict {str: 1-D array}
    """
    # Convert time to float seconds
    t0        = time_leg[0]
    t_sec     = (time_leg - t0).astype(float) / 1e9

    # Cumulative spatial position using trapezoidal TAS integration
    dt        = np.diff(t_sec)
    tas_mid   = (tas_leg[:-1] + tas_leg[1:]) / 2.0
    dx_steps  = tas_mid * dt                          # step lengths (m)
    x_raw     = np.concatenate([[0.0], np.cumsum(dx_steps)])

    # Uniform spatial grid
    dx_uniform = float(np.median(np.diff(x_raw)))
    x_end      = x_raw[-1]
    x_uniform  = np.arange(0.0, x_end + dx_uniform, dx_uniform)

    # Gap mask: for each uniform grid point, distance to nearest raw point
    ins       = np.searchsorted(x_raw, x_uniform)
    ins       = np.clip(ins, 0, len(x_raw) - 1)
    dist_r    = np.abs(x_uniform - x_raw[ins])
    ins_l     = np.clip(ins - 1, 0, len(x_raw) - 1)
    dist_l    = np.abs(x_uniform - x_raw[ins_l])
    nearest   = np.minimum(dist_r, dist_l)
    gap_mask  = nearest > gap_threshold_m

    regridded = {}
    for name, data in data_dict.items():
        interp           = np.interp(x_uniform, x_raw, data.astype(float))
        interp[gap_mask] = np.nan
        regridded[name]  = interp

    return x_uniform, dx_uniform, regridded


# ── MSE ────────────────────────────────────────────────────────────────

def compute_mse(temp_c, mr_gkg, alt_m):
    """Moist static energy in kJ/kg."""
    T_K = temp_c + 273.15
    q   = mr_gkg / 1000.0   # g/kg → kg/kg
    return (CP * T_K + G * alt_m + LV * q) / 1000.0


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print(f"Reading {SOURCE_FILE} ...")
    ds   = xr.open_dataset(SOURCE_FILE)
    time = ds.time.values

    flight_starts, flight_ends = identify_flights(time)
    n_flights = len(flight_starts)
    print(f"Found {n_flights} flights; analysing flights 6–27 (equatorial).\n")

    all_legs = []  # list of dicts

    for fi in range(6, min(28, n_flights)):
        s, e = int(flight_starts[fi]), int(flight_ends[fi])
        alt  = ds.ALTITUDE.values[s:e]
        lat  = ds.LATITUDE.values[s:e]
        lon  = ds.LONGITUDE.values[s:e]

        legs = find_level_straight_legs(alt, lat, lon)
        if not legs:
            continue

        for ls, le in legs:
            gi_s = s + ls
            gi_e = s + le

            time_leg = time[gi_s:gi_e]
            tas_leg  = ds['TAS1'].values[gi_s:gi_e].astype(float)

            raw_temp = ds['TEMP1'].values[gi_s:gi_e].astype(float)
            raw_mr2  = ds['MR2'].values[gi_s:gi_e].astype(float)
            raw_uw   = ds['UW'].values[gi_s:gi_e].astype(float)
            raw_vw   = ds['VW'].values[gi_s:gi_e].astype(float)
            raw_alt  = ds['ALTITUDE'].values[gi_s:gi_e].astype(float)

            # Spatial regrid
            data_raw = {
                'TEMP1':    raw_temp,
                'MR2':      raw_mr2,
                'UW':       raw_uw,
                'VW':       raw_vw,
                'ALTITUDE': raw_alt,
                'TAS1':     tas_leg,
            }
            x_uniform, dx_uniform, rg = regrid_spatial(time_leg, tas_leg, data_raw)

            # Compute derived quantities on the regridded grid
            T_K     = rg['TEMP1'] + 273.15                         # K
            h_kJkg  = compute_mse(rg['TEMP1'], rg['MR2'], rg['ALTITUDE'])  # kJ/kg

            # NaN fraction check
            analysis_vars_rg = ['TEMP1', 'MR2', 'UW', 'VW']
            nan_fracs  = {v: float(np.isnan(rg[v]).mean()) for v in analysis_vars_rg}
            max_nan    = max(nan_fracs.values())

            if max_nan > MAX_NAN_FRACTION:
                print(f"  SKIP flight={fi} alt={np.nanmean(raw_alt):.0f}m "
                      f"— NaN fraction {max_nan:.2f} > {MAX_NAN_FRACTION}")
                continue

            mean_alt  = float(np.nanmean(rg['ALTITUDE']))
            min_alt   = float(np.nanmin(rg['ALTITUDE']))
            max_alt   = float(np.nanmax(rg['ALTITUDE']))
            std_alt   = float(np.nanstd(rg['ALTITUDE']))
            mean_tas  = float(np.nanmean(rg['TAS1']))
            n_samples = len(x_uniform)

            all_legs.append({
                'flight_id':   fi,
                'mean_alt':    mean_alt,
                'min_alt':     min_alt,
                'max_alt':     max_alt,
                'std_alt':     std_alt,
                'mean_tas':    mean_tas,
                'dx':          dx_uniform,
                'n':           n_samples,
                'nan_frac':    max_nan,
                # standardized variable arrays (display units)
                'T':  T_K,           # K
                'qv': rg['MR2'],     # g/kg
                'h':  h_kJkg,        # kJ/kg
                'u':  rg['UW'],      # m/s
                'v':  rg['VW'],      # m/s
            })

    n_legs = len(all_legs)
    print(f"\nRetained {n_legs} legs.")

    # ── Step 0: altitude statistics ──────────────────────────────────
    mean_alts = np.array([leg['mean_alt'] for leg in all_legs])
    print(f"Altitude distribution of airplane legs:")
    print(f"  min={mean_alts.min():.0f}  p5={np.percentile(mean_alts,5):.0f}  "
          f"p25={np.percentile(mean_alts,25):.0f}  median={np.median(mean_alts):.0f}  "
          f"p75={np.percentile(mean_alts,75):.0f}  p95={np.percentile(mean_alts,95):.0f}  "
          f"max={mean_alts.max():.0f}  (all in m)")

    # Per-leg stats
    print("\nPer-leg summary:")
    for i, leg in enumerate(all_legs):
        print(f"  [{i:02d}] flight={leg['flight_id']}  alt={leg['mean_alt']:.0f}m  "
              f"n={leg['n']}pts  dx={leg['dx']:.1f}m  NaN={leg['nan_frac']:.3f}")

    # ── Determine median dx across all legs ──────────────────────────
    all_dx = np.array([leg['dx'] for leg in all_legs])
    global_dx = float(np.median(all_dx))
    print(f"\nMedian dx across legs: {global_dx:.2f} m")

    # ── Build NC file ─────────────────────────────────────────────────
    max_samples = max(leg['n'] for leg in all_legs)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(str(OUTPUT_FILE), 'w') as dst:
        # Global attributes
        dst.setncattr('source_file', str(SOURCE_FILE))
        dst.setncattr('creation_date', datetime.utcnow().isoformat() + 'Z')
        dst.setncattr('dx_m', global_dx)
        dst.setncattr('gap_threshold_m', GAP_THRESHOLD_M)
        dst.setncattr('alt_deviation_max_m', ALT_DEVIATION_MAX)
        dst.setncattr('straightness_min', STRAIGHTNESS_MIN)
        dst.setncattr('min_leg_length_pts', MIN_LEG_LENGTH)
        dst.setncattr('max_nan_fraction', MAX_NAN_FRACTION)
        dst.setncattr('description',
            'Airplane legs from MAESTRO ATR-42, spatially regridded. '
            'Variables in standardized names (T, qv, h, u, v).')
        dst.setncattr('constants',
            f'CP={CP} J/kg/K, G={G} m/s2, LV={LV} J/kg')
        dst.setncattr('flight_selection', 'equatorial flights 6-27')

        # Dimensions
        dst.createDimension('leg',    n_legs)
        dst.createDimension('sample', max_samples)

        # ── Leg-level coordinate variables ───────────────────────────
        def _leg_var(name, data, units, long_name):
            v = dst.createVariable(name, 'f4', ('leg',))
            v[:] = data
            v.units = units
            v.long_name = long_name
            return v

        _leg_var('altitude',     [l['mean_alt'] for l in all_legs], 'm',   'mean altitude of leg')
        _leg_var('altitude_min', [l['min_alt']  for l in all_legs], 'm',   'min altitude of leg')
        _leg_var('altitude_max', [l['max_alt']  for l in all_legs], 'm',   'max altitude of leg')
        _leg_var('altitude_std', [l['std_alt']  for l in all_legs], 'm',   'std of altitude within leg')
        _leg_var('mean_tas',     [l['mean_tas'] for l in all_legs], 'm/s', 'mean true airspeed of leg')
        _leg_var('leg_length',   [l['n']        for l in all_legs], '1',   'number of spatial samples')
        _leg_var('nan_fraction', [l['nan_frac'] for l in all_legs], '1',   'max NaN fraction across analysis vars')

        v = dst.createVariable('flight_id', 'i4', ('leg',))
        v[:] = [l['flight_id'] for l in all_legs]
        v.units = '1'
        v.long_name = 'flight index (0-based)'

        # ── 2-D data variables (leg × sample) ────────────────────────
        fill = np.float32(np.nan)
        var_meta = {
            'T':  ('K',    'air temperature'),
            'qv': ('g/kg', 'water vapor mixing ratio'),
            'h':  ('kJ/kg','moist static energy'),
            'u':  ('m/s',  'eastward wind component'),
            'v':  ('m/s',  'northward wind component'),
        }
        for vname, (units, long_name) in var_meta.items():
            nv = dst.createVariable(vname, 'f4', ('leg', 'sample'),
                                    fill_value=fill)
            nv.units = units
            nv.long_name = long_name
            for i, leg in enumerate(all_legs):
                arr = leg[vname].astype(np.float32)
                n   = leg['n']
                nv[i, :n]  = arr
                nv[i, n:]  = fill

    print(f"\nWrote {OUTPUT_FILE}")
    print(f"  Dimensions: leg={n_legs}, sample={max_samples}")


if __name__ == '__main__':
    main()
