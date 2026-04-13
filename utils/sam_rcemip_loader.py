import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (RCEMIP_DATA_DIR, SAM_RCEMIP_LARGE_TIMESTEPS, SAM_RCEMIP_SMALL_TIMESTEPS,
                    MAX_HEIGHT, MIN_HEIGHT, HORIZ_STRIDE, get_unit_factor,
                    resolve_standard_variable)

_CP = 1004.0   # J/(kg·K)
_G  = 9.81     # m/s²
_LV = 2.5e6    # J/kg

# File naming differs between experiments
_SAM_RCEMIP_FILE_PATTERNS = {
    'RCE_large300':     'SAM_CRM/{exp}/3D/SAM_CRM_{exp}_3D_{ts}.nc',
    'RCE_small_les300': 'SAM_CRM/{exp}/3D/RCEMIP_SST300_480x480x146-200m-2s_480_{ts}.nc',
}

_SAM_RCEMIP_DEFAULT_TIMESTEPS = {
    'RCE_large300':     SAM_RCEMIP_LARGE_TIMESTEPS,
    'RCE_small_les300': SAM_RCEMIP_SMALL_TIMESTEPS,
}


def _get_filepath(experiment, ts):
    pattern = _SAM_RCEMIP_FILE_PATTERNS[experiment]
    return f'{RCEMIP_DATA_DIR}/{pattern.format(exp=experiment, ts=ts)}'


def load_sam_rcemip_variable(variable, experiment='RCE_large300', timesteps=None, horiz_stride=None):
    """Load a SAM RCEMIP 3D variable across timesteps.

    Parameters
    ----------
    variable : str
        Standardized name (qv, T, u, v) or native SAM RCEMIP name (hus, ta, QV...).
    experiment : str
        'RCE_large300' or 'RCE_small_les300'.
    timesteps : list of str, optional
        Timestep strings to load. Defaults to all available for the experiment.
    horiz_stride : int, optional
        Subsample every nth point along the y axis. Defaults to HORIZ_STRIDE.

    Returns
    -------
    z : ndarray, shape (nz,)
    data_4d : ndarray, shape (n_timesteps, ny, nx, nz)
    dx : float
        Horizontal grid spacing in metres (accounting for horiz_stride).
    """
    # Resolve standard variable name to SAM RCEMIP native name
    dataset_key = 'SAM_RCEMIP_large' if 'large' in experiment else 'SAM_RCEMIP_small'
    native = resolve_standard_variable(variable, dataset_key)
    if native is None:
        raise ValueError(
            f"Variable '{variable}' is not available for SAM_RCEMIP {experiment}. "
            f"Try 'qv' or 'T' instead."
        )

    if experiment not in _SAM_RCEMIP_FILE_PATTERNS:
        raise ValueError(f"Unknown experiment '{experiment}'. "
                         f"Choose from {list(_SAM_RCEMIP_FILE_PATTERNS)}")
    if timesteps is None:
        timesteps = _SAM_RCEMIP_DEFAULT_TIMESTEPS[experiment]
    if horiz_stride is None:
        horiz_stride = HORIZ_STRIDE

    if native == '__mse_computed__':
        return _load_sam_rcemip_mse(experiment, dataset_key, timesteps, horiz_stride)

    variable = native

    data_list = []
    z = None

    for ts in timesteps:
        filepath = _get_filepath(experiment, ts)
        with nc.Dataset(filepath, 'r') as ds:
            if z is None:
                z = ds.variables['z'][:]
                x = ds.variables['x'][:]
                dx = float(x[1] - x[0])
            var_data = ds.variables[variable]
            # File dims: (time, z, y, x) or (z, y, x) for some vars in large domain.
            # Stride on y; transpose to (nx, ny_strided, nz) so non-strided dim is axis=1.
            if var_data.ndim == 4:
                data = var_data[0, :, ::horiz_stride, :]   # (nz, ny_strided, nx)
            else:
                data = var_data[:, ::horiz_stride, :]       # (nz, ny_strided, nx)
            data = data.transpose(2, 1, 0)                  # (nx, ny_strided, nz)
            data_list.append(data)

    data_4d = np.stack(data_list, axis=0)      # (n_timesteps, nx, ny_strided, nz)
    data_4d = data_4d * get_unit_factor(variable)
    return z, data_4d, dx


def _load_sam_rcemip_mse(experiment, dataset_key, timesteps, horiz_stride):
    """Compute MSE = Cp*T + g*z + Lv*qv and return in display units (K)."""
    T_native  = resolve_standard_variable('T',  dataset_key)
    qv_native = resolve_standard_variable('qv', dataset_key)

    def _read(ds, vname):
        vd = ds.variables[vname]
        if vd.ndim == 4:
            d = vd[0, :, ::horiz_stride, :]
        else:
            d = vd[:, ::horiz_stride, :]
        return np.array(d, dtype=float).transpose(2, 1, 0)  # (nx, ny_strided, nz)

    data_list = []
    z = None

    for ts in timesteps:
        filepath = _get_filepath(experiment, ts)
        with nc.Dataset(filepath, 'r') as ds:
            if z is None:
                z = ds.variables['z'][:]
                x = ds.variables['x'][:]
                dx = float(x[1] - x[0])
            T_K  = _read(ds, T_native)   # K
            qv   = _read(ds, qv_native)  # kg/kg
            z_3d = z[np.newaxis, np.newaxis, :]
            mse  = _CP * T_K + _G * z_3d + _LV * qv  # J/kg
            data_list.append(mse)

    data_4d = np.stack(data_list, axis=0) * get_unit_factor('h')  # → K
    return z, data_4d, dx


def load_sam_rcemip_variable_interpolated(variable, experiment='RCE_large300',
                                          timesteps=None, horiz_stride=None,
                                          min_height=None, max_height=None, vert_spacing=None):
    """Load SAM RCEMIP variable and interpolate to a uniform vertical grid.

    Returns (z_uniform, data_interp, dx) with shape (n_timesteps, ny, nx, nz_uniform).
    """
    if min_height is None:
        min_height = MIN_HEIGHT
    if max_height is None:
        max_height = MAX_HEIGHT
    z_orig, data_4d, dx = load_sam_rcemip_variable(variable, experiment, timesteps, horiz_stride)
    if vert_spacing is None:
        vert_spacing = float(np.round(np.mean(np.diff(z_orig))))

    z_uniform = np.arange(min_height, max_height + vert_spacing, vert_spacing)
    interp_func = interp1d(z_orig, data_4d, axis=-1, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    data_interp = interp_func(z_uniform)

    return z_uniform, data_interp, dx
