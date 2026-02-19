import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (RCEMIP_DATA_DIR, SAM_RCEMIP_LARGE_TIMESTEPS, SAM_RCEMIP_SMALL_TIMESTEPS,
                    MAX_HEIGHT, MIN_HEIGHT, HORIZ_STRIDE, get_unit_factor)

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
        Variable name in the NetCDF file (e.g. 'hus', 'ta', 'ua', 'QV').
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
    if experiment not in _SAM_RCEMIP_FILE_PATTERNS:
        raise ValueError(f"Unknown experiment '{experiment}'. "
                         f"Choose from {list(_SAM_RCEMIP_FILE_PATTERNS)}")
    if timesteps is None:
        timesteps = _SAM_RCEMIP_DEFAULT_TIMESTEPS[experiment]
    if horiz_stride is None:
        horiz_stride = HORIZ_STRIDE

    data_list = []
    z = None

    for ts in timesteps:
        filepath = _get_filepath(experiment, ts)
        with nc.Dataset(filepath, 'r') as ds:
            if z is None:
                z = ds.variables['z'][:]
                x = ds.variables['x'][:]
                dx = float(x[1] - x[0]) * horiz_stride
            var_data = ds.variables[variable]
            # File dims: (time, z, y, x) or (z, y, x) for some vars in large domain.
            # Transpose to (ny_strided, nx, nz) to match loader convention (z last).
            if var_data.ndim == 4:
                data = var_data[0, :, ::horiz_stride, :]   # (nz, ny_strided, nx)
            else:
                data = var_data[:, ::horiz_stride, :]       # (nz, ny_strided, nx)
            data = data.transpose(1, 2, 0)                  # (ny_strided, nx, nz)
            data_list.append(data)

    data_4d = np.stack(data_list, axis=0)      # (n_timesteps, ny_strided, nx, nz)
    data_4d = data_4d * get_unit_factor(variable)
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
    if vert_spacing is None:
        vert_spacing = 100

    z_orig, data_4d, dx = load_sam_rcemip_variable(variable, experiment, timesteps, horiz_stride)

    z_uniform = np.arange(min_height, max_height + vert_spacing, vert_spacing)
    interp_func = interp1d(z_orig, data_4d, axis=-1, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    data_interp = interp_func(z_uniform)

    return z_uniform, data_interp, dx
