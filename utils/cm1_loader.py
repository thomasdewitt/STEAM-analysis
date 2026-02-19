import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (RCEMIP_DATA_DIR, CM1_LARGE_TIMESTEPS, CM1_SMALL_TIMESTEPS,
                    MAX_HEIGHT, MIN_HEIGHT, HORIZ_STRIDE, get_unit_factor)

# File naming pattern: CM1/{exp}/3D/CM1_{exp}_3D_allvars_{ts}.nc
_CM1_DEFAULT_TIMESTEPS = {
    'RCE_large300':     CM1_LARGE_TIMESTEPS,
    'RCE_small_les300': CM1_SMALL_TIMESTEPS,
}


def _get_filepath(experiment, ts):
    return f'{RCEMIP_DATA_DIR}/CM1/{experiment}/3D/CM1_{experiment}_3D_allvars_{ts}.nc'


def load_cm1_variable(variable, experiment='RCE_large300', timesteps=None, horiz_stride=None):
    """Load a CM1 RCEMIP 3D variable across timesteps.

    Parameters
    ----------
    variable : str
        Variable name in the NetCDF file (e.g. 'hus', 'ta', 'ua', 'wa').
    experiment : str
        'RCE_large300' or 'RCE_small_les300'.
    timesteps : list of str, optional
        Timestep strings to load (e.g. ['hour1860', 'hour2040']).
        Defaults to all available for the experiment.
    horiz_stride : int, optional
        Subsample every nth point along the j (y) axis. Defaults to HORIZ_STRIDE.

    Returns
    -------
    z : ndarray, shape (nz,)
    data_4d : ndarray, shape (n_timesteps, nj, ni, nz)
    dx : float
        Horizontal grid spacing in metres (accounting for horiz_stride).
    """
    if experiment not in _CM1_DEFAULT_TIMESTEPS:
        raise ValueError(f"Unknown experiment '{experiment}'. "
                         f"Choose from {list(_CM1_DEFAULT_TIMESTEPS)}")
    if timesteps is None:
        timesteps = _CM1_DEFAULT_TIMESTEPS[experiment]
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
            # File dims: (time, nk, nj, ni) = (time, z, y, x).
            # Transpose to (nj_strided, ni, nz) to match loader convention (z last).
            data = ds.variables[variable][0, :, ::horiz_stride, :]  # (nz, nj_strided, ni)
            data = data.transpose(1, 2, 0)                           # (nj_strided, ni, nz)
            data_list.append(data)

    data_4d = np.stack(data_list, axis=0)      # (n_timesteps, nj_strided, ni, nz)
    data_4d = data_4d * get_unit_factor(variable)
    return z, data_4d, dx


def load_cm1_variable_interpolated(variable, experiment='RCE_large300',
                                   timesteps=None, horiz_stride=None,
                                   min_height=None, max_height=None, vert_spacing=None):
    """Load CM1 variable and interpolate to a uniform vertical grid.

    Returns (z_uniform, data_interp, dx) with shape (n_timesteps, nj, ni, nz_uniform).
    """
    if min_height is None:
        min_height = MIN_HEIGHT
    if max_height is None:
        max_height = MAX_HEIGHT
    if vert_spacing is None:
        vert_spacing = 100

    z_orig, data_4d, dx = load_cm1_variable(variable, experiment, timesteps, horiz_stride)

    z_uniform = np.arange(min_height, max_height + vert_spacing, vert_spacing)
    interp_func = interp1d(z_orig, data_4d, axis=-1, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    data_interp = interp_func(z_uniform)

    return z_uniform, data_interp, dx
