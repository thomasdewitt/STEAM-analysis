import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (SAM_TWPICE_DATA_DIR, SAM_TWPICE_TIMESTEPS, MAX_HEIGHT, MIN_HEIGHT,
                    HORIZ_STRIDE, get_unit_factor, resolve_standard_variable)


def load_sam_twpice_variable(variable, timesteps=None, horiz_stride=None):
    """Load a SAM TWPICE 3D variable across timesteps.

    Parameters
    ----------
    variable : str
        Standardized name (qv, qt, h, T, u, v) or native SAM TWPICE name.

    Returns (z, data_4d, dx) where data_4d has shape (n_timesteps, ny, nx, nz).
    """
    native = resolve_standard_variable(variable, 'SAM_TWPICE')
    if native is None:
        raise ValueError(f"Variable '{variable}' is not available for SAM_TWPICE dataset.")
    variable = native
    if timesteps is None:
        timesteps = SAM_TWPICE_TIMESTEPS
    if horiz_stride is None:
        horiz_stride = HORIZ_STRIDE

    data_list = []
    z = None

    for ts in timesteps:
        filepath = f'{SAM_TWPICE_DATA_DIR}/OUT_3D.{variable}/TWPICE_LPT_3D_{variable}_{ts}.nc'
        with nc.Dataset(filepath, 'r') as ds:
            if z is None:
                z = ds.variables['z'][:]
                x = ds.variables['x'][:]
                dx = float(x[1] - x[0])
            data = ds.variables[variable][0, :, ::horiz_stride, :]
            data_list.append(data)

    data_4d = np.stack(data_list, axis=0)
    data_4d = data_4d * get_unit_factor(variable)
    return z, data_4d, dx


def load_sam_twpice_variable_interpolated(variable, timesteps=None, horiz_stride=None,
                                          min_height=None, max_height=None, vert_spacing=None):
    """Load SAM TWPICE variable and interpolate to uniform vertical grid.

    Returns (z_uniform, data_interp, dx) with shape (n_timesteps, ny, nx, nz_uniform).
    """
    if min_height is None:
        min_height = MIN_HEIGHT
    if max_height is None:
        max_height = MAX_HEIGHT
    z_orig, data_4d, dx = load_sam_twpice_variable(variable, timesteps, horiz_stride)
    if vert_spacing is None:
        vert_spacing = float(np.round(np.mean(np.diff(z_orig))))

    z_uniform = np.arange(min_height, max_height + vert_spacing, vert_spacing)
    interp_func = interp1d(z_orig, data_4d, axis=-1, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    data_interp = interp_func(z_uniform)

    return z_uniform, data_interp, dx
