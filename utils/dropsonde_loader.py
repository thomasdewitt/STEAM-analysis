import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from config import DROPSONDE_FILE, MAX_HEIGHT, MIN_HEIGHT, get_unit_factor


def load_dropsonde_variable(variable):
    """Load a dropsonde variable.

    Returns (z, data_2d) where data_2d has shape (n_sondes, n_alt).
    Data is returned in display units.
    """
    filepath = _ROOT / DROPSONDE_FILE
    with nc.Dataset(filepath, 'r') as ds:
        z = ds.variables['altitude'][:]
        data = ds.variables[variable][:]

    # Reshape to 4D: (n_sondes, 1, 1, n_alt) to match SAM/STEAM convention
    data = (data * get_unit_factor(variable))[:, np.newaxis, np.newaxis, :]
    return z, data, None  # no horizontal spacing


def load_dropsonde_variable_interpolated(variable, min_height=None, max_height=None,
                                         vert_spacing=None):
    """Load dropsonde variable and interpolate to uniform vertical grid.

    Returns (z_uniform, data_4d) with shape (n_sondes, 1, 1, nz_uniform).
    """
    if min_height is None:
        min_height = MIN_HEIGHT
    if max_height is None:
        max_height = MAX_HEIGHT
    if vert_spacing is None:
        vert_spacing = 100

    z_orig, data_4d, dx = load_dropsonde_variable(variable)

    z_uniform = np.arange(min_height, max_height + vert_spacing, vert_spacing)
    interp_func = interp1d(z_orig, data_4d, axis=-1, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    data_interp = interp_func(z_uniform)

    return z_uniform, data_interp, dx
