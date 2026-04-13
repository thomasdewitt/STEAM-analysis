import numpy as np
import netCDF4 as nc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from config import get_unit_factor, resolve_standard_variable

AIRPLANE_FILE = 'data/airplane_legs.nc'


def load_airplane_variable(variable, alt_min=None, alt_max=None):
    """Load a spatially-regridded airplane leg variable.

    Parameters
    ----------
    variable : str
        Standardized variable name: T, qv, h, u, v.
    alt_min, alt_max : float, optional
        Keep only legs whose mean altitude is within [alt_min, alt_max] metres.

    Returns
    -------
    altitude_per_leg : ndarray, shape (n_legs,)
        Mean altitude (m) for each retained leg.
    data_2d : ndarray, shape (n_legs, max_spatial_samples)
        NaN-padded spatial series in display units.
    dx : float
        Uniform spatial grid spacing in metres (from NC global attribute).
    """
    native = resolve_standard_variable(variable, 'airplane')
    if native is None:
        raise ValueError(
            f"Variable '{variable}' is not available for airplane dataset. "
            f"Available: T, qv, h, u, v."
        )

    filepath = _ROOT / AIRPLANE_FILE
    with nc.Dataset(str(filepath), 'r') as ds:
        dx        = float(ds.getncattr('dx_m'))
        altitudes = ds.variables['altitude'][:]          # (n_legs,)
        data_raw  = ds.variables[native][:]              # (n_legs, max_samples)

    # Optional altitude filter
    if alt_min is not None or alt_max is not None:
        lo = alt_min if alt_min is not None else -np.inf
        hi = alt_max if alt_max is not None else np.inf
        mask = (altitudes >= lo) & (altitudes <= hi)
        altitudes = altitudes[mask]
        data_raw  = data_raw[mask]

    # Apply unit conversion (airplane variables already in display units so
    # factor is 1.0 for all, but we honour get_unit_factor for consistency)
    factor   = get_unit_factor(f'airplane_{native}')
    data_out = np.array(data_raw, dtype=float) * factor

    return altitudes, data_out, dx
