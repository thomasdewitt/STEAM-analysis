import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STEAM_DATA_DIR, STEAM_FILE_PATTERN, MAX_HEIGHT, MIN_HEIGHT, HORIZ_STRIDE, get_unit_factor


def load_steam_variable(variable, seed_files=None, horiz_stride=None,
                        data_dir=None, file_pattern=None):
    """Load a STEAM 3D variable across seed files.

    Returns (z, data_4d) where data_4d has shape (n_seeds, nx, ny, nz).

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory to search for files (used only when seed_files is None).
        Defaults to STEAM_DATA_DIR from config.
    file_pattern : str, optional
        Glob pattern for auto-discovery (used only when seed_files is None).
        Defaults to 'steam_twpice_seed_*.nc'.
    """
    if seed_files is None:
        _dir = Path(data_dir) if data_dir is not None else Path(STEAM_DATA_DIR)
        _pat = file_pattern if file_pattern is not None else STEAM_FILE_PATTERN
        seed_files = sorted(_dir.glob(_pat))
        if not seed_files:
            raise FileNotFoundError(
                f"No STEAM files matching '{_pat}' in {_dir}\n"
                f"Set STEAM_DATA_DIR in config.py or pass data_dir= to the loader."
            )
    if horiz_stride is None:
        horiz_stride = HORIZ_STRIDE

    data_list = []
    z = None

    _base_dir = Path(data_dir) if data_dir is not None else Path(STEAM_DATA_DIR)
    for fpath in seed_files:
        filepath = Path(fpath) if Path(fpath).is_absolute() else _base_dir / fpath
        with nc.Dataset(str(filepath), 'r') as ds:
            if z is None:
                z = ds.variables['z'][:]
                x = ds.variables['x'][:]
                dx = float(x[1] - x[0]) * horiz_stride
            data = ds.variables[variable][:, ::horiz_stride, :]
            data_list.append(data)

    data_4d = np.stack(data_list, axis=0)
    data_4d = data_4d * get_unit_factor(variable)
    return z, data_4d, dx


def load_steam_variable_interpolated(variable, seed_files=None, horiz_stride=None,
                                     min_height=None, max_height=None, vert_spacing=None,
                                     data_dir=None, file_pattern=None):
    """Load STEAM variable and interpolate to uniform vertical grid.

    Returns (z_uniform, data_interp) with shape (n_seeds, nx, ny, nz_uniform).
    """
    if min_height is None:
        min_height = MIN_HEIGHT
    if max_height is None:
        max_height = MAX_HEIGHT
    if vert_spacing is None:
        vert_spacing = 100

    z_orig, data_4d, dx = load_steam_variable(variable, seed_files, horiz_stride,
                                               data_dir=data_dir, file_pattern=file_pattern)

    z_uniform = np.arange(min_height, max_height + vert_spacing, vert_spacing)
    interp_func = interp1d(z_orig, data_4d, axis=-1, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    data_interp = interp_func(z_uniform)

    return z_uniform, data_interp, dx
