import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (STEAM_DATA_DIR, STEAM_FILE_PATTERN, MAX_HEIGHT, MIN_HEIGHT,
                    HORIZ_STRIDE, get_unit_factor, resolve_standard_variable)


_GROUP_SHORTCUTS = {
    'parent': [None],
    'root':   [None],
    '/':      [None],
    'strips': ['refinements/strip_center', 'refinements/strip_edge'],
    'cubes':  ['refinements/cube_center',  'refinements/cube_edge'],
}


def _resolve_group_spec(group):
    """Normalize a group argument into a list of group paths (None = root)."""
    if group is None:
        return [None]
    if isinstance(group, str):
        if group in _GROUP_SHORTCUTS:
            return _GROUP_SHORTCUTS[group]
        return [group]
    return [g if g not in _GROUP_SHORTCUTS else _GROUP_SHORTCUTS[g][0] for g in group]


def load_steam_variable(variable, seed_files=None, horiz_stride=None,
                        data_dir=None, file_pattern=None, group=None):
    """Load a STEAM 3D variable across seed files.

    Returns (z, data_4d, dx) where data_4d has shape (n_seeds*n_groups, nx, ny, nz).

    Parameters
    ----------
    variable : str
        Standardized name (qv, qt, h, T) or native STEAM name.
    data_dir : str or Path, optional
        Directory to search for files (used only when seed_files is None).
        Defaults to STEAM_DATA_DIR from config.
    file_pattern : str, optional
        Glob pattern for auto-discovery (used only when seed_files is None).
        Defaults to STEAM_FILE_PATTERN.
    group : str, list[str], or None
        netCDF group(s) to read. None (default) → root group (parent domain).
        Shortcuts: 'parent'/'root'/'/' (root), 'strips' (both strip subgroups),
        'cubes' (both cube subgroups). Explicit paths like
        'refinements/strip_center' also work. A list reads and concatenates
        multiple groups along the leading axis; all must share the same z grid
        and (nx, ny) shape.
    """
    native = resolve_standard_variable(variable, 'STEAM')
    if native is None:
        raise ValueError(f"Variable '{variable}' is not available for STEAM dataset.")
    variable = native
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

    groups = _resolve_group_spec(group)

    data_list = []
    z = None
    dx = None

    _base_dir = Path(data_dir) if data_dir is not None else Path(STEAM_DATA_DIR)
    for fpath in seed_files:
        filepath = Path(fpath) if Path(fpath).is_absolute() else _base_dir / fpath
        with nc.Dataset(str(filepath), 'r') as ds:
            for g in groups:
                grp = ds if g is None else ds[g]
                if z is None:
                    z = grp.variables['z'][:]
                    # Prefer the group's dx attribute; fall back to x[1]-x[0]
                    # (the coord diff fails when x has length 1).
                    if hasattr(grp, 'dx'):
                        dx = float(grp.dx)
                    else:
                        x = grp.variables['x'][:]
                        dx = float(x[1] - x[0])
                # Stride along y (axis 1); x (axis 0) stays full density — x is
                # the long horizontal axis in both parent and transposed strips.
                data = grp.variables[variable][:, ::horiz_stride, :]
                data_list.append(data)

    data_4d = np.stack(data_list, axis=0)
    data_4d = data_4d * get_unit_factor(variable)
    return z, data_4d, dx


def load_steam_variable_interpolated(variable, seed_files=None, horiz_stride=None,
                                     min_height=None, max_height=None, vert_spacing=None,
                                     data_dir=None, file_pattern=None, group=None):
    """Load STEAM variable and interpolate to uniform vertical grid.

    Returns (z_uniform, data_interp, dx) with shape
    (n_seeds*n_groups, nx, ny, nz_uniform). See `load_steam_variable` for the
    `group` argument.
    """
    if min_height is None:
        min_height = MIN_HEIGHT
    if max_height is None:
        max_height = MAX_HEIGHT
    z_orig, data_4d, dx = load_steam_variable(variable, seed_files, horiz_stride,
                                               data_dir=data_dir, file_pattern=file_pattern,
                                               group=group)
    if vert_spacing is None:
        vert_spacing = float(np.round(np.mean(np.diff(z_orig))))

    z_uniform = np.arange(min_height, max_height + vert_spacing, vert_spacing)
    interp_func = interp1d(z_orig, data_4d, axis=-1, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    data_interp = interp_func(z_uniform)

    return z_uniform, data_interp, dx
