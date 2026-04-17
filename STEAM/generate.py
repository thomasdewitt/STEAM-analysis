"""Nested STEAM simulation generator.

Hierarchy:
    parent (nx² domain, full height)
    └── strips (2 positions: center, edge)      — narrow in x, full in y
        └── cube (1 per strip, centered)        — narrow in x and y, optional z slab

Prints grid-size / memory estimates per level up front, then runs the
full tree. Tracks wall-clock runtime and peak RSS. Renders glimpses of
the parent and each cube leaf into ../renders/steam_refine/.

Refinement factors default to 128 because powers of 2 cascade cleanly
(finest_k = 4·dx at every level, so the next level's minimum narrow
cells = 4). Non-power-of-2 factors get auto-snapped per level.

Usage:
    python STEAM/generate.py
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from steam.simulate import simulate, refine
from steam.thermodynamics import compute_diagnostics, recover_diagnostics

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROFILE_DATASET  = 'Dropsonde_extrap'
BASE_SEED        = 30
NSEEDS           = 3

# Parent: domain size and outer_scale are fixed; change NX/NY to sweep resolution.
NX, NY           = 1024, 1024
DOMAIN_WIDTH     = 4_000_000.0          # m  (constant, 17 000 km)
DOMAIN_HEIGHT    = 20_000.0              # m
OUTER_SCALE      = DOMAIN_WIDTH / 2      # m  (half domain width)
N_SIZE_CLASSES   = 10
SPARSITY_FACTORS = (1, 1, 1)
SURFACE_PRESSURE = 101_325.0
H_MAX, H_MIN     = 400 * 1004, 250 * 1004
QT_MIN, QT_MAX   = 0.0, 30 / 1000

# Spheroscale: 'constant' (ls = SPHEROSCALE_CONST) or 'linear' (ramps with z).
SPHEROSCALE_MODE  = 'linear'
SPHEROSCALE_CONST = 10.0                 # m  (if 'constant')
SPHEROSCALE_SFC   = 100.0                # m  (if 'linear', at z = 0)
SPHEROSCALE_TOP   = 1.0                  # m  (if 'linear', at z = DOMAIN_HEIGHT)

# Grid anisotropy: 'canonical' | 'piecewise_isotropic_below_spheroscale'
ANISOTROPY = 'piecewise_isotropic_below_spheroscale'

# Refinement levels. Each level carves a subdomain of its parent.
# narrow_cells: requested cells in the narrow axis of the parent grid
#               (auto-snapped to a valid integer-tile count).
# long_cells:   cells in the long axis; None means the full parent extent.
# refine:       refinement factor (output_nx/cells in the narrow axis).
# positions:    sequence of narrow-axis positions — 'center' or 'edge'.
STRIP = dict(narrow_cells=2, long_cells=None, refine=16, n_classes=6,
             positions=('center', 'edge'), z_min=None, z_max=None)
# CUBE is always centered in both axes of its parent strip.
CUBE  = dict(narrow_cells=4, long_cells=4,    refine=256, n_classes=8,
             z_min=1_000.0, z_max=2_000.0)

# Paths
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).resolve().parent / 'data'
RENDER_DIR = REPO_ROOT / 'renders' / 'steam_refine'
PROFILES   = REPO_ROOT / 'data' / 'rcemip_profiles.nc'
NC_STEM    = 'nested_refine'   # per-seed file: {NC_STEM}_seed{N:03d}.nc
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_spheroscale_profile(heights: np.ndarray) -> np.ndarray:
    if SPHEROSCALE_MODE == 'constant':
        return np.full_like(heights, SPHEROSCALE_CONST, dtype=np.float64)
    if SPHEROSCALE_MODE == 'linear':
        return np.interp(heights, [0.0, DOMAIN_HEIGHT],
                         [SPHEROSCALE_SFC, SPHEROSCALE_TOP]).astype(np.float64)
    raise ValueError(f"Unknown SPHEROSCALE_MODE: {SPHEROSCALE_MODE!r}")


def k_z(k: float, ls: float) -> float:
    """Vertical scale at horizontal scale k. Matches steam._k_z."""
    canonical = ls * (k / ls) ** (5 / 9)
    if ANISOTROPY == 'piecewise_isotropic_below_spheroscale' and k < ls:
        return k
    return canonical


def snap_cells(requested: int, parent_dx: float, finest_k: float,
               parent_extent_cells: int) -> int:
    """Snap requested cells to an integer multiple of finest_k/parent_dx."""
    n_tiles = max(1, round(requested * parent_dx / finest_k))
    cells_per_tile = finest_k / parent_dx
    cells = int(round(n_tiles * cells_per_tile))
    return min(cells, parent_extent_cells)


def estimate_nz(finest_k: float, ls_profile: np.ndarray,
                z_min: float, z_max: float) -> int:
    """Rough nz estimate: domain height / (k_z at finest k, avg over z)."""
    ls_avg = float(np.mean(ls_profile))
    dz_finest = k_z(finest_k, ls_avg) / 2.0
    return max(1, int(np.ceil((z_max - z_min) / dz_finest)))


def fmt_mem(cells: int, nz: int, peak_factor: float = 2.5) -> str:
    steady_gb = cells * nz * 4 * 2 / 1e9          # 2 float32 fields
    return f"steady {steady_gb:.2f} GB / peak ~{steady_gb * peak_factor:.1f} GB"


def print_estimates(ls_profile: np.ndarray) -> None:
    print('\n=== Grid-size estimates ===')
    dx = DOMAIN_WIDTH / NX
    finest_k = OUTER_SCALE / 2 ** (N_SIZE_CLASSES - 1)
    nz = estimate_nz(finest_k, ls_profile, 0.0, DOMAIN_HEIGHT)
    print(f'Parent:     {NX}×{NY} × nz~{nz}   dx={dx/1000:.2f}km  '
          f'L={OUTER_SCALE/1000:.0f}km  finest_k={finest_k/1000:.2f}km  '
          f'[{fmt_mem(NX * NY, nz)}]')

    # Walk the tree symbolically. Each level inherits outer_scale = parent_finest_k.
    p_dx, p_fk = dx, finest_k
    for label, cfg, parent_narrow, parent_long in [
        ('Strip',   STRIP, NX, NY),
        ('Cube',    CUBE,  None, None),   # parent is the strip, computed below
    ]:
        if parent_narrow is None:
            parent_narrow = narrow_grid_prev
            parent_long   = long_grid_prev

        requested_narrow = cfg['narrow_cells']
        requested_long   = cfg['long_cells'] or parent_long
        cells_narrow = snap_cells(requested_narrow, p_dx, p_fk, parent_narrow)
        cells_long   = snap_cells(requested_long,   p_dx, p_fk, parent_long)

        new_dx = p_dx / cfg['refine']
        new_outer = p_fk
        new_finest_k = new_outer / 2 ** (cfg['n_classes'] - 1)
        narrow_grid = int(round(cells_narrow * cfg['refine']))
        long_grid   = int(round(cells_long   * cfg['refine']))

        z0 = 0.0            if cfg['z_min'] is None else cfg['z_min']
        z1 = DOMAIN_HEIGHT  if cfg['z_max'] is None else cfg['z_max']
        nz_i = estimate_nz(new_finest_k, ls_profile, z0, z1)

        snap_note = ''
        if cells_narrow != requested_narrow or cells_long != requested_long:
            snap_note = f'  (cells snap {requested_narrow}→{cells_narrow}, ' \
                        f'{requested_long}→{cells_long})'
        nyq_warn = '' if new_finest_k >= 2 * new_dx else \
                   f'  WARNING: finest_k {new_finest_k:.2f} < 2·dx {2*new_dx:.2f}'

        print(f'{label:<10} {narrow_grid}×{long_grid} × nz~{nz_i}   '
              f'dx={new_dx:.2f}m  L={new_outer/1000:.2f}km  '
              f'finest_k={new_finest_k:.2f}m  '
              f'[{fmt_mem(narrow_grid * long_grid, nz_i)}]'
              f'{snap_note}{nyq_warn}')

        narrow_grid_prev, long_grid_prev = narrow_grid, long_grid
        p_dx, p_fk = new_dx, new_finest_k
    print()


def read_group_info(nc_path: Path, group: str) -> tuple[int, int, float, float]:
    """Return (nx, ny, dx, finest_k) for a group."""
    ds = netCDF4.Dataset(nc_path, 'r')
    grp = ds if group == '/' else ds[group]
    nx = len(grp.dimensions['x'])
    ny = len(grp.dimensions['y'])
    dx = float(grp.dx)
    k_vals = grp.variables['k_values'][:]
    finest_k = float(k_vals.min())
    ds.close()
    return nx, ny, dx, finest_k


_DIAG_ATTRS = {
    'T':  ('K',     'temperature'),
    'qv': ('kg/kg', 'water vapor mixing ratio'),
    'qc': ('kg/kg', 'cloud liquid water mixing ratio'),
    'qi': ('kg/kg', 'cloud ice mixing ratio'),
    'p':  ('Pa',    'pressure'),
}


def compute_diagnostics_for_group(nc_path: Path, group: str) -> None:
    """recover_diagnostics + write T/qv/qc/qi/p into the group (public API
    compute_diagnostics only supports the root group)."""
    with netCDF4.Dataset(nc_path, 'r+') as ds:
        grp = ds if group == '/' else ds[group]
        h = grp.variables['h'][:]
        qt = grp.variables['qt'][:]
        z = grp.variables['z'][:]
        sp = float(grp.surface_pressure)
        diag = recover_diagnostics(h, qt, z, sp)
        for name, arr in diag.items():
            if name in grp.variables:
                v = grp.variables[name]
            else:
                v = grp.createVariable(name, 'f4', ('x', 'y', 'z'),
                                       zlib=True, complevel=4)
                units, long_name = _DIAG_ATTRS[name]
                v.units = units
                v.long_name = long_name
            v[:] = arr


def position_slice(axis_len: int, cells: int, position: str) -> tuple[int, int]:
    """Return (start, stop) for a narrow region at 'center' or 'edge'."""
    if position == 'center':
        start = (axis_len - cells) // 2
    elif position == 'edge':
        start = 0
    else:
        raise ValueError(f"Unknown position {position!r}")
    return start, start + cells


def run_refine(nc_path: Path, parent_group: str, cfg: dict,
               narrow_axis_positions: tuple[str, str | None],
               output_group: str, level_idx: int,
               seed: int) -> tuple[str, int, int]:
    """Carve a subdomain out of parent_group and run refine().

    narrow_axis_positions: (x_position, y_position_or_None).
      If y_position is None, the region spans the full parent in y.
    """
    parent_nx, parent_ny, parent_dx, parent_fk = read_group_info(nc_path, parent_group)

    # Narrow (x) cells
    cells_x_req = cfg['narrow_cells']
    cells_x = snap_cells(cells_x_req, parent_dx, parent_fk, parent_nx)
    x_start, x_stop = position_slice(parent_nx, cells_x, narrow_axis_positions[0])

    # Long (y) cells: full extent if long_cells is None, else narrow in y too
    if cfg['long_cells'] is None:
        y_start, y_stop = 0, parent_ny
    else:
        cells_y_req = cfg['long_cells']
        cells_y = snap_cells(cells_y_req, parent_dx, parent_fk, parent_ny)
        y_pos = narrow_axis_positions[1] or 'center'
        y_start, y_stop = position_slice(parent_ny, cells_y, y_pos)

    new_dx = parent_dx / cfg['refine']
    print(f"  refine parent='{parent_group}' → '{output_group}'  "
          f"x[{x_start}:{x_stop}] y[{y_start}:{y_stop}]  "
          f"new_dx={new_dx:.3f}m  classes={cfg['n_classes']}")

    refine(
        nc_path,
        x_start=x_start, x_stop=x_stop,
        y_start=y_start, y_stop=y_stop,
        dx=new_dx, dy=new_dx,
        parent_group=parent_group,
        output_group=output_group,
        n_size_classes=cfg['n_classes'],
        seed=seed + 1000 * level_idx + hash(output_group) % 997,
        z_min=cfg['z_min'], z_max=cfg['z_max'],
        anisotropy=ANISOTROPY,
    )
    compute_diagnostics_for_group(nc_path, output_group)
    return output_group, x_stop - x_start, y_stop - y_start


def run_glimpse(nc_path: Path, group: str | None, suffix: str) -> None:
    cmd = ['glimpse', str(nc_path), '-o', str(RENDER_DIR)]
    if group:
        cmd += ['--group', group]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    default = RENDER_DIR / f"cloudyview_glimpse_top_view_{nc_path.stem}.png"
    renamed = RENDER_DIR / f"glimpse_{suffix}.png"
    if renamed.exists():
        renamed.unlink()
    default.rename(renamed)


# ── Per-seed pipeline ─────────────────────────────────────────────────────────
def run_one_seed(seed: int, h_profile: np.ndarray, qt_profile: np.ndarray,
                 profile_dz: float, ls_profile: np.ndarray) -> float:
    """Run parent + strips + cubes + glimpses for a single seed.
    Returns wall-clock seconds for this seed."""
    t0 = time.monotonic()
    nc_path = DATA_DIR / f'{NC_STEM}_seed{seed:03d}.nc'
    if nc_path.exists():
        nc_path.unlink()

    print(f"\n{'#'*72}\n# Seed {seed} → {nc_path.name}\n{'#'*72}")
    print(f"=== Parent: {NX}×{NY} @ dx={DOMAIN_WIDTH/NX/1000:.2f}km ===")
    simulate(
        h_profile=h_profile, qt_profile=qt_profile,
        nx=NX, ny=NY,
        dx=DOMAIN_WIDTH / NX, dy=DOMAIN_WIDTH / NY,
        outer_scale=OUTER_SCALE,
        spheroscale=ls_profile,
        domain_height=DOMAIN_HEIGHT,
        profile_dz=profile_dz,
        output_path=nc_path,
        sparsity_factors=SPARSITY_FACTORS,
        surface_pressure=SURFACE_PRESSURE,
        seed=seed,
        h_max=H_MAX, h_min=H_MIN,
        qt_min=QT_MIN, qt_max=QT_MAX,
        n_size_classes=N_SIZE_CLASSES,
        anisotropy=ANISOTROPY,
    )
    compute_diagnostics(nc_path)

    cube_groups: list[str] = []
    for strip_pos in STRIP['positions']:
        strip_group = f'refinements/strip_{strip_pos}'
        print(f"\n=== Strip [{strip_pos}] ===")
        run_refine(nc_path, '/', STRIP, (strip_pos, None),
                   strip_group, level_idx=1, seed=seed)

        cube_group = f'refinements/cube_{strip_pos}'
        print(f"\n=== Cube [{strip_pos}] ===")
        run_refine(nc_path, strip_group, CUBE, ('center', 'center'),
                   cube_group, level_idx=2, seed=seed)
        cube_groups.append(cube_group)

    print(f"\n=== Rendering glimpses to {RENDER_DIR} ===")
    run_glimpse(nc_path, group=None, suffix=f'parent_seed{seed:03d}')
    for g in cube_groups:
        tag = g.replace('refinements/', '').replace('/', '_')
        run_glimpse(nc_path, group=g, suffix=f'{tag}_seed{seed:03d}')

    elapsed = time.monotonic() - t0
    print(f"\nSeed {seed} runtime: {elapsed/60:.2f} min ({elapsed:.1f} s)")
    return elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.monotonic()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(PROFILES)
    prof = ds.sel(dataset=PROFILE_DATASET)
    heights    = ds['height'].values
    h_profile  = prof['mse'].values
    qt_profile = prof['qt'].values
    profile_dz = float(heights[1] - heights[0])
    mask = heights <= DOMAIN_HEIGHT
    heights    = heights[mask]
    h_profile  = h_profile[mask]
    qt_profile = qt_profile[mask]

    ls_profile = build_spheroscale_profile(heights)
    print(f'Config: ANISOTROPY={ANISOTROPY!r}, '
          f'SPHEROSCALE_MODE={SPHEROSCALE_MODE!r}, '
          f'ls(sfc)={ls_profile[0]:.2f}m  ls(top)={ls_profile[-1]:.2f}m, '
          f'NSEEDS={NSEEDS}')
    print_estimates(ls_profile)

    for i in range(NSEEDS):
        run_one_seed(BASE_SEED + i, h_profile, qt_profile, profile_dz, ls_profile)

    elapsed = time.monotonic() - t0
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is in kilobytes.
    print(f"\n{'='*72}")
    print(f"Total runtime ({NSEEDS} seed{'s' if NSEEDS != 1 else ''}): "
          f"{elapsed/60:.2f} min ({elapsed:.1f} s)")
    print(f"Peak RSS: {peak_rss_kb/1e6:.2f} GB")


if __name__ == '__main__':
    main()
