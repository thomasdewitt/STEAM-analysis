"""Visual test for 3-level recursive STEAM refinement.

Runs N coarse parent simulations (different seeds), each with two
recursive refinements zooming into a centered subdomain. Produces
one glimpse (top-down) image per level, with a box showing the
region that was refined into the next level.

Usage:
    python STEAM/visual_test_refine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, '/Users/thomas/code-and-data/turbulon-model')

import numpy as np
import netCDF4
import xarray as xr
import subprocess
from steam import simulate, refine
from steam.thermodynamics import recover_diagnostics

# ── Parent simulation parameters ─────────────────────────────────────────────
PROFILE_DATASET  = 'Dropsonde_extrap'
NSIMS            = 10
NX, NY           = 256, 256
DX, DY           = 15000.0, 15000.0
OUTER_SCALE      = DX * 256
SPHEROSCALE      = 20.0
DOMAIN_HEIGHT    = 20000.0
SPARSITY_FACTORS = (1, 1, 1)
SURFACE_PRESSURE = 101325.0
H_MAX = 400 * 1004
H_MIN = 250 * 1004
QT_MIN = 0
QT_MAX = 30 / 1000
N_SIZE_CLASSES = 15
BASE_SEED = 3

# ── Refinement parameters (per level) ────────────────────────────────────────
# Each entry: (cells_per_side, output_nx, n_size_classes, z_min, z_max)
#   cells_per_side: number of parent-grid cells to extract (centered)
#   output_nx: target output grid cells per side
#   n_size_classes: size classes for this refinement level
#   z_min, z_max: optional vertical bounds (m); None = inherit from parent
#
# Constraints: cells * parent_dx must be an integer multiple of parent's
# finest k. With these params:
#   L0 finest_k = 10,000m, dx=5000 → cells must be even
#   L1 finest_k ≈ 781m,   dx≈391  → cells must be even
REFINEMENTS = [
    (20, 256, 8, None, None),   
    (20, 256, 8, 0, 10000),     
    (150, 1024, 8, 0, 4000),     
]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parent / 'data'
PROFILES  = Path(__file__).resolve().parent.parent / 'data' / 'rcemip_profiles.nc'
OUTPUT_DIR = DATA_DIR


def compute_diagnostics_group(nc_path, group_name):
    """Run thermodynamic recovery on a NetCDF group and write results there."""
    ds = netCDF4.Dataset(nc_path, "r+")
    grp = ds[group_name]

    h = grp.variables["h"][:]
    qt = grp.variables["qt"][:]
    z = grp.variables["z"][:]
    sp = float(grp.surface_pressure)

    result = recover_diagnostics(h, qt, z, sp)

    nx, ny, nz = h.shape
    for name, (units, long_name) in [
        ("T", ("K", "temperature")),
        ("qv", ("kg/kg", "water vapor mixing ratio")),
        ("qc", ("kg/kg", "cloud liquid water mixing ratio")),
        ("qi", ("kg/kg", "cloud ice mixing ratio")),
        ("p", ("Pa", "pressure")),
    ]:
        if name not in grp.variables:
            v = grp.createVariable(name, "f4", ("x", "y", "z"),
                                   zlib=True, complevel=4,
                                   chunksizes=(min(64, nx), min(64, ny), nz))
            v.units = units
            v.long_name = long_name
        else:
            v = grp.variables[name]
        v[:] = result[name].astype(np.float32)

    ds.close()
    print(f"  Diagnostics written to '{group_name}'")


def run_glimpse(nc_path, output_dir, group=None, suffix=""):
    """Run glimpse and rename the output."""
    cmd = ["glimpse", str(nc_path), "-o", str(output_dir)]
    if group:
        cmd += ["--group", group]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    stem = nc_path.stem
    default_name = f"cloudyview_glimpse_top_view_{stem}.png"
    out_png = output_dir / default_name
    if suffix:
        renamed = output_dir / f"refine_glimpse_{suffix}.png"
        if renamed.exists():
            renamed.unlink()
        out_png.rename(renamed)
        out_png = renamed
    return out_png


def draw_child_box(glimpse_png, parent_nx, parent_ny, x_start, x_stop, y_start, y_stop):
    """Draw a rectangle on a glimpse image showing the child refinement region."""
    from PIL import Image, ImageDraw

    img = Image.open(glimpse_png)
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Approximate matplotlib plot area margins
    margin_l = int(w * 0.125)
    margin_r = int(w * 0.89)
    margin_t = int(h * 0.07)
    margin_b = int(h * 0.89)
    plot_w = margin_r - margin_l
    plot_h = margin_b - margin_t

    # Fractional position of the child region within the parent grid
    px_x = margin_l + int(x_start / parent_nx * plot_w)
    px_y = margin_t + int(y_start / parent_ny * plot_h)
    px_x2 = margin_l + int(x_stop / parent_nx * plot_w)
    px_y2 = margin_t + int(y_stop / parent_ny * plot_h)

    draw.rectangle([px_x, px_y, px_x2, px_y2], outline='#ff3333', width=3)
    img.save(glimpse_png)


def read_group_grid_info(nc_path, group='/'):
    """Read nx, dx, finest_k from a group."""
    ds = netCDF4.Dataset(nc_path, "r")
    grp = ds if group == '/' else ds[group]
    nx = len(grp.dimensions["x"])
    ny = len(grp.dimensions["y"])
    dx = float(grp.dx)
    dy = float(grp.dy)
    k_values = grp.variables["k_values"][:]
    finest_k = float(k_values[-1])
    ds.close()
    return nx, ny, dx, dy, finest_k


def run_one_simulation(sim_id, seed, h_profile, qt_profile, profile_dz):
    """Run parent + refinements for a single seed, render glimpses."""
    nc_path = DATA_DIR / f'refine_test_{sim_id:03d}.nc'

    # ── Level 0: Parent simulation ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Sim {sim_id}] Level 0 (parent): {NX}x{NY} @ dx={DX:.0f}m")
    print(f"  Domain: {NX*DX/1000:.0f} km x {NY*DY/1000:.0f} km")
    print(f"{'='*60}")
    simulate(
        h_profile=h_profile, qt_profile=qt_profile,
        nx=NX, ny=NY, dx=DX, dy=DY,
        outer_scale=OUTER_SCALE, spheroscale=SPHEROSCALE,
        domain_height=DOMAIN_HEIGHT, profile_dz=profile_dz,
        output_path=nc_path,
        sparsity_factors=SPARSITY_FACTORS,
        surface_pressure=SURFACE_PRESSURE,
        seed=seed,
        h_max=H_MAX, h_min=H_MIN,
        qt_min=QT_MIN, qt_max=QT_MAX,
        n_size_classes=N_SIZE_CLASSES,
    )
    from steam.thermodynamics import compute_diagnostics
    compute_diagnostics(nc_path)

    # ── Refinement levels ────────────────────────────────────────────────
    groups = ['/']
    # Store child region info per level so we can draw boxes
    child_regions = {}

    for level, (cells_requested, output_nx, n_cls, z_min, z_max) in enumerate(REFINEMENTS, start=1):
        parent_group = groups[-1]
        parent_nx, parent_ny, parent_dx, parent_dy, finest_k = \
            read_group_grid_info(nc_path, parent_group)

        # Snap cells to nearest valid value
        cells_per_tile = finest_k / parent_dx
        n_tiles = max(1, round(cells_requested * parent_dx / finest_k))
        cells = int(round(n_tiles * cells_per_tile))
        cells = min(cells, parent_nx)
        if cells != cells_requested:
            print(f"  (adjusted cells {cells_requested} → {cells} for "
                  f"integer-multiple constraint)")

        # Center the subdomain
        x_start = (parent_nx - cells) // 2
        x_stop = x_start + cells
        y_start = (parent_ny - cells) // 2
        y_stop = y_start + cells

        # Record child region for the parent's glimpse
        child_regions[parent_group] = (parent_nx, parent_ny, x_start, x_stop, y_start, y_stop)

        inner_extent_x = cells * parent_dx
        new_dx = inner_extent_x / output_nx
        new_dy = new_dx

        print(f"\n{'='*60}")
        print(f"[Sim {sim_id}] Level {level}: {cells}x{cells} cells from "
              f"{'root' if parent_group == '/' else parent_group}")
        print(f"  → {output_nx}x{output_nx} @ dx={new_dx:.1f}m "
              f"({parent_dx/new_dx:.0f}x finer, {DX/new_dx:.0f}x vs root)")
        print(f"  Extent: {inner_extent_x/1000:.1f} km, "
              f"slice [{x_start}:{x_stop}, {y_start}:{y_stop}]")
        print(f"{'='*60}")

        if finest_k <= 2 * new_dx:
            print(f"  SKIPPING: parent finest_k ({finest_k:.1f}m) <= 2*dx "
                  f"({2*new_dx:.1f}m), no room to refine further")
            continue

        refine(
            nc_path,
            x_start=x_start, x_stop=x_stop,
            y_start=y_start, y_stop=y_stop,
            dx=new_dx, dy=new_dy,
            parent_group=parent_group,
            n_size_classes=n_cls,
            seed=seed + 1000 * level,
            z_min=z_min,
            z_max=z_max,
        )

        group_name = f"refinements/r{level - 1}"
        compute_diagnostics_group(nc_path, group_name)
        groups.append(group_name)

    # ── Render glimpse for each level ────────────────────────────────────
    level_names = ['L0_root'] + [f'L{i}' for i in range(1, len(groups))]

    print(f"\n{'='*60}")
    print(f"[Sim {sim_id}] Rendering glimpses")
    print(f"{'='*60}")

    for group, name in zip(groups, level_names):
        grp_arg = None if group == '/' else group
        suffix = f"{name}_sim{sim_id:03d}"
        g_png = run_glimpse(nc_path, OUTPUT_DIR, group=grp_arg, suffix=suffix)

        # Draw box showing child region if this level was refined
        if group in child_regions:
            parent_nx, parent_ny, x_start, x_stop, y_start, y_stop = child_regions[group]
            draw_child_box(g_png, parent_nx, parent_ny, x_start, x_stop, y_start, y_stop)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load profiles
    ds = xr.open_dataset(PROFILES)
    prof = ds.sel(dataset=PROFILE_DATASET)
    h_profile = prof['mse'].values
    qt_profile = prof['qt'].values
    heights = ds['height'].values
    profile_dz = float(heights[1] - heights[0])
    mask = heights <= DOMAIN_HEIGHT
    h_profile = h_profile[mask]
    qt_profile = qt_profile[mask]

    for i in range(NSIMS):
        seed = BASE_SEED + i
        run_one_simulation(sim_id=i, seed=seed, h_profile=h_profile,
                           qt_profile=qt_profile, profile_dz=profile_dz)


if __name__ == '__main__':
    main()
