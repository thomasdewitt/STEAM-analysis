"""Visual test for 3-level recursive STEAM refinement.

Runs a coarse parent simulation, then two recursive refinements, each
zooming into a centered subdomain of the previous level. Produces:
  - glimpse (top-down) images for each level
  - witness (ground-perspective) renders for each level

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
NX, NY           = 256, 256
DX, DY           = 5000.0, 5000.0
OUTER_SCALE      = DX * 256
SPHEROSCALE      = 100.0
DOMAIN_HEIGHT    = 20000.0
SPARSITY_FACTORS = (1, 1, 1)
SURFACE_PRESSURE = 101325.0
H_MAX = 400 * 1004
H_MIN = 250 * 1004
QT_MIN = 0
QT_MAX = 30 / 1000
N_SIZE_CLASSES = 30
SEED = 3

# ── Refinement parameters (per level) ────────────────────────────────────────
# Each entry: (cells_per_side, output_nx, n_size_classes)
#   cells_per_side: number of parent-grid cells to extract (centered)
#   output_nx: target output grid cells per side
#   n_size_classes: size classes for this refinement level
#
# Constraints: cells * parent_dx must be an integer multiple of parent's
# finest k. With these params:
#   L0 finest_k = 10,000m, dx=5000 → cells must be even
#   L1 finest_k ≈ 781m,   dx≈391  → cells must be even
REFINEMENTS = [
    (20, 256, 8),   # L1: 20 root cells → 256, dx≈391m (13x finer)
    (40, 256, 8),   # L2: 40 L1 cells  → 256, dx≈61m  (82x total)
]

# ── Witness camera parameters (tweak these!) ─────────────────────────────────
# Camera position in relative coords: ±1 = domain edge, z: -1=ground, +1=top
CAMERA_POSITION  = (0.0, -0.95, -0.999)
CAMERA_AZIMUTH   = 0       # degrees: 0=North, 90=East, 180=South, 270=West
CAMERA_ELEVATION = 25      # degrees above horizon
CAMERA_FOV       = 100     # field of view in degrees
WITNESS_QUALITY  = 'high'  # min, low, medium, high
WITNESS_SIZE     = (1600, 1000)  # or None to use quality preset

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


def run_cloudyview(nc_path, tool, output_dir, group=None, suffix=""):
    """Run a cloudyview tool (glimpse or witness) and rename the output."""
    cmd = [tool, str(nc_path)]
    if tool == "witness":
        cmd.append(WITNESS_QUALITY)
    cmd += ["-o", str(output_dir)]
    if group:
        cmd += ["--group", group]
    if tool == "witness":
        cx, cy, cz = CAMERA_POSITION
        cmd += ["--camera-position", str(cx), str(cy), str(cz)]
        cmd += ["--camera-azimuth", str(CAMERA_AZIMUTH)]
        cmd += ["--camera-elevation", str(CAMERA_ELEVATION)]
        cmd += ["--fov", str(CAMERA_FOV)]
        if WITNESS_SIZE:
            cmd += ["--size", str(WITNESS_SIZE[0]), str(WITNESS_SIZE[1])]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    stem = nc_path.stem
    default_name = {
        "glimpse": f"cloudyview_glimpse_top_view_{stem}.png",
        "witness": f"witness_{stem}.png",
    }[tool]
    out_png = output_dir / default_name
    if suffix:
        renamed = output_dir / f"refine_{tool}_{suffix}.png"
        if renamed.exists():
            renamed.unlink()
        out_png.rename(renamed)
        out_png = renamed
    return out_png


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


def build_nested_composite(nc_path, glimpse_pngs, groups, output_dir):
    """Build a high-res composite with all refinement levels nested in root.

    Each refinement is placed at its correct position relative to the root
    domain by chaining parent_x_offset / parent_y_offset through the group
    hierarchy. The root glimpse is upscaled to 4000px for clarity.
    """
    from PIL import Image, ImageDraw

    # Read root domain extent
    ds = netCDF4.Dataset(nc_path, "r")
    root_nx = len(ds.dimensions["x"])
    root_ny = len(ds.dimensions["y"])
    root_dx = float(ds.dx)
    root_dy = float(ds.dy)
    domain_x = root_nx * root_dx
    domain_y = root_ny * root_dy

    # For each refinement group, compute absolute offset and extent
    # relative to root domain (chain through parent hierarchy)
    group_info = {}
    for group in groups[1:]:  # skip root
        grp = ds[group]
        x_coords = grp.variables["x"][:]
        y_coords = grp.variables["y"][:]
        extent_x = float(x_coords[-1] - x_coords[0]) + float(grp.dx)
        extent_y = float(y_coords[-1] - y_coords[0]) + float(grp.dy)
        # x_coords[0] is the absolute offset (set in refine())
        abs_x = float(x_coords[0])
        abs_y = float(y_coords[0])
        group_info[group] = (abs_x, abs_y, extent_x, extent_y)
    ds.close()

    # Upscale root image to high resolution
    target_size = 4000
    root_img = Image.open(glimpse_pngs[0])
    aspect = root_img.size[0] / root_img.size[1]
    if aspect >= 1:
        new_w, new_h = target_size, int(target_size / aspect)
    else:
        new_w, new_h = int(target_size * aspect), target_size
    composite = root_img.resize((new_w, new_h), Image.LANCZOS)
    draw = ImageDraw.Draw(composite)

    # Detect plot area bounds from the image
    # glimpse matplotlib: approximate margins
    margin_l = int(new_w * 0.125)
    margin_r = int(new_w * 0.89)
    margin_t = int(new_h * 0.07)
    margin_b = int(new_h * 0.89)
    plot_w = margin_r - margin_l
    plot_h = margin_b - margin_t

    colors = ['#ff3333', '#33ff33', '#3399ff']

    for i, (group, g_png) in enumerate(zip(groups[1:], glimpse_pngs[1:])):
        abs_x, abs_y, extent_x, extent_y = group_info[group]

        frac_x = extent_x / domain_x
        frac_y = extent_y / domain_y
        offset_frac_x = abs_x / domain_x
        offset_frac_y = abs_y / domain_y

        px_x = margin_l + int(offset_frac_x * plot_w)
        px_y = margin_t + int(offset_frac_y * plot_h)
        px_w = max(1, int(frac_x * plot_w))
        px_h = max(1, int(frac_y * plot_h))

        # Draw border
        color = colors[i % len(colors)]
        border = 3
        draw.rectangle([px_x - border, px_y - border,
                         px_x + px_w + border - 1, px_y + px_h + border - 1],
                        outline=color, width=border)

        # Paste refined glimpse
        fine_img = Image.open(g_png)
        fine_resized = fine_img.resize((px_w, px_h), Image.LANCZOS)
        composite.paste(fine_resized, (px_x, px_y))

    out_path = output_dir / "refine_composite_all_levels.png"
    composite.save(out_path, quality=95)
    print(f"  Nested composite: {out_path}")


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

    nc_path = DATA_DIR / 'refine_test.nc'

    # ── Level 0: Parent simulation ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Level 0 (parent): {NX}x{NY} @ dx={DX:.0f}m")
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
        seed=SEED,
        h_max=H_MAX, h_min=H_MIN,
        qt_min=QT_MIN, qt_max=QT_MAX,
        n_size_classes=N_SIZE_CLASSES,
    )
    from steam.thermodynamics import compute_diagnostics
    compute_diagnostics(nc_path)

    # ── Refinement levels ────────────────────────────────────────────────────
    groups = ['/']
    for level, (cells, output_nx, n_cls) in enumerate(REFINEMENTS, start=1):
        parent_group = groups[-1]
        parent_nx, parent_ny, parent_dx, parent_dy, finest_k = \
            read_group_grid_info(nc_path, parent_group)

        # Center the subdomain
        x_start = (parent_nx - cells) // 2
        x_stop = x_start + cells
        y_start = (parent_ny - cells) // 2
        y_stop = y_start + cells

        inner_extent_x = cells * parent_dx
        new_dx = inner_extent_x / output_nx
        new_dy = new_dx

        print(f"\n{'='*60}")
        print(f"Level {level}: {cells}x{cells} cells from "
              f"{'root' if parent_group == '/' else parent_group}")
        print(f"  → {output_nx}x{output_nx} @ dx={new_dx:.1f}m "
              f"({parent_dx/new_dx:.0f}x finer, {DX/new_dx:.0f}x vs root)")
        print(f"  Extent: {inner_extent_x/1000:.1f} km, "
              f"slice [{x_start}:{x_stop}, {y_start}:{y_stop}]")
        print(f"{'='*60}")

        refine(
            nc_path,
            x_start=x_start, x_stop=x_stop,
            y_start=y_start, y_stop=y_stop,
            dx=new_dx, dy=new_dy,
            parent_group=parent_group,
            n_size_classes=n_cls,
            seed=SEED + 1000 * level,
        )

        group_name = f"refinements/r{level - 1}"
        compute_diagnostics_group(nc_path, group_name)
        groups.append(group_name)

    # ── Render all levels ────────────────────────────────────────────────────
    level_names = ['L0_root'] + [f'L{i}' for i in range(1, len(groups))]

    print(f"\n{'='*60}")
    print("Rendering glimpse + witness for each level")
    print(f"{'='*60}")

    glimpse_pngs = []
    for group, name in zip(groups, level_names):
        grp_arg = None if group == '/' else group
        g_png = run_cloudyview(nc_path, "glimpse", OUTPUT_DIR,
                               group=grp_arg, suffix=name)
        glimpse_pngs.append(g_png)
        run_cloudyview(nc_path, "witness", OUTPUT_DIR,
                       group=grp_arg, suffix=name)

    # ── Composite: nest all refinement levels into the root glimpse ──────────
    print(f"\n{'='*60}")
    print("Building nested composite glimpse")
    print(f"{'='*60}")
    build_nested_composite(nc_path, glimpse_pngs, groups, OUTPUT_DIR)

    # ── Print witness commands for manual use ────────────────────────────────
    cx, cy, cz = CAMERA_POSITION
    print(f"\n{'='*60}")
    print("Witness commands (copy-paste to tweak camera):")
    print(f"{'='*60}")
    for group, name in zip(groups, level_names):
        cmd = f"witness {nc_path} {WITNESS_QUALITY}"
        cmd += f" --camera-position {cx} {cy} {cz}"
        cmd += f" --camera-azimuth {CAMERA_AZIMUTH}"
        cmd += f" --camera-elevation {CAMERA_ELEVATION}"
        cmd += f" --fov {CAMERA_FOV}"
        if group != '/':
            cmd += f" --group {group}"
        if WITNESS_SIZE:
            cmd += f" --size {WITNESS_SIZE[0]} {WITNESS_SIZE[1]}"
        cmd += f" -o {OUTPUT_DIR}"
        print(f"\n# {name}")
        print(cmd)


if __name__ == '__main__':
    main()
