"""Drive witness renders for every discovered STEAM cube subgroup.

Reuses the ocean FIF wave field across all renders and each cube's sigma
field across its 6 views, so per-view cost is just the render kernel.
"""
from __future__ import annotations

import time
from pathlib import Path

import netCDF4
import numpy as np
from PIL import Image as PILImage

from cloudyview import io, optical_depth, config
from cloudyview.angles import direction_from_azimuth_elevation
from cloudyview.domain import compute_domain_geometry
from cloudyview.ocean_fif import dummy_fif_arrays, generate_fif_normals
from cloudyview.witness import NestedLevel, render_nested

REPO = Path(__file__).resolve().parents[2]
RENDER_DIR = REPO / 'renders' / 'witness_cubes'
NC_DIR = REPO / 'STEAM' / 'data'

IMG_SIZE = (1600, 1200)
CUBE_GROUP_CANDIDATES = ('cube_center', 'cube_edge')

# Two rings of four views, all placed 20% outside the horizontal domain edge
# and aimed at the domain centre (view elevation follows from geometry).
#   down_*  — slightly above the top of the domain (cam_z = 1.1*bmax_z).
#   up_*    — just above the ocean surface (cam_z ≈ 0).
VIEWS = [
    ('down_n', ( 0.0,   1.2,   1.2)),
    ('down_e', ( 1.2,   0.0,   1.2)),
    ('down_s', ( 0.0,  -1.2,   1.2)),
    ('down_w', (-1.2,   0.0,   1.2)),
    ('up_n',   ( 0.0,   1.2,  -0.99)),
    ('up_e',   ( 1.2,   0.0,  -0.99)),
    ('up_s',   ( 0.0,  -1.2,  -0.99)),
    ('up_w',   (-1.2,   0.0,  -0.99)),
]


def discover_cubes():
    """Return [(nc_path, group, tag), ...] for every available cube group."""
    cubes = []
    for nc_path in sorted(NC_DIR.glob('nested_refine_seed*.nc')):
        seed = nc_path.stem.split('seed')[-1]
        try:
            ds = netCDF4.Dataset(nc_path, 'r')
        except OSError as e:
            print(f"[skip] {nc_path.name}: {e}")
            continue
        try:
            present = set(ds['refinements'].groups) if 'refinements' in ds.groups else set()
        finally:
            ds.close()
        for sub in CUBE_GROUP_CANDIDATES:
            if sub in present:
                cubes.append((nc_path, f'refinements/{sub}', f'{sub}_seed{seed}'))
    return cubes


def load_cube_level(nc_path: Path, group: str, ext_mult: float) -> NestedLevel:
    data = io.load_and_validate(str(nc_path), dataset_group=group)
    lw_da = data['liquid_water_data']
    iw_da = data['ice_water_data']
    x_coord = data['x_coord']
    y_coord = data['y_coord']
    z_coord = data['z_coord']

    lw = lw_da.values[0] if 'time' in lw_da.dims else lw_da.values
    iw = None
    if iw_da is not None:
        iw_arr = iw_da.values[0] if 'time' in iw_da.dims else iw_da.values
        if np.max(iw_arr) >= 1e-6:
            iw = iw_arr

    sigma_ext = optical_depth.compute_extinction_field(
        lw, z_coord, re=10.0, iwc=iw, re_ice=30.0)
    sigma = np.ascontiguousarray((sigma_ext * ext_mult).astype(np.float64))

    nx_d, ny_d, nz_d = lw.shape
    geom = compute_domain_geometry(x_coord, y_coord, z_coord, nx_d, ny_d, nz_d)
    xv = np.asarray(x_coord, dtype=np.float64)
    yv = np.asarray(y_coord, dtype=np.float64)
    zv = np.asarray(z_coord, dtype=np.float64)
    bmin = np.array([xv.min() - 0.5 * geom.dx,
                     yv.min() - 0.5 * geom.dy,
                     zv.min() - 0.5 * abs(zv[1] - zv[0])], dtype=np.float64)
    bmax = np.array([xv.max() + 0.5 * geom.dx,
                     yv.max() + 0.5 * geom.dy,
                     zv.max() + 0.5 * abs(zv[-1] - zv[-2])], dtype=np.float64)
    return NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name=group)


def camera_basis(pos_rel, bmin, bmax):
    cam = np.empty(3, dtype=np.float64)
    cam[0] = bmin[0] + (pos_rel[0] + 1.0) * 0.5 * (bmax[0] - bmin[0])
    cam[1] = bmin[1] + (pos_rel[1] + 1.0) * 0.5 * (bmax[1] - bmin[1])
    cam[2] = (pos_rel[2] + 1.0) * 0.5 * bmax[2]
    center = 0.5 * (bmin + bmax)
    forward = center - cam
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    return cam, forward, right, up


def main() -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    cubes = discover_cubes()
    print(f"Discovered {len(cubes)} cube groups:")
    for nc_path, group, tag in cubes:
        print(f"  {nc_path.name} :: {group}  ->  {tag}")

    witness_cfg = config.get_witness_config()
    cam_cfg = witness_cfg['camera']
    sun_cfg = witness_cfg['sun']
    render_cfg = witness_cfg['rendering']
    ocean_cfg = render_cfg['ocean']

    ocean_enabled = ocean_cfg['enabled']
    camera_fov = cam_cfg['fov']
    n_light_steps = render_cfg['n_light_steps']
    exposure = render_cfg['exposure']
    ext_mult = render_cfg['extinction_multiplier']

    sun_dir = direction_from_azimuth_elevation(sun_cfg['azimuth'], sun_cfg['elevation'])

    print("\nPrecomputing ocean FIF wave field (shared across all renders)...")
    t0 = time.perf_counter()
    if ocean_enabled:
        fif_normals = generate_fif_normals()
    else:
        _z, _, _o = dummy_fif_arrays()
        fif_normals = (_z, _z, _o, 1.0)
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    for nc_path, group, tag in cubes:
        pending = [v for v in VIEWS
                   if not (RENDER_DIR / f'witness_{tag}_{v[0]}.png').exists()]
        if not pending:
            print(f"\n[{tag}] all views present, skip load")
            continue

        print(f"\n[{tag}] loading {nc_path.name} :: {group}")
        t0 = time.perf_counter()
        level = load_cube_level(nc_path, group, ext_mult)
        print(f"  load + sigma: {time.perf_counter() - t0:.1f}s  "
              f"grid={level.sigma.shape}")
        ocean_z = (ocean_cfg['height'] + 1.0) * 0.5 * level.bmax[2]

        for view, pos in pending:
            out = RENDER_DIR / f'witness_{tag}_{view}.png'
            cam, fwd, right, up = camera_basis(pos, level.bmin, level.bmax)
            t0 = time.perf_counter()
            image = render_nested(
                [level],
                camera_position=tuple(cam),
                camera_forward=tuple(fwd),
                camera_right=tuple(right),
                camera_up=tuple(up),
                sun_direction=tuple(sun_dir),
                image_size=IMG_SIZE,
                fov_degrees=camera_fov,
                n_light_steps=n_light_steps,
                exposure=exposure,
                ocean_enabled=ocean_enabled,
                ocean_z=ocean_z,
                fif_normals=fif_normals,
                verbose=False,
            )
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            PILImage.fromarray(img_uint8).save(str(out))
            print(f"  [{view}] -> {out.name} ({time.perf_counter() - t0:.1f}s)")


if __name__ == '__main__':
    main()
