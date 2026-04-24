"""Behold renders of strip up_along views.

Matches the witness ``up_along`` framing: camera at the horizontal centre
of the strip (pos_rel_x = 0, pos_rel_y = 0) just above the ocean surface
(pos_rel_z = -0.99), looking along +x (az = 90°) with a small upward
pitch (el = +8°).

Currently renders:
    strip_center, seed 003
    strip_edge,   seed 001

Render budget = behold 'medium' preset (max_depth=64, rr_depth=16) with
spp bumped to 1024 and size at 1200x800, same as behold_cubes.

Same ``pad_*_to_ground`` hack as behold_cubes: write a flat temp nc
covering the strip group so behold's rel_pos and the witness pos_rel
conventions coincide numerically. Strip's bmin_z is already below 0 so
padding reduces to a no-op copy, but we reuse the same code path.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NC_DIR = REPO / 'STEAM' / 'data'
OUT_DIR = Path(__file__).resolve().parent

# (nc_path, group, tag)
DATASETS = [
    (NC_DIR / 'nested_refine_seed003.nc', 'refinements/strip_center', 'strip_center_seed003'),
    (NC_DIR / 'nested_refine_seed001.nc', 'refinements/strip_edge',   'strip_edge_seed001'),
]

# (name, pos_rel, azimuth_deg, elevation_deg). az=90 → look along +x
# (strip long axis); el=+8° matches the witness up_along tilt.
VIEWS = [
    ('up_along', (0.0, 0.0, -0.99), 90.0, 8.0),
]

SPP        = 512
SIZE       = (600, 400)
MAX_DEPTH  = 64
RR_DEPTH   = 16
STEP_SPP   = 32

# Strips are long in x; camera sits at the x-centre. Truncate to ±100 km
# around the centre so paths don't crawl off down the strip forever.
TRUNCATE_X_M = 100_000.0


def pad_strip_to_ground(src_path: Path, src_group: str, dst_path: Path) -> None:
    """Copy qc/qi/coords from src_path[src_group] to a flat temp nc at
    dst_path, prepending zero-filled cells in z so the new domain bottom
    sits at z ≈ 0. If the source already starts at or below z=0 this is
    a plain copy. Assumes uniform vertical spacing."""
    with netCDF4.Dataset(src_path, 'r') as src:
        grp = src[src_group]
        x = np.asarray(grp.variables['x'][:], dtype=np.float64)
        y = np.asarray(grp.variables['y'][:], dtype=np.float64)
        z = np.asarray(grp.variables['z'][:], dtype=np.float64)
        qc = np.asarray(grp.variables['qc'][:])
        qc_units = grp.variables['qc'].units
        has_qi = 'qi' in grp.variables
        if has_qi:
            qi = np.asarray(grp.variables['qi'][:])
            qi_units = grp.variables['qi'].units

    x_centre = 0.5 * (float(x[0]) + float(x[-1]))
    keep_x = np.abs(x - x_centre) <= TRUNCATE_X_M
    if not keep_x.all():
        n_drop = int((~keep_x).sum())
        print(f"  truncate x: keeping {int(keep_x.sum())}/{x.size} cells within "
              f"±{TRUNCATE_X_M/1000:.0f} km of centre (dropped {n_drop})")
        x = x[keep_x]
        qc = qc[keep_x, :, :]
        if has_qi:
            qi = qi[keep_x, :, :]

    dz = float(z[1] - z[0])
    bottom_orig = float(z[0] - dz / 2)
    n_pad = max(0, int(np.ceil(bottom_orig / dz)))
    pad_z = z[0] - dz * np.arange(1, n_pad + 1, dtype=np.float64)[::-1]
    z_new = np.concatenate([pad_z, z])

    nx, ny, nz_old = qc.shape
    pad_shape = (nx, ny, n_pad)
    qc_new = np.concatenate([np.zeros(pad_shape, dtype=qc.dtype), qc], axis=2)
    if has_qi:
        qi_new = np.concatenate([np.zeros(pad_shape, dtype=qi.dtype), qi], axis=2)

    print(f"  pad: prepending {n_pad} zero cells at dz={dz:.3f}m  "
          f"(original floor z={bottom_orig:.1f}m → new floor z={z_new[0] - dz/2:.3f}m)")

    with netCDF4.Dataset(dst_path, 'w') as dst:
        dst.createDimension('x', nx)
        dst.createDimension('y', ny)
        dst.createDimension('z', nz_old + n_pad)
        vx = dst.createVariable('x', 'f8', ('x',))
        vy = dst.createVariable('y', 'f8', ('y',))
        vz = dst.createVariable('z', 'f8', ('z',))
        vx[:] = x
        vy[:] = y
        vz[:] = z_new
        vqc = dst.createVariable('qc', qc.dtype, ('x', 'y', 'z'))
        vqc.units = qc_units
        vqc[:] = qc_new
        if has_qi:
            vqi = dst.createVariable('qi', qi.dtype, ('x', 'y', 'z'))
            vqi.units = qi_units
            vqi[:] = qi_new


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    default_out = OUT_DIR / f'behold_ground_view_max_depth={MAX_DEPTH}_rr_depth={RR_DEPTH}.png'

    for nc_path, group, tag in DATASETS:
        pending = [v for v in VIEWS
                   if not (OUT_DIR / (
                       f'behold_{tag}_{v[0]}_'
                       f'{SIZE[0]}x{SIZE[1]}_spp{SPP}_md{MAX_DEPTH}_rr{RR_DEPTH}.png')).exists()]
        if not pending:
            print(f"\n[{tag}] all views present, skip")
            continue

        with tempfile.TemporaryDirectory(prefix='behold_pad_', dir='/var/tmp') as tmpdir:
            tmp_nc = Path(tmpdir) / f'{tag}_padded.nc'
            print(f"\nPadding {nc_path.name}[{group}] → {tmp_nc}")
            pad_strip_to_ground(nc_path, group, tmp_nc)

            for name, pos, az, el in pending:
                print(f"\n[{tag}:{name}] pos_rel={pos}  az={az:.1f}°  el={el:.1f}°")
                cmd = [
                    'behold', str(tmp_nc), 'custom', '--gpu',
                    '--output', str(OUT_DIR),
                    '--size', str(SIZE[0]), str(SIZE[1]),
                    '--spp', str(SPP),
                    '--max-depth', str(MAX_DEPTH),
                    '--rr-depth', str(RR_DEPTH),
                    '--camera-position', f'{pos[0]:.6f}', f'{pos[1]:.6f}', f'{pos[2]:.6f}',
                    '--camera-azimuth', f'{az:.6f}',
                    '--camera-elevation', f'{el:.6f}',
                    '--progress-interval', str(STEP_SPP),
                ]
                print('  ' + ' '.join(cmd))
                subprocess.run(cmd, check=True)

                final = OUT_DIR / (
                    f'behold_{tag}_{name}_'
                    f'{SIZE[0]}x{SIZE[1]}_spp{SPP}_md{MAX_DEPTH}_rr{RR_DEPTH}.png'
                )
                if final.exists():
                    final.unlink()
                default_out.rename(final)
                for ckpt in OUT_DIR.glob(
                        f'behold_ground_view_max_depth={MAX_DEPTH}_rr_depth={RR_DEPTH}_spp*.png'):
                    ckpt.unlink()
                print(f"  -> {final.name}")


if __name__ == '__main__':
    main()
