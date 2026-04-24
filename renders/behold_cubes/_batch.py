"""Behold renders of seed 001 cube_edge from the south face.

Two views, both with the camera centred on the south face of the cube
(pos_rel_x = 0, pos_rel_y = −1), looking north (az = 0):
    south_up50   — camera at ground level, elevation +50°
    south_down50 — camera at cube top,    elevation −50°

Render budget = behold 'medium' preset (max_depth=64, rr_depth=16) with
spp bumped to 1024 and size at 1200x800. Runs in 'custom' mode so all
four knobs are pinned explicitly.

Hack — behold places the domain bottom on the ground (ocean at world
z≈0), but STEAM cubes start a few hundred metres above ground. We
sidestep that by writing a temp nc whose z-axis is extended downward to
z=0 with zero-filled qc/qi cells, then pointing behold at the temp
file. With bmin_z = 0 the witness pos_rel and behold rel_pos
conventions coincide numerically.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NC_PATH = REPO / 'STEAM' / 'data' / 'nested_refine_seed001.nc'
GROUP = 'refinements/cube_edge'
OUT_DIR = Path(__file__).resolve().parent
TAG = 'cube_edge_seed001'

# (name, pos_rel, azimuth_deg, elevation_deg). az=0 → look north (into
# the cube from its south face).
VIEWS = [
    ('south_up50',   ( 0.0, -1.0, -0.99), 0.0,  50.0),
    ('south_down50', ( 0.0, -1.0,  1.00), 0.0, -50.0),
]

# behold 'medium' preset, spp bumped to 1024, rendered at 1200x800.
SPP        = 1024
SIZE       = (1200, 800)
MAX_DEPTH  = 64
RR_DEPTH   = 16
STEP_SPP   = 32          # samples accumulated per progress step


def pad_cube_to_ground(src_path: Path, src_group: str, dst_path: Path) -> None:
    """Copy qc/qi/coords from src_path[src_group] to a flat temp nc at
    dst_path, prepending zero-filled cells in z so the new domain bottom
    sits at z ≈ 0. Assumes uniform vertical spacing."""
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

    with tempfile.TemporaryDirectory(prefix='behold_pad_', dir='/var/tmp') as tmpdir:
        tmp_nc = Path(tmpdir) / 'cube_edge_padded.nc'
        print(f"Padding {NC_PATH.name}[{GROUP}] → {tmp_nc}")
        pad_cube_to_ground(NC_PATH, GROUP, tmp_nc)

        default_out = OUT_DIR / f'behold_ground_view_max_depth={MAX_DEPTH}_rr_depth={RR_DEPTH}.png'

        for name, pos, az, el in VIEWS:
            print(f"\n[{name}] pos_rel={pos}  az={az:.1f}°  el={el:.1f}°")
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
                f'behold_{TAG}_{name}_'
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
