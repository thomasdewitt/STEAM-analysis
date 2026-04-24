"""Witness renders of the same two south-face views as `_batch.py`.

Reproduces the two behold views on seed 001 `cube_edge` using witness.
Positions match byte-for-byte: witness handles lifted domains natively
(rel_z=-1 is physical z=0) so no z-padding hack is needed.

To match behold's frustum we have to translate its FOV spec:
  - behold uses Mitsuba's perspective sensor with fov_axis='x' (the
    default), so its fov=100 is the *horizontal* FOV.
  - witness applies fov to the *vertical* axis and scales x by aspect.
Behold rendered at 1200x800 (aspect 1.5). To match that frustum with
witness we use the same size and convert:
    vertical_fov = 2·atan(tan(100°/2) / 1.5) ≈ 76.93°.

Views (az=0 → look north into the cube from its south face):
    south_up50   — camera near ground (rel_z=-0.99), elevation +50°
    south_down50 — camera at cube top (rel_z=+1.00),  elevation -50°
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NC_PATH = REPO / 'STEAM' / 'data' / 'nested_refine_seed001.nc'
GROUP = 'refinements/cube_edge'
OUT_DIR = Path(__file__).resolve().parent
TAG = 'cube_edge_seed001'
QUALITY = 'high'           # for defaults other than size (exposure, n_light_steps, etc.)
SIZE = (1200, 800)         # match behold _batch.py
BEHOLD_FOV_H = 100.0       # behold fov (horizontal, Mitsuba default)
FOV = math.degrees(2.0 * math.atan(
    math.tan(math.radians(BEHOLD_FOV_H) * 0.5) * SIZE[1] / SIZE[0]
))

VIEWS = [
    ('south_up50',   ( 0.0, -1.0, -0.99), 0.0,  50.0),
    ('south_down50', ( 0.0, -1.0,  1.00), 0.0, -50.0),
]

WITNESS_BIN = REPO / '.venv' / 'bin' / 'witness'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    default_out = OUT_DIR / f'witness_{NC_PATH.stem}.png'

    for name, pos, az, el in VIEWS:
        print(f"\n[{name}] pos_rel={pos}  az={az:.1f}°  el={el:.1f}°  fov_v={FOV:.3f}°")
        cmd = [
            str(WITNESS_BIN), str(NC_PATH), QUALITY,
            '--output', str(OUT_DIR),
            '--group', GROUP,
            '--size', str(SIZE[0]), str(SIZE[1]),
            '--camera-position', f'{pos[0]:.6f}', f'{pos[1]:.6f}', f'{pos[2]:.6f}',
            '--camera-azimuth', f'{az:.6f}',
            '--camera-elevation', f'{el:.6f}',
            '--fov', f'{FOV:.6f}',
        ]
        print('  ' + ' '.join(cmd))
        subprocess.run(cmd, check=True)

        final = OUT_DIR / f'witness_{TAG}_{name}_{SIZE[0]}x{SIZE[1]}_fov{BEHOLD_FOV_H:.0f}h.png'
        if final.exists():
            final.unlink()
        shutil.move(str(default_out), str(final))
        print(f"  -> {final.name}")


if __name__ == '__main__':
    main()
