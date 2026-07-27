#!/usr/bin/env python3
"""Strip nest inside the square domain, for the multifractal-params appendix.

Level-1 refinement of steam_square_icon_lem_snap0.nc: a thin strip
spanning the full 6144 km x-extent (so x stays periodic with the parent
extent as period, no x-halo) and 48 km wide in y, refined from dx = 3 km
to dx = 375 m (16384 cells along x, three octaves below the parent's
finest class: k = 3 km, 1.5 km, 750 m). CPU device per Thomas's
production guidance (memory over VRAM). Output is written as a group
(refinements/r0) inside the parent file.

Usage: python run_strip_nest.py [parent.nc]
"""

import sys
import time
from pathlib import Path

from steam.simulate import refine

HERE = Path(__file__).parent
PARENT = (Path(sys.argv[1]) if len(sys.argv) > 1
          else HERE / "runs" / "steam_square_icon_lem_snap0.nc")

# Parent grid: 2048 x 2048 at dx = 3 km. Strip: all of x, 16 parent cells
# (48 km) centered in y.
X_START, X_STOP = 0, 2048
Y_CENTER = 1024
Y_HALF_CELLS = 8
NEW_DX = 375.0

started = time.perf_counter()
out = refine(
    PARENT,
    X_START, X_STOP,
    Y_CENTER - Y_HALF_CELLS, Y_CENTER + Y_HALF_CELLS,
    NEW_DX, NEW_DX,
    device="cpu",
)
print(f"strip nest written into {out} in {time.perf_counter() - started:.0f} s",
      flush=True)
