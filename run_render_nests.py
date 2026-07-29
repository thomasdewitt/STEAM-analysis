#!/usr/bin/env python3
"""Render nests per main.tex Sect. 'Nested domain simulations' (2026-07-29).

From the icon_lem production square (member m00):
  nest A: 24 x 24 km, full depth, dx = 46.875 m  (512 x 512; 6 octaves
          below the parent's 3 km grid). Site: the 8x8-parent-cell window
          whose cloud fraction at z ~ 1 km is closest to 0.2.
  nest B: refine of nest A. 6 x 6 km, lowest 2 km, dx ~ 2.93 m
          (2048 x 2048; 4 more octaves). Site: the 6-km window of nest A
          with cloud fraction at 1 km closest to 0.2.

Both nests contain sub-spheroscale turbulons at the fine end (l_s = 10 m);
anisotropy is inherited (piecewise isotropic below the spheroscale).
CPU device (memory over VRAM, production guidance). Restartable: skips
a nest whose group already has diagnostics.

Usage: python run_render_nests.py [parent.nc]
"""

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4

from steam.simulate import refine
from steam.thermodynamics import compute_diagnostics

HERE = Path(__file__).parent
PARENT = (Path(sys.argv[1]) if len(sys.argv) > 1
          else HERE / "runs" / "steam_sq10_icon_lem_m00.nc")
CLOUD_KGKG = 0.01e-3
CF_TARGET = 0.2


def _group_done(path, group):
    ds = netCDF4.Dataset(path)
    try:
        g = ds
        for part in group.split("/"):
            if part not in g.groups:
                return False
            g = g.groups[part]
        return "qc" in g.variables
    finally:
        ds.close()


def _cloud_mask_at(path, group, z_target):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    g = ds
    if group != "/":
        for part in group.split("/"):
            g = g.groups[part]
    z = g.variables["z"][:]
    k = int(np.argmin(np.abs(z - z_target)))
    cond = g.variables["qc"][:, :, k] + g.variables["qi"][:, :, k]
    ds.close()
    return cond > CLOUD_KGKG


def _best_window(mask, win):
    """Top-left index (ix, iy) of the win x win window with CF closest to target."""
    from numpy.lib.stride_tricks import sliding_window_view
    cf = sliding_window_view(mask.astype(np.float32), (win, win)).mean(axis=(2, 3))
    ix, iy = np.unravel_index(np.argmin(np.abs(cf - CF_TARGET)), cf.shape)
    print(f"  window ({ix},{iy}) of {cf.shape}, CF = {cf[ix, iy]:.3f}", flush=True)
    return int(ix), int(iy)


def main():
    # --- nest A: 24 km @ 46.875 m, sited on the parent's 1-km cloud mask ---
    if _group_done(PARENT, "refinements/render_a"):
        print("nest A exists, skipping", flush=True)
    else:
        mask = _cloud_mask_at(PARENT, "/", 1000.0)
        ix, iy = _best_window(mask, 8)          # 8 x 3 km = 24 km
        t0 = time.perf_counter()
        refine(PARENT, ix, ix + 8, iy, iy + 8, 46.875, 46.875,
               output_group="refinements/render_a", device="cpu")
        compute_diagnostics(PARENT, group="refinements/render_a", compress=True)
        print(f"nest A done in {time.perf_counter() - t0:.0f} s", flush=True)

    # --- nest B: 6 km @ ~2.93 m, lowest 2 km, sited on nest A ---
    if _group_done(PARENT, "refinements/render_b"):
        print("nest B exists, skipping", flush=True)
    else:
        mask = _cloud_mask_at(PARENT, "refinements/render_a", 1000.0)
        win = 128                                # 128 x 46.875 m = 6 km
        ix, iy = _best_window(mask, win)
        t0 = time.perf_counter()
        refine(PARENT, ix, ix + win, iy, iy + win, 46.875 / 16, 46.875 / 16,
               parent_group="refinements/render_a",
               output_group="refinements/render_b",
               z_max=2000.0, device="cpu")
        compute_diagnostics(PARENT, group="refinements/render_b", compress=True)
        print(f"nest B done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
