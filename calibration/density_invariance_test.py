#!/usr/bin/env python3
"""Item-34 mechanism test: does interpolation cause the 1/n_c violation?

The production cascade coarsens the grid per class and interpolates the
flux between classes (n_c x more zoom hops per octave at higher cascade
density). This test runs the SAME flux cascade on ONE fixed grid with
per-class physical kernels and per-class center sparsity -- no
inter-class interpolation at all. If realized C1 is then density-
invariant, interpolation smoothing is the mechanism; if C1 still scales
as 1/n_c, the envelope overlap between 2^(1/n_c)-spaced classes (or the
additive composition itself) is responsible.

Isotropic kernels on a thin 3D slab; C1 estimated exactly as in
flux_c1_calibration.py (box-coarse-grained K(2)).
"""

import importlib.util
from pathlib import Path

import numpy as np

from steam.simulate import (
    _turbulon_envelope, _extremal_levy, LEVY_LOG_MEAN, FLUX_ALPHA, CONVOLVE,
    SUPPORT_FACTOR,
)

spec = importlib.util.spec_from_file_location(
    "cal", Path(__file__).parent / "flux_c1_calibration.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)

NX = NY = 512
NZ = 32
OUTER = 64.0     # grid units; dx = 1
FINEST = 4.0
C_VALUES = (0.03, 0.05)
N_C_VALUES = (1, 2, 4)
BOX_SIZES = (2, 4, 8, 16, 32)
REALIZATIONS = 3


def run_fixed_grid_cascade(c, n_c, seed):
    octaves = np.log2(OUTER / FINEST)
    n_classes = int(round(octaves)) * n_c + 1
    k_values = OUTER / 2.0 ** (np.arange(n_classes) / n_c)
    per_class_scale = np.float32(c / n_c ** (1.0 / FLUX_ALPHA))
    shift = np.float32(LEVY_LOG_MEAN * float(per_class_scale) ** FLUX_ALPHA)

    rng_master = np.random.SeedSequence(seed).spawn(n_classes)
    flux = np.ones((NX, NY, NZ), dtype=np.float32)
    for i, k in enumerate(k_values):
        rng = np.random.default_rng(rng_master[i])
        entering_mean = float(flux.mean(dtype=np.float64))
        # Per-class physical kernel on the FIXED grid (isotropic).
        kernel = _turbulon_envelope(
            float(k), 1.0, 1.0, 1.0, support_factor=SUPPORT_FACTOR)
        # Sparse centers every k/2 grid cells.
        s = max(1, int(round(k / 2.0)))
        gamma = np.zeros((NX, NY, NZ), dtype=np.float32)
        idx = np.ix_(np.arange(0, NX, s), np.arange(0, NY, s),
                     np.arange(0, NZ, s))
        n_draw = len(range(0, NX, s)) * len(range(0, NY, s)) * len(range(0, NZ, s))
        gamma[idx] = _extremal_levy(FLUX_ALPHA, n_draw, rng).reshape(
            len(range(0, NX, s)), len(range(0, NY, s)), len(range(0, NZ, s)))
        gamma *= per_class_scale
        noise = np.expm1(gamma - shift)
        noise[gamma == 0.0] = np.float32(0.0)
        noise *= flux
        flux += CONVOLVE(noise, kernel)
        np.maximum(flux, np.float32(0.0), out=flux)
        volume_mean = float(flux.mean(dtype=np.float64))
        if volume_mean > 0:
            flux *= np.float32(entering_mean / volume_mean)
    return flux


def main():
    print(f"fixed-grid cascade, no inter-class interpolation "
          f"({NX}x{NY}x{NZ}, outer {OUTER:g} -> finest {FINEST:g})")
    for n_c in N_C_VALUES:
        for c in C_VALUES:
            c1s = []
            for r in range(REALIZATIONS):
                flux = run_fixed_grid_cascade(c, n_c, seed=100 + r)
                est = cal.estimate_c1(flux, BOX_SIZES, FLUX_ALPHA)
                c1s.append(est["C1"])
            c1s = np.asarray(c1s)
            print(f"n_c={n_c} c={c}: C1 = {c1s.mean():.4f} "
                  f"+/- {c1s.std(ddof=1):.4f}", flush=True)
    print("\nExact compensation predicts equal C1 across n_c at fixed c.")


if __name__ == "__main__":
    main()
