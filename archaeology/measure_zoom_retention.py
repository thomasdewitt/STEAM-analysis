#!/usr/bin/env python3
"""Measure the trilinear-regrid retention factor per class (2026-07-27, v2).

Thomas's ruling: "the interpolation reducing variance thing should
probably be fixed ... measure or derive the variance factor and then
multiply each scale by it ... Except last scale ofc."

v2 protocol (v1 was confounded by z-node alignment at a single level and
by native-vs-final Haar discretization): synthetic single-class deposits
(Rademacher +/-1 at the model's centers, model kernel) are carried to the
final grid twice with the SAME shapes chain from _compute_all_grids:

  Arm A (model):     trilinear 3D regrid per hop (steam.zoom_trilinear)
  Arm B (reference): exact Fourier x,y upsampling per hop (periodic,
                     band-limited -> lossless) + identical trilinear z

  R_i = Mhat1_x(A, lag k_i) / Mhat1_x(B, lag k_i)

with the Haar fluctuation pooled over ALL y and z (one scaleinvariance
call, periodic in x). Both arms share the measurement and the z chain, so
R isolates the horizontal piecewise-linearization loss — the part that is
well-posed as a per-class scalar. Because dyadic linear-interpolation
chains are near-idempotent, R is expected ~hop-independent (one-time
loss); the finest class has R = 1 identically (never regridded).

The compensation multiplies each class amplitude by 1/R_i.

Run with the turbulon-analysis venv (steam = turbulon-model HEAD).
"""

from pathlib import Path

import numpy as np
import scaleinvariance as si

import importlib
sm = importlib.import_module("steam.simulate")
from steam.utils import zoom_trilinear

HERE = Path(__file__).resolve().parent.parent
N_SEEDS = 3


def fourier_xy_resize(field, nx_new, ny_new):
    """Exact band-limited x,y upsampling (periodic), z untouched."""
    nx, ny, nz = field.shape
    spec = np.fft.rfft2(field.astype(np.float64), axes=(0, 1))
    out = np.zeros((nx_new, ny_new // 2 + 1, nz), dtype=spec.dtype)
    kx = nx // 2
    out[:kx, : spec.shape[1], :] = spec[:kx]
    out[nx_new - (nx - kx):, : spec.shape[1], :] = spec[kx:]
    result = np.fft.irfft2(out, s=(nx_new, ny_new), axes=(0, 1))
    result *= (nx_new * ny_new) / (nx * ny)
    return result.astype(np.float32)


def class_grids(nx, dx, outer_scale, domain_height=20_000.0, ls=30.0):
    n_classes = int(round(np.log2(outer_scale / (2 * dx)))) + 1
    k_values = outer_scale / 2.0 ** np.arange(n_classes)
    z_profile = np.arange(0.0, domain_height + 25.0, 50.0)
    grids = sm._compute_all_grids(
        k_values, nx * dx, nx * dx, domain_height, (1, 1, 1),
        np.full(z_profile.size, ls), z_profile,
        anisotropy="canonical",
    )
    return k_values, grids


def retention(nx=512, dx=1_000_000.0 / 512, outer_scale=500_000.0):
    k_values, grids = class_grids(nx, dx, outer_scale)
    n_classes = len(k_values)
    kernel = sm._turbulon_envelope(1, 1 / 2, 1 / 2, 1 / 2,
                                   support_factor=sm.SUPPORT_FACTOR)
    shapes = [(int(grids["nx"][i]), int(grids["ny"][i]), int(grids["nz"][i]))
              for i in range(n_classes)]

    print(f"config: {nx} cells, outer {outer_scale/1000:.0f} km, "
          f"{n_classes} classes, shapes {shapes[0]}..{shapes[-1]}")
    print("k [km]   hops   R = M1(trilinear)/M1(fourier-xy)")
    r_all = np.ones(n_classes)
    for i in range(n_classes - 1):          # finest class: R = 1
        ratios = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + 97 * i + seed)
            nx_k, ny_k, nz_k = shapes[i]
            noise = np.zeros((nx_k, ny_k, nz_k), dtype=np.float32)
            sl = (slice(0, nx_k, 2), slice(0, ny_k, 2), slice(0, nz_k, 2))
            noise[sl] = rng.choice(np.float32([-1.0, 1.0]),
                                   size=noise[sl].shape)
            deposit = sm.CONVOLVE(noise, kernel)

            a = deposit
            b = deposit.copy()
            for j in range(i + 1, n_classes):
                a = zoom_trilinear(a, shapes[j])
                b = fourier_xy_resize(b, shapes[j][0], shapes[j][1])
                b = zoom_trilinear(b, shapes[j])   # x,y already match: z only
            lag = np.array([int(round(k_values[i] / (k_values[-1] / 2)))])
            _, Fa = si.haar_fluctuation(np.asarray(a, dtype=np.float64),
                                        order=1.0, axis=0, lags=lag,
                                        periodic=True)
            _, Fb = si.haar_fluctuation(np.asarray(b, dtype=np.float64),
                                        order=1.0, axis=0, lags=lag,
                                        periodic=True)
            ratios.append(float(Fa[0]) / float(Fb[0]))
        r_all[i] = float(np.mean(ratios))
        hops = n_classes - 1 - i
        print(f"{k_values[i]/1000:7.1f}  {hops:4d}   "
              f"{r_all[i]:.4f} +/- {np.std(ratios):.4f}")
    print(f"{k_values[-1]/1000:7.1f}     0   1.0000 (never regridded)")
    return k_values, r_all


def main():
    print("=== driver config (8 classes, 512) ===")
    k1, r1 = retention()
    print("\n=== check config (9 classes, 1024 @ 2000 km) ===")
    k2, r2 = retention(nx=1024, dx=2_000_000.0 / 1024, outer_scale=1_000_000.0)
    np.savez(HERE / "stats" / "zoom_retention.npz",
             k_small=k1, r_small=r1, k_check=k2, r_check=r2)
    print("\nwrote stats/zoom_retention.npz")


if __name__ == "__main__":
    main()
