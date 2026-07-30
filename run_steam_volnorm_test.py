#!/usr/bin/env python3
"""Gate test for the flux-renorm ruling (2026-07-27, audit B14).

Thomas's ruling: the per-class flux renormalization should RESTORE THE
ENTERING MEAN (clip-bias corrector) with VOLUME scope, replacing the
per-level unit-mean renorm — gated on this paired check. This script runs
one RCEMIP member (icon_lem snap0, seed 1000, c for C1 = 0.05 from the
2026-07-27 calibration) with a monkeypatched `_advance_flux` implementing
the ruled renorm; the paired per-level member is steam_96_icon_lem_snap0
from run_steam_96.py at the same c and seed. simulate.py itself is
untouched until the gate passes.

The variant is a verbatim copy of steam.simulate._advance_flux except:
  - the entering volume mean is captured before the increment;
  - after the clip, the field is rescaled by a single scalar restoring
    that entering mean (volume scope), instead of per-level unit-mean.

Writes runs/steam_volnorm_icon_lem_snap0.nc + stats npz.
"""

from pathlib import Path

import numpy as np
import netCDF4

import importlib
_steam_simulate = importlib.import_module("steam.simulate")
C1_TARGET = 0.05
_steam_simulate.FLUX_SCALE = (C1_TARGET / 3.097) ** (1 / 1.8)

from steam.simulate import (
    _sparse_levy, LEVY_LOG_MEAN, FLUX_ALPHA, CONVOLVE,
)
from steam.thermodynamics import compute_diagnostics
from steam.constants import specific_heat_dry_air as cp


def _advance_flux_entering_mean_volume(
    flux, rng, kernel, flux_noise_scale, n_scale_classes_per_dyad,
    sparsity_factors, n_zero=0, zero_bottom=False, zero_top=False,
    device='cpu',
):
    """steam.simulate._advance_flux with the ruled renorm (see module doc)."""
    s_x, s_y, s_z = sparsity_factors
    per_class_scale = flux_noise_scale / n_scale_classes_per_dyad ** (1.0 / FLUX_ALPHA)
    shift = np.float32(LEVY_LOG_MEAN * per_class_scale ** FLUX_ALPHA)
    per_class_scale = np.float32(per_class_scale)

    entering_mean = float(flux.mean(dtype=np.float64))

    gamma = _sparse_levy(*flux.shape, s_x, s_y, s_z, FLUX_ALPHA, rng)
    if zero_bottom and n_zero > 0:
        gamma[:, :, :n_zero] = 0
    if zero_top and n_zero > 0:
        gamma[:, :, -n_zero:] = 0

    gamma *= per_class_scale
    noise = np.expm1(gamma - shift)
    noise[gamma == 0.0] = np.float32(0.0)

    mean_abs = np.abs(noise).sum() / max(np.count_nonzero(noise), 1)
    scalar_amplitude = noise * flux
    if mean_abs > 0:
        scalar_amplitude *= np.float32(1.0 / mean_abs)

    noise *= flux
    flux += CONVOLVE(noise, kernel, device=device)

    n_clipped = int(np.count_nonzero(flux < 0))
    np.maximum(flux, np.float32(0.0), out=flux)
    volume_mean = float(flux.mean(dtype=np.float64))
    if volume_mean > 0:
        flux *= np.float32(entering_mean / volume_mean)
    else:
        flux[:] = np.float32(entering_mean)

    diagnostics = {
        'n_clipped': n_clipped,
        'n_points': int(flux.size),
        'clip_fraction': n_clipped / flux.size,
    }
    return scalar_amplitude, diagnostics


_steam_simulate._advance_flux = _advance_flux_entering_mean_volume

from steam.simulate import simulate  # noqa: E402  (after monkeypatch)

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"
CLOUD_KGKG = 0.01e-3
SPHEROSCALE_SURFACE = 100.0
SPHEROSCALE_TOP = 1.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0


def steam_stats(path, out_path):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    h = ds.variables["h"][:]
    qt = ds.variables["qt"][:]
    cond = ds.variables["qc"][:] + ds.variables["qi"][:]
    ds.close()
    nz = z.size
    out = {k: np.empty(nz) for k in
           ("h_mean", "h_var", "qt_mean", "qt_var", "cloud_fraction")}
    for k in range(nz):
        hk = h[:, :, k].astype(np.float64)
        qtk = qt[:, :, k].astype(np.float64)
        out["h_mean"][k] = hk.mean()
        out["h_var"][k] = hk.var()
        out["qt_mean"][k] = qtk.mean()
        out["qt_var"][k] = qtk.var()
        out["cloud_fraction"][k] = np.mean(cond[:, :, k] > CLOUD_KGKG)
    np.savez(out_path, z=z, **out)


def main():
    model, i = "icon_lem", 0
    src = np.load(STATS / f"{model}_snap{i}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    z = src["z_profile"]
    spheroscale = SPHEROSCALE_SURFACE + (
        SPHEROSCALE_TOP - SPHEROSCALE_SURFACE) * z / DOMAIN_HEIGHT
    RUNS.mkdir(exist_ok=True)
    out_nc = RUNS / f"steam_volnorm_{model}_snap{i}.nc"
    simulate(
        h_profile, qt_profile,
        nx=2048, ny=128, dx=3000.0, dy=3000.0,
        outer_scale=96000.0,
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=float(src["surface_pressure"]),
        seed=1000 + i,
        h_min=h_profile.min() - 10 * cp,
        h_max=h_profile.max() + 10 * cp,
        qt_min=0.0, qt_max=max(0.03, 1.5 * qt_profile.max()),
        compress=True,
        device="cuda",
    )
    compute_diagnostics(str(out_nc), compress=True)
    steam_stats(out_nc, STATS / f"steam_volnorm_{model}_snap{i}.npz")
    print("volnorm test member done")


if __name__ == "__main__":
    main()
