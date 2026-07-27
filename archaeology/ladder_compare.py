#!/usr/bin/env python3
"""Compare the realized amplitude ladder: ensemble-norm (71fd792, GOOD)
vs realized-mean norm (544fbb0, BAD, taper+projection disabled).

Wraps CONVOLVE to record, per class and scalar, the realized mean
absolute amplitude <|A|> over nonzero centers at ~7 km and the Haar
fluctuation of the deposited increment (lag = 2 cells on the class's own
grid). After the run, the final field's Haar at the class lags.

The question (2026-07-27 toggle result): the good code normalized W by
an analytic E[W] reference known to be ~3x off; the bad code enforces
<|A|> = C_k exactly and the field comes out flat. If the good code's
realized <|A|> ladder has a materially different log-slope than C_k's
H = 0.45, the enforced design ladder itself is not the one that yields
a k^H field, and the fix is theory-side, not code-side.

Usage (checkout the era in ~/code-and-data/turbulon-egu first):
    .venv/bin/python archaeology/ladder_compare.py <label>
"""

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
EGU_REPO = Path.home() / "code-and-data" / "turbulon-egu"
CP = 1004.0
LEVEL_M = 7000.0
DOMAIN_HEIGHT = 20_000.0

sm = importlib.import_module("steam.simulate")
STEAM_REPO = next(p for p in Path(sm.__file__).resolve().parents
                  if (p / ".git").exists())
print(f"steam from {STEAM_REPO}")

if os.environ.get("NO_FLUX"):      # F ≡ 1 exactly: no flux noise increments
    sm.FLUX_SCALE = 0.0

if os.environ.get("NO_GRADW"):     # Ŵ ≡ 1: disable gradient weighting
    _real_grad = sm._gradient_components

    def _flat_grad(field_3d, dx, dy, z_coords):
        return (np.ones_like(field_3d), np.zeros_like(field_3d))

    sm._gradient_components = _flat_grad

if os.environ.get("NO_SFLUX"):     # S_k without its flux factor
    _real_advance = sm._advance_flux

    def _advance_noflux_scalar(flux, *args, **kwargs):
        F0 = flux.copy()
        scalar_amplitude, diag = _real_advance(flux, *args, **kwargs)
        scalar_amplitude /= np.where(F0 > 0, F0, np.float32(1.0))
        return scalar_amplitude, diag

    sm._advance_flux = _advance_noflux_scalar

# Disable taper/projection where they exist, isolating the norm change.
if hasattr(sm, "_bound_taper"):
    sm._bound_taper = lambda running_sum, b, mn, mx: np.float32(1.0)
if hasattr(sm, "_project_onto_bounds"):
    sm._project_onto_bounds = lambda *a, **k: None

records = []
real_convolve = sm.CONVOLVE


def recording_convolve(field, kernel, device="cpu"):
    try:
        result = real_convolve(field, kernel, device=device)
    except TypeError:                       # EGU-era: no device kwarg
        result = real_convolve(field, kernel)
    nz = field.shape[2]
    iz = min(nz - 1, int(round(LEVEL_M / DOMAIN_HEIGHT * nz)))
    level = np.asarray(field[:, :, iz], dtype=np.float64)
    nonzero = level[level != 0.0]
    records.append({
        "shape": field.shape,
        "amp": float(np.abs(nonzero).mean()) if nonzero.size else np.nan,
        "inc": np.asarray(result[:, :, iz], dtype=np.float32),
    })
    return result


sm.CONVOLVE = recording_convolve


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "era"
    head = subprocess.run(["git", "-C", str(STEAM_REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "-C", str(STEAM_REPO), "diff", "--quiet"]).returncode:
        head += "+dirty"
    prof = np.load(HERE / "stats" / "icon_lem_snap0.npz")
    z = prof["z_profile"]
    out = Path(tempfile.gettempdir()) / f"ladder_{label}.nc"
    if out.exists():
        out.unlink()

    import inspect
    kwargs = dict(
        h_profile=prof["h_profile"], qt_profile=prof["qt_profile"],
        nx=512, ny=512, dx=1_000_000.0 / 512, dy=1_000_000.0 / 512,
        outer_scale=500_000.0,
        spheroscale=np.full(z.size, 30.0),
        domain_height=DOMAIN_HEIGHT, profile_dz=50.0,
        output_path=str(out),
        sparsity_factors=(1, 1, 1),
        surface_pressure=float(prof["surface_pressure"]),
        seed=7,
        h_min=250.0 * CP, h_max=420.0 * CP, qt_min=0.0, qt_max=0.03,
        anisotropy="canonical", compress=False, device="cpu",
    )
    accepted = set(inspect.signature(sm.simulate).parameters)
    sm.simulate(**{k: v for k, v in kwargs.items() if k in accepted})

    import netCDF4
    import scaleinvariance as si
    n_classes_calls = len(records)
    # Infer calls per class (flux + h + qt = 3, or h + qt = 2).
    k_values = None
    for per in (3, 2):
        if n_classes_calls % per == 0:
            n_classes = n_classes_calls // per
            kv = 500_000.0 / 2.0 ** np.arange(n_classes)
            if kv[-1] >= 2 * kwargs["dx"] * 0.9:
                k_values, per_class = kv, per
                break
    assert k_values is not None, f"cannot infer classes from {n_classes_calls} calls"
    offset_h = per_class - 2   # h is second-to-last call of each class

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        zf = ds.variables["z"][:]
        izf = int(np.argmin(np.abs(zf - LEVEL_M)))
        final_h = np.asarray(ds.variables["h"][:, :, izf], dtype=np.float64)
        dx_final = float(ds.dx)

    print(f"\n=== {label} [{head}] — h at ~7 km, {n_classes} classes, "
          f"{per_class} CONVOLVE calls/class ===")
    print("k [km]    <|A|>        Mhat1(incr, lag=k)")
    amps, deps = [], []
    for i in range(n_classes):
        rec = records[per_class * i + offset_h]
        lags, F = si.haar_fluctuation(rec["inc"], order=1.0, axis=0,
                                      lags=np.array([2]), periodic=True)
        amps.append(rec["amp"])
        deps.append(float(F[0]))
        print(f"{k_values[i]/1000:8.1f}  {amps[-1]:.4e}   {deps[-1]:.4e}")
    amps, deps = np.array(amps), np.array(deps)
    ok = np.isfinite(amps)
    slope_amp = np.polyfit(np.log(k_values[ok]), np.log(amps[ok]), 1)[0]
    slope_dep = np.polyfit(np.log(k_values), np.log(deps), 1)[0]

    lag_cells = np.unique((k_values / dx_final).astype(int))
    lag_cells = np.sort(lag_cells[lag_cells >= 2])
    lags, Ff = si.haar_fluctuation(final_h, order=1.0, axis=0,
                                   lags=lag_cells, periodic=True)
    slope_final = np.polyfit(np.log(lags * dx_final), np.log(Ff), 1)[0]
    print(f"slopes: <|A|> {slope_amp:+.3f}   deposits {slope_dep:+.3f}   "
          f"final field {slope_final:+.3f}   (H design = +0.45)")
    np.savez(HERE / "stats" / f"ladder_compare_{label}.npz",
             k_values=k_values, amps=amps, deposits=deps,
             final_lags_m=lags * dx_final, final_F=Ff, head=head)
    out.unlink()


if __name__ == "__main__":
    main()
