#!/usr/bin/env python3
"""Where does the k^H_h ladder break? (2026-07-27, the flat-xi(1) problem)

Instruments a small channel run (512 x 128 at dx = 3 km, outer 96 km,
constant 10 m spheroscale, c = 0.101, icon_lem profile) by wrapping
steam.simulate.CONVOLVE. Per size class and scalar this records:

  1. the realized mean absolute turbulon amplitude <|A|> over the sparse
     centers at the level nearest 7 km — Thomas's hypothesis is that
     these do NOT follow the appendix ladder C_k ~ k^H_h;
  2. the mean absolute Haar fluctuation (scaleinvariance, lag = 2 cells
     = k on the class's own grid) of the class's deposited increment at
     the same level — the per-class composition;
  3. after the run, the FINAL field's Haar fluctuation at each class
     scale at the same level — accumulation/cancellation effects.

Comparing the log-slopes of (1), (2), (3) against H_h = 0.45 localizes
the break: amplitude law vs deposit vs accumulation.
"""

from pathlib import Path

import numpy as np
import scaleinvariance as si

import importlib
sm = importlib.import_module("steam.simulate")
sm.H_h = 0.45
C1_TARGET = 0.05
sm.FLUX_SCALE = (C1_TARGET / 3.097) ** (1 / 1.8)

from steam.constants import specific_heat_dry_air as cp

HERE = Path(__file__).parent
STATS = HERE.parent / "stats"
LEVEL_M = 7000.0
DOMAIN_HEIGHT = 20000.0

records = []          # one dict per CONVOLVE call
real_convolve = sm.convolve_fft_xy_oa_z


def recording_convolve(field, kernel, device="cpu"):
    result = real_convolve(field, kernel, device=device)
    nz = field.shape[2]
    iz = min(nz - 1, int(round(LEVEL_M / DOMAIN_HEIGHT * nz)))
    level = np.asarray(field[:, :, iz], dtype=np.float64)
    nonzero = level[level != 0.0]
    records.append({
        "shape": field.shape,
        "amp_absmean_7km": float(np.abs(nonzero).mean()) if nonzero.size else np.nan,
        "n_centers_7km": int(nonzero.size),
        "increment_7km": np.asarray(result[:, :, iz], dtype=np.float32),
    })
    return result


sm.CONVOLVE = recording_convolve

from steam.simulate import simulate  # noqa: E402


def main():
    src = np.load(STATS / "icon_lem_snap0.npz")
    z = src["z_profile"]
    out = Path("/tmp/diag_ladder.nc")
    if out.exists():
        out.unlink()
    simulate(
        src["h_profile"], src["qt_profile"],
        nx=512, ny=128, dx=3000.0, dy=3000.0,
        outer_scale=96000.0,
        spheroscale=np.full(z.size, 10.0),
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=50.0,
        output_path=str(out),
        surface_pressure=float(src["surface_pressure"]),
        seed=7,
        h_min=float(src["h_profile"].min()) - 10 * cp,
        h_max=float(src["h_profile"].max()) + 1.0,
        qt_min=0.0, qt_max=0.03,
        compress=False,
        device="cpu",
    )

    # Calls per class: flux, h, qt.
    n_classes = len(records) // 3
    k_values = 96000.0 / 2.0 ** np.arange(n_classes)
    print(f"\n{n_classes} classes, k = {k_values.tolist()}")

    import netCDF4
    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        zf = ds.variables["z"][:]
        izf = int(np.argmin(np.abs(zf - LEVEL_M)))
        final = {name: np.asarray(ds.variables[name][:, :, izf],
                                  dtype=np.float64)
                 for name in ("h", "qt")}
        dx_final = 3000.0

    for si_name, offset in (("h", 1), ("qt", 2)):
        print(f"\n=== {si_name} ===")
        print("k [km]   <|A|>@7km    Mhat1(incr, lag=k)   n_centers")
        amps, deposits = [], []
        for i in range(n_classes):
            rec = records[3 * i + offset]
            inc = rec["increment_7km"]
            # Haar at lag = 2 cells = k on the class's own grid, periodic x.
            lags, F = si.haar_fluctuation(inc, order=1.0, axis=0,
                                          lags=np.array([2]), periodic=True)
            amps.append(rec["amp_absmean_7km"])
            deposits.append(float(F[0]))
            print(f"{k_values[i]/1000:7.0f}  {amps[-1]:.4e}   {deposits[-1]:.4e}"
                  f"        {rec['n_centers_7km']}")
        amps = np.array(amps)
        deposits = np.array(deposits)
        slope_amp = np.polyfit(np.log(k_values), np.log(amps), 1)[0]
        slope_dep = np.polyfit(np.log(k_values), np.log(deposits), 1)[0]

        # Final-field Haar at each class scale (lag = k / dx cells).
        lag_cells = (k_values / dx_final).astype(int)
        lag_cells = np.unique(lag_cells[lag_cells >= 2])
        lags, Ff = si.haar_fluctuation(final[si_name], order=1.0, axis=0,
                                       lags=np.sort(lag_cells),
                                       periodic=True)
        slope_final = np.polyfit(np.log(lags * dx_final),
                                 np.log(Ff), 1)[0]
        print(f"log-slope vs k: amplitudes {slope_amp:+.3f}, "
              f"deposits {slope_dep:+.3f}, final field {slope_final:+.3f} "
              f"(design H_h = +0.45)")
        print("final-field Mhat1 at class lags:",
              [f"{v:.3e}" for v in Ff])


if __name__ == "__main__":
    main()
