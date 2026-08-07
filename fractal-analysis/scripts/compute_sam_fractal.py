#!/usr/bin/env python3
"""The four fractal metrics for the two SAM cases, TWPICE and GATE.

Identical machinery to compute_fractal_metrics.py -- same objscale calls,
same default parameters, same three albedo thresholds -- run on the LES
hosts instead of the STEAM square campaign, so the two figures can be read
against each other.

RESOLUTION. Both cases are taken at their NATIVE 2048^2 x 100 m, not
coarsened onto STEAM's grid. Fractal dimensions and size-distribution
exponents are scaling measures, so the same reasoning applies as to the
fluctuation functions: coarsening would discard exactly the small-scale
structure the exponents describe. It does mean the range of scales here
(0.1 to 205 km) is not the square campaign's (1 to 2048 km).

ONE SNAPSHOT EACH. The square campaign pools many members before fitting;
here there is a single field per case, so every exponent rests on one
realization. objscale's 30-count floor per size bin is the thing to watch --
a nan means a distribution had too few populated bins, and pooling is the
only honest fix. Treat the numbers as indicative rather than measured, and
read the scaling functions in the figure before quoting any of them.

Optical depth comes from cloudyview's SAM relationships, the same call
run_steam_simulations.py makes for the STEAM squares. GATE's liquid/ice split is
SAM's linear ramp applied at native resolution (see make_input_profiles.py).

Usage: python compute_sam_fractal.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import netCDF4
import objscale

from albedo import ALBEDO_THRESHOLDS, tau_for_albedo, threshold_tag
from compute_fractal_metrics import metrics_at

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # fractal-analysis/
REPO = BASE.parent
OUTPUT = BASE / "output"
OUT = OUTPUT / "sam_fractal_metrics.npz"

sys.path.insert(0, str(REPO))
from make_input_profiles import read_var, liquid_fraction   # noqa: E402

# cloudyview is not installed in this venv; load the one module by path, as
# run_steam_simulations.py does.
_spec = importlib.util.spec_from_file_location(
    "cv_optical_depth",
    Path.home() / "code-and-data/cloudyview/cloudyview/optical_depth.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

TWPICE = REPO / "data" / "twpice"
TWPICE_SNAPSHOT = "0000003450"
GATE = (REPO / "data" / "gate"
        / "GATE_IDEAL_S_2048x2048x256_100m_2s_2048_0000041400.nc")
CASES = ("twpice", "gate")
DX_KM = 0.1
# objscale's default: one snapshot per case, so the correlation
# integral is not thinned the way the pooled square campaign's is.
POINT_REDUCTION_FACTOR = 1
# The two size distributions rest on counts per size bin, which one snapshot
# per case supports poorly -- objscale warns that the fits span barely a
# decade. Set False to compute the two dimensions only; plot_sam_fractal.py
# carries the same flag and must agree with whatever was computed.
RUN_SAM_DISTRIBUTIONS = True


def twpice_tau():
    """Column optical depth of the TWPICE snapshot, native 100 m."""
    from make_input_profiles import _twpice_field
    z = np.asarray(read_var(TWPICE / f"TWPICE_LPT_3D_QV_{TWPICE_SNAPSHOT}.nc",
                            "z"), dtype=np.float64)
    # cloudyview wants (nx, ny, nz) in g/kg; the files are (y, x, z).
    qc = np.moveaxis(_twpice_field(
        TWPICE / f"TWPICE_LPT_3D_QC_{TWPICE_SNAPSHOT}.nc", "QC", True), 0, -1)
    qi = np.moveaxis(_twpice_field(
        TWPICE / f"TWPICE_LPT_3D_QI_{TWPICE_SNAPSHOT}.nc", "QI", True), 0, -1)
    return cv.vertically_integrated_optical_depth(qc, z, iwc=qi)


def gate_tau():
    """Column optical depth of the GATE snapshot, native 100 m.

    QN is the combined condensate, split by SAM's linear ramp at native
    resolution before the integration.
    """
    with netCDF4.Dataset(GATE) as ds:
        ds.set_auto_mask(False)
        z = np.asarray(ds.variables["z"][:], dtype=np.float64)
        T = np.asarray(ds.variables["TABS"][0], dtype=np.float64)
        liquid = liquid_fraction(T).astype(np.float32)
        del T
        qn = np.asarray(ds.variables["QN"][0], dtype=np.float32)   # g/kg
    qc = np.moveaxis(qn * liquid, 0, -1)
    qi = np.moveaxis(qn * (1.0 - liquid), 0, -1)
    del qn, liquid
    return cv.vertically_integrated_optical_depth(qc, z, iwc=qi)


TAU_OF = {"twpice": twpice_tau, "gate": gate_tau}


def main():
    out = {"cases": np.array(CASES), "dx_km": DX_KM,
           "thresholds": np.array(ALBEDO_THRESHOLDS),
           "distributions": RUN_SAM_DISTRIBUTIONS,
           "point_reduction_factor": POINT_REDUCTION_FACTOR,
           "objscale_version": objscale.__version__}

    for case in CASES:
        print(f"=== {case} ===", flush=True)
        tau = TAU_OF[case]()
        if not np.all(np.isfinite(tau)):
            raise SystemExit(f"non-finite tau for {case}")
        print(f"  tau {tau.shape}, max {tau.max():.1f}", flush=True)
        x_sizes = np.full(tau.shape, DX_KM)
        y_sizes = np.full(tau.shape, DX_KM)

        for R in ALBEDO_THRESHOLDS:
            cut = tau_for_albedo(R)
            tag = f"{case}_{threshold_tag(R)}"
            mask = (tau > cut).astype(np.float32)
            print(f"  albedo > {R:g} (tau > {cut:.3f}), "
                  f"cover {mask.mean():.4f}", flush=True)
            for key, value in metrics_at(
                    [mask], x_sizes, y_sizes,
                    point_reduction_factor=POINT_REDUCTION_FACTOR,
                    distributions=RUN_SAM_DISTRIBUTIONS).items():
                out[f"{tag}_{key}"] = value
            line = (f"    D_f {out[f'{tag}_D_f']:.3f}   "
                    f"D_e {out[f'{tag}_D_e']:.3f}")
            if RUN_SAM_DISTRIBUTIONS:
                line += (f"   tau_area {out[f'{tag}_tau_area']:.3f}   "
                         f"tau_per {out[f'{tag}_tau_per']:.3f}")
            print(line, flush=True)
        del tau

    OUTPUT.mkdir(exist_ok=True)
    np.savez(OUT, **out)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
