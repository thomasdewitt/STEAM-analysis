#!/usr/bin/env python3
"""Compute the paper's four fractal metrics for a set of runs in runs/square/.

The four metrics, as defined in the main text (Sect. "cloud geometry"):

  D_f        individual fractal dimension -- how convoluted a single cloud
             edge is, from perimeter vs. size scaling across objects
  D_e        ensemble fractal dimension -- the correlation dimension of the
             cloud-edge point set, capturing both edge roughness and how
             clouds are organized across scales
  tau_area   cloud area size-distribution exponent
  tau_per    cloud perimeter size-distribution exponent (nested perimeter)

All four come from objscale with DEFAULT parameters throughout, following
DeWitt & Garrett (2024) and DeWitt et al. (2026). No fitting ranges, bin
counts or thresholds are overridden.

Clouds are albedo masks of the vertically integrated optical depth stored
in each member's parent group by run_paper_squares.py, at each of three
thresholds R = 0.1, 0.2, 0.3 -- the same thresholds as the satellite
retrievals of DeWitt et al. (2026), so the exponents are directly
comparable. See albedo.py for the two-stream mapping; every metric is
computed independently at each threshold.

Every matched file is pooled into ONE ensemble and passed to objscale in a
single call per metric. This matters: these estimators are regressions, not
linear operations, so computing per file and averaging gives a different
(wrong) answer.

Edit PATTERN below to choose which runs are read.

Usage: python compute_fractal_metrics.py
"""

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import netCDF4
import objscale

from albedo import ALBEDO_THRESHOLDS, tau_for_albedo, threshold_tag

# Which files in runs/square/ to read and pool into one ensemble. Override
# on the command line to do another set; the output is named after it, so
# the sets do not overwrite each other.
PATTERN = "sq1km_C1large*.nc"

# Correlation-integral sampling: thinned 10x for the pooled square
# campaign, where ten members already oversample the domain.
POINT_REDUCTION_FACTOR = 10

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "square"
HERE = Path(__file__).resolve().parent


def out_path(pattern):
    """fractal_metrics_<set>.npz, from the pattern's set tag."""
    tag = pattern.split("_")[1].split("*")[0]
    return HERE / f"fractal_metrics_{tag}.npz"


def load_tau(pattern):
    """Load the stored column optical depth and pixel size from each file."""
    paths = sorted(RUNS.glob(pattern))
    if not paths:
        raise SystemExit(f"no files in {RUNS} match {pattern!r}")
    taus, dx_km = [], None
    for path in paths:
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            if "parent" not in ds.groups:
                raise SystemExit(f"{path.name} has no parent group")
            parent = ds.groups["parent"]
            if "tau" not in parent.variables:
                raise SystemExit(
                    f"{path.name} has no parent tau -- it predates the "
                    f"optical-depth step in run_paper_squares.py")
            tau = parent.variables["tau"][:]
            dx = float(parent.getncattr("dx")) / 1000.0    # m -> km
        if dx_km is None:
            dx_km = dx
        elif abs(dx - dx_km) > 1e-9:
            raise SystemExit(
                f"{path.name} has dx = {dx} km but the first file has "
                f"{dx_km} km; pool only runs on a common grid")
        taus.append(tau)
        print(f"  {path.name}: {tau.shape}", flush=True)
    return taus, dx_km, [p.name for p in paths]


@contextmanager
def warned():
    """Record whether objscale warned during a call.

    objscale warns when a fit rests on too few bins or too narrow a range.
    That is exactly the caveat a reader of the table needs attached to the
    specific number, so it is captured per metric and carried through to the
    figure and the LaTeX table rather than scrolling past in a log.
    """
    flag = [False]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield flag
    flag[0] = any(issubclass(c.category, UserWarning) for c in caught)
    for c in caught:
        print(f"      objscale: {c.message}", flush=True)


def metrics_at(masks, x_sizes, y_sizes,
               point_reduction_factor=POINT_REDUCTION_FACTOR,
               distributions=True):
    """The four metrics and their scaling functions for one mask set.

    point_reduction_factor thins the correlation-integral sampling only. It
    is 10 here because ten pooled members already oversample the domain;
    a single-snapshot caller should pass objscale's default of 1.

    distributions=False skips tau_area and tau_per. The two dimensions are
    measured per object and survive a single snapshot; the size
    distributions are counts per size bin and do not, so a caller with one
    field may want the dimensions without them.
    """
    with warned() as w_De:
        D_e, C_bins, C_l = objscale.ensemble_correlation_dimension(
            masks, x_sizes=x_sizes, y_sizes=y_sizes,
            point_reduction_factor=point_reduction_factor, return_C_l=True)
    with warned() as w_Df:
        D_f, ind_log_length, ind_log_perimeter = \
            objscale.individual_fractal_dimension(
                masks, x_sizes=x_sizes, y_sizes=y_sizes, return_values=True)
    result = dict(
        D_e=D_e, C_bins=C_bins, C_l=C_l, warn_D_e=w_De[0],
        D_f=D_f, ind_log_length=ind_log_length,
        ind_log_perimeter=ind_log_perimeter, warn_D_f=w_Df[0],
        cover=float(np.mean([m.mean() for m in masks])))
    if not distributions:
        return result

    with warned() as w_area:
        tau_area, (area_log_bins, area_log_counts) = \
            objscale.finite_array_powerlaw_exponent(
                masks, "area", x_sizes=x_sizes, y_sizes=y_sizes,
                return_counts=True)
    with warned() as w_per:
        tau_per, (per_log_bins, per_log_counts) = \
            objscale.finite_array_powerlaw_exponent(
                masks, "nested perimeter", x_sizes=x_sizes, y_sizes=y_sizes,
                return_counts=True)
    result.update(
        tau_area=tau_area, area_log_bins=area_log_bins,
        area_log_counts=area_log_counts, warn_tau_area=w_area[0],
        tau_per=tau_per, per_log_bins=per_log_bins,
        per_log_counts=per_log_counts, warn_tau_per=w_per[0])
    return result


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else PATTERN
    out_file = out_path(pattern)
    print(f"reading {pattern} from {RUNS}", flush=True)
    taus, dx_km, names = load_tau(pattern)
    print(f"{len(taus)} members at dx = {dx_km:g} km\n", flush=True)

    # One grid shared by every member (all the same shape).
    x_sizes = np.full(taus[0].shape, dx_km)
    y_sizes = np.full(taus[0].shape, dx_km)

    out = dict(pattern=pattern, members=np.array(names), n_members=len(taus),
               dx_km=dx_km, objscale_version=objscale.__version__,
               thresholds=np.array(ALBEDO_THRESHOLDS))

    for R in ALBEDO_THRESHOLDS:
        cut = tau_for_albedo(R)
        tag = threshold_tag(R)
        masks = [(t > cut).astype(np.float32) for t in taus]
        cover = float(np.mean([m.mean() for m in masks]))
        print(f"--- albedo > {R:g}  (tau > {cut:.3f}), cover {cover:.3f} ---",
              flush=True)
        for key, value in metrics_at(masks, x_sizes, y_sizes).items():
            out[f"{tag}_{key}"] = value
        print(f"  D_f {out[f'{tag}_D_f']:.3f}   D_e {out[f'{tag}_D_e']:.3f}   "
              f"tau_area {out[f'{tag}_tau_area']:.3f}   "
              f"tau_per {out[f'{tag}_tau_per']:.3f}", flush=True)

    np.savez(out_file, **out)
    if not all(np.isfinite(out[f"{threshold_tag(R)}_{k}"])
               for R in ALBEDO_THRESHOLDS
               for k in ("D_f", "D_e", "tau_area", "tau_per")):
        print("\nNOTE: a nan exponent usually means too few size bins "
              "cleared objscale's 30-count floor. Pool more members.")
    print(f"\nwrote {out_file.name}")


if __name__ == "__main__":
    main()
