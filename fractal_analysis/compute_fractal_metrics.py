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

Clouds are the tau > 1 mask of the vertically integrated optical depth
stored in each member's parent group by run_paper_squares.py.

Every matched file is pooled into ONE ensemble and passed to objscale in a
single call per metric. This matters: these estimators are regressions, not
linear operations, so computing per file and averaging gives a different
(wrong) answer.

Edit PATTERN below to choose which runs are read.

Usage: python compute_fractal_metrics.py
"""

from pathlib import Path

import numpy as np
import netCDF4
import objscale

# Which files in runs/square/ to read and pool into one ensemble.
PATTERN = "sq1km_C1large*.nc"

TAU_THRESHOLD = 1.0        # tau > this is cloud

POINT_REDUCTION_FACTOR = 10

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "square"
OUT = Path(__file__).resolve().parent / "fractal_metrics.npz"


def load_masks(pattern):
    """Load the tau > 1 cloud mask and pixel size from each matched file."""
    paths = sorted(RUNS.glob(pattern))
    if not paths:
        raise SystemExit(f"no files in {RUNS} match {pattern!r}")
    masks, dx_km = [], None
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
        mask = (tau > TAU_THRESHOLD).astype(np.float32)
        masks.append(mask)
        print(f"  {path.name}: {mask.shape}, cloud cover {mask.mean():.3f}",
              flush=True)
    return masks, dx_km, [p.name for p in paths]


def main():
    print(f"reading {PATTERN} from {RUNS}", flush=True)
    masks, dx_km, names = load_masks(PATTERN)
    print(f"{len(masks)} members at dx = {dx_km:g} km\n", flush=True)

    # Pixel sizes in km, one grid shared by every member (all same shape).
    x_sizes = np.full(masks[0].shape, dx_km)
    y_sizes = np.full(masks[0].shape, dx_km)

    print("ensemble fractal dimension D_e (correlation dimension)...",
          flush=True)
    D_e, C_bins, C_l = objscale.ensemble_correlation_dimension(
        masks, x_sizes=x_sizes, y_sizes=y_sizes,
        point_reduction_factor=POINT_REDUCTION_FACTOR, return_C_l=True)

    print("individual fractal dimension D_f...", flush=True)
    D_f, ind_log_length, ind_log_perimeter = objscale.individual_fractal_dimension(
        masks, x_sizes=x_sizes, y_sizes=y_sizes, return_values=True)

    print("area size distribution tau_area...", flush=True)
    tau_area, (area_log_bins, area_log_counts) = \
        objscale.finite_array_powerlaw_exponent(
            masks, "area", x_sizes=x_sizes, y_sizes=y_sizes,
            return_counts=True)

    print("nested-perimeter size distribution tau_per...", flush=True)
    tau_per, (per_log_bins, per_log_counts) = \
        objscale.finite_array_powerlaw_exponent(
            masks, "nested perimeter", x_sizes=x_sizes, y_sizes=y_sizes,
            return_counts=True)

    cover = float(np.mean([m.mean() for m in masks]))

    np.savez(
        OUT,
        pattern=PATTERN, members=np.array(names), n_members=len(masks),
        dx_km=dx_km, tau_threshold=TAU_THRESHOLD, cover=cover,
        objscale_version=objscale.__version__,
        D_e=D_e, C_bins=C_bins, C_l=C_l,
        D_f=D_f, ind_log_length=ind_log_length,
        ind_log_perimeter=ind_log_perimeter,
        tau_area=tau_area, area_log_bins=area_log_bins,
        area_log_counts=area_log_counts,
        tau_per=tau_per, per_log_bins=per_log_bins,
        per_log_counts=per_log_counts,
    )

    print(f"\n{len(masks)} members, cloud cover {cover:.3f}")
    print(f"  D_f      = {D_f:.3f}")
    print(f"  D_e      = {D_e:.3f}")
    print(f"  tau_area = {tau_area:.3f}")
    print(f"  tau_per  = {tau_per:.3f}")
    if not np.all(np.isfinite([D_f, D_e, tau_area, tau_per])):
        print("\nNOTE: a nan exponent usually means too few size bins "
              "cleared objscale's 30-count floor. Pool more members.")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
