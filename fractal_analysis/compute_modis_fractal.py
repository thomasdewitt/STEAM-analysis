#!/usr/bin/env python3
"""The four fractal metrics for the MODIS archive of DeWitt et al. (2026).

Identical machinery to compute_fractal_metrics.py and compute_sam_fractal.py
-- the same objscale calls at their default parameters, the same three
thresholds -- so the retrieval, the LES and STEAM all enter the comparison
table through one pipeline. The only thing that changes is where the cloud
mask comes from: here it is band 1 reflectance thresholded directly, rather
than a two-stream albedo computed from a modelled column optical depth. That
is the point of having moved the model masks onto albedo in the first place
(see albedo.py); the thresholds mean the same thing on both sides.

The loader in modis.py is written from the file specification rather than
taken from the code behind that paper, so these numbers are a reproduction
of its Table 1 and not a restatement of it.

SOLAR ZENITH. Run twice. The default leaves the L1B reflectance as stored,
which is rho*cos(theta_0); `--sza` divides the cosine out, giving a
bidirectional reflectance factor comparable to the overhead-sun two-stream
albedo used on the model side. The correction only ever raises reflectance,
so it can only raise cloud cover at a fixed threshold, and the two runs
bracket the choice rather than settling it.

ONE GRID. objscale's ensemble estimators take a single x_sizes/y_sizes pair
for every array, so all 72 granules are cut to a common window (see
modis.py) and one footprint grid is used throughout. The granule-to-granule
spread in that grid is measured and reported rather than assumed away: MODIS
scan geometry repeats every granule, so it should be small, and if it is not
then the single-grid premise is what broke.

Usage: python compute_modis_fractal.py [--sza]
"""

import sys
from pathlib import Path

import numpy as np
import objscale

import modis
from albedo import ALBEDO_THRESHOLDS, threshold_tag
from compute_fractal_metrics import metrics_at

HERE = Path(__file__).resolve().parent

# The archive is 72 granules of ~2.5e6 pixels, far more edge points than the
# correlation integral needs; the square campaign uses 10 for ten members.
POINT_REDUCTION_FACTOR = 100


def out_path(solar_correction):
    tag = "_sza" if solar_correction else ""
    return HERE / f"modis_fractal_metrics{tag}.npz"


def load_archive(solar_correction):
    """(masks-ready reflectance list, x_sizes, y_sizes, diagnostics)."""
    pairs = modis.granules()
    print(f"{len(pairs)} granules in {modis.ARCHIVE}", flush=True)

    rows, j0, j1 = modis.common_window(pairs)
    print(f"common window: {rows} rows x {j1 - j0} columns "
          f"(cols {j0}-{j1}), sensor zenith <= "
          f"{modis.MAX_SENSOR_ZENITH_DEG:g} deg", flush=True)
    window = (slice(0, rows), slice(j0, j1))

    fields, x_sum, y_sum = [], None, None
    x_lo = x_hi = y_lo = y_hi = None
    invalid = 0

    for n, (l1b_path, geo_path) in enumerate(pairs, 1):
        lat, lon = modis.read_geolocation(geo_path)
        # Footprints come from the full swath, then are cropped: a centred
        # difference at the window edge needs the column outside it.
        x_full, y_full = modis.pixel_sizes_km(lat, lon)
        x, y = x_full[window], y_full[window]
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise SystemExit(f"{geo_path.name}: non-finite pixel footprint "
                             f"inside the common window")
        if x_sum is None:
            x_sum, y_sum = x.copy(), y.copy()
            x_lo, x_hi, y_lo, y_hi = x.copy(), x.copy(), y.copy(), y.copy()
        else:
            x_sum += x
            y_sum += y
            np.minimum(x_lo, x, out=x_lo)
            np.maximum(x_hi, x, out=x_hi)
            np.minimum(y_lo, y, out=y_lo)
            np.maximum(y_hi, y, out=y_hi)

        R, valid = modis.read_reflectance(l1b_path, geo_path, solar_correction)
        R, valid = R[window], valid[window]
        invalid += int((~valid).sum())
        # nan, not zero: metrics_at hands the nan to the object-based
        # estimators so a cloud running off the edge of a gap is dropped
        # rather than counted as a small complete one, and zero-fills only
        # for the correlation dimension, which cannot represent a gap.
        fields.append(np.where(valid, R, np.nan).astype(np.float32))
        if n % 12 == 0 or n == len(pairs):
            print(f"  read {n}/{len(pairs)}", flush=True)

    x_sizes = (x_sum / len(pairs)).astype(np.float64)
    y_sizes = (y_sum / len(pairs)).astype(np.float64)
    # How well one grid stands in for all 72. Reported as a distribution,
    # not a maximum: the worst pixel in the archive runs to tens of percent,
    # but it is a handful of pixels of geolocation jitter, and quoting it
    # alone would condemn a premise that holds across the other 99.8%.
    relative = np.concatenate([((x_hi - x_lo) / x_sizes).ravel(),
                               ((y_hi - y_lo) / y_sizes).ravel()])
    spread = float(np.percentile(relative, 99))
    print(f"footprints {x_sizes.min():.3f}-{x_sizes.max():.3f} km along scan, "
          f"{y_sizes.min():.3f}-{y_sizes.max():.3f} km along track", flush=True)
    print(f"granule-to-granule footprint spread (the single-grid premise): "
          f"median {np.median(relative) * 100:.3f}%, p99 {spread * 100:.2f}%, "
          f"max {relative.max() * 100:.1f}%; "
          f"{(relative > 0.05).mean() * 100:.3f}% of pixels over 5%",
          flush=True)
    print(f"pixels without a measurement: "
          f"{invalid / (len(pairs) * x_sizes.size) * 100:.4f}%", flush=True)
    return fields, x_sizes, y_sizes, spread, [p.name for _, p in pairs]


def main():
    solar_correction = "--sza" in sys.argv[1:]
    print(f"solar zenith correction: {solar_correction}\n", flush=True)
    fields, x_sizes, y_sizes, spread, names = load_archive(solar_correction)

    out = dict(n_granules=len(fields), granules=np.array(names),
               solar_correction=solar_correction,
               footprint_spread=spread,
               x_size_km_range=np.array([x_sizes.min(), x_sizes.max()]),
               y_size_km_range=np.array([y_sizes.min(), y_sizes.max()]),
               max_sensor_zenith=modis.MAX_SENSOR_ZENITH_DEG,
               point_reduction_factor=POINT_REDUCTION_FACTOR,
               objscale_version=objscale.__version__,
               thresholds=np.array(ALBEDO_THRESHOLDS))

    for R in ALBEDO_THRESHOLDS:
        tag = threshold_tag(R)
        # np.where, not `f > R`: a comparison against nan is False, which
        # would quietly turn every gap into clear sky and undo the whole
        # point of carrying nan this far.
        masks = [np.where(np.isnan(f), np.nan, f > R).astype(np.float32)
                 for f in fields]
        cover = float(np.mean([np.nanmean(m) for m in masks]))
        print(f"\n--- reflectance > {R:g}, cover {cover:.4f} ---", flush=True)
        for key, value in metrics_at(
                masks, x_sizes, y_sizes,
                point_reduction_factor=POINT_REDUCTION_FACTOR).items():
            out[f"{tag}_{key}"] = value
        del masks
        print(f"  D_f {out[f'{tag}_D_f']:.3f}   D_e {out[f'{tag}_D_e']:.3f}   "
              f"tau_area {out[f'{tag}_tau_area']:.3f}   "
              f"tau_per {out[f'{tag}_tau_per']:.3f}", flush=True)

    path = out_path(solar_correction)
    np.savez(path, **out)
    print(f"\nwrote {path.name}")


if __name__ == "__main__":
    main()
