#!/usr/bin/env python3
"""Read MODIS band 1 reflectance and per-pixel footprints from L1B granules.

Written from the MOD021KM/MOD03 file specifications rather than ported from
the code behind DeWitt et al. (2026), deliberately: recomputing the
retrieval independently makes the comparison in fractal_table.tex a
reproduction of that paper's numbers rather than a restatement of them. If
the exponents come out on top of the published ones, that is evidence; if
they do not, the disagreement is the interesting result and should not be
explained away by shared code.

WHAT IS READ. Band 1 (0.645 um) from `EV_250_Aggr1km_RefSB`, the 250 m
bands aggregated onto the 1 km grid, so every field here is 2030 x 1354 at
nominal 1 km. Geolocation, solar zenith, and the pixel footprints come from
the matching MOD03 granule.

SOLAR ZENITH. The L1B `reflectance_scales` give pi*L / (E_sun / d^2) -- the
solar zenith angle is NOT divided out, so the stored quantity is
rho * cos(theta_0), smaller than the bidirectional reflectance factor by up
to a factor of ten near the terminator. Whether to divide it out is a real
choice and it moves the thresholds, so it is a parameter here rather than a
buried convention: `read_reflectance(..., solar_correction=True)` divides by
cos(theta_0), giving a quantity comparable to the overhead-sun two-stream
albedo the model side uses (see albedo.py); False leaves the L1B value
alone. compute_modis_fractal.py runs both.

PIXEL SIZE. MODIS scans +-55 deg, so a pixel at the swath edge covers about
4.8 km along-scan and 2.0 km along-track against 1 km x 1 km at nadir. That
is a factor of five in area across a single granule, which objscale's
x_sizes/y_sizes exist to absorb -- passing a uniform 1 km would put the edge
of the swath four scale octaves away from where it belongs. Both are
measured from the MOD03 latitude/longitude by great-circle distance between
neighbouring pixel centres, not from a nominal scan-angle formula.

The bow-tie is the one wrinkle. Adjacent 10-detector scans overlap at the
swath edge, so a centred along-track difference straddling a scan boundary
measures the overlap rather than the pixel. Along-track differences are
therefore taken only between rows within the same scan; every row has at
least one such neighbour, since scans are ten rows deep. Along-scan
differences need no such care.

TRIMMING. The outer swath is cut at 60 degrees sensor zenith. Sensor zenith
is a function of scan angle and so of column, to well under a pixel, which
is what makes the cut a contiguous column window and lets the granule stay
a rectangle. `common_window` intersects those windows over the archive and
takes the shortest granule's row count, so every scene enters the ensemble
on one grid -- objscale's ensemble estimators take a single x_sizes/y_sizes
pair for all arrays, so a common grid is required, not merely tidy.
"""

from pathlib import Path

import numpy as np
from pyhdf.SD import SD, SDC

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / "data" / "MODIS_data"

BAND1_SDS = "EV_250_Aggr1km_RefSB"
BAND1_INDEX = 0                 # `band_names` on that SDS is "1,2"

# MODIS L1B reserves the top of the unsigned range for status codes; only
# `valid_range` carries data. 65533 is a saturated detector, which is a
# measurement -- the scene was brighter than the band could record -- so it
# becomes the largest representable reflectance instead of a dropout. Every
# other code is a genuine dropout.
L1B_SATURATED = 65533

# Below this the cosine correction divides by less than ~0.1 and amplifies
# both the reflectance and its error without bound. Terminator pixels are
# dropped rather than corrected.
MAX_SOLAR_ZENITH_DEG = 84.0

# The outer swath is trimmed at this view angle. Beyond it a pixel is several
# km across, its footprint is a smeared parallelogram rather than a square,
# and it sees cloud sides as much as cloud tops -- none of which the
# footprint areas passed to objscale can repair.
MAX_SENSOR_ZENITH_DEG = 60.0

DETECTORS_PER_SCAN = 10         # 1 km bands: 203 scans x 10 rows = 2030
EARTH_RADIUS_KM = 6371.0088     # IUGG mean radius


def granules(archive=ARCHIVE):
    """[(l1b_path, geo_path)] for every granule with both files present.

    Paired on the acquisition stamp `A<yyyyddd>.<hhmm>`, which is the part of
    the filename shared by the two products; the trailing production date is
    not.
    """
    def stamped(product):
        out = {}
        for path in sorted((archive / product).rglob(f"{product}.A*.hdf")):
            parts = path.name.split(".")
            out[f"{parts[1]}.{parts[2]}"] = path
        return out

    l1b, geo = stamped("MOD021KM"), stamped("MOD03")
    missing = sorted(set(l1b) - set(geo))
    if missing:
        raise SystemExit(
            f"{len(missing)} MOD021KM granules have no matching MOD03 "
            f"geolocation ({', '.join(missing[:3])}...); the pixel "
            f"footprints cannot be computed without it")
    return [(l1b[key], geo[key]) for key in sorted(l1b)]


def _select(path, name):
    """One SDS as a plain array, with its attributes."""
    sd = SD(str(path), SDC.READ)
    try:
        sds = sd.select(name)
        try:
            return np.asarray(sds.get()), dict(sds.attributes())
        finally:
            sds.endaccess()
    finally:
        sd.end()


def _scaled(path, name):
    """An MOD03 angle field in degrees, invalid entries as nan."""
    raw, attrs = _select(path, name)
    lo, hi = attrs["valid_range"]
    out = raw.astype(np.float64) * attrs.get("scale_factor", 1.0)
    out[(raw < lo) | (raw > hi)] = np.nan
    return out


def read_reflectance(l1b_path, geo_path, solar_correction):
    """Band 1 reflectance, and a mask of the pixels that carry a measurement.

    Returns (reflectance, valid). Invalid entries of `reflectance` are nan;
    the caller decides what to do with them, because objscale will not accept
    a nan and the honest options (treat as clear, or drop the granule)
    differ in kind.
    """
    raw, attrs = _select(l1b_path, BAND1_SDS)
    raw = raw[BAND1_INDEX]
    scale = float(np.asarray(attrs["reflectance_scales"]).ravel()[BAND1_INDEX])
    offset = float(np.asarray(attrs["reflectance_offsets"]).ravel()[BAND1_INDEX])
    lo, hi = attrs["valid_range"]

    valid = (raw >= lo) & (raw <= hi)
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    out[valid] = scale * (raw[valid].astype(np.float64) - offset)

    # A saturated detector is a bright pixel, not a missing one.
    saturated = raw == L1B_SATURATED
    out[saturated] = scale * (hi - offset)
    valid |= saturated

    if solar_correction:
        sza = _scaled(geo_path, "SolarZenith")
        usable = np.isfinite(sza) & (sza <= MAX_SOLAR_ZENITH_DEG)
        out = np.where(usable, out / np.cos(np.radians(np.where(usable, sza, 0.0))),
                       np.nan)
        valid &= usable

    return out, valid


def _great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two arrays of pixel centres, in km."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _centres_from_neighbours(gaps):
    """Per-pixel spacing from the gaps between neighbours along one axis.

    `gaps[i]` is the distance from row i to row i+1, possibly nan. A pixel's
    size is the mean of the gaps on either side of it, skipping nans, so a
    pixel with one usable neighbour takes that gap and one with none is nan.
    """
    left = np.concatenate([np.full((1,) + gaps.shape[1:], np.nan), gaps])
    right = np.concatenate([gaps, np.full((1,) + gaps.shape[1:], np.nan)])
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack([left, right]), axis=0)


def pixel_sizes_km(lat, lon):
    """(x_sizes, y_sizes) in km: along-scan on axis 1, along-track on axis 0.

    objscale's convention, from `locations_x[h//2, w-1] - locations_x[h//2, 0]`
    in _fractal_dimensions.py: x runs along the columns.
    """
    n_rows = lat.shape[0]
    if n_rows % DETECTORS_PER_SCAN:
        raise ValueError(
            f"{n_rows} rows is not a whole number of {DETECTORS_PER_SCAN}-"
            f"detector scans; the bow-tie correction assumes the 1 km grid")

    along_scan = _great_circle_km(lat[:, :-1], lon[:, :-1],
                                 lat[:, 1:], lon[:, 1:])
    x_sizes = _centres_from_neighbours(along_scan.T).T

    along_track = _great_circle_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    # Blank the gaps that straddle a scan boundary: those measure the
    # overlap between two scans, not the depth of a pixel.
    boundary = (np.arange(n_rows - 1) % DETECTORS_PER_SCAN) == DETECTORS_PER_SCAN - 1
    along_track[boundary] = np.nan
    y_sizes = _centres_from_neighbours(along_track)

    return x_sizes, y_sizes


def sensor_zenith_window(geo_path):
    """(j0, j1, n_rows): the columns within MAX_SENSOR_ZENITH_DEG.

    Sensor zenith depends on the scan angle, so within a granule it is a
    function of column and the retained columns form one contiguous block.
    That is checked rather than assumed -- a granule where the kept columns
    are not contiguous would mean the premise is wrong and the crop is
    silently discarding good data somewhere else.
    """
    sensor = _scaled(geo_path, "SensorZenith")
    with np.errstate(invalid="ignore"):
        keep = np.nanmedian(sensor, axis=0) <= MAX_SENSOR_ZENITH_DEG
    columns = np.flatnonzero(keep)
    if columns.size == 0:
        raise SystemExit(f"{geo_path.name}: no column is within "
                         f"{MAX_SENSOR_ZENITH_DEG} deg of nadir")
    j0, j1 = int(columns[0]), int(columns[-1]) + 1
    if columns.size != j1 - j0:
        raise SystemExit(
            f"{geo_path.name}: columns within {MAX_SENSOR_ZENITH_DEG} deg "
            f"are not contiguous ({columns.size} kept over a span of "
            f"{j1 - j0}); sensor zenith is not behaving as a function of "
            f"column and the crop would drop good data")
    return j0, j1, sensor.shape[0]


def common_window(pairs):
    """(rows, j0, j1) shared by every granule: tightest window, fewest rows.

    Granules differ a little in length, and their nadir column can shift by
    a pixel, so the ensemble is cut to what they all have. Rows are taken
    from the top, in whole scans, since the bow-tie correction needs the row
    index to keep meaning its detector.
    """
    j0, j1, rows = 0, np.inf, np.inf
    for _, geo_path in pairs:
        a, b, n = sensor_zenith_window(geo_path)
        j0, j1, rows = max(j0, a), min(j1, b), min(rows, n)
    rows = int(rows) // DETECTORS_PER_SCAN * DETECTORS_PER_SCAN
    return rows, int(j0), int(j1)


def read_geolocation(geo_path):
    """(lat, lon) in degrees, invalid entries as nan."""
    out = []
    for name in ("Latitude", "Longitude"):
        raw, attrs = _select(geo_path, name)
        arr = raw.astype(np.float64)
        lo, hi = attrs["valid_range"]
        arr[(arr < lo) | (arr > hi)] = np.nan
        out.append(arr)
    return tuple(out)
