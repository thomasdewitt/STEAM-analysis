#!/usr/bin/env python3
"""Horizontal Haar fluctuation functions and local slopes, at native
resolution, for both comparison cases.

For each host, each of h and qt, and each of 5 and 10 km: the order-1 Haar
fluctuation along the long horizontal axis of a single level, for the host
and for both STEAM runs driven by its profile. Writes scaling_stats.npz.

NATIVE RESOLUTION, deliberately, and unlike the profile and PDF figures:
nothing is coarsened on either side. The host and STEAM curves therefore
start at different smallest lags -- 2 x 100 m against 2 x 200 m for TWPICE,
2 x 3 km against 2 x 6 km for the channels -- and where they overlap they are
measuring the same physical scales. A scaling function is the one place where
matching resolutions would destroy the thing being measured.

The transform runs along the longer horizontal axis with periodic=True, so
the shorter axis is pooled as independent realizations in a single call
(these estimators are not linear; per-line loops and averaging would give a
different, wrong answer). Lags come back in cells and are converted to
metres.

The level mean is subtracted before the transform. The Haar kernel is
zero-mean so this changes nothing analytically, but h is O(3e5) J/kg while
its fluctuations are O(1e3), and differencing two half-window means at that
ratio is exactly where precision goes. scaleinvariance is also put in
float64 for the same reason.

Local slope is the OLS slope of log10 F against log10 r over the half-decade
window centred on each lag, which is main.tex's definition.

Usage: python compute_scaling.py [case ...]      (default: twpice rcemip)
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4
import scaleinvariance as si

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "runs" / "hydro"
TWPICE = REPO / "data" / "twpice"
OUT = HERE / "scaling_stats.npz"

sys.path.insert(0, str(REPO))
from make_input_profiles import ADAPTERS, read_var          # noqa: E402

SETS = ("c005", "c017")
VARS = ("h", "qt")
LEVELS = (5000.0, 10000.0)
SNAPSHOT = "0000003450"
HALF_DECADE = 0.25          # +/- this many dex about each lag

CASES = {
    "twpice": (("twpice",), 100.0, 200.0),
    "rcemip": (("sam", "cm1", "ukmo_casim", "ukmo_ra1t", "ukmo_ra1t_nocloud",
                "scale", "ucla", "icon_lem", "icon_nwp"), 3000.0, 6000.0),
}

si.set_numerical_precision("float64")


def haar(field2d, dx):
    """Order-1 Haar fluctuation along the longer axis, lags in metres."""
    axis = int(np.argmax(field2d.shape))
    lags, F = si.haar_fluctuation(field2d - field2d.mean(), order=1.0,
                                  axis=axis, periodic=True)
    return np.asarray(lags, float) * dx, np.asarray(F, float)


def local_slope(lags, F):
    """OLS slope of log10 F vs log10 r over a half-decade window per lag."""
    good = np.isfinite(F) & (F > 0)
    lr, lF = np.log10(lags[good]), np.log10(F[good])
    slope = np.full(lags.size, np.nan)
    idx = np.flatnonzero(good)
    for n, i in enumerate(idx):
        w = np.abs(lr - lr[n]) <= HALF_DECADE
        if w.sum() >= 3:
            slope[i] = np.polyfit(lr[w], lF[w], 1)[0]
    return slope


def twpice_level(z_target):
    """TWPICE h and qt at one level, native 100 m, read level by level.

    A whole field is 4.3 GB and only two levels are wanted, so each variable
    is sliced in the file. h is cp times the archived MSE. The MSE file's
    horizontal axes are (x, y) and the mixing ratios' are (y, x); with a
    square domain and a horizontally isotropic statistic that only sets which
    of two equivalent directions the transform runs along.
    """
    z = np.asarray(read_var(TWPICE / f"TWPICE_LPT_3D_QV_{SNAPSHOT}.nc", "z"),
                   dtype=np.float64)
    k = int(np.argmin(np.abs(z - z_target)))

    def level(name):
        with netCDF4.Dataset(TWPICE / f"TWPICE_LPT_3D_{name}_{SNAPSHOT}.nc") as ds:
            ds.set_auto_mask(False)
            return np.asarray(ds.variables[name][0, :, :, k],
                              dtype=np.float64)

    h = cp * level("MSE")
    qt = (level("QV") + level("QC") + level("QI")) * 1e-3
    return {"h": h, "qt": qt}, float(z[k])


def host_level(host, z_target):
    """Host h and qt at the level nearest z_target, at native resolution."""
    if host == "twpice":
        return twpice_level(z_target)
    z, T, qv, qc, qi, _ = ADAPTERS[host](0)
    z = np.asarray(z, dtype=np.float64)
    k = int(np.argmin(np.abs(z - z_target)))
    h = cp * T[k].astype(np.float64) + g * z[k] + Lv * qv[k]
    qt = np.asarray(qv[k], np.float64) + qc[k] + qi[k]
    return {"h": h, "qt": qt}, float(z[k])


def steam_level(host, set_tag, z_target):
    """STEAM h and qt at the level nearest z_target."""
    with netCDF4.Dataset(RUNS / f"{host}_{set_tag}.nc") as ds:
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(np.float64)
        k = int(np.argmin(np.abs(z - z_target)))
        fields = {v: np.asarray(ds.variables[v][:, :, k], np.float64)
                  for v in VARS}
    return fields, float(z[k])


def do_case(case, out):
    hosts, host_dx, steam_dx = CASES[case]
    out[f"{case}_hosts"] = np.array(hosts)
    for host in hosts:
        for z_target in LEVELS:
            tag = f"{z_target / 1000:.0f}km"
            sources = [("host", host_level(host, z_target), host_dx)]
            for s in SETS:
                sources.append((s, steam_level(host, s, z_target), steam_dx))
            for name, (fields, z_used), dx in sources:
                for v in VARS:
                    lags, F = haar(fields[v], dx)
                    key = f"{case}_{host}_{v}_{tag}_{name}"
                    out[f"{key}_lags"] = lags
                    out[f"{key}_F"] = F
                    out[f"{key}_slope"] = local_slope(lags, F)
                out[f"{case}_{host}_{tag}_{name}_z"] = z_used
            print(f"  {case} {host} {tag} done", flush=True)


def main():
    cases = sys.argv[1:] or list(CASES)
    out = {"sets": np.array(SETS), "vars": np.array(VARS),
           "levels": np.array(LEVELS), "cases": np.array(cases)}
    for case in cases:
        if case not in CASES:
            raise SystemExit(f"unknown case {case!r} (have {list(CASES)})")
        do_case(case, out)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
