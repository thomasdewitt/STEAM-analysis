#!/usr/bin/env python3
"""Horizontal Haar fluctuation functions and local slopes, at native
resolution, for both comparison cases.

For each host, each of h and qt, and each of 5 and 10 km: the order-1 Haar
fluctuation along the long horizontal axis of a single level, for the host
and for both STEAM runs driven by its profile. Writes scaling_stats.npz.

NATIVE RESOLUTION, deliberately, and unlike the profile and PDF figures:
nothing is coarsened on either side. The host and STEAM curves therefore
start at different smallest lags -- 2 x 100 m against 2 x 200 m for the SAM
cases, 2 x 3 km against 2 x 6 km for the channels -- and where they overlap
they are measuring the same physical scales. A scaling function is the one
place where matching resolutions would destroy the thing being measured.

The twpice case covers both SAM LES, TWPICE and GATE, as the profile and PDF
figures do; the rcemip case covers the nine channels.

The transform runs along the longer horizontal axis with periodic=True, so
the shorter axis is pooled as independent realizations in a single call
(these estimators are not linear; per-line loops and averaging would give a
different, wrong answer). Lags come back in cells and are converted to
metres.

All three archived channel snapshots are used: F_1 is computed per snapshot
and averaged, which at order 1 with equal window counts per snapshot IS the
pooled mean. (Until 2026-08-06 only the first snapshot was used -- a bug
against the stated methodology.) TWPICE and GATE have one snapshot each.

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
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
RUNS = REPO / "runs" / "hydro"
TWPICE = REPO / "data" / "twpice"
OUT = OUTPUT / "scaling_stats.npz"

sys.path.insert(0, str(REPO))
from make_input_profiles import ADAPTERS, read_var          # noqa: E402

SETS = ("c005", "c017")
VARS = ("h", "qt")
LEVELS = (5000.0, 10000.0)
SNAPSHOT = "0000003450"
HALF_DECADE = 0.25          # +/- this many dex about each lag

GATE_FILE = "GATE_IDEAL_S_2048x2048x256_100m_2s_2048_0000041400.nc"   # 23 h

# case -> (hosts, host dx [m]). STEAM's dx is not listed: it is read from each
# run's own `dx` attribute, so changing a domain in run_steam_simulations.py
# does not leave a stale number here.
CASES = {
    "twpice": (("twpice", "gate"), 100.0),
    "rcemip": (("sam", "cm1", "ukmo_casim", "ukmo_ra1t", "ukmo_ra1t_nocloud",
                "scale", "ucla", "icon_lem", "icon_nwp"), 3000.0),
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


def gate_level(z_target):
    """GATE h and qt at one level, native 100 m, sliced in the file.

    SAM's liquid/ice ramp, which the profile and PDF figures apply to the
    archived QN, is not needed here: only qv and the total condensate enter h
    and qt, so qt is qv + QN however the condensate is partitioned.
    """
    with netCDF4.Dataset(REPO / "data" / "gate" / GATE_FILE) as ds:
        ds.set_auto_mask(False)
        z = np.asarray(ds.variables["z"][:], dtype=np.float64)
        k = int(np.argmin(np.abs(z - z_target)))
        T = np.asarray(ds.variables["TABS"][0, k], dtype=np.float64)
        qv = np.asarray(ds.variables["QV"][0, k], dtype=np.float64) * 1e-3
        qn = np.asarray(ds.variables["QN"][0, k], dtype=np.float64) * 1e-3
    h = cp * T + g * z[k] + Lv * qv
    return {"h": h, "qt": qv + qn}, float(z[k])


SINGLE_SNAPSHOT = {"twpice": twpice_level, "gate": gate_level}


def host_levels(host, z_target):
    """Per-snapshot host h and qt at the level nearest z_target.

    Returns a list of {var: 2D field} dicts, one per snapshot (one for each
    SAM case, three for the channels), and the level height used.
    """
    if host in SINGLE_SNAPSHOT:
        fields, z_used = SINGLE_SNAPSHOT[host](z_target)
        return [fields], z_used
    per_snapshot, z_used = [], None
    for snap in range(3):
        z, T, qv, qc, qi, _ = ADAPTERS[host](snap)
        z = np.asarray(z, dtype=np.float64)
        k = int(np.argmin(np.abs(z - z_target)))
        z_used = float(z[k])
        h = cp * T[k].astype(np.float64) + g * z[k] + Lv * qv[k]
        qt = np.asarray(qv[k], np.float64) + qc[k] + qi[k]
        per_snapshot.append({"h": h, "qt": qt})
    return per_snapshot, z_used


def steam_level(host, set_tag, z_target):
    """STEAM h and qt at the level nearest z_target, with the run's own dx.

    dx comes from the file rather than from a table here, so a run regenerated
    on a different domain is measured on the grid it actually has.
    """
    with netCDF4.Dataset(RUNS / f"{host}_{set_tag}.nc") as ds:
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(np.float64)
        k = int(np.argmin(np.abs(z - z_target)))
        fields = {v: np.asarray(ds.variables[v][:, :, k], np.float64)
                  for v in VARS}
        dx = float(ds.dx)
    return fields, float(z[k]), dx


def do_case(case, out):
    hosts, host_dx = CASES[case]
    out[f"{case}_hosts"] = np.array(hosts)
    for host in hosts:
        for z_target in LEVELS:
            tag = f"{z_target / 1000:.0f}km"
            fields_list, z_host = host_levels(host, z_target)
            sources = [("host", fields_list, z_host, host_dx)]
            for s in SETS:
                fields, z_steam, steam_dx = steam_level(host, s, z_target)
                sources.append((s, [fields], z_steam, steam_dx))
            for name, fields_list, z_used, dx in sources:
                for v in VARS:
                    # Mean of the per-snapshot F_1: at order 1 with equal
                    # window counts per snapshot this is the pooled mean.
                    Fs = []
                    for fields in fields_list:
                        lags, F = haar(fields[v], dx)
                        Fs.append(F)
                    F = np.mean(Fs, axis=0)
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
    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
