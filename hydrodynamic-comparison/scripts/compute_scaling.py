#!/usr/bin/env python3
"""Horizontal Haar fluctuation functions and local slopes, at native
resolution, for both comparison cases.

For each host, each of h and qt, and each of 5 and 10 km: the order-1 Haar
fluctuation along the long horizontal axis of a single level, for the host
and for every STEAM run driven by its profile. Writes scaling_stats.npz.

STEAM sources are tagged `<set>_<lscale>`: three flux amplitudes crossed
with the outer scales that are distinct for the case. The channels have two
-- L at the long axis (6144 km) or the short one (384 km) -- so six runs
per host; TWPICE and GATE are square, so the two coincide and they have
three. This is the axis the scaling functions show most directly: a
short-L run's cascade is five dyads deep against the long-L run's nine, so
its fluctuation function has no variance left to accumulate past 384 km
while the long-L one keeps climbing to the domain scale.

NATIVE RESOLUTION, deliberately, and unlike the profile and PDF figures:
nothing is coarsened on either side. The host and STEAM curves therefore
start at different smallest lags -- 2 x 100 m against 2 x 200 m for the SAM
cases, 2 x 3 km against 2 x 6 km for the channels -- and where they overlap
they are measuring the same physical scales. A scaling function is the one
place where matching resolutions would destroy the thing being measured.

The gigales case covers both SAM LES, TWPICE and GATE, as the profile and PDF
figures do; the rcemip case covers the nine channels.

The transform runs along the longer horizontal axis with periodic=True, so
the shorter axis is pooled as independent realizations in a single call
(these estimators are not linear; per-line loops and averaging would give a
different, wrong answer). Lags come back in cells and are converted to
metres.

EVERYTHING THAT POOLS IS PASSED IN ONE CALL (2026-08-10). The three
archived channel snapshots, and the five members of each gigaLES ensemble,
are stacked onto a trailing axis and handed to the estimator whole, so the
statistic is over the pooled sample by construction. This used to compute
F_1 per snapshot and average them, which gives the identical number -- the
estimator returns a raw moment and the window counts are equal, verified to
machine precision at orders 0.5 through 3 -- but only while the order stays
at 1, and nothing at the call site said so. (Until 2026-08-06 only the
first snapshot was used at all, a bug against the stated methodology.)

The level mean is subtracted before the transform. The Haar kernel is
zero-mean so this changes nothing analytically, but h is O(3e5) J/kg while
its fluctuations are O(1e3), and differencing two half-window means at that
ratio is exactly where precision goes. scaleinvariance is also put in
float64 for the same reason.

Local slope is the OLS slope of log10 F against log10 r over the half-decade
window centred on each lag, which is main.tex's definition.

Usage: python compute_scaling.py [case ...]      (default: gigales rcemip)
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

SETS = ("c002", "c005", "c017")
VARS = ("h", "qt")
LEVELS = (5000.0, 10000.0)
SNAPSHOT = "0000003450"
HALF_DECADE = 0.25          # +/- this many dex about each lag

GATE_FILE = "GATE_IDEAL_S_2048x2048x256_100m_2s_2048_0000041400.nc"   # 23 h

# case -> (hosts, host dx [m], outer-scale cases, STEAM members per config).
# STEAM's dx and its realized L are not listed: both are read from each run's
# own attributes, so changing a domain in run_rcemip_simulations.py does not
# leave a stale number here.
# The gigaLES hosts carry a five-member realization ensemble; the channels
# are one run each. Their outer-scale axis is degenerate -- the domains are
# square, so L is the extent either way -- but the tag stays in the source
# names so both cases key the same way.
CASES = {
    "gigales": (("twpice", "gate"), 100.0, ("Llong",), 5),
    "rcemip": (("sam", "cm1", "ukmo_casim", "ukmo_ra1t", "ukmo_ra1t_nocloud",
                "scale", "ucla", "icon_lem", "icon_nwp"), 3000.0,
               ("Llong", "Lshort"), 1),
}

si.set_numerical_precision("float64")


def haar(field, dx, axis=None):
    """Order-1 Haar fluctuation along one horizontal axis, lags in metres.

    `field` may carry axes beyond the two horizontal ones -- a member axis
    for the gigaLES ensembles -- and every axis that is not the transform
    axis is pooled as independent realizations inside the single call. That
    is exactly what pooling means here, and it is why the whole ensemble is
    handed over at once rather than looped: these estimators are not linear,
    so a per-member loop and an average is a different quantity.

    The axis is passed explicitly where there is a member axis. Defaulting
    to the longest is right for a bare 2-D channel field, but with five
    members stacked it is right only by accident of 1024 > 5.
    """
    if axis is None:
        axis = int(np.argmax(field.shape))
    lags, F = si.haar_fluctuation(field - field.mean(), order=1.0,
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
    is sliced in the file. h is cp times the archived MSE.

    Every one of these files is (time, y, x, z), the MSE included -- its
    dimensions are NAMED ('time', 'x', 'y', 'z') but its data is laid out
    like the others, established pointwise against the archived TABS on
    2026-08-10 (make_input_profiles._twpice_field). This function was always
    right about it by doing nothing: the slices below are taken as stored, so
    h runs along the same horizontal direction as qt. The remark that used to
    sit here -- that the two were transposed relative to each other and it did
    not matter on a square domain -- was wrong on the first half and moot on
    the second.
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


def steam_levels(host, set_tag, lscale, z_target, n_members):
    """STEAM h and qt at the level nearest z_target, one dict per member.

    Returns a list so the caller treats an ensemble and a single run the
    same way -- the whole list is stacked and handed to the estimator in one
    call, which is what pooling over realizations means here.

    dx and the realized outer scale come from the files rather than from a
    table, so a run regenerated on a different domain is measured on the
    grid it actually has, and every member is checked to agree.
    """
    if n_members > 1:
        paths = [RUNS / f"{host}_{set_tag}_m{m:02d}.nc"
                 for m in range(n_members)]
    else:
        paths = [RUNS / f"{host}_{set_tag}_{lscale}.nc"]

    out, z_used, dx, outer_scale = [], None, None, None
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path.name} not found")
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            z = ds.variables["z"][:].astype(np.float64)
            k = int(np.argmin(np.abs(z - z_target)))
            out.append({v: np.asarray(ds.variables[v][:, :, k], np.float64)
                        for v in VARS})
            this = (float(z[k]), float(ds.dx), float(ds.outer_scale))
        if z_used is None:
            z_used, dx, outer_scale = this
        elif this != (z_used, dx, outer_scale):
            raise SystemExit(
                f"{path.name}: level {this} disagrees with member 00 "
                f"{(z_used, dx, outer_scale)}; these cannot be pooled")
    return out, z_used, dx, outer_scale


def do_case(case, out):
    hosts, host_dx, lscales, n_members = CASES[case]
    out[f"{case}_hosts"] = np.array(hosts)
    out[f"{case}_lscales"] = np.array(lscales)
    for host in hosts:
        for z_target in LEVELS:
            tag = f"{z_target / 1000:.0f}km"
            fields_list, z_host = host_levels(host, z_target)
            sources = [("host", fields_list, z_host, host_dx)]
            for lscale in lscales:
                for s in SETS:
                    members, z_steam, steam_dx, L = steam_levels(
                        host, s, lscale, z_target, n_members)
                    # Keyed by case, not by lscale alone: `Llong` means
                    # 6144 km for a channel and 204.8 km for a square SAM
                    # domain, so a single L_<lscale> would be whichever case
                    # ran last. Within a case every host shares the geometry.
                    key = f"{case}_L_{lscale}"
                    if key in out and abs(out[key] - L) > 1e-6:
                        raise SystemExit(
                            f"{case} {host} {s} {lscale}: outer scale {L} m "
                            f"disagrees with {out[key]} m from an earlier "
                            f"host; this case is not one geometry")
                    out[key] = L
                    sources.append((f"{s}_{lscale}", members, z_steam,
                                    steam_dx))
            for name, fields_list, z_used, dx in sources:
                for v in VARS:
                    # Every snapshot or member stacked onto a trailing axis
                    # and passed in ONE call, so the pooling is over the
                    # whole sample by construction. This used to compute F_1
                    # per snapshot and average, which is the same number --
                    # the estimator returns a raw moment and the window
                    # counts are equal -- but only while the order stays at
                    # 1, and nothing said so at the call site.
                    #
                    # The transform axis is taken from the 2-D level shape
                    # before stacking, so it is the long horizontal axis for
                    # a channel and never the member axis.
                    axis = int(np.argmax(fields_list[0][v].shape))
                    stack = np.stack([f[v] for f in fields_list], axis=-1)
                    lags, F = haar(stack, dx, axis=axis)
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
