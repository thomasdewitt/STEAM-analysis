#!/usr/bin/env python3
"""Matched RCEMIP-channel statistics, the same quantities as the TWPICE side
but across the nine hosts.

For each RCE_large300 channel host: per-level standard deviations of h, qt,
qc and qi, cloud fraction, and the single-level fields the PDFs are drawn
from -- for the host itself and for all six STEAM runs driven by its
profile. Everything lands in rcemip_stats.npz keyed by host.

SIX STEAM runs per host, not two (2026-08-08): three flux amplitudes
crossed with two outer scales, L set to the channel's long axis (6144 km)
or its short one (384 km). STEAM keys carry both tags,
`..._<set>_<lscale>`.

The host keys do NOT carry an outer-scale tag, and the host side is reduced
once rather than per outer scale. Nothing about the host depends on L, and
nothing about the matching does either: STEAM's dz follows k_z(2 dx), a
function of the finest class, so both outer scales land on the same 78
levels and produce the same z_levels and the same coarsening factors. That
is asserted against the runs rather than assumed -- an L case whose z axis
disagrees is refused, not silently matched on the first one's grid.

MATCHING, in two steps (2026-08-10), the standing rule from common.py.

FIRST the hosts are coarsened in 2 x 2 x 2 blocks -- vertically as well as
horizontally -- before any one-point statistic is taken, to keep the
standard deviations off the host's own grid scale where numerical artifacts
live (main.tex, one-point statistics). Horizontally that is 3 km -> 6 km,
STEAM's spacing, the factor taken from each run's own dx attribute rather
than written down here. Vertically it halves the level count.

THEN STEAM is matched to that grid per level: whichever field is locally
finer is block-averaged by the nearest integer factor bringing the two
spacings closest together. Before the hosts were coarsened vertically the
rule pointed both ways -- the host finer near the surface, STEAM finer
aloft. Coarsening the host doubles its spacing everywhere, so the balance
shifts toward STEAM doing the averaging; the factors used are saved per
level rather than asserted here, since they follow from the two grids.

Host fields come from the make_input_profiles.py adapters, so the
conventions match the profiles that drove the runs: mixing ratios rather
than specific humidities, h = cp*T + g*z + Lv*qv with steam's constants,
qt = qv + qc + qi with no precipitating water. T is cast to float64 before
h is formed -- it is K-scale, where float32 accumulators drift.

All three archived timesteps of each host are used, matching the driving
profiles, and every statistic is POOLED across them: the quantity reported
at a level is the statistic of all three snapshots' cells taken as one
population, not a combination of the three per-snapshot values. (Until
2026-08-06 only the first snapshot was used. Until 2026-08-10 the standard
deviations were the mean of the three per-snapshot standard deviations,
which is biased low -- it averages standard deviations rather than
variances, and it cannot see the level mean moving between snapshots.)

MESONH is excluded (documented RCEMIP hus error) and ICON_AES for having no
usable z, leaving nine.

Usage: python compute_rcemip_stats.py [host ...]     (default: all nine)
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

from common import (VARS, CLOUD_KGKG, coarsen_factor, coarsen_xyz,
                    level_planes, match_factors, pdf_slices, reduce_source)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
RUNS = REPO / "runs" / "hydro"
OUT = OUTPUT / "rcemip_stats.npz"

sys.path.insert(0, str(REPO))
from make_input_profiles import ADAPTERS                   # noqa: E402

HOSTS = ("sam", "cm1", "ukmo_casim", "ukmo_ra1t", "ukmo_ra1t_nocloud",
         "scale", "ucla", "icon_lem", "icon_nwp")
SETS = ("c002", "c005", "c017")
LSCALES = ("Llong", "Lshort")
HOST_DX = 3000.0               # RCE_large300, by protocol
SNAPSHOTS = (0, 1, 2)


def host_fields(host, snapshot, xy_coarsen):
    """Host h, qt, qc, qi on STEAM's grid at one snapshot, and its z.

    Coarsened in f x f x f blocks, vertically as well as horizontally, and
    the z axis with them so the fields and their own coordinate stay on one
    grid. h is formed at native resolution and coarsened after, which is
    the same number either way -- it is linear in T, z and qv.
    """
    z, T, qv, qc, qi, _ = ADAPTERS[host](snapshot)
    z = np.asarray(z, dtype=np.float64)
    h = cp * T.astype(np.float64) + g * z[:, None, None] + Lv * qv
    qt = qv + qc + qi
    fields = {"h": h, "qt": qt, "qc": qc, "qi": qi}
    coarse = {k: coarsen_xyz(v, xy_coarsen) for k, v in fields.items()}
    z = coarsen_xyz(z, xy_coarsen)
    print(f"  {host} host {T.shape} -> {coarse['h'].shape}", flush=True)
    # z last, to match STEAM's layout for the level reductions.
    return z, {k: np.moveaxis(v, 0, -1) for k, v in coarse.items()}


def steam_fields(host, set_tag, lscale):
    """STEAM h, qt, qc, qi each (nx, ny, nz), its z axis, dx, and realized L.

    dx and outer_scale are read off the run rather than written down here,
    so a run regenerated on a different geometry is described by the geometry
    it actually has.
    """
    ds = netCDF4.Dataset(RUNS / f"{host}_{set_tag}_{lscale}.nc")
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    fields = {v: ds.variables[v][:] for v in VARS}
    fields["h"] = fields["h"].astype(np.float64)
    dx = float(ds.dx)
    outer_scale = float(ds.outer_scale)
    ds.close()
    return z, fields, dx, outer_scale


def do_host(host, out):
    z_steam, steam_f, steam_dx, L0 = steam_fields(host, SETS[0], LSCALES[0])
    xy_coarsen = coarsen_factor(steam_dx, HOST_DX)
    out[f"dx_{host}"] = steam_dx
    out[f"xy_coarsen_{host}"] = xy_coarsen

    # Host side: one snapshot's FIELD in memory at a time, but every
    # snapshot's level planes kept, POOLED across snapshots (2026-08-10).
    # Only the planes are kept, ~90 MB a snapshot against gigabytes for the
    # field, which is what makes it possible to hand the whole pooled sample
    # to numpy in one reduction instead of combining per-snapshot summaries.
    # Until this date the standard deviations were the mean of the three
    # per-snapshot values, which is a different and smaller number; the PDF
    # slices were already pooled by stacking, and cloud fraction happened to
    # agree because the counts are equal, so only the stds move.
    planes, pooled = [], {}
    z_levels = n_steam = n_host = None
    for snap in SNAPSHOTS:
        z_host, host_f = host_fields(host, snap, xy_coarsen)
        if z_levels is None:
            z_levels = z_host[z_host <= z_steam[-1]]
            n_steam, n_host = match_factors(z_levels, z_steam)
            out[f"z_{host}"] = z_levels
            out[f"n_steam_{host}"] = n_steam
            out[f"n_host_{host}"] = n_host
        planes.append(level_planes(z_host, host_f, z_levels, n_host))
        for k, a in pdf_slices(z_host, host_f, z_levels, n_host).items():
            pooled.setdefault(k, []).append(a)
        del host_f

    # The pooled sample, handed to numpy whole: axis 0 is the snapshot and
    # axes 2,3 are the horizontal, so reducing over all three is the
    # statistic of every snapshot's cells at that level as one population.
    for v in VARS:
        stack = np.stack([p[v] for p in planes])       # (snap, lev, ny, nx)
        out[f"std_{v}_{host}_host"] = stack.std(axis=(0, 2, 3),
                                                dtype=np.float64)
    cond = (np.stack([p["qc"] for p in planes])
            + np.stack([p["qi"] for p in planes]))
    out[f"cf_{host}_host"] = (cond >= CLOUD_KGKG).mean(axis=(0, 2, 3))
    del planes, cond
    for k, arrs in pooled.items():
        out[f"pdf_{k}_{host}_host"] = np.stack(arrs)

    for lscale in LSCALES:
        for tag in SETS:
            if (lscale, tag) == (LSCALES[0], SETS[0]):
                z_s, fields, L = z_steam, steam_f, L0
            else:
                z_s, fields, _, L = steam_fields(host, tag, lscale)
                # The whole matching -- z_levels and both sets of coarsening
                # factors -- was derived from the first run's z axis. Reusing
                # it for a run on a different vertical grid would quietly
                # compare two things measured at different heights.
                if z_s.shape != z_steam.shape or \
                        not np.allclose(z_s, z_steam, rtol=0, atol=1e-6):
                    raise SystemExit(
                        f"{host} {tag} {lscale}: STEAM z axis differs from "
                        f"{SETS[0]} {LSCALES[0]}; these runs cannot share "
                        f"one set of matching factors")
            out[f"L_{lscale}"] = L
            std, cf, slices = reduce_source(z_s, fields, z_levels, n_steam)
            for v in VARS:
                out[f"std_{v}_{host}_{tag}_{lscale}"] = std[v]
            out[f"cf_{host}_{tag}_{lscale}"] = cf
            for k, a in slices.items():
                out[f"pdf_{k}_{host}_{tag}_{lscale}"] = a

    print(f"  {host}: {z_levels.size} levels to {z_levels[-1]:.0f} m, "
          f"host {xy_coarsen}x{xy_coarsen} coarsened to {steam_dx:.0f} m, "
          f"STEAM coarsened by {sorted(set(n_steam.tolist()))}, "
          f"host by {sorted(set(n_host.tolist()))}", flush=True)


def main():
    hosts = sys.argv[1:] or list(HOSTS)
    for host in hosts:
        if host not in HOSTS:
            raise SystemExit(f"unknown host {host!r} (have {list(HOSTS)})")

    out = {"hosts": np.array(hosts), "sets": np.array(SETS),
           "lscales": np.array(LSCALES)}
    for host in hosts:
        do_host(host, out)

    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
