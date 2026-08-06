#!/usr/bin/env python3
"""Matched RCEMIP-channel statistics, the same quantities as the TWPICE side
but across the nine hosts.

For each RCE_large300 channel host: per-level standard deviations of h, qt,
qc and qi, cloud fraction, and the single-level fields the PDFs are drawn
from -- for the host itself and for both STEAM runs driven by its profile.
Everything lands in rcemip_stats.npz keyed by host.

MATCHING is the standing rule, from common.py. Horizontally the hosts are
block-averaged 2x2, 3 km -> 6 km, which is STEAM's spacing. Vertically the
hosts are on the stretched RCEMIP grid -- 75 m at the surface, ~270 m by
1 km, then 500 m through the free troposphere (250 m for the three UKMO runs
on their 98-level grid) -- against STEAM's uniform 257 m. So the rule points
both ways here: near the surface the host is the finer field and gets
averaged, aloft STEAM is finer and gets averaged, and around 1 km and for
UKMO throughout the two already agree and neither moves.

Host fields come from the make_input_profiles.py adapters, so the
conventions match the profiles that drove the runs: mixing ratios rather
than specific humidities, h = cp*T + g*z + Lv*qv with steam's constants,
qt = qv + qc + qi with no precipitating water. T is cast to float64 before
h is formed -- it is K-scale, where float32 accumulators drift.

Only the first archived timestep of each host is used. The hosts each have
three, and the driving profiles average all three, but the inter-model
spread this figure is about is far larger than the inter-snapshot spread.

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

from common import VARS, coarsen_xy, match_factors, reduce_source

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
SETS = ("c005", "c017")
XY_COARSEN = 2                 # host 3 km -> 6 km
SNAPSHOT = 0


def host_fields(host):
    """Host h, qt, qc, qi on the 6 km grid, each (nx, ny, nz), and its z."""
    z, T, qv, qc, qi, _ = ADAPTERS[host](SNAPSHOT)
    z = np.asarray(z, dtype=np.float64)
    h = cp * T.astype(np.float64) + g * z[:, None, None] + Lv * qv
    qt = qv + qc + qi
    fields = {"h": h, "qt": qt, "qc": qc, "qi": qi}
    coarse = {k: coarsen_xy(v, XY_COARSEN) for k, v in fields.items()}
    print(f"  {host} host {T.shape} -> {coarse['h'].shape}", flush=True)
    # z last, to match STEAM's layout for the level reductions.
    return z, {k: np.moveaxis(v, 0, -1) for k, v in coarse.items()}


def steam_fields(host, set_tag):
    """STEAM h, qt, qc, qi, each (nx, ny, nz), and its z axis."""
    ds = netCDF4.Dataset(RUNS / f"{host}_{set_tag}.nc")
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    fields = {v: ds.variables[v][:] for v in VARS}
    fields["h"] = fields["h"].astype(np.float64)
    ds.close()
    return z, fields


def do_host(host, out):
    z_host, host_f = host_fields(host)
    z_steam, steam_f = steam_fields(host, SETS[0])

    z_levels = z_host[z_host <= z_steam[-1]]
    n_steam, n_host = match_factors(z_levels, z_steam)
    out[f"z_{host}"] = z_levels
    out[f"n_steam_{host}"] = n_steam
    out[f"n_host_{host}"] = n_host

    std, cf, slices = reduce_source(z_host, host_f, z_levels, n_host)
    for v in VARS:
        out[f"std_{v}_{host}_host"] = std[v]
    out[f"cf_{host}_host"] = cf
    for k, a in slices.items():
        out[f"pdf_{k}_{host}_host"] = a

    for tag in SETS:
        z_s, fields = (z_steam, steam_f) if tag == SETS[0] \
            else steam_fields(host, tag)
        std, cf, slices = reduce_source(z_s, fields, z_levels, n_steam)
        for v in VARS:
            out[f"std_{v}_{host}_{tag}"] = std[v]
        out[f"cf_{host}_{tag}"] = cf
        for k, a in slices.items():
            out[f"pdf_{k}_{host}_{tag}"] = a

    print(f"  {host}: {z_levels.size} levels to {z_levels[-1]:.0f} m, "
          f"STEAM coarsened by {sorted(set(n_steam.tolist()))}, "
          f"host by {sorted(set(n_host.tolist()))}", flush=True)


def main():
    hosts = sys.argv[1:] or list(HOSTS)
    for host in hosts:
        if host not in HOSTS:
            raise SystemExit(f"unknown host {host!r} (have {list(HOSTS)})")

    out = {"hosts": np.array(hosts), "sets": np.array(SETS)}
    for host in hosts:
        do_host(host, out)

    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
