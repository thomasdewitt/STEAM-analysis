#!/usr/bin/env python3
"""The realization ensemble for the two SAM cases: five members per amplitude.

A single STEAM realization carries the sampling noise of one draw, and on
these domains that noise is not small -- measured over a ten-member trial
on GATE, the per-level std(h) at 5-10 km spanned 40-80% of its own value
and std(qc) aloft spanned more than a factor of two. The profile figures
were reporting that as if it were the model. This generates the ensemble;
compute_ensemble_stats.py reduces it.

Runs and statistics are split into two scripts, as everywhere else in the
repo, which means the ensemble is KEPT rather than reduced on the way past.
That is what lets the statistics be taken with plain numpy over the pooled
sample instead of assembled from per-member summaries -- see
compute_ensemble_stats.py.

WHAT IS KEPT, and why not everything. A full run is 9.06 GB, so thirty of
them would be 272 GB against roughly 360 GB free -- and 544 GB at the ten
members first planned, which is what set the size at five. Eight 3-D fields
are written; six are kept. The figures read h, qt, qc and qi; T and p are
kept for analysis not yet written (2026-08-10), which is a cheaper bet than
regenerating thirty members to get them later. Only flux and qv are
dropped, and compute_diagnostics rebuilds those from a rerun.

Sizes: measured at 2.98 GB a member with the four figure fields, so the
ensemble was 83 GB on disk. T and p have not been measured -- p is smooth
and should compress hard, T behaves like h -- so budget roughly 4-4.5 GB a
member and 130 GB for the thirty.

A keeper written before this list changed does NOT have the newer fields
and nothing backfills it: the working file it was stripped from is gone.
Adding a field means regenerating the members that need it.

CHUNKED ONE LEVEL PER CHUNK. compute_ensemble_stats.py reads a horizontal
level at a time across all five members, and with the default chunking a
single-level read has to decompress chunks spanning many levels -- the same
trap that made an earlier version of this reduction take seven minutes per
member instead of fifty seconds. chunksizes=(nx, ny, 1) makes a level read
exactly one chunk.

Members differ only in seed. The geometry, the driving profile, the bounds
and the outer scale are the ones the channels get in
run_rcemip_simulations.py, which is where these two hosts were generated
until 2026-08-10; both scripts still write to runs/hydro/, and what
separates them is the host set and the member axis, not the output.

Usage: python run_gigales_simulations.py [HOST [SET]]
  no args          -> both hosts, every amplitude, five members each
  gate             -> that host, every amplitude
  gate c005        -> that host and amplitude
"""

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4

import importlib
_steam_simulate = importlib.import_module("steam.simulate")

from steam.simulate import simulate
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv

BASE = Path(__file__).resolve().parent.parent   # hydrodynamic-comparison/
REPO = BASE.parent
RUNS = REPO / "runs" / "hydro"
PROFILES = REPO / "runs" / "input_profiles"

# host -> (nx, ny, dx [m]). Both SAM cases are square, so the outer scale is
# the domain extent either way and there is no outer-scale axis here.
HOSTS = {
    "twpice": (1024, 1024, 200.0),
    "gate":   (1024, 1024, 200.0),
}
SETS = {"c002": 0.02, "c005": 0.05, "c017": 0.17}
N_MEMBERS = 5

KEEP = ("h", "qt", "qc", "qi", "T", "p")
SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
DEVICE = "cuda"

# A block of its own, clear of the packed 5000-block the single-realization
# runs use (5000 + 10*host + 4*lscale + set, topping out at 5106), so no
# member silently reproduces a run that already exists under another name.
SEED_BASE = 7000


def member_seed(host, set_tag, member):
    return (SEED_BASE + 1000 * list(HOSTS).index(host)
            + 100 * list(SETS).index(set_tag) + member)


def keeper_path(host, set_tag, member):
    return RUNS / f"{host}_{set_tag}_m{member:02d}.nc"


def working_path(host, set_tag, member):
    return RUNS / f"work_{host}_{set_tag}_m{member:02d}.nc"


def strip_to_keeper(work_nc, out_nc):
    """Copy the four kept fields and their axes, one level per chunk.

    Each field is read in ONE piece, 2.16 GB, rather than level by level.
    The working file is chunked [64, 64, 516] -- the whole z column in every
    chunk -- so a single-level read has to decompress every chunk in the
    variable, and doing that 516 times per field runs to hours. Reading
    whole and letting HDF5 rechunk on the way out is one decompression per
    field.
    """
    with netCDF4.Dataset(work_nc) as src, \
            netCDF4.Dataset(out_nc, "w") as dst:
        src.set_auto_mask(False)
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim))
        for name in (*KEEP, "x", "y", "z"):
            if name not in src.variables:
                continue
            var = src.variables[name]
            chunks = ((var.shape[0], var.shape[1], 1)
                      if var.ndim == 3 else None)
            # complevel 1: this is a working archive read once per figure,
            # and level 4 spends minutes a field for a few percent.
            new = dst.createVariable(name, var.dtype, var.dimensions,
                                     zlib=True, complevel=1,
                                     chunksizes=chunks)
            new.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
            new[...] = var[...]


def run_member(host, set_tag, member):
    out_nc = keeper_path(host, set_tag, member)
    if out_nc.exists():
        print(f"  {out_nc.name} exists, skipping", flush=True)
        return

    nx, ny, dx = HOSTS[host]
    work_nc = working_path(host, set_tag, member)
    if work_nc.exists():
        work_nc.unlink()

    src = np.load(PROFILES / f"{host}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    spheroscale = np.full(src["z_profile"].size, SPHEROSCALE_CONSTANT)
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface, float(h_profile.max()))
    h_lower = float(h_profile.min()) - 10.0 * cp

    _steam_simulate.FLUX_SCALE = SETS[set_tag]
    t0 = time.perf_counter()
    RUNS.mkdir(parents=True, exist_ok=True)
    simulate(
        h_profile, qt_profile,
        nx=nx, ny=ny, dx=dx, dy=dx,
        outer_scale=max(nx * dx, ny * dx),
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(work_nc),
        surface_pressure=surface_pressure,
        seed=member_seed(host, set_tag, member),
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device=DEVICE,
    )
    compute_diagnostics(str(work_nc), compress=True, device=DEVICE)
    strip_to_keeper(work_nc, out_nc)
    work_nc.unlink()
    print(f"  {out_nc.name} done in {time.perf_counter() - t0:.0f} s "
          f"({out_nc.stat().st_size / 1e9:.2f} GB kept)", flush=True)


def main():
    hosts = [sys.argv[1]] if len(sys.argv) > 1 else list(HOSTS)
    sets = [sys.argv[2]] if len(sys.argv) > 2 else list(SETS)
    for host in hosts:
        if host not in HOSTS:
            raise SystemExit(f"unknown host {host!r} (have {list(HOSTS)})")
        for set_tag in sets:
            if set_tag not in SETS:
                raise SystemExit(f"unknown set {set_tag!r} "
                                 f"(have {list(SETS)})")
            print(f"=== {host} {set_tag} (c = {SETS[set_tag]}), "
                  f"{N_MEMBERS} members ===", flush=True)
            for member in range(N_MEMBERS):
                run_member(host, set_tag, member)
    print("ensemble generation complete", flush=True)


if __name__ == "__main__":
    main()
