#!/usr/bin/env python3
"""Matched STEAM runs for the hydrodynamic comparison.

One simulation per host per flux-noise amplitude per outer scale. Each run
sits on its host's own domain, driven by that host's mean h and qt profile,
at twice the host's horizontal grid spacing. Nothing else varies across the
set.

Hosts: SAM-TWPICE (a driven, sheared deep-convection case), SAM-GATE
(idealized maritime deep convection, same 2048^2 x 100 m geometry) and the
nine usable RCE_large300 channels (MESONH excluded for its documented hus error, ICON_AES
for having no usable z).

Grid. STEAM runs at dx = 2 x the host's dx: 200 m for TWPICE (host 100 m) and
6 km for the channels (host 3 km). The channels are all pinned to one
geometry, 1024 x 64 at 6 km = 6144 x 384 km, rather than tracking each host's
own 1984-2048 x 128-144: the hosts differ by at most 3% in extent, and a
dyadic grid keeps the class ladder landing on 2*dx exactly. TWPICE matches its
host's 204.8 km extent exactly.

The vertical spacing is NOT a free knob -- with a constant spheroscale it
follows from dx through the aspect-ratio scaling, dz = k_z(2 dx)/2 -- so it
lands where it lands and this script coarsens nothing. What comes out:

    twpice     1024 x 1024 x 516   dx  200 m   dz  38.8 m
    channels   1024 x   64 x  78   dx 6000 m   dz 256.8 m

Against the hosts that bears on how these get read, and it points opposite
ways for the two groups. TWPICE is 100 m horizontal and 50-100 m vertical, so
STEAM is ~5x finer vertically. The channels are on the RCEMIP stretched grid
-- 500 m through the free troposphere, 250 m for the three UKMO runs on their
98-level grid -- so STEAM is ~2x finer than most of them and already matched
to UKMO.

OUTER SCALE is the second axis (2026-08-08), and it is what doubles the
channel comparison. L is set to a domain dimension, either of two:

    Llong    L = the longest horizontal extent    6144 km for the channels
    Lshort   L = the shortest horizontal extent    384 km for the channels

Llong is what this script has always done and its runs are unchanged. The
realized L is printed per run rather than recorded here, so moving the rule
cannot leave a wrong number behind; downstream reads it back off each run's
own `outer_scale` attribute.

simulate() requires each extent to be an integer multiple of L or smaller
than it, and both cases satisfy that: at Llong the long axis is one tile and
the short axis is a narrow strip, with the coarsest classes kernel-folded
onto it; at Lshort the long axis is exactly 16 tiles and the short axis is
one, so neither axis is a strip. What actually differs is the depth of the
class ladder, L/2dx falling from 512 to 32 -- nine dyads of cascade against
five. The vertical grid does NOT move: dz follows k_z(2 dx), which is a
function of the finest class, not of L. So both cases land on the same 78
levels and the matching machinery downstream is untouched.

TWPICE and GATE are square -- 1024 x 1024 at 200 m -- so their longest and
shortest extents are the same 204.8 km and the two cases coincide. They run
Llong only, and there is nothing to compare on that side; the doubling is
the nine channels.

Everything else is the frozen config: constant 10 m spheroscale, anchored
bounds (supplement S2 as amended 2026-07-27), domain top 20 km, H_h and
lambda at package defaults.

Input profiles come from runs/input_profiles/, built by make_input_profiles.py
at the repo root. Output is one full .nc per run in runs/hydro/ -- h, qt, flux
and the diagnostics, at native resolution, since the scaling functions want it
that way. Restartable: a run whose .nc exists is skipped.

Usage: python run_steam_simulations.py [HOST [SET [LSCALE]]]
  no args                 -> every host, every set, every outer scale
  twpice                  -> that host, every set (Llong only, being square)
  sam c005                -> that host and amplitude, both outer scales
  sam c005 Lshort         -> that single run
"""

import sys
import time
from pathlib import Path

import numpy as np

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

# host -> STEAM's (nx, ny, dx [m]).
CHANNEL = (1024, 64, 6000.0)        # 6144 x 384 km

HOSTS = {
    "twpice":            (1024, 1024, 200.0),      # 204.8 km square
    "sam":               CHANNEL,
    "cm1":               CHANNEL,
    "ukmo_casim":        CHANNEL,
    "ukmo_ra1t":         CHANNEL,
    "ukmo_ra1t_nocloud": CHANNEL,
    "scale":             CHANNEL,
    "ucla":              CHANNEL,
    "icon_lem":          CHANNEL,
    "icon_nwp":          CHANNEL,
    # Appended rather than grouped with twpice: the seed is derived from
    # position in this dict, so inserting higher up would silently
    # re-randomize every run below it.
    "gate":              (1024, 1024, 200.0),      # as twpice
}

SETS = {  # set tag -> c, the flux noise amplitude (steam.simulate.FLUX_SCALE)
    "c005": 0.05,
    "c017": 0.17,
    # Appended, not sorted into place: as with HOSTS, the seed carries this
    # dict's ordering, so putting c002 first would re-randomize every
    # existing run. The tag is c x 100, zero-padded to three digits, as in
    # small-domain/ -- so c002 is 0.02.
    "c002": 0.02,
}

# outer scale tag -> which horizontal extent L is set to. Llong FIRST: the
# seed carries its index, so Llong at 0 leaves every pre-existing run's seed
# where it was.
LSCALES = {
    "Llong": max,
    "Lshort": min,
}

SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
DEVICE = "cuda"


def lscales_for(host):
    """The outer-scale cases that are distinct for this host.

    A square domain has one horizontal extent, so Llong and Lshort would name
    the same simulation; it runs Llong alone rather than writing the same
    field to two filenames.
    """
    nx, ny, dx = HOSTS[host]
    if nx == ny:
        return ["Llong"]
    return list(LSCALES)


def run_one(host, set_tag, lscale):
    out_nc = RUNS / f"{host}_{set_tag}_{lscale}.nc"
    if out_nc.exists():
        print(f"{out_nc.name} exists, skipping", flush=True)
        return

    nx, ny, dx = HOSTS[host]
    outer_scale = LSCALES[lscale](nx * dx, ny * dx)

    src = np.load(PROFILES / f"{host}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    spheroscale = np.full(src["z_profile"].size, SPHEROSCALE_CONSTANT)
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    # Anchored bounds: every realization sees the prescribed 300 K SST, so qt
    # is capped at surface saturation. The gz term puts stratospheric <h>
    # above any surface value, so the upper h bound is the LARGER of
    # surface-saturation MSE and the profile max; the lower allows a 10 K
    # deficit below the coldest point of the profile.
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                  float(h_profile.max()))
    h_lower = float(h_profile.min()) - 10.0 * cp

    _steam_simulate.FLUX_SCALE = SETS[set_tag]
    print(f"=== {host} {set_tag} {lscale} === nx={nx} ny={ny} dx={dx:.0f} m, "
          f"L={outer_scale / 1000:.1f} km, c={SETS[set_tag]}", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    simulate(
        h_profile, qt_profile,
        nx=nx, ny=ny, dx=dx, dy=dx,
        outer_scale=outer_scale,
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=surface_pressure,
        # Stride 10 per host, 4 per outer scale, 1 per amplitude -- so the
        # outer-scale term is 0 for Llong and every run that predates this
        # axis keeps the seed it was generated with.
        seed=(5000 + 10 * list(HOSTS).index(host)
              + 4 * list(LSCALES).index(lscale)
              + list(SETS).index(set_tag)),
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device=DEVICE,
    )
    compute_diagnostics(str(out_nc), compress=True, device=DEVICE)
    print(f"{out_nc.name} done in {time.perf_counter() - t0:.0f} s "
          f"({out_nc.stat().st_size / 1e9:.2f} GB)", flush=True)


def main():
    hosts = [sys.argv[1]] if len(sys.argv) > 1 else list(HOSTS)
    sets = [sys.argv[2]] if len(sys.argv) > 2 else list(SETS)
    asked = [sys.argv[3]] if len(sys.argv) > 3 else None
    for host in hosts:
        if host not in HOSTS:
            raise SystemExit(f"unknown host {host!r} (have {list(HOSTS)})")
        for set_tag in sets:
            if set_tag not in SETS:
                raise SystemExit(
                    f"unknown set {set_tag!r} (have {list(SETS)})")
            for lscale in (asked or lscales_for(host)):
                if lscale not in LSCALES:
                    raise SystemExit(
                        f"unknown outer scale {lscale!r} "
                        f"(have {list(LSCALES)})")
                if lscale not in lscales_for(host):
                    raise SystemExit(
                        f"{host} is square, so {lscale} is the same "
                        f"simulation as Llong; run that instead")
                run_one(host, set_tag, lscale)
    print("generation complete", flush=True)


if __name__ == "__main__":
    main()
