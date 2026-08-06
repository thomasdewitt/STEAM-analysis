#!/usr/bin/env python3
"""Matched STEAM runs for the hydrodynamic comparison.

One simulation per host per flux-noise amplitude. Each run sits on its host's
own domain, driven by that host's mean h and qt profile, at twice the host's
horizontal grid spacing. Nothing else varies across the set.

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

Outer scale: L = (longest domain dimension) / 2 for TWPICE, / 4 for the
channels. The channels' short axis is far below L, so it is a strip axis --
the coarsest classes are kernel-folded onto it.

Everything else is the frozen config: constant 10 m spheroscale, anchored
bounds (supplement S2 as amended 2026-07-27), domain top 20 km, H_h and
lambda at package defaults.

Input profiles come from runs/input_profiles/, built by make_input_profiles.py
at the repo root. Output is one full .nc per run in runs/hydro/ -- h, qt, flux
and the diagnostics, at native resolution, since the scaling functions want it
that way. Restartable: a run whose .nc exists is skipped.

Usage: python generate.py [HOST [SET]]
  no args            -> every host, every set
  twpice             -> that host, every set
  twpice c005        -> that single run
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

# host -> STEAM's (nx, ny, dx [m]) and the divisor d in
# L = (longest domain dimension) / d.
CHANNEL = (1024, 64, 6000.0, 4)     # 6144 x 384 km, L = 1536 km

HOSTS = {
    "twpice":            (1024, 1024, 200.0, 2),   # 204.8 km sq, L = 102.4 km
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
    "gate":              (1024, 1024, 200.0, 2),   # as twpice
}

SETS = {  # set tag -> c, the flux noise amplitude (steam.simulate.FLUX_SCALE)
    "c005": 0.05,
    "c017": 0.17,
}

SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
DEVICE = "cuda"


def run_one(host, set_tag):
    out_nc = RUNS / f"{host}_{set_tag}.nc"
    if out_nc.exists():
        print(f"{out_nc.name} exists, skipping", flush=True)
        return

    nx, ny, dx, divisor = HOSTS[host]
    outer_scale = max(nx * dx, ny * dx) / divisor

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
                  float(h_profile.max()) + 1.0)
    h_lower = float(h_profile.min()) - 10.0 * cp

    _steam_simulate.FLUX_SCALE = SETS[set_tag]
    print(f"=== {host} {set_tag} === nx={nx} ny={ny} dx={dx:.0f} m, "
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
        seed=5000 + 10 * list(HOSTS).index(host) + list(SETS).index(set_tag),
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device=DEVICE,
    )
    compute_diagnostics(str(out_nc), compress=True)
    print(f"{out_nc.name} done in {time.perf_counter() - t0:.0f} s "
          f"({out_nc.stat().st_size / 1e9:.2f} GB)", flush=True)


def main():
    hosts = [sys.argv[1]] if len(sys.argv) > 1 else list(HOSTS)
    sets = [sys.argv[2]] if len(sys.argv) > 2 else list(SETS)
    for host in hosts:
        if host not in HOSTS:
            raise SystemExit(f"unknown host {host!r} (have {list(HOSTS)})")
        for set_tag in sets:
            if set_tag not in SETS:
                raise SystemExit(
                    f"unknown set {set_tag!r} (have {list(SETS)})")
            run_one(host, set_tag)
    print("generation complete", flush=True)


if __name__ == "__main__":
    main()
