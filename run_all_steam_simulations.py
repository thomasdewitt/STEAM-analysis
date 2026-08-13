#!/usr/bin/env python3
"""Run every STEAM simulation this repository defines, one campaign at a time.

Each subfolder owns one or more run scripts that generate its own runs and
nothing else. This calls them in sequence, in the same interpreter this
script was started with, and stops at the first failure.

The order is cheapest first, so a broken environment shows up in minutes
rather than after the square campaign has run:

  1. hydrodynamic-comparison -- two scripts, RCEMIP channels then the gigaLES
                               ensemble, cheapest first within the campaign
                               too: a channel run is 0.09 GB against 4.3 GB
                               for a gigaLES member
  2. fractal-analysis        -- the square campaign (the longest overall: two
                               sets of ten members, each with two nests)
  3. small-domain            -- the nested visualization runs, one per flux
                               amplitude per spheroscale, the most expensive
                               per simulation and the only ones that want the
                               machine to themselves

Every campaign skips work already on disk, so a rerun after an interruption
resumes rather than starting over. Nothing here parallelizes: each campaign's
working set is sized against the whole machine.

Usage: python run_all_steam_simulations.py [CAMPAIGN ...]
  no args        -> all three, in the order above
  small-domain   -> that campaign only (folder name, or any unique prefix)
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

CAMPAIGNS = {
    "small-domain": ("run_steam_simulations.py",),
    "hydrodynamic-comparison": ("run_rcemip_simulations.py",
                                "run_gigales_simulations.py"),
    "fractal-analysis": ("run_steam_simulations.py",),
}


def scripts_for(campaign):
    paths = [REPO / campaign / "scripts" / name
             for name in CAMPAIGNS[campaign]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{campaign} is missing scripts/{missing}")
    return paths


def resolve(name):
    """Accept a folder name or any unambiguous prefix of one."""
    if name in CAMPAIGNS:
        return name
    matches = [c for c in CAMPAIGNS if c.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"unknown campaign {name!r} "
        f"({'ambiguous' if matches else 'no match'}; have {list(CAMPAIGNS)})")


def run(campaign):
    print(f"\n{'=' * 70}\n=== {campaign}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    for script in scripts_for(campaign):
        print(f"\n--- {script.name}", flush=True)
        t1 = time.perf_counter()
        # cwd is the script's own directory: these scripts import their
        # siblings as flat modules, which only works from there.
        result = subprocess.run([sys.executable, script.name],
                                cwd=script.parent)
        if result.returncode != 0:
            raise SystemExit(
                f"\n{campaign}/{script.name} failed "
                f"(exit {result.returncode}) after "
                f"{time.perf_counter() - t1:.0f} s; stopping before the rest")
    print(f"\n=== {campaign} complete "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)


def main():
    campaigns = [resolve(a) for a in sys.argv[1:]] or list(CAMPAIGNS)
    t0 = time.perf_counter()
    for campaign in campaigns:
        run(campaign)
    print(f"\nall requested campaigns complete "
          f"({time.perf_counter() - t0:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
