#!/usr/bin/env python3
"""Run every STEAM simulation this repository defines, one campaign at a time.

Each subfolder owns a scripts/run_steam_simulations.py that generates its own
runs and nothing else. This calls them in sequence, in the same interpreter
this script was started with, and stops at the first failure.

The order is cheapest first, so a broken environment shows up in minutes
rather than after the square campaign has run:

  1. hydrodynamic-comparison -- the host-matched channel and square runs
  2. fractal-analysis        -- the square campaign (the longest overall: two
                               sets of ten members, each with two nests)
  3. small-domain            -- two nested visualization runs, the most
                               expensive per simulation and the only ones
                               that want the machine to themselves

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

# Cheapest first; see the module docstring.
CAMPAIGNS = ("hydrodynamic-comparison", "fractal-analysis", "small-domain")


def script_for(campaign):
    path = REPO / campaign / "scripts" / "run_steam_simulations.py"
    if not path.exists():
        raise SystemExit(f"{campaign} has no scripts/run_steam_simulations.py")
    return path


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
    script = script_for(campaign)
    print(f"\n{'=' * 70}\n=== {campaign}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    # cwd is the script's own directory: these scripts import their siblings
    # as flat modules, which only works from there.
    result = subprocess.run([sys.executable, script.name], cwd=script.parent)
    dt = time.perf_counter() - t0
    if result.returncode != 0:
        raise SystemExit(
            f"\n{campaign} failed (exit {result.returncode}) after {dt:.0f} s; "
            f"stopping before the remaining campaigns")
    print(f"\n=== {campaign} complete ({dt:.0f} s)", flush=True)


def main():
    campaigns = [resolve(a) for a in sys.argv[1:]] or list(CAMPAIGNS)
    t0 = time.perf_counter()
    for campaign in campaigns:
        run(campaign)
    print(f"\nall requested campaigns complete "
          f"({time.perf_counter() - t0:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
