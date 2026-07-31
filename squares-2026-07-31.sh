#!/bin/bash
# Square-domain campaign, 2026-07-31 (Thomas). Committed config:
# H_h = 0.45, lambda = 0.28927, C1 = 0.05, constant 10 m spheroscale,
# 2048^2 at dx = 3 km, outer scale 1536 km, 3 members each of the
# CF-extreme profile pair ukmo_ra1t (low) and icon_nwp (high).
#
# Strictly serial: one process, one peak at a time. Restartable --
# squares whose .nc exists are skipped, nests whose group exists are
# skipped -- so rerunning resumes where it stopped.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

# Memory safety net on the run steps (55G of 60G, no swap): an OOM kill
# lands inside the scope, never on the login session, and the driver just
# redoes that member on the next run.
CAP="systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 --same-dir"

echo "=== [1/3] squares + strip nests ($(date +%H:%M:%S)) ==="
$CAP $PY -u run_production_squares.py

echo "=== [2/3] render nests, m00 of each profile ($(date +%H:%M:%S)) ==="
$CAP $PY -u run_render_nests.py runs/steam_sq10_ukmo_ra1t_m00.nc
$CAP $PY -u run_render_nests.py runs/steam_sq10_icon_nwp_m00.nc

echo "=== [3/3] figures ($(date +%H:%M:%S)) ==="
$PY -u make_strip_appendix.py runs/steam_sq10_ukmo_ra1t_m00.nc refinements/r0
$PY -u make_fractal_square.py
$PY -u make_square_level_haar.py

echo "=== CAMPAIGN DONE ($(date +%H:%M:%S)) ==="
