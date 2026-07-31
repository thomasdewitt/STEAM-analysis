#!/bin/bash
# Production regeneration under the 2026-07-28-evening code (joint norm,
# INTERPOLATION_COMPENSATION on scalars AND flux, per-class increments on
# square parents, lambda re-anchored). GATED on the c->C1 re-fit
# (flux compensation moved the calibration); update FLUX_SCALE first.
# bounded amplitude-preserving add, ZOOM_RETENTION comp). Pre-fix output
# deleted first. Squares trimmed to 3 members. Restartable (existing outputs skipped),
# so rerunning this script resumes where it stopped.
#
#   1. RCEMIP channel comparison  (27 runs, GPU, ~15 min)
#   2. production squares + strip nests (GPU square then its own CPU nest,
#      one member at a time)
#   3. analysis: deltas, channel fractal + PDFs, square-level Haar
#      (the diagnostic that caught the bug — validation), square
#      fractal/size distributions, strip-nest multifractal appendix
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

# Safety net on the run steps: a single process with one peak at a time, now
# that run_production_squares.py runs each square and then its own nest
# sequentially (the square-against-nest pipelining, and with it the
# co-residency hazard this cap was written for, is gone). The victim is inside
# the scope and the driver is restartable -- members whose .nc exists are
# skipped, nests whose group exists are skipped -- so an OOM-killed member is
# simply redone on the next run, and the login session is protected absolutely.
CAP="systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 --same-dir"

echo "=== [1/3] RCEMIP channels ($(date +%H:%M)) ==="
$CAP $PY run_steam.py

echo "=== [2/3] squares + nests ($(date +%H:%M)) ==="
$CAP $PY run_production_squares.py

echo "=== [3/3] analysis ($(date +%H:%M)) ==="
$PY make_deltas.py
$PY make_fractal.py
$PY make_pdfs.py
$PY make_square_level_haar.py
$PY make_fractal_square.py
$PY make_strip_appendix.py runs/steam_sq10_icon_lem_m00.nc refinements/r0

echo "=== ALL DONE ($(date +%H:%M)) ==="
