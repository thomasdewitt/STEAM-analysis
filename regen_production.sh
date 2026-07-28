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
#   2. production squares + strip nests (20 GPU squares ~3 min each,
#      CPU nests pipelined behind them)
#   3. analysis: deltas, channel fractal + PDFs, square-level Haar
#      (the diagnostic that caught the bug — validation), square
#      fractal/size distributions, strip-nest multifractal appendix
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "=== [1/3] RCEMIP channels ($(date +%H:%M)) ==="
$PY run_steam.py

echo "=== [2/3] squares + nests ($(date +%H:%M)) ==="
$PY run_production_squares.py

echo "=== [3/3] analysis ($(date +%H:%M)) ==="
$PY make_deltas.py
$PY make_fractal.py
$PY make_pdfs.py
$PY make_square_level_haar.py
$PY make_fractal_square.py
$PY make_strip_appendix.py runs/steam_sq10_icon_lem_m00.nc refinements/r0

echo "=== ALL DONE ($(date +%H:%M)) ==="
