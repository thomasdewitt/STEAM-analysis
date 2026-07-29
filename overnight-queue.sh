#!/bin/bash
# Strictly serial job queue (2026-07-29, after two OOMs from stacked jobs).
# Everything restartable; safe to rerun the whole script.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "=== [1/7] remaining strip nests ($(date +%H:%M)) ==="
$PY -u run_production_squares.py

echo "=== [2/7] C1=0.01 l_s=1 channels ($(date +%H:%M)) ==="
$PY -u run_steam.py

echo "=== [3/7] strip-appendix figures ($(date +%H:%M)) ==="
$PY -u make_strip_appendix.py runs/steam_sq10_icon_lem_m00.nc refinements/r0

echo "=== [4/7] render nest B ($(date +%H:%M)) ==="
$PY -u run_render_nests.py

echo "=== [5/7] diag stats + profile/pdf/delta/slope refresh ($(date +%H:%M)) ==="
$PY -u make_diag_stats.py
$PY -u make_profiles_ensemble.py
$PY -u make_deltas.py
$PY -u make_pdfs.py
rm -f stats/local_slope.npz          # force recompute with 5th config
$PY -u make_local_slope.py

echo "=== [6/7] geometry refresh ($(date +%H:%M)) ==="
$PY -u make_fractal.py
$PY -u make_fractal_square.py
$PY -u make_square_level_haar.py

echo "=== [7/7] nest B renders ($(date +%H:%M)) ==="
~/code-and-data/cloudyview/.venv/bin/python -u make_renders.py

echo "=== QUEUE DONE ($(date +%H:%M)) ==="
