#!/bin/bash
# Matched STEAM-vs-host campaign, 2026-07-31 (Thomas). One STEAM sim per
# comparison host on that host's geometry: SAM-TWPICE and the four RCEMIP
# RCE_small_les300 LES. Committed config: H_h = 0.45, C1 = 0.05, constant
# 10 m spheroscale, anchored bounds, L = L_x/4, domain top 20 km.
#
# Strictly serial: one process, one peak at a time. Restartable -- a model
# whose stats .npz exists is skipped, so rerunning resumes where it stopped.
# The four LES run first (minutes each); TWPICE is the expensive one.
#
# Step 4 deletes the runs/*.nc (Thomas's ruling): the stats and figures are
# the deliverable and the .nc are ~100 GB of scratch.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

# Memory safety net on the run step (55G of 60G, no swap): an OOM kill lands
# inside the scope, never on the login session.
CAP="systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 --same-dir"

echo "=== [1/4] matched sims ($(date +%H:%M:%S)) ==="
$CAP $PY -u run_steam_matched.py les_cm1 les_sam les_dales les_icon_lem twpice

echo "=== [2/4] anomaly PDFs, compute ($(date +%H:%M:%S)) ==="
$CAP $PY -u make_pdfs_matched.py compute

echo "=== [3/4] figures ($(date +%H:%M:%S)) ==="
$PY -u make_profiles_matched.py
$PY -u make_pdfs_matched.py figure

echo "=== [4/4] deleting matched .nc ($(date +%H:%M:%S)) ==="
du -shc runs/steam_matched_*.nc
rm -f runs/steam_matched_*.nc

echo "=== CAMPAIGN DONE ($(date +%H:%M:%S)) ==="
