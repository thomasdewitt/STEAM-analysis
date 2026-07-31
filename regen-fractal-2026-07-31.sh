#!/bin/bash
# Regeneration for the 2026-07-31 fractal + glimpse deliverables.
#
# The squares and matched sims were deleted after the earlier figures
# (plots matter, output is disposable), so the .nc have to come back before
# the fractal masks and the glimpse renders can be made. Strictly serial:
# one GPU peak at a time. Restartable -- both drivers skip on an existing .nc.
#
# Only the squares themselves are regenerated, NOT the strip nests in
# refinements/r0: nothing in today's deliverables reads them, they are ~11
# min of CPU each, and the .nc are deleted again at the end. The strip
# appendix figure already exists from the campaign run.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
CAP="systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 --same-dir"

echo "=== [1/2] squares ($(date +%H:%M:%S)) ==="
$CAP $PY -u -c "
from run_production_squares import MODELS, N_MEMBERS, run_square
import time
for model in MODELS:
    for member in range(N_MEMBERS):
        t0 = time.perf_counter()
        run_square(model, member)
        print(f'TIMING square {model} m{member:02d}: {time.perf_counter()-t0:.0f} s', flush=True)
"

echo "=== [2/2] matched sims ($(date +%H:%M:%S)) ==="
$CAP $PY -u run_steam_matched.py les_cm1 les_sam les_dales les_icon_lem twpice

echo "=== REGEN DONE ($(date +%H:%M:%S)) ==="
