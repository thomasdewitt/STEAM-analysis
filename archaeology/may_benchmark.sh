#!/bin/bash
# Overnight May-checkpoint benchmark at the production config
# (Thomas, 2026-07-27 night). Restartable.
set -e
cd "$(dirname "$0")/.."
EGU_PY=$HOME/code-and-data/turbulon-egu/.venv/bin/python
PY=.venv/bin/python

echo "=== May sims (3 squares + strip nest, old code, CPU) ==="
$EGU_PY archaeology/may_production_benchmark.py

echo "=== square-level overlay ==="
$PY archaeology/may_square_overlay.py

echo "=== strip M-hat figures (same estimator, tag=may) ==="
$PY make_strip_appendix.py runs/archaeology/may_sq_m00.nc refinements/r0 \
    --tag may --profiles stats/icon_lem_snap0.npz

echo "=== MAY BENCHMARK DONE ==="
