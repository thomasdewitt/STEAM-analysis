# turbulon-analysis

The complete analysis for the STEAM paper's RCEMIP comparison. The sister repo
`turbulon-model` holds the STEAM model itself and the paper source (`paper/`).
Reproducibility ruling: it's one repo for model, one for analysis, and the full
paper is complete with those two repos.

## Pipeline (channel / RCE_large300)

Run in order:

1. `extract_stats.py` — per-level h/qt mean+variance, cloud fraction, and the
   STEAM input profile for each host model snapshot -> `stats/<model>_snap<i>.npz`
2. `run_steam.py` — the frozen STEAM config against each extracted profile ->
   `runs/steam_<model>_snap<i>.nc` + `stats/steam_<model>_snap<i>.npz`
3. figures, any order:
   - `make_deltas.py` — STEAM$-$host profile deltas vs inter-LES spread
   - `make_pdfs.py` — anomaly PDFs
   - `make_fractal.py` — tau>1 mask correlation dimension + size distributions

## Small-domain ensemble (RCE_small_les300)

`small_pipeline.py` — self-contained version for the four small-domain models
(SAM, CM1, DALES, ICON-LEM). Phases as args (default all):
`python small_pipeline.py hosts steam figures`. Writes `stats_small/` and the
`*_small.png` figures.

## Data

`data/` and `runs/` are gitignored. The host originals live on the Expansion
drive under `hydrodynamic-model-output/RCEMIP/`. Copy the files each adapter in
`extract_stats.py` / `small_pipeline.py` globs into `data/<model>/`.

The `stats/` and `stats_small/` `.npz` are gitignored (regenerate them with the
pipeline). The `figs/*.png` outputs are committed as the figure record.
