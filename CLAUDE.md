# turbulon-analysis

Analysis for the STEAM paper. The sister repo `turbulon-model` holds the STEAM
model itself and the paper source (`paper/`). Reproducibility ruling: one repo
for the model, one for the analysis, and the full paper is complete with those
two.

**Scope note (2026-08-05).** This repo was narrowed to the square-domain cloud
geometry analysis. The RCEMIP channel comparison pipeline — `extract_stats.py`,
`run_steam*.py`, the `make_*` figure scripts, the archaeology and calibration
probes — was deleted, along with the per-snapshot `stats/` archive. It is all in
git history at `362d59e~1` if any of it is needed again.

## Pipeline

Run in order:

1. `make_input_profiles.py` — mean h and qt profiles for each host, averaged
   over every archived timestep, on a uniform 50 m grid to 20 km, plus surface
   pressure -> `runs/input_profiles/<host>.npz`. All 14 hosts by default, or
   name them on the command line.
2. `fractal_analysis/run_paper_squares.py` — the square campaign. One file per
   member into `runs/square/`, holding qc and qi only plus the parent's 2D
   vertically integrated optical depth. `RUN_NESTS` switches both nests on or
   off together; `SETS` maps a set tag onto the flux noise amplitude directly.
3. `fractal_analysis/compute_fractal_metrics.py` — the paper's four metrics
   (D_f, D_e, tau_area, tau_per) over the members matched by `PATTERN`, pooled
   into one ensemble -> `fractal_analysis/fractal_metrics.npz`.
4. `fractal_analysis/plot_fractal_metrics.py` — the four scaling functions.

Conventions that matter in step 3: every matched member is passed to objscale
in a single call per metric, because these estimators are regressions and
computing per file then averaging gives a different, wrong answer. objscale
runs with default parameters throughout, and the perimeter distribution uses
`'nested perimeter'`.

## hydrodynamic-comparison/

The matched comparison against SAM-TWPICE and the nine RCE_large300 channels.
`generate.py` runs one STEAM simulation per host per flux amplitude, on the
host's domain at twice its horizontal spacing; then a compute/plot pair per
figure, with `common.py` holding the matching rule and the styling.

Two resolution conventions, deliberately different:

- **Profiles and PDFs** (`compute_{twpice,rcemip}_stats.py`) are matched. The
  hosts are block-averaged 2x2 horizontally onto STEAM's spacing, and per
  level whichever field is finer vertically is averaged by the nearest
  integer factor that closes the gap — so the factor varies with height, and
  points at STEAM in some places and at the host in others.
- **Scaling functions** (`compute_scaling.py`) are at native resolution on
  both sides. Matching resolutions there would destroy the thing being
  measured; the curves simply start at different smallest lags.

Cloud fraction is condensate >= 0.01 g/kg, thresholded after coarsening.
`qt` excludes precipitating water throughout, and K-scale fields are cast to
float64 on read rather than at each reduction.

Scripts live in `scripts/`, cached statistics in `output/`, figures in
`figs/`; the simulations themselves go to `runs/hydro/` with everything else
regenerable.

## small-domain/

`generate.py` writes two finely resolved runs for visualization —
20.48 x 7.68 km at dx = 10 m, surface to 5 km, on the square campaign's
ukmo_ra1t profile, one per flux amplitude, into `runs/small-domain/`. The
vertical spacing follows dx through the aspect ratio (dz = 7.34 m,
681 levels), so the peak working set is ~46 GiB and the runs want the machine
to themselves.

## Dependencies

`uv sync` against `pyproject.toml`. `steam` and `objscale` are editable path
dependencies on their sibling repos.

`cloudyview` is **not** managed here: `run_paper_squares.py` loads
`optical_depth.py` directly from `~/code-and-data/cloudyview/` by file path,
because it is not installed in this venv. The campaign fails at import, not
mid-run, if that repo is missing.

## Exclusions

MESONH is excluded entirely: its archived 3D `hus` is a documented RCEMIP data
error (Known RCEMIP Bugs doc on Expansion, Sec. 17). ICON_AES is excluded for
having no usable z. The small-domain (RCE_small_les300) analysis was dropped
2026-07-21.

## Data

`data/` and `runs/` are gitignored, as are `*.nc`, `*.npz`, `*.png`, `*.pdf`
and `*.log`. The host originals live on the Expansion drive under
`hydrodynamic-model-output/RCEMIP/`; copy the files each adapter in
`make_input_profiles.py` globs into `data/<host>/`.

Everything under `runs/` is regenerable — including `runs/input_profiles/`,
which is cheap to rebuild from `data/`. Figures under `figs/` are committed as
the figure record, which means force-adding them past the `*.png` rule.
