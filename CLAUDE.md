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
2. `fractal-analysis/scripts/run_steam_simulations.py` — the square campaign.
   One file per member into `runs/square/`, holding the parent square whole —
   every variable, refinement state included — plus the parent's 2D vertically
   integrated optical depth, and the two nests stripped to qc and qi. The
   keeper still cannot seed a new nest: `refine` also reads the
   `class_increments` groups, which stay behind with the working file. `RUN_NESTS` switches both
   nests on or off together; `SETS` maps a set tag onto the flux noise
   amplitude directly.
3. `fractal-analysis/scripts/compute_fractal_metrics.py` — the paper's four
   metrics (D_f, D_e, tau_area, tau_per) over the members matched by
   `PATTERN`, pooled into one ensemble ->
   `fractal-analysis/output/fractal_metrics_<set>.npz`.
4. `fractal-analysis/scripts/plot_fractal_metrics.py` — the four scaling
   functions.

Each campaign subfolder lays out the same way: `scripts/` for code, `output/`
for cached statistics, `figs/` for figures and the text tables beside them.
Every subfolder's run generator is `scripts/run_steam_simulations.py`;
`run_all_steam_simulations.py` at the repo root calls all three in sequence,
cheapest first, skipping whatever is already on disk.

Conventions that matter in step 3: every matched member is passed to objscale
in a single call per metric, because these estimators are regressions and
computing per file then averaging gives a different, wrong answer. objscale
runs with default parameters throughout, and the perimeter distribution uses
`'nested perimeter'`.

## hydrodynamic-comparison/

The matched comparison against SAM-TWPICE, SAM-GATE and the nine
RCE_large300 channels. GATE is hour 23, the last hour with complete data (20
and 24 h exist only as 210-level gap files rebuilt from `.dat`); it archives
only the combined condensate `QN`, so the liquid/ice split uses SAM's linear
ramp — all liquid at 0 C, all ice at -38 C — applied at native resolution,
before coarsening, since it does not commute with the average.
`scripts/run_steam_simulations.py` runs one STEAM simulation per host per
flux amplitude, on the host's domain at twice its horizontal spacing; then a compute/plot pair per
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

`scripts/run_steam_simulations.py` writes one finely resolved run per flux
amplitude for visualization, on the square campaign's ukmo_ra1t profile, into
`runs/small-domain/`. A 40.96 x 20.48 km parent at dx = 20 m, surface to 5 km,
carries a centered 2.56 x 2.56 km nest at dx = 5 m to full depth
(`refinements/r0`).

The vertical spacing follows dx through the aspect ratio, dz = k_z(2 dx) / 2:
the parent is (2048, 1024, 463) at dz = 10.80 m, a 3.62 GiB field and ~38 GiB
at the cascade's peak, and the nest is (512, 512, 1001) at dz = 5.00 m,
0.98 GiB and ~10 GiB. Doubling dx from the earlier 10 m run is what makes the
wider footprint fit — it halves the level count, so the parent field is
actually smaller than the 2048 x 768 run it replaces. The nest's dz stops
falling with dx because 2 dx has reached the 10 m spheroscale and the finest
class is isotropic. The runs want the machine to themselves.

## Dependencies

`uv sync` against `pyproject.toml`. `steam` and `objscale` are editable path
dependencies on their sibling repos.

`cloudyview` is **not** managed here: the square campaign's
`run_steam_simulations.py` loads
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
which is cheap to rebuild from `data/`. Figures are **not** committed
(2026-08-07): the top-level `figs/` record was deleted and the `*.png` /
`*.pdf` ignore rules now stand unforced, so each subfolder's `figs/` is local
and the plotting scripts are the record. The text tables written beside the
figures (`*_fractal_metrics.txt`, `fractal_table.tex`) are tracked, since they
carry the numbers the paper quotes.
