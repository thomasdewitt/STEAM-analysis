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
   One file per member into `runs/square/`, holding every parent variable but
   each 3D one at three levels only — those nearest `PARENT_LEVELS` = 5, 10
   and 15 km — plus the parent's 2D vertically integrated optical depth, and
   the two nests stripped to qc and qi at full depth. The keeper still cannot
   seed a new nest: `refine` also reads the `class_increments` groups, which
   stay behind with the working file. `RUN_NESTS` switches both nests on or
   off together; `SETS` maps a set tag onto the flux noise amplitude directly.

   **Vertical thinning (2026-08-07).** Full-depth keepers were 21.6 GB each
   — the eleven parent 3D fields are 3.30 GiB raw apiece at 2048² x 211 and
   the nine non-condensate ones compress only 1.16–1.8x — so twenty came to
   431 GB against 340 GB free. Three levels measure 0.38 GB per member, 7.6 GB
   over twenty. Nothing downstream reads a parent 3D field: `compute_fractal_
   metrics.py` takes `tau` and `dx` and nothing else, and `tau` is still the
   full-column integral, computed in the working file before the thinning.
   The parent's `z`, `dz` and `spheroscale` are cut to match; `C_h_k` and
   `C_qt_k` are on `nz_k_max` and stay whole. The keeper records
   `parent_z_levels` at the root and `level_targets` / `level_indices` /
   `source_nz` on the parent group, and a keeper written under a different
   vertical spec is refused rather than pooled — pre-thinning members must be
   moved or deleted.
3. `fractal-analysis/scripts/compute_fractal_metrics.py` — the paper's four
   metrics (D_f, D_e, tau_area, tau_per) over the members matched by
   `PATTERN`, pooled into one ensemble ->
   `fractal-analysis/output/fractal_metrics_<set>.npz`.
4. `fractal-analysis/scripts/plot_fractal_metrics.py` — the four scaling
   functions, one figure per set.

**Flux amplitudes (2026-08-08).** One naming convention across all three
campaigns: the set tag is `c` followed by c x 100 zero-padded to three
digits, so `c002` = 0.02, `c005` = 0.05, `c017` = 0.17. The square
campaign's `C1small`/`C1large` were renamed to `c005`/`c017` — the
amplitudes did not move, so the existing keepers still match
`campaign_spec`, and that campaign's member seed never depended on the tag.
`c002` was added to all three. Renaming a set tag in
`hydrodynamic-comparison/` or `small-domain/` is not free the same way:
their seeds are derived from position in the `SETS` dict, so new tags go on
the end.

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
flux amplitude per outer scale, on the host's domain at twice its horizontal
spacing; then a compute/plot pair per figure, with `common.py` holding the
matching rule and the styling. Runs are `runs/hydro/<host>_<set>_<L>.nc`.

**Outer scale (2026-08-08).** The second axis, and what doubles the channel
comparison. `Llong` sets L to the longest horizontal extent, `Lshort` to the
shortest — 6144 km and 384 km for the channels. `simulate` wants each extent
to be an integer multiple of L or smaller than it: at `Llong` the long axis
is one tile and the short axis a folded strip, at `Lshort` the long axis is
exactly 16 tiles and neither is a strip. The class ladder is what changes,
L/2dx falling from 512 to 32 — nine dyads of cascade against five. The
vertical grid does not move, because dz follows k_z(2 dx) and the finest
class is the same either way; both cases land on the same 78 levels, which
is what lets the host side be reduced once and asserted against, not
recomputed per L. TWPICE and GATE are square, so their two cases coincide
and they run `Llong` alone.

Figures split on the outer scale rather than pooling it: `rcemip_profiles_
<L>`, `rcemip_pdfs_<L>`, `rcemip_scaling_<L>`. Flux amplitude is pooled into
the single STEAM envelope on the profile and PDF figures and is a colour on
the line figures; the outer scale is neither, because merging a 6144 km
cascade with a 384 km one into one band would hide the distinction the axis
exists to show. `plot_rcemip.members()` refuses any STEAM tag it has no
grouping for, and refuses a figure handed more than one outer scale.

The RCEMIP profile, PDF and scaling figures all draw two envelopes — the
min-to-max across hosts and across STEAM runs — rather than one line per
run, pooling the flux amplitudes into the STEAM band and never the outer
scale. The twpice figures keep one line per run: two LES cases are not a
population. Profiles carry a bar beside each panel, green where the two
envelopes overlap and red where they are non-overlapping. The scaling
bands take no interpolation — within a band every run shares dx and
domain, so the lag axes are identical and the min/max is pointwise, which
`curves()` checks rather than assumes.

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
amplitude per spheroscale for visualization: a parent carrying one centered
nest (`refinements/r0`), on a single host's profile with anchored bounds.
`RUN_NEST` switches the nest on or off; with it off the parent is written
without refinement state, so turning the flag back on means rerunning the
parent, and `run_nest` refuses such a file rather than failing deeper in
`refine`. The geometry, the profile host and the tags live in the script's
constants and are not repeated here — they are still moving.

**Spheroscale (2026-08-10).** The axis these runs exist to show, tagged by the
spheroscale in metres zero-padded to four digits. Each case carries its own
outer scale, because one L cannot serve both: k_z,L = l_s (L/l_s)^H_z rises
with the spheroscale and `simulate` refuses any config whose k_z,L reaches the
domain top. Raising the top instead is not the cheap way out, since dz follows
the aspect ratio down — a taller domain buys room in levels the cascade then
carries. So the coarser spheroscale runs the shorter L and the shallower class
ladder, and the two pictures differ in cascade depth as well; that is the price
of the axis rather than something the runs hide. The anchored bounds use the
prescribed 300 K SST whatever the host, as `hydrodynamic-comparison/` does.

**One seed for the campaign (2026-08-10).** `SEED` at the top of the script is
what every run uses — no per-set or per-spheroscale offset, as there was when
the axis went in. These are pictures read side by side, so they should differ
by the knob varied and not by the realization underneath. It buys matching
realizations, not identical fields: two runs agree only as far as their class
ladders do, so the amplitudes at one spheroscale are the same cascade rescaled
while the spheroscale cases share a starting point and diverge.

**Stripping (2026-08-10).** Each case runs into `work_small_<set>_<sphero>.nc`
and ends as a keeper `small_<set>_<sphero>.nc` holding qc and qi alone, in a
`parent` group and a `nest` group, on the square campaign's precedent and using
the same `KEEP_AUX` list. The working file is tens of GB — most of it the
cascade state and the per-class increments the nest is cut from — and is
deleted once the keeper is written, which costs the same thing it costs there:
the keeper cannot seed a further nest. Restartable at stage granularity, a
keeper with no working file marking a case complete. The keeper records the
config it was made under (`config_spec`, plus `profile_host`, `run_nest` and
`kept_variables`) and a keeper that disagrees with the config now in force is
refused rather than counted complete — including on `SEED`, so redrawing the
campaign from a new realization names the runs to redo instead of silently
keeping the old pictures. A pre-stripper full run sitting under the keeper's
name is caught by the missing `kept_variables`. The runs want the machine to
themselves.

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
which is cheap to rebuild from `data/`.

**Figures: PDFs tracked, PNGs not** (2026-08-07). Each subfolder's `figs/`
carries both, and the PDF of each is force-added past the `*.pdf` ignore rule
— they are the vector originals the paper's `\includegraphics` takes, and 0.44
MB across the nine of them. The PNGs stay local: they are the same plots at
3.5 MB, for looking at rather than for typesetting. A new figure is not
tracked until someone `git add -f`s its PDF, so add it in the same commit as
the script that draws it. The text tables written beside the figures
(`*_fractal_metrics.txt`, `fractal_table.tex`) are tracked as well, since they
carry the numbers the paper quotes.
