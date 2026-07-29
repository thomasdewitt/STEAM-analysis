# Overnight regeneration + figure construction — plan (Thomas, 2026-07-28 ~17:00)

Timer fires at midnight. Thomas's spec (verbatim intent): regenerate everything
under the finalized 2026-07-28 code and build ALL figures described in main.tex
so he can start from them tomorrow. "If they are in the main already I'd like
them." Use objscale + scaleinvariance wherever possible.

## Runs

1. **Delete stale outputs** — every `runs/steam_*` and `stats/steam_*` predates
   tonight's code (joint norm + INTERPOLATION_COMPENSATION on scalars AND flux +
   lambda 0.7699 + FLUX_SCALE re-fit). Regenerate all.
2. **RCEMIP channel ensemble** (run_steam.py, modify): 2x2 config grid,
   C1 in {0.03, 0.1} -> FLUX_SCALE = (C1/1.681)^(1/1.8); spheroscale constant
   {3 m, 10 m}. All models x 3 snapshots x 4 configs (~108 GPU runs, est. ~1-1.5 h).
   Filenames: `steam_{model}_snap{i}_C1{003|010}_ls{3|10}.nc/.npz`. Keep
   H_h = 0.45, anchored bounds, everything else as run_steam.py has it.
3. **Squares + strip nests** (run_production_squares.py, as-is: C1=0.05, ls=10,
   save_class_increments=True, icon_lem + ukmo_ra1t).
4. **Render nests** (NEW script `run_render_nests.py`), per main.tex L251:
   from an icon_lem square member: nest A = 24x24 km full depth, dx 46.875 m
   (512x512); nest B (refine of A) = 6x6 km, lowest 2 km, dx ~2.93 m
   (2048x2048). Site chosen so nest B has cloud fraction ~0.2 at z = 1 km
   (scan nest-A qc+qi>0.01 g/kg at 1 km in 6-km windows, pick closest to 0.2).
   Nest B contains sub-spheroscale turbulons (main L369 says the render is the
   only sub-l_s sim) — anisotropy already piecewise; fine.

## Figures (figs/, PNG + PDF; dataviz skill before plotting)

A. **Prognostic std profiles** (main Fig () at L235): std h', std qt' vs z.
   STEAM ensemble (12 members/model or pooled across models — pool per-model:
   host line + STEAM percentile shading 5-95% + median across that host's 12).
B. **Diagnostic mean+std profiles** (L236): cloud fraction, qc, qi, T, p —
   same percentile-shading treatment vs host spread.
C. **Fluctuation functions + local exponents** (L238): Mexican hat
   (scaleinvariance.wavelet_fluctuation, order=1, along x, periodic=True),
   h and qt at ONE mid-level (~5 km), hosts vs STEAM; second panel local
   exponent = slope over half-decade window. Also diagnostics version (L240)
   for the same level (qc or T — do T and qc).
D. **Cloud geometry, channels** (make_fractal.py, extend): tau>1 masks,
   objscale defaults; correlation dimension + INDIVIDUAL fractal dimension
   only (his ruling) — hosts vs STEAM configs.
E. **Cloud geometry, squares** (make_fractal_square.py, extend): correlation
   dim, individual fractal dim, PLUS nested (finite-domain) size-distribution
   exponents beta and alpha — perimeter and area, objscale defaults.
   C_l scaling plot (L242's "count of cloud-edge pixels vs r").
F. **Renders** (L253): cloudyview witness (ray-marching tier) on nest B
   (and a nest-A wide shot). qc+qi condensate, sensible sun angle; a few
   azimuths, pick best 2.
G. Existing regen-script figures (deltas, pdfs, square-level Haar, strip
   appendix) — regenerate via regen_production.sh stage 3 equivalents,
   updated for new filenames.

## Order (GPU serial, CPU pipelined)

squares first (needed by nests + renders, longest per-run), RCEMIP ensemble
next, render nests (CPU-heavy refine ok alongside), then analysis + figures.
Actually: RCEMIP first (~1.5 h) then squares, so if something breaks in the
modified run_steam.py it fails early. Nests after their parent square exists
(icon_lem m00). Restartable: keep the exists-skip pattern everywhere.

## Notes
- Load the objscale AND scaleinvariance Skills before writing analysis code
  (Thomas's explicit ask, 17:05).
- Percentile shading: np.percentile across ensemble members at each z.
- make_deltas/make_pdfs expect old filenames — update glob patterns.
- Log everything to overnight-2026-07-28.log; commit at natural boundaries.
- Do NOT push, no PRs. Leave a morning summary in this file + chat.
