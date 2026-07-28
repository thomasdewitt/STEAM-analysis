#!/usr/bin/env python3
"""Curvature dissection: one instrumented 2048^2 production square per variant.

Thomas's 2026-07-28 ruling: variance level is fine, but the SLOPES must
match — especially at small scales, which should be unaffected by the
clip where the field is far from the bounds. This probe runs the exact
production icon_lem config (seed 2000, same as steam_sq10_icon_lem_m00)
with the machinery toggled per variant, and records the cascade's
internals:

  - per-class deposited increment at ~7 km (CONVOLVE wrapper),
  - realized <|W|> ladder over nonzero centers at that level,
  - the running perturbation at ~7 km BEFORE and AFTER each class's
    mean-preserving projection (_project_onto_bounds wrapper),
  - bound-taper activity at that level (mean g, fraction g<1, g==0).

Variants:
  stock            production code as-is
  nocomp           ZOOM_RETENTION compensation disabled (all-ones)
  nobounds         taper == 1 and projection disabled (bounds machinery off)
  nocomp_nobounds  both

Usage: .venv/bin/python archaeology/curvature_probe.py VARIANT
Writes runs/archaeology/curv_VARIANT.nc and stats/curv_VARIANT.npz.
"""

import importlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs" / "archaeology"
STATS = HERE / "stats"
LEVEL_M = 7000.0
CP = 1004.0
LV = 2.5e6

VARIANT = sys.argv[1]
assert VARIANT in ("stock", "nocomp", "nobounds", "nocomp_nobounds",
                   "b1", "b2", "renormtaper", "noproj",
                   "renormtaper_noproj", "ampproj"), VARIANT

MODEL_REPO = Path.home() / "code-and-data" / "turbulon-model"

# The taper-inside-norm reorder cannot be monkeypatched (inline in
# cascade_loop), so patch a throwaway copy of the package and import that.
NORM_BLOCK_OLD = """\
            W *= S_k            # joint pattern W*S_k (sparse at centers)
            inner = _inner_view(W, window)
            level_sum = np.abs(inner).sum(axis=(0, 1), dtype=np.float64)
            level_cnt = np.count_nonzero(inner, axis=(0, 1))
            level_mean = (level_sum / np.maximum(level_cnt, 1)).astype(np.float32)
            W /= np.where(level_mean > 0, level_mean, np.float32(1.0))[None, None, :]

            # Bound taper, applied AFTER the norm: proximity of one part of
            # the field to a bound reduces that level's total variance
            # rather than redistributing it to other parts of the level.
            W *= _bound_taper(running_sum, b_i, phi_min, phi_max)
            del running_sum
"""
NORM_BLOCK_NEW = """\
            W *= S_k            # joint pattern W*S_k (sparse at centers)
            # VARIANT renormtaper: taper INSIDE the norm (May philosophy) —
            # bound proximity redistributes the level's amplitude instead of
            # suppressing it; <|A|> = C_k is enforced on the tapered pattern.
            W *= _bound_taper(running_sum, b_i, phi_min, phi_max)
            del running_sum
            inner = _inner_view(W, window)
            level_sum = np.abs(inner).sum(axis=(0, 1), dtype=np.float64)
            level_cnt = np.count_nonzero(inner, axis=(0, 1))
            level_mean = (level_sum / np.maximum(level_cnt, 1)).astype(np.float32)
            W /= np.where(level_mean > 0, level_mean, np.float32(1.0))[None, None, :]
"""

# VARIANT ampproj: taper inside the norm (pattern shaping) + the
# amplitude-preserving bounded add replacing the per-class projection.
# The operator iterates demean -> clip-to-caps -> rescale-to-A0 per
# level, with a final demean+clip pass so bounds are exact; A0 is the
# increment's own pre-clip mean-abs (bounding decoupled from
# calibration). Near the feasibility ceiling 2*min(<phi>-lo, hi-<phi>)
# the rescale stalls against the clip: the level delivers the maximum
# realizable amplitude instead of the design value.
AMPPROJ_DEPOSIT_OLD = """\
            W *= C_k_i          # 1D broadcast: mean amplitude C_k(z)

            # Interpolation-retention compensation: amplify this class's
            # deposit by the inverse of the measured retention of the
            # regrid chain it has yet to traverse, so the FINAL grid
            # carries the designed amplitude ladder. The finest class is
            # never regridded (factor 1).
            m = min(n_classes - 1 - i, len(ZOOM_RETENTION) - 1)
            W *= np.float32(1.0 / ZOOM_RETENTION[m])

            # Convolve and accumulate (periodic x,y; zero-padded z)
            perturbation_field += CONVOLVE(W, kernel, device=device)
            del W

            # Mean-preserving projection onto the bounds, so subsequent
            # classes' gradients see a field already within [φ_min, φ_max].
            _project_onto_bounds(perturbation_field, mean_1d, phi_min, phi_max,
                                 window=window)
"""
AMPPROJ_DEPOSIT_NEW = """\
            W *= C_k_i          # 1D broadcast: mean amplitude C_k(z)

            m = min(n_classes - 1 - i, len(ZOOM_RETENTION) - 1)
            W *= np.float32(1.0 / ZOOM_RETENTION[m])

            # VARIANT ampproj: amplitude-preserving bounded add.
            inc = CONVOLVE(W, kernel, device=device)
            del W
            _bounded_amplitude_add(perturbation_field, mean_1d, inc,
                                   phi_min, phi_max, window=window)
            del inc
"""
AMPPROJ_FUNC = '''

def _bounded_amplitude_add(pert, mean_1d, inc, lo, hi, window=None, n_iter=10):
    """Add inc to pert with zero level mean, preserved level mean-abs
    amplitude, and bounds respected (VARIANT ampproj prototype).

    Iterates demean -> clip to pointwise caps -> rescale to the pre-clip
    amplitude A0, then a final demean+clip so bounds are exact. Levels
    whose caps never bind converge in one pass (scale 1). Where A0
    exceeds the feasibility ceiling the rescale stalls against the clip
    and the level delivers its maximum realizable amplitude.
    """
    lo32 = np.float32(lo)
    hi32 = np.float32(hi)
    nz = pert.shape[2]
    for lev in range(nz):
        phi = pert[:, :, lev] + mean_1d[lev]
        cl = lo32 - phi
        ch = hi32 - phi
        d = inc[:, :, lev].astype(np.float32).copy()
        dw = d if window is None else d[window[0]:window[1], window[2]:window[3]]
        a0 = float(np.abs(dw).mean(dtype=np.float64))
        if a0 <= 0.0:
            continue
        for _ in range(n_iter):
            d -= np.float32(dw.mean(dtype=np.float64))
            np.clip(d, cl, ch, out=d)
            m_abs = float(np.abs(dw).mean(dtype=np.float64))
            if m_abs <= 0.0:
                break
            scale = min(a0 / m_abs, 2.0)
            if abs(scale - 1.0) < 1e-4:
                break
            d *= np.float32(scale)
        d -= np.float32(dw.mean(dtype=np.float64))
        np.clip(d, cl, ch, out=d)
        pert[:, :, lev] += d
'''

if VARIANT.startswith("renormtaper") or VARIANT == "ampproj":
    tmp_pkg = Path("/tmp/steam_variant_pkg")
    if tmp_pkg.exists():
        shutil.rmtree(tmp_pkg)
    tmp_pkg.mkdir(parents=True)
    shutil.copytree(MODEL_REPO / "steam", tmp_pkg / "steam")
    src_file = tmp_pkg / "steam" / "simulate.py"
    src = src_file.read_text()
    assert NORM_BLOCK_OLD in src, "norm block drifted; update probe"
    src = src.replace(NORM_BLOCK_OLD, NORM_BLOCK_NEW)
    if VARIANT == "ampproj":
        assert AMPPROJ_DEPOSIT_OLD in src, "deposit block drifted; update probe"
        src = src.replace(AMPPROJ_DEPOSIT_OLD, AMPPROJ_DEPOSIT_NEW)
        src += AMPPROJ_FUNC
    src_file.write_text(src)
    sys.path.insert(0, str(tmp_pkg))
    sm = importlib.import_module("steam.simulate")
    assert "VARIANT renormtaper" in Path(sm.__file__).read_text()
else:
    sm = importlib.import_module("steam.simulate")
    STEAM_REPO = next(p for p in Path(sm.__file__).resolve().parents
                      if (p / ".git").exists())
    assert STEAM_REPO.name == "turbulon-model", f"steam from {STEAM_REPO}"

# Production calibration (run_production_squares.py)
sm.H_h = 0.45
C1_TARGET = 0.05
sm.FLUX_SCALE = (C1_TARGET / 3.097) ** (1 / 1.8)

if "nocomp" in VARIANT:
    sm.ZOOM_RETENTION = (1.0,)
if VARIANT == "b1":
    sm.BOUND_BUFFER_MULTIPLE = 1
if VARIANT == "b2":
    sm.BOUND_BUFFER_MULTIPLE = 2

BOUNDS_OFF = "nobounds" in VARIANT
PROJ_OFF = BOUNDS_OFF or "noproj" in VARIANT
real_taper = sm._bound_taper
real_project = sm._project_onto_bounds

# ---------------------------------------------------------------- grids stash
GRIDS = {}
real_cascade = sm.cascade_loop


def stash_cascade(*args, **kwargs):
    GRIDS["grids"] = args[3]
    return real_cascade(*args, **kwargs)


sm.cascade_loop = stash_cascade


def class_iz(i):
    z = GRIDS["grids"]["z_arrays"][i]
    return int(np.argmin(np.abs(np.asarray(z) - LEVEL_M)))


REC = {"amp": {"h": [], "qt": []}, "inc": {"h": [], "qt": []},
       "ps_pre": {"h": [], "qt": []}, "ps_post": {"h": [], "qt": []},
       "taper": {"h": [], "qt": []}}

# ---------------------------------------------------------------- convolve
real_convolve = sm.CONVOLVE
conv_count = [0]


def recording_convolve(field, kernel, device="cpu"):
    result = real_convolve(field, kernel, device=device)
    i, sub = conv_count[0] // 3, conv_count[0] % 3
    conv_count[0] += 1
    if sub != 0:                       # 0 = flux noise, 1 = h, 2 = qt
        name = "h" if sub == 1 else "qt"
        iz = class_iz(i)
        level = np.asarray(field[:, :, iz], dtype=np.float64)
        nonzero = level[level != 0.0]
        REC["amp"][name].append(
            float(np.abs(nonzero).mean()) if nonzero.size else np.nan)
        REC["inc"][name].append(np.asarray(result[:, :, iz], dtype=np.float32))
    return result


sm.CONVOLVE = recording_convolve

# ---------------------------------------------------------------- taper
taper_count = [0]


def recording_taper(running_sum, b, phi_min, phi_max):
    i = taper_count[0] // 2
    name = "h" if taper_count[0] % 2 == 0 else "qt"
    taper_count[0] += 1
    iz = class_iz(i)
    if BOUNDS_OFF:
        REC["taper"][name].append((1.0, 0.0, 0.0))
        return np.float32(1.0)
    g = real_taper(running_sum, b, phi_min, phi_max)
    gl = np.asarray(g[:, :, iz], dtype=np.float64)
    REC["taper"][name].append(
        (float(gl.mean()), float((gl < 1.0).mean()), float((gl == 0.0).mean())))
    return g


sm._bound_taper = recording_taper

# ---------------------------------------------------------------- projection
proj_count = [0]


def recording_project(perturbation_field, mean_1d, phi_min, phi_max, window=None):
    i = proj_count[0] // 2
    name = "h" if proj_count[0] % 2 == 0 else "qt"
    proj_count[0] += 1
    iz = class_iz(i)
    REC["ps_pre"][name].append(
        np.asarray(perturbation_field[:, :, iz], dtype=np.float32))
    if not PROJ_OFF:
        real_project(perturbation_field, mean_1d, phi_min, phi_max, window=window)
    REC["ps_post"][name].append(
        np.asarray(perturbation_field[:, :, iz], dtype=np.float32))


sm._project_onto_bounds = recording_project


# ---------------------------------------------------------------- run
def main():
    from steam.thermodynamics import _saturation_mixing_ratio
    src = np.load(STATS / "icon_lem_snap0.npz")
    sp = float(src["surface_pressure"])
    qts = float(_saturation_mixing_ratio(300.0, sp))
    h_upper = max(CP * 300.0 + LV * qts, float(src["h_profile"].max()) + 1.0)
    h_lower = float(src["h_profile"].min()) - 10.0 * CP
    z = src["z_profile"]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"curv_{VARIANT}.nc"
    if out.exists():
        out.unlink()

    import os
    test = bool(os.environ.get("CURV_TEST"))
    n = 512 if test else 2048
    t0 = time.monotonic()
    sm.simulate(
        src["h_profile"], src["qt_profile"],
        nx=n, ny=n, dx=3000.0, dy=3000.0,
        outer_scale=n * 3000.0 / 4,
        spheroscale=np.full(z.size, 10.0),
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=20_000.0,
        profile_dz=50.0,
        output_path=str(out),
        surface_pressure=sp,
        seed=2000,
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qts,
        compress=False,
        device="cuda",
    )
    print(f"{VARIANT}: simulate {time.monotonic() - t0:.0f} s", flush=True)

    grids = GRIDS["grids"]
    n_classes = len(grids["k"])
    save = {
        "k_values": np.asarray(grids["k"], dtype=np.float64),
        "dx_k": np.asarray(grids["dx"], dtype=np.float64),
        "iz_k": np.array([class_iz(i) for i in range(n_classes)]),
        "z_at_iz": np.array([float(grids["z_arrays"][i][class_iz(i)])
                             for i in range(n_classes)]),
    }
    for name in ("h", "qt"):
        save[f"amp_{name}"] = np.array(REC["amp"][name])
        save[f"taper_{name}"] = np.array(REC["taper"][name])   # (n, 3)
        for i in range(n_classes):
            save[f"inc_{name}_{i}"] = REC["inc"][name][i]
            if i < len(REC["ps_pre"][name]):     # ampproj: projection unused
                save[f"ps_pre_{name}_{i}"] = REC["ps_pre"][name][i]
                save[f"ps_post_{name}_{i}"] = REC["ps_post"][name][i]
    np.savez_compressed(STATS / f"curv_{VARIANT}.npz", **save)
    print(f"wrote stats/curv_{VARIANT}.npz "
          f"({n_classes} classes, conv calls {conv_count[0]})", flush=True)


if __name__ == "__main__":
    main()
