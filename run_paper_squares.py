#!/usr/bin/env python3
"""Paper production campaign: paired square ensembles + two nests per member.

Two sets of 10 members each, identical except for the flux noise scale
(paired by seed 2000+member across the sets):

  C1003: C1 = 0.03  ->  c = (0.03/1.681)^(1/1.8)
  C1010: C1 = 0.10  ->  c = (0.10/1.681)^(1/1.8)

Member config (2026-08-04 ruling): TWP-ICE snap0 profile, 2048 x 2048 at
dx = 1 km (2048 km square), outer scale L = 1024 km under the NEW
convention L = (longest domain dimension) / 2 (run_production_squares.py
used the old /4 -- overridden here on Thomas's ruling), constant 10 m
spheroscale, H_h and lambda at package defaults, anchored bounds
(supp S2 as amended 2026-07-27), domain top 20 km, serial CPU
(production ruling: CUDA squares are realization-changing).

Per member, strictly serially:
  1. square (save_for_refinement=True) + compute_diagnostics
  2. nest A: centered 64x64 km, full depth, dx = 62.5 m
     (parent cells 992:1056), save_for_refinement=True, + diagnostics
  3. nest B: refines nest A, centered 16x16 km, lowest 4 km,
     dx = 15.625 m (nest-A cells 384:640), + diagnostics
  4. extraction: 2D tau field (make_fractal_square.py methodology),
     steam_stats + diag stats for the square, keepers file with both
     nest groups (h, qt, T, qc, qi + coords/dz/spheroscale only,
     blosc-zstd)
  5. delete the parent .nc, log wall time and df

Restartable at stage granularity: keepers + tau present with the parent
.nc absent marks a member complete; while the parent exists, the square,
each nest group, each diagnostics pass and each extraction product is
skipped if already present.

Usage: python run_paper_squares.py [SET [MEMBER]]
  no args         -> full sweep (C1003 m00..09, then C1010 m00..09)
  C1003           -> that set only
  C1003 0         -> that single member
"""

import importlib.util
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4

import importlib
_steam_simulate = importlib.import_module("steam.simulate")

from steam.simulate import simulate, refine
from steam.thermodynamics import compute_diagnostics, _saturation_mixing_ratio
from steam.constants import specific_heat_dry_air as cp
from steam.constants import latent_heat_vaporization as Lv
from steam.output import compression_kwargs

from run_steam import steam_stats
from make_diag_stats import do_steam

HERE = Path(__file__).parent
STATS = HERE / "stats"
RUNS = HERE / "runs"

SETS = {  # set tag -> C1 target; c = (C1/1.681)^(1/1.8), 2026-07-28 re-fit
    "C1003": 0.03,
    "C1010": 0.10,
}
N_MEMBERS = 10
NX = 2048
DX = 1000.0
OUTER_SCALE = 1024e3       # NEW convention (2026-08-04): L = longest dim / 2
SPHEROSCALE_CONSTANT = 10.0
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0
SQUARE_SHAPE = (2048, 2048, 211)   # ruled expectation; mismatch is fatal

# Nest A: centered 64x64 km, full depth, target dx = 62.5 m
NEST_A = dict(x_start=992, x_stop=1056, y_start=992, y_stop=1056,
              dx=62.5, dy=62.5)
NEST_A_GROUP = "refinements/r0"
# Nest B: refines nest A, centered 16x16 km, lowest 4 km, dx = 15.625 m
NEST_B = dict(x_start=384, x_stop=640, y_start=384, y_stop=640,
              dx=15.625, dy=15.625, z_min=0.0, z_max=4000.0)
NEST_B_GROUP = "refinements/r1"

TAU_THRESHOLD = 1.0        # applied downstream; the field itself is stored

# cloudyview optical depth, loaded the way make_fractal_square.py does
_spec = importlib.util.spec_from_file_location(
    "cv_optical_depth",
    Path.home() / "code-and-data/cloudyview/cloudyview/optical_depth.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

# Keepers: only these variables (plus coordinates) survive into the
# standalone nest file. flux, qv, p, perturbation states and
# class_increments are deliberately dropped (2026-08-04 ruling).
KEEP_VARS = ("h", "qt", "T", "qc", "qi")
KEEP_AUX = ("x", "y", "z", "z_profile", "dz", "spheroscale", "p_bottom")


def parent_path(set_tag, member):
    return RUNS / f"steam_sq1km_{set_tag}_m{member:02d}.nc"


def keepers_path(set_tag, member):
    return RUNS / f"nests_sq1km_{set_tag}_m{member:02d}.nc"


def tau_path(set_tag, member):
    return STATS / f"tau_sq1km_{set_tag}_m{member:02d}.npz"


def group_shape(ds, group):
    grp = ds
    for part in group.split("/"):
        grp = grp.groups[part]
    return tuple(len(grp.dimensions[d]) for d in ("x", "y", "z")), grp


def check_shape(what, got, expect_xy, z_range):
    ok = (got[0] == expect_xy and got[1] == expect_xy
          and z_range[0] <= got[2] <= z_range[1])
    if not ok:
        raise RuntimeError(
            f"SHAPE MISMATCH for {what}: got {got}, expected "
            f"({expect_xy}, {expect_xy}, z in {z_range}). Stopping this "
            f"path rather than improvising (campaign spec).")
    print(f"  {what} shape {got} OK", flush=True)


def run_square(set_tag, member, out_nc):
    if not out_nc.exists():
        src = np.load(STATS / "twpice_snap0.npz")
        h_profile = src["h_profile"]
        qt_profile = src["qt_profile"]
        z = src["z_profile"]
        spheroscale = np.full(z.size, SPHEROSCALE_CONSTANT)
        surface_pressure = float(src["surface_pressure"])
        qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
        # Anchored bounds (supp S2 as amended 2026-07-27)
        h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                      float(h_profile.max()) + 1.0)
        h_lower = float(h_profile.min()) - 10.0 * cp
        RUNS.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        simulate(
            h_profile, qt_profile,
            nx=NX, ny=NX, dx=DX, dy=DX,
            outer_scale=OUTER_SCALE,
            spheroscale=spheroscale,
            anisotropy="piecewise_isotropic_below_spheroscale",
            domain_height=DOMAIN_HEIGHT,
            profile_dz=PROFILE_DZ,
            output_path=str(out_nc),
            surface_pressure=surface_pressure,
            seed=2000 + member,
            h_min=h_lower, h_max=h_upper,
            qt_min=0.0, qt_max=qt_sat_surface,
            compress=True,
            device="cpu",
            save_for_refinement=True,
        )
        print(f"square {set_tag} m{member:02d} simulated "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    else:
        print(f"square {set_tag} m{member:02d} exists, skipping", flush=True)
    with netCDF4.Dataset(out_nc) as ds:
        shape = tuple(len(ds.dimensions[d]) for d in ("x", "y", "z"))
        has_T = "T" in ds.variables
    check_shape("square", shape, SQUARE_SHAPE[0],
                (SQUARE_SHAPE[2], SQUARE_SHAPE[2]))
    if not has_T:
        compute_diagnostics(str(out_nc), compress=True)
        print(f"square {set_tag} m{member:02d} diagnostics done", flush=True)


def run_nest(out_nc, which, spec_kwargs, group, parent_group, expect_xy,
             z_range, save_for_refinement):
    with netCDF4.Dataset(out_nc) as ds:
        exists = ("refinements" in ds.groups
                  and group.split("/")[1] in ds.groups["refinements"].groups)
        if exists:
            shape, grp = group_shape(ds, group)
            has_T = "T" in grp.variables
    if not exists:
        t0 = time.perf_counter()
        refine(str(out_nc), parent_group=parent_group, output_group=group,
               device="cpu", compress=True,
               save_for_refinement=save_for_refinement, **spec_kwargs)
        print(f"nest {which} done ({time.perf_counter() - t0:.0f} s)",
              flush=True)
        with netCDF4.Dataset(out_nc) as ds:
            shape, grp = group_shape(ds, group)
            has_T = "T" in grp.variables
    else:
        print(f"nest {which} exists, skipping", flush=True)
    check_shape(f"nest {which}", shape, expect_xy, z_range)
    if not has_T:
        compute_diagnostics(str(out_nc), group=group, compress=True)
        print(f"nest {which} diagnostics done", flush=True)


def extract_tau(out_nc, out_npz):
    if out_npz.exists():
        print(f"{out_npz.name} exists, skipping", flush=True)
        return
    # Exactly make_fractal_square.py's tau computation (vertically
    # integrated optical depth from qc + qi in g/kg); threshold tau = 1
    # is applied downstream, the 2D field itself is stored.
    ds = netCDF4.Dataset(out_nc)
    ds.set_auto_mask(False)
    z = ds.variables["z"][:].astype(np.float64)
    lwc = ds.variables["qc"][:] * 1000.0   # (x, y, z), g/kg
    iwc = ds.variables["qi"][:] * 1000.0
    ds.close()
    tau = cv.vertically_integrated_optical_depth(lwc, z, iwc=iwc)
    if not np.all(np.isfinite(tau)):
        raise RuntimeError(f"non-finite tau for {out_nc.name}")
    cover = float((tau > TAU_THRESHOLD).mean())
    np.savez(out_npz, tau=tau.astype(np.float32), dx=DX,
             tau_threshold=TAU_THRESHOLD)
    print(f"wrote {out_npz.name} (tau>1 cover {cover:.3f})", flush=True)


def copy_keeper_group(src, dst):
    """Copy one nest group keeping only the keeper variables."""
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    names = [n for n in (*KEEP_AUX, *KEEP_VARS) if n in src.variables]
    dims_needed = {d for n in names for d in src.variables[n].dimensions}
    for name, dim in src.dimensions.items():
        if name in dims_needed:
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
    for name in names:
        var = src.variables[name]
        chunks = var.chunking()
        chunks = None if chunks == "contiguous" else chunks
        out = dst.createVariable(
            name, var.dtype, var.dimensions, chunksizes=chunks,
            **compression_kwargs(True, chunks or var.shape))
        out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        if var.ndim == 3:                       # copy big fields in slabs
            nx = var.shape[0]
            step = max(1, nx // 8)
            for i0 in range(0, nx, step):
                out[i0:i0 + step] = var[i0:i0 + step]
        else:
            out[...] = var[...]


def extract_keepers(out_nc, out_keep):
    if out_keep.exists():
        print(f"{out_keep.name} exists, skipping", flush=True)
        return
    tmp = out_keep.with_suffix(".nc.tmp")
    with netCDF4.Dataset(out_nc) as src, \
            netCDF4.Dataset(tmp, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.source_parent = out_nc.name
        for label, group in (("nest_a", NEST_A_GROUP),
                             ("nest_b", NEST_B_GROUP)):
            grp = src
            for part in group.split("/"):
                grp = grp.groups[part]
            copy_keeper_group(grp, dst.createGroup(label))
    tmp.rename(out_keep)
    print(f"wrote {out_keep.name} "
          f"({out_keep.stat().st_size / 1e9:.2f} GB)", flush=True)


def verify_or_scrap(out_nc):
    """Delete a parent .nc that a crash left partially written.

    NetCDF groups cannot be deleted, so any integrity failure means the
    whole parent is scrapped and the member reruns from the square. Each
    completed stage is verified only as far as its structure shows: the
    square must carry its refinement state and all 10 increment classes;
    an existing nest group must carry its fields (and, for nest A, the
    14-class ladder nest B re-weights).
    """
    if not out_nc.exists():
        return
    try:
        with netCDF4.Dataset(out_nc) as ds:
            assert "flux_state" in ds.variables, "square lacks flux_state"
            n_inc = len(ds.groups["class_increments"].groups)
            assert n_inc == 10, f"square increments incomplete ({n_inc}/10)"
            refinements = (ds.groups["refinements"].groups
                           if "refinements" in ds.groups else {})
            for tag, n_ladder in (("r0", 14), ("r1", None)):
                if tag not in refinements:
                    continue
                grp = refinements[tag]
                assert "qt" in grp.variables, f"{tag} lacks qt"
                if n_ladder is not None:
                    n = len(grp.groups["class_increments"].groups)
                    assert n == n_ladder, \
                        f"{tag} increments incomplete ({n}/{n_ladder})"
    except Exception as exc:
        print(f"  {out_nc.name} failed integrity ({exc}); "
              f"deleting and rerunning the member", flush=True)
        out_nc.unlink()


def run_member(set_tag, member):
    out_nc = parent_path(set_tag, member)
    out_keep = keepers_path(set_tag, member)
    out_tau = tau_path(set_tag, member)
    if out_keep.exists() and out_tau.exists() and not out_nc.exists():
        print(f"member {set_tag} m{member:02d} complete, skipping", flush=True)
        return
    verify_or_scrap(out_nc)
    print(f"=== member {set_tag} m{member:02d} ===", flush=True)
    t0 = time.perf_counter()
    c1 = SETS[set_tag]
    _steam_simulate.FLUX_SCALE = (c1 / 1.681) ** (1 / 1.8)
    print(f"  C1 = {c1}, FLUX_SCALE = {_steam_simulate.FLUX_SCALE:.4f}",
          flush=True)

    run_square(set_tag, member, out_nc)
    run_nest(out_nc, "A", NEST_A, NEST_A_GROUP, "/", 1024, (900, 1060), True)
    run_nest(out_nc, "B", NEST_B, NEST_B_GROUP, NEST_A_GROUP, 1024,
             (380, 470), False)

    # Extraction (all products before the parent is deleted)
    extract_tau(out_nc, out_tau)
    stats_npz = STATS / f"steam_sq1km_{set_tag}_m{member:02d}.npz"
    if not stats_npz.exists():
        steam_stats(out_nc, stats_npz)
        print(f"wrote {stats_npz.name}", flush=True)
    do_steam(out_nc)                # -> stats/diag_steam_sq1km_..., skips
    extract_keepers(out_nc, out_keep)

    # Delete the parent (campaign's own file only), then log disk state
    out_nc.unlink()
    print(f"deleted {out_nc.name}", flush=True)
    usage = shutil.disk_usage(RUNS)
    try:
        peak_kb = int(next(l for l in open("/proc/self/status")
                           if l.startswith("VmHWM")).split()[1])
        peak = f", peak RSS {peak_kb / 1024**2:.1f} GiB"
    except Exception:
        peak = ""
    print(f"member {set_tag} m{member:02d} done in "
          f"{time.perf_counter() - t0:.0f} s "
          f"(disk free {usage.free / 1e9:.0f} GB{peak})", flush=True)


def main():
    sets = [sys.argv[1]] if len(sys.argv) > 1 else list(SETS)
    members = [int(sys.argv[2])] if len(sys.argv) > 2 else range(N_MEMBERS)
    for set_tag in sets:
        if set_tag not in SETS:
            raise SystemExit(f"unknown set {set_tag!r} (have {list(SETS)})")
        for member in members:
            run_member(set_tag, member)
    print("campaign portion complete", flush=True)


if __name__ == "__main__":
    main()
