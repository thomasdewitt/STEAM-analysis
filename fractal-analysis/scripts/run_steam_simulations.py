#!/usr/bin/env python3
"""Paper production campaign: square ensembles with two nests per member.

Runs the simulations and nothing else. One file per member lands in
runs/square/, carrying every parent variable but only three levels of each 3D
one -- those nearest PARENT_LEVELS -- plus the parent's 2D vertically
integrated optical depth, taken over the full column before the thinning, and
the two nests stripped to qc and qi (the condensate the fractal analysis
needs), at full depth. No statistics, no other directories written.

Member config (2026-08-04 rulings): ukmo_ra1t profile (the least-cloudy of the
comparison set; was TWP-ICE until its m00 delivered tau>1 cover 0.95), 2048 x
2048 at dx = 1 km (2048 km square), outer scale L = OUTER_SCALE below (the
convention has been the longest domain dimension, whole or halved), constant
10 m spheroscale, H_h and lambda at package defaults, anchored bounds (supp
S2 as amended 2026-07-27), domain top 20 km, CUDA.

A keeper on disk is only reused if the run attributes it recorded still match
that config -- see campaign_spec -- so moving a knob mid-campaign stops the
run rather than pooling two experiments.

RUN_NESTS switches the two nests on or off together. With it off, the square is
run without refinement state as well, and the keeper file holds the parent
group alone.

Per member, strictly serially:
  1. square (save_for_refinement=RUN_NESTS) + compute_diagnostics
  2. nest A: centered 32x32 km, full depth, dx = 62.5 m
     (parent cells 1008:1040), save_for_refinement=True, + diagnostics.
     (Was 64 km at 1024^2; halved 2026-08-04 after the sweep was
     OOM-killed at 53.5 GB in-process.)
  3. nest B: refines nest A, centered 8x8 km, z = 1-5 km,
     dx = 7.8125 m (nest-A cells 192:320), + diagnostics.
  4. write the keeper file -- the parent thinned to PARENT_LEVELS plus its tau
     (cloudyview, unthresholded, full column), qc/qi for each nest -- then
     delete the working .nc

The working .nc is deleted because it is ~50 GB per member with its refinement
state; ten members would be half a terabyte. The keeper file is the product.

Vertical thinning (2026-08-07). The parent's eleven 3D fields are stored at
three levels only -- the grid levels nearest 5, 10 and 15 km -- because the
full column would not fit the campaign on disk. Measured on the one full-depth
member: 2048^2 x 211 float32 is 3.30 GiB raw per field and the nine
non-condensate fields compress only 1.16-1.8x, so the keeper came to 21.6 GB,
and twenty of them 431 GB against 340 GB free. Three levels measure 0.38 GB
per member, 7.6 GB over twenty. Nothing downstream reads a parent 3D field --
compute_fractal_metrics.py takes tau and dx and nothing else -- and tau is
still integrated over the whole column, in the working file, before the
thinning happens.

netCDF cannot delete a variable or a level in place, so the thinning is done
where the keeper is written from the working file rather than as a later pass
over the keeper; that is the same point in the run and it avoids writing
20 GB per member only to discard it.

Thinning applies to the parent group alone. The nests keep their full depth,
and the parent's own z, dz and spheroscale are cut to the same three levels so
the group stays self-describing. C_h_k and C_qt_k are on nz_k_max, a separate
dimension of the original length, and stay whole -- they are the scale-class
amplitude ladder, not a field, and they cost nothing.

What the keeper drops besides levels is the class_increments groups, 12.8 GB
across the ten classes, three quarters of which is the finest class alone.
That is why the keeper cannot seed a new nest, since refine() reads them;
cutting a further nest means rerunning the square.

Restartable at stage granularity: a keeper file with no working .nc marks a
member complete; while the working file exists, the square, each nest group and
each diagnostics pass is skipped if already present.

Input profiles come from runs/input_profiles/, built by make_input_profiles.py
at the repo root.

Usage: python run_steam_simulations.py [SET [MEMBER]]
  no args         -> full sweep (all sets, N_MEMBERS members each)
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

# cloudyview is not installed in this venv; load the one module by path.
_spec = importlib.util.spec_from_file_location(
    "cv_optical_depth",
    Path.home() / "code-and-data/cloudyview/cloudyview/optical_depth.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # fractal-analysis/
REPO = BASE.parent
RUNS = REPO / "runs" / "square"
PROFILES = REPO / "runs" / "input_profiles"

PROFILE_HOST = "ukmo_ra1t"

SETS = {  
    "c002": 0.02,
    "c005": 0.05,
    "c017": 0.17,
}
N_MEMBERS = 10
NX = 2048
DX = 1000.0
OUTER_SCALE = 2048e3
SPHEROSCALE_CONSTANT = 10
DOMAIN_HEIGHT = 20000.0
PROFILE_DZ = 50.0

DEVICE = 'cuda'
RUN_NESTS = False

# Nest A: centered 32x32 km, full depth, target dx = 62.5 m.
NEST_A = dict(x_start=1008, x_stop=1040, y_start=1008, y_stop=1040,
              dx=62.5, dy=62.5)
NEST_A_GROUP = "refinements/r0"
# Nest B: refines nest A, centered 8x8 km, z = 1-5 km (elevated, so it stores
# p_bottom from the parent pressure), dx = 7.8125 m -- one octave deeper,
# finest class 15.625 m approaching the 10 m spheroscale.
NEST_B = dict(x_start=192, x_stop=320, y_start=192, y_stop=320,
              dx=7.8125, dy=7.8125, z_min=1000.0, z_max=5000.0)
NEST_B_DX = 7.8125
NEST_B_GROUP = "refinements/r1"

# Every parent variable is copied, but each 3D one only at the levels nearest
# these heights (2026-08-07; see the vertical-thinning note in the module
# docstring). The nests are still stripped to condensate, at full depth: they
# exist to be looked at, not analyzed further.
#
# This does NOT make the keeper refinable, and did not before the thinning
# either. refine() also wants the class_increments groups, which stay behind
# with the working file. Cutting a new nest still means rerunning the square.
PARENT_LEVELS = (5000.0, 10000.0, 15000.0)

KEEP_VARS = ("qc", "qi")
KEEP_AUX = ("x", "y", "z", "z_profile", "dz", "spheroscale", "p_bottom")
NEST_KEEP = (*KEEP_AUX, *KEEP_VARS)

# Reported only, as a sanity signal while the campaign runs; the stored tau
# field is unthresholded, so any threshold can be applied downstream.
TAU_THRESHOLD = 1.0


def campaign_spec(set_tag):
    """The run attributes a keeper must match to be a member of this ensemble.

    RUN_NESTS and PARENT_LEVELS are checked separately because they change the
    product's contents; these change the physics behind it. A keeper written
    before one of these knobs moved is a different experiment, and since
    compute_fractal_metrics.py pools everything matching its PATTERN into a
    single regression, pooling the two would be silent.
    """
    return {"dx": DX, "nx": NX, "ny": NX, "outer_scale": OUTER_SCALE,
            "flux_noise_scale": SETS[set_tag]}


def spec_mismatches(ds, set_tag):
    """Attributes of an existing keeper that disagree with the config."""
    bad = {}
    for attr, want in campaign_spec(set_tag).items():
        got = getattr(ds, attr, None)
        if got is None or abs(float(got) - want) > 1e-9 * max(1.0, abs(want)):
            bad[attr] = ("missing" if got is None else f"{float(got):g}", want)
    return bad


def working_path(set_tag, member):
    return RUNS / f"work_sq1km_{set_tag}_m{member:02d}.nc"


def keeper_path(set_tag, member):
    return RUNS / f"sq1km_{set_tag}_m{member:02d}.nc"


def group_shape(ds, group):
    grp = ds
    for part in group.split("/"):
        grp = grp.groups[part]
    return tuple(len(grp.dimensions[d]) for d in ("x", "y", "z")), grp




def run_square(set_tag, member, out_nc):
    if not out_nc.exists():
        src = np.load(PROFILES / f"{PROFILE_HOST}.npz")
        h_profile = src["h_profile"]
        qt_profile = src["qt_profile"]
        z = src["z_profile"]
        spheroscale = np.full(z.size, SPHEROSCALE_CONSTANT)
        surface_pressure = float(src["surface_pressure"])
        qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
        # Anchored bounds (supp S2 as amended 2026-07-27)
        h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                      float(h_profile.max()))
        h_lower = float(h_profile.min()) - 10.0 * cp
        RUNS.mkdir(parents=True, exist_ok=True)
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
            device=DEVICE,
            save_for_refinement=RUN_NESTS,
        )
        print(f"square {set_tag} m{member:02d} simulated "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    else:
        print(f"square {set_tag} m{member:02d} exists, skipping", flush=True)
    with netCDF4.Dataset(out_nc) as ds:
        shape = tuple(len(ds.dimensions[d]) for d in ("x", "y", "z"))
        has_T = "T" in ds.variables
    if not has_T:
        compute_diagnostics(str(out_nc), compress=True, device=DEVICE)
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
               device=DEVICE, compress=True,
               save_for_refinement=save_for_refinement, **spec_kwargs)
        print(f"nest {which} done ({time.perf_counter() - t0:.0f} s)",
              flush=True)
        with netCDF4.Dataset(out_nc) as ds:
            shape, grp = group_shape(ds, group)
            has_T = "T" in grp.variables
    else:
        print(f"nest {which} exists, skipping", flush=True)
    if not has_T:
        compute_diagnostics(str(out_nc), group=group, compress=True, device=DEVICE)
        print(f"nest {which} diagnostics done", flush=True)


def level_indices(z):
    """Grid indices nearest PARENT_LEVELS, sorted ascending.

    Fatal if two targets land on the same level, which would mean the grid is
    far coarser than the campaign spec and the thinning is not doing what the
    name says.
    """
    z = np.asarray(z, dtype=np.float64)
    idx = sorted({int(np.abs(z - target).argmin()) for target in PARENT_LEVELS})
    if len(idx) != len(PARENT_LEVELS):
        raise RuntimeError(
            f"the {len(PARENT_LEVELS)} targets {PARENT_LEVELS} collapsed onto "
            f"{len(idx)} distinct levels of a {z.size}-level grid. Stopping "
            f"rather than silently storing fewer levels than asked for.")
    return np.array(idx, dtype=int)


def copy_keeper_group(src, dst, keep=None, z_index=None):
    """Copy one group; keep=None copies every variable, else only `keep`.

    z_index, when given, is an ascending index array into the group's z
    dimension: that dimension is created at its length and every variable
    using it is thinned to those levels. Variables on a different dimension
    that happens to be the same length (nz_k_max) are untouched.
    """
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    names = (list(src.variables) if keep is None
             else [n for n in keep if n in src.variables])
    dims_needed = {d for n in names for d in src.variables[n].dimensions}
    for name, dim in src.dimensions.items():
        if name not in dims_needed:
            continue
        if name == "z" and z_index is not None:
            dst.createDimension(name, len(z_index))
        else:
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
    for name in names:
        var = src.variables[name]
        thin = z_index is not None and "z" in var.dimensions
        zaxis = var.dimensions.index("z") if thin else None
        shape = tuple(len(dst.dimensions[d]) for d in var.dimensions)
        chunks = var.chunking()
        # Clamp rather than rescale: a thinned field's stored z is shorter
        # than the chunk the working file used.
        chunks = (None if chunks == "contiguous"
                  else [min(c, n) for c, n in zip(chunks, shape)])
        out = dst.createVariable(
            name, var.dtype, var.dimensions, chunksizes=chunks,
            **compression_kwargs(True, chunks or shape))
        out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        if var.ndim == 3:                       # copy big fields in slabs
            nx = var.shape[0]
            step = max(1, nx // 8)
            for i0 in range(0, nx, step):
                slab = var[i0:i0 + step]
                out[i0:i0 + step] = (np.take(slab, z_index, axis=zaxis)
                                     if thin else slab)
        else:
            data = var[...]
            out[...] = np.take(data, z_index, axis=zaxis) if thin else data


def write_parent_tau(src, dst_group):
    """Store the parent square's 2D vertically integrated optical depth.

    cloudyview's SAM relationships, from qc + qi in g/kg -- the same
    computation the fractal analysis used to do downstream, done once here so
    the campaign product carries it. Written for the parent regardless of
    RUN_NESTS. The whole condensate volume is held in memory (~3.5 GB per
    field at 2048^2 x 211) and handed to cloudyview whole.
    """
    z = src.variables["z"][:].astype(np.float64)
    lwc = src.variables["qc"][:] * 1000.0        # (x, y, z), g/kg
    iwc = src.variables["qi"][:] * 1000.0
    tau = cv.vertically_integrated_optical_depth(lwc, z, iwc=iwc)
    del lwc, iwc
    if not np.all(np.isfinite(tau)):
        raise RuntimeError("non-finite tau for the parent square")
    var = dst_group.createVariable(
        "tau", "f4", ("x", "y"),
        **compression_kwargs(True, tau.shape))
    var[...] = tau.astype(np.float32)
    var.long_name = "vertically integrated optical depth (qc + qi)"
    var.units = "1"
    var.source = "cloudyview.optical_depth.vertically_integrated_optical_depth"
    cover = float((tau > TAU_THRESHOLD).mean())
    print(f"  parent tau written (tau>{TAU_THRESHOLD:g} cover {cover:.3f})",
          flush=True)


def write_keeper(out_nc, out_keep):
    if out_keep.exists():
        print(f"{out_keep.name} exists, skipping", flush=True)
        return
    tmp = out_keep.with_suffix(".nc.tmp")
    with netCDF4.Dataset(out_nc) as src, netCDF4.Dataset(tmp, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.source_parent = out_nc.name
        # Recorded so a rerun under the other setting is caught rather than
        # silently accepted as a complete member (see run_member).
        dst.run_nests = int(RUN_NESTS)
        z_index = level_indices(src.variables["z"][:])
        # Recorded for the same reason as run_nests: so a keeper written under
        # a different vertical spec is caught rather than silently pooled.
        dst.parent_z_levels = len(z_index)
        parent = dst.createGroup("parent")
        copy_keeper_group(src, parent, z_index=z_index)
        z_kept = np.asarray(src.variables["z"][:], dtype=np.float64)[z_index]
        parent.level_targets = np.array(PARENT_LEVELS, dtype="f8")
        parent.level_indices = z_index.astype("i4")
        parent.source_nz = len(src.dimensions["z"])
        parent.level_note = (
            "3D fields stored only at the levels nearest level_targets; "
            "level_indices are into the original source_nz grid. tau is the "
            "full-column integral, taken before the thinning.")
        print("  parent thinned to z = "
              + ", ".join(f"{v:.1f} m (k={k})" for v, k in zip(z_kept, z_index))
              + f" of {parent.source_nz}", flush=True)
        write_parent_tau(src, parent)
        nests = (("nest_a", NEST_A_GROUP), ("nest_b", NEST_B_GROUP)) \
            if RUN_NESTS else ()
        for label, group in nests:
            grp = src
            for part in group.split("/"):
                grp = grp.groups[part]
            copy_keeper_group(grp, dst.createGroup(label), keep=NEST_KEEP)
    tmp.rename(out_keep)
    print(f"wrote {out_keep.name} "
          f"({out_keep.stat().st_size / 1e9:.2f} GB)", flush=True)


def verify_or_scrap(out_nc):
    """Delete a working .nc that a crash left partially written.

    NetCDF groups cannot be deleted, so any integrity failure means the whole
    working file is scrapped and the member reruns from the square. Each
    completed stage is verified only as far as its structure shows: the square
    must carry its refinement state and all 10 increment classes; an existing
    nest group must carry its fields (and, for nest A, the 14-class ladder
    nest B re-weights).

    Under RUN_NESTS = False the square carries no refinement state, so only
    its presence is checked -- and a leftover file that does carry nests is
    scrapped, since it was made under the other setting.
    """
    if not out_nc.exists():
        return
    try:
        with netCDF4.Dataset(out_nc) as ds:
            if not RUN_NESTS:
                assert "refinements" not in ds.groups, \
                    "working file has nests but RUN_NESTS is False"
                return
            assert "flux_state" in ds.variables, "square lacks flux_state"
            n_inc = len(ds.groups["class_increments"].groups)
            assert n_inc == 10, f"square increments incomplete ({n_inc}/10)"
            refinements = (ds.groups["refinements"].groups
                           if "refinements" in ds.groups else {})
            assert set(refinements) <= {"r0", "r1"}, \
                f"unexpected refinement groups {sorted(refinements)}"
            for tag, spec_nx, spec_dx, n_ladder in (
                    ("r0", 512, 62.5, 14), ("r1", 1024, NEST_B_DX, None)):
                if tag not in refinements:
                    continue
                grp = refinements[tag]
                assert "qt" in grp.variables, f"{tag} lacks qt"
                nx = len(grp.dimensions["x"])
                dx = float(grp.getncattr("dx"))
                assert nx == spec_nx and abs(dx - spec_dx) < 1e-6, \
                    (f"{tag} is an old-spec nest (nx={nx}, dx={dx}; "
                     f"spec {spec_nx}, {spec_dx})")
                if n_ladder is not None:
                    n = len(grp.groups["class_increments"].groups)
                    assert n == n_ladder, \
                        f"{tag} increments incomplete ({n}/{n_ladder})"
    except Exception as exc:
        print(f"  {out_nc.name} failed integrity ({exc}); "
              f"deleting and rerunning the member", flush=True)
        out_nc.unlink()


def run_member(set_tag, member):
    out_nc = working_path(set_tag, member)
    out_keep = keeper_path(set_tag, member)
    if out_keep.exists() and not out_nc.exists():
        with netCDF4.Dataset(out_keep) as ds:
            made_with = bool(getattr(ds, "run_nests", 1))
            n_levels = int(getattr(ds, "parent_z_levels", 0))
            bad = spec_mismatches(ds, set_tag)
        if bad:
            detail = ", ".join(f"{a} = {got} (this run: {want:g})"
                               for a, (got, want) in bad.items())
            raise RuntimeError(
                f"{out_keep.name} was made under a different campaign spec: "
                f"{detail}. Move or delete it rather than pooling two "
                f"experiments in runs/square/.")
        if made_with != RUN_NESTS:
            raise RuntimeError(
                f"{out_keep.name} was made with RUN_NESTS={made_with}, but "
                f"this run has RUN_NESTS={RUN_NESTS}. Move or delete it "
                f"rather than mixing the two products in runs/square/.")
        if n_levels != len(PARENT_LEVELS):
            raise RuntimeError(
                f"{out_keep.name} carries {n_levels or 'all'} parent levels, "
                f"but this run stores {len(PARENT_LEVELS)} "
                f"(PARENT_LEVELS={PARENT_LEVELS}). Move or delete it rather "
                f"than mixing the two products in runs/square/. A pre-"
                f"2026-08-07 keeper is full-depth and reports 'all'.")
        print(f"member {set_tag} m{member:02d} complete, skipping", flush=True)
        return
    verify_or_scrap(out_nc)
    print(f"=== member {set_tag} m{member:02d} ===", flush=True)
    t0 = time.perf_counter()
    _steam_simulate.FLUX_SCALE = SETS[set_tag]
    print(f"  FLUX_SCALE = {_steam_simulate.FLUX_SCALE:.5f}", flush=True)

    run_square(set_tag, member, out_nc)
    if RUN_NESTS:
        run_nest(out_nc, "A", NEST_A, NEST_A_GROUP, "/", 512, (900, 1060), True)
        run_nest(out_nc, "B", NEST_B, NEST_B_GROUP, NEST_A_GROUP, 1024,
                 (550, 750), False)
    write_keeper(out_nc, out_keep)

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
