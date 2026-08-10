#!/usr/bin/env python3
"""A small, finely resolved STEAM domain for visualization.

One run per flux amplitude per spheroscale, each a parent carrying a centered
nest, with anchored bounds and H_h and lambda at package defaults. This is a
rendering target, not an analysis product -- the point is a domain fine enough
to look at closely, and the nest is where that happens.

SPHEROSCALE is the axis these runs exist to show, and it drags two other knobs
with it, which is why each case carries its own outer scale rather than
sharing one.

The vertical outer scale k_z,L = l_s (L/l_s)^{H_z} grows with the spheroscale,
and simulate() refuses any config whose k_z,L reaches the domain top. Raising
the top instead of shortening L is not the cheap way out: dz = k_z(2 dx)/2 is
set by dx and by which side of the spheroscale the finest class falls on, so a
taller domain buys room in levels the cascade then has to carry. The coarser
spheroscale therefore runs the shorter L and the shallower class ladder, and
the two pictures differ in cascade depth as well as in spheroscale. That is the
price of the axis rather than something the runs hide.

RUN_NEST switches the nest on or off. With it off the parent is run without
refinement state as well, so the file cannot seed a nest later -- turning the
flag back on means rerunning the parent, and run_nest refuses a parent written
without it rather than failing deeper in refine().

STRIPPING. Each case runs into a working file and ends as a keeper carrying qc
and qi alone, for the parent and for the nest, on the square campaign's
precedent. The working file is tens of GB, most of it the cascade state and the
per-class increments the nest is cut from, and a picture needs none of it; it
is deleted once the keeper is written. As with the square campaign's keeper,
what that costs is the ability to cut a further nest later -- refine() reads
the increments, and they stay behind with the working file.

Restartable at stage granularity: a keeper with no working file marks a case
complete, and while the working file exists the parent and the nest are each
skipped if already present. A keeper that disagrees with the config now in
force is refused rather than counted complete, so moving a knob mid-campaign
stops the run instead of leaving a stale picture on disk under a name that
says otherwise.

Runs are serial and the parent's working set is close enough to the machine's
limit that they should not be run alongside anything large.

Usage: python run_steam_simulations.py [SET [SPHERO]]
  no args         -> every amplitude at every spheroscale
  c005            -> that amplitude, both spheroscales
  c005 s0010      -> that single run
"""

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

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # small-domain/
REPO = BASE.parent
RUNS = REPO / "runs" / "small-domain"
PROFILES = REPO / "runs" / "input_profiles"

PROFILE_HOST = "cm1"      
SETS = {"c002": 0.02, "c005": 0.05, "c017": 0.17}

# spheroscale tag -> (l_s [m], outer scale [m]). The tag is the spheroscale in
# metres, zero-padded to four digits. Each case carries its own L because a
# single L cannot serve both; see the module docstring. Order is free now that
# every run shares SEED -- nothing is derived from position in this dict.
SPHEROSCALES = {
    "s0010": (10.0, 40960.0),
    "s1000": (1000.0, 10240.0),
}

NX, NY = 2048, 512
DX = 20.0
# NX, NY = 2048/4, 512/4
# DX = 80.0
DOMAIN_HEIGHT = 5500.0
PROFILE_DZ = 50.0
DEVICE = "cuda"

SEED = 7002


RUN_NEST = True

# The nest, in its OWN cells at its own spacing -- set these three and nothing
# else. The parent-cell window refine() wants is derived and centered, so the
# nest cannot be placed off the edge of the parent by hand.
NEST_NX, NEST_NY = 512, 256        # nest cells, so 2:1 at NEST_DX below
NEST_DX = 5.0                      # nest spacing [m]
NEST_GROUP = "refinements/r0"


def nest_window(parent_cells, nest_cells, axis):
    """Centered parent-cell [start, stop) spanning nest_cells of the nest.

    refine() cuts on parent cells, so a nest extent that is not a whole number
    of them has no window to ask for; that and a nest wider than the parent are
    both config errors rather than something to round into shape.
    """
    span = nest_cells * NEST_DX / DX
    if abs(span - round(span)) > 1e-9:
        raise SystemExit(
            f"nest {axis} extent {nest_cells * NEST_DX:.4g} m "
            f"({nest_cells} x {NEST_DX:g} m) is {span:g} parent cells at "
            f"dx = {DX:g} m, not a whole number of them")
    span = int(round(span))
    if span > parent_cells:
        raise SystemExit(
            f"nest {axis} extent {nest_cells * NEST_DX:.4g} m is {span} "
            f"parent cells, wider than the parent's {parent_cells}")
    start = (parent_cells - span) // 2
    return start, start + span


NEST_X_START, NEST_X_STOP = nest_window(NX, NEST_NX, "x")
NEST_Y_START, NEST_Y_STOP = nest_window(NY, NEST_NY, "y")
NEST = dict(x_start=NEST_X_START, x_stop=NEST_X_STOP,
            y_start=NEST_Y_START, y_stop=NEST_Y_STOP,
            dx=NEST_DX, dy=NEST_DX)

# What the keeper carries, as in fractal-analysis/. The aux names are copied
# where present, so a group without one (p_bottom, on a nest that starts at the
# surface) is not an error.
KEEP_VARS = ("qc", "qi")
KEEP_AUX = ("x", "y", "z", "z_profile", "dz", "spheroscale", "p_bottom")
KEEP = (*KEEP_AUX, *KEEP_VARS)


def working_path(set_tag, sphero_tag):
    return RUNS / f"work_small_{set_tag}_{sphero_tag}.nc"


def keeper_path(set_tag, sphero_tag):
    return RUNS / f"small_{set_tag}_{sphero_tag}.nc"


def config_spec(set_tag, sphero_tag):
    """The run attributes a keeper must match to be this case's picture.

    Everything here is recorded by simulate() itself except spheroscale, which
    it writes as a profile variable rather than an attribute; write_keeper puts
    the constant on the root so the comparison stays a scalar one.
    """
    sphero_m, outer_scale = SPHEROSCALES[sphero_tag]
    return {"nx": NX, "ny": NY, "dx": DX, "outer_scale": outer_scale,
            "domain_height": DOMAIN_HEIGHT, "seed": SEED,
            "flux_noise_scale": SETS[set_tag],
            "spheroscale_constant": sphero_m}


def spec_mismatches(ds, set_tag, sphero_tag):
    """Attributes of an existing keeper that disagree with the config."""
    bad = {}
    for attr, want in config_spec(set_tag, sphero_tag).items():
        got = getattr(ds, attr, None)
        if got is None or abs(float(got) - want) > 1e-9 * max(1.0, abs(want)):
            bad[attr] = ("missing" if got is None else f"{float(got):g}", want)
    for attr, want in (("profile_host", PROFILE_HOST),
                       ("run_nest", int(RUN_NEST))):
        got = getattr(ds, attr, None)
        if got is None or type(want)(got) != want:
            bad[attr] = ("missing" if got is None else str(got), want)
    return bad


def run_nest(out_nc):
    """Cut the centered nest, if the file does not already carry it."""
    with netCDF4.Dataset(out_nc) as ds:
        exists = ("refinements" in ds.groups
                  and "r0" in ds.groups["refinements"].groups)
        refinable = "flux_state" in ds.variables
    if exists:
        print(f"  nest exists in {out_nc.name}, skipping", flush=True)
        return
    if not refinable:
        raise RuntimeError(
            f"{out_nc.name} carries no refinement state, so it was written "
            f"with RUN_NEST = False. Delete it and rerun the parent rather "
            f"than reporting a nest this file cannot produce.")
    print(f"  nest {NEST_NX} x {NEST_NY} at dx = {NEST_DX:g} m "
          f"({NEST_NX * NEST_DX / 1000:.2f} x {NEST_NY * NEST_DX / 1000:.2f} "
          f"km), parent cells x {NEST_X_START}:{NEST_X_STOP}, "
          f"y {NEST_Y_START}:{NEST_Y_STOP}", flush=True)
    t0 = time.perf_counter()
    refine(str(out_nc), parent_group="/", output_group=NEST_GROUP,
           device=DEVICE, compress=True, **NEST)
    compute_diagnostics(str(out_nc), group=NEST_GROUP, compress=True, device=DEVICE)
    print(f"  nest done in {time.perf_counter() - t0:.0f} s", flush=True)


def run_parent(set_tag, sphero_tag, out_nc):
    if out_nc.exists():
        print(f"{out_nc.name} exists, skipping the parent", flush=True)
        return

    sphero_m, outer_scale = SPHEROSCALES[sphero_tag]
    src = np.load(PROFILES / f"{PROFILE_HOST}.npz")
    h_profile = src["h_profile"]
    qt_profile = src["qt_profile"]
    spheroscale = np.full(src["z_profile"].size, sphero_m)
    surface_pressure = float(src["surface_pressure"])
    qt_sat_surface = float(_saturation_mixing_ratio(300.0, surface_pressure))
    # Anchored bounds, as the square campaign sets them.
    h_upper = max(cp * 300.0 + Lv * qt_sat_surface,
                  float(h_profile.max()))
    h_lower = float(h_profile.min()) - 10.0 * cp

    _steam_simulate.FLUX_SCALE = SETS[set_tag]
    print(f"=== {set_tag} {sphero_tag} === {NX} x {NY} at dx = {DX:.0f} m "
          f"({NX * DX / 1000:.2f} x {NY * DX / 1000:.2f} km), "
          f"top {DOMAIN_HEIGHT / 1000:.0f} km, L = {outer_scale / 1000:.2f} km, "
          f"l_s = {sphero_m:.0f} m, c = {SETS[set_tag]}, "
          f"nest {'on' if RUN_NEST else 'off'}", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    simulate(
        h_profile, qt_profile,
        nx=NX, ny=NY, dx=DX, dy=DX,
        outer_scale=outer_scale,
        spheroscale=spheroscale,
        anisotropy="piecewise_isotropic_below_spheroscale",
        domain_height=DOMAIN_HEIGHT,
        profile_dz=PROFILE_DZ,
        output_path=str(out_nc),
        surface_pressure=surface_pressure,
        seed=SEED,
        h_min=h_lower, h_max=h_upper,
        qt_min=0.0, qt_max=qt_sat_surface,
        compress=True,
        device=DEVICE,
        save_for_refinement=RUN_NEST,
    )
    compute_diagnostics(str(out_nc), compress=True, device=DEVICE)
    print(f"{out_nc.name} parent done in {time.perf_counter() - t0:.0f} s "
          f"({out_nc.stat().st_size / 1e9:.1f} GB)", flush=True)


def copy_group(src, dst):
    """Copy one group's KEEP variables, with its attributes and dimensions."""
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    names = [n for n in KEEP if n in src.variables]
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
            step = max(1, var.shape[0] // 8)
            for i0 in range(0, var.shape[0], step):
                out[i0:i0 + step] = var[i0:i0 + step]
        else:
            out[...] = var[...]


def write_keeper(out_nc, out_keep, sphero_tag):
    if out_keep.exists():
        print(f"{out_keep.name} exists, skipping", flush=True)
        return
    tmp = out_keep.with_suffix(".nc.tmp")
    with netCDF4.Dataset(out_nc) as src, netCDF4.Dataset(tmp, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.source_parent = out_nc.name
        # Recorded so a keeper made under other settings is caught rather than
        # counted complete; see config_spec.
        dst.profile_host = PROFILE_HOST
        dst.run_nest = int(RUN_NEST)
        dst.spheroscale_constant = SPHEROSCALES[sphero_tag][0]
        dst.kept_variables = " ".join(KEEP_VARS)
        copy_group(src, dst.createGroup("parent"))
        if RUN_NEST:
            grp = src
            for part in NEST_GROUP.split("/"):
                grp = grp.groups[part]
            copy_group(grp, dst.createGroup("nest"))
    tmp.rename(out_keep)
    print(f"wrote {out_keep.name} "
          f"({out_keep.stat().st_size / 1e9:.2f} GB)", flush=True)


def complete(out_keep, set_tag, sphero_tag):
    """True if this keeper is this case's picture; fatal if it is another's."""
    with netCDF4.Dataset(out_keep) as ds:
        if not getattr(ds, "kept_variables", ""):
            raise RuntimeError(
                f"{out_keep.name} carries no kept_variables attribute, so it "
                f"is a full pre-stripper run under the keeper's name. Move or "
                f"delete it rather than passing it off as a stripped keeper.")
        bad = spec_mismatches(ds, set_tag, sphero_tag)
    if bad:
        detail = ", ".join(f"{a} = {got} (this run: {want})"
                           for a, (got, want) in bad.items())
        raise RuntimeError(
            f"{out_keep.name} was made under a different config: {detail}. "
            f"Move or delete it rather than leaving a stale picture on disk.")
    return True


def run_case(set_tag, sphero_tag):
    out_nc = working_path(set_tag, sphero_tag)
    out_keep = keeper_path(set_tag, sphero_tag)
    if out_keep.exists() and not out_nc.exists():
        complete(out_keep, set_tag, sphero_tag)
        print(f"case {set_tag} {sphero_tag} complete, skipping", flush=True)
        return
    run_parent(set_tag, sphero_tag, out_nc)
    if RUN_NEST:
        run_nest(out_nc)
    write_keeper(out_nc, out_keep, sphero_tag)
    out_nc.unlink()
    print(f"deleted {out_nc.name}", flush=True)


def main():
    args = sys.argv[1:]
    tags = [args[0]] if args else list(SETS)
    spheros = [args[1]] if len(args) > 1 else list(SPHEROSCALES)
    for tag in tags:
        if tag not in SETS:
            raise SystemExit(f"unknown set {tag!r} (have {list(SETS)})")
        for sphero_tag in spheros:
            if sphero_tag not in SPHEROSCALES:
                raise SystemExit(f"unknown spheroscale {sphero_tag!r} "
                                 f"(have {list(SPHEROSCALES)})")
            run_case(tag, sphero_tag)
    print("small-domain generation complete", flush=True)


if __name__ == "__main__":
    main()
