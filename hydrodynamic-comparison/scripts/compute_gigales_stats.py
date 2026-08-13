#!/usr/bin/env python3
"""Matched gigaLES statistics: per-level standard deviations, cloud
fraction, and the single-level fields the PDFs are drawn from.

Reads each matched LES snapshot and the STEAM realization ensemble
generated from its profile by run_gigales_simulations.py, brings them onto
a common resolution, and writes everything plot_gigales.py needs into
gigales_stats.npz, keyed by case.

T JOINED THE STANDARD DEVIATIONS ON 2026-08-10, computed exactly like h,
qt, qc and qi -- same coarsening, same matching, same pooling over members.
The PDFs are unchanged, still the original four variables. Pressure was added
the same day and dropped again; see common.py for why.

On the STEAM side T is read straight off each member, where
compute_diagnostics wrote it from the same column solve as qc and qi.
Members generated before run_gigales_simulations.py's KEEP list grew do not
carry it and cannot be backfilled -- the working file they were stripped
from is gone -- so this script refuses such an ensemble by name rather than
quietly dropping a panel.

On the HOST side both cases archive a temperature directly:

  twpice   TABS, copied off Expansion on 2026-08-10 -- the archived
           temperature, on the same grid and the same timestep as the MSE and
           mixing-ratio files already here. It is read rather than recovered
           by inverting MSE, so T is the host's own field here exactly as it
           is for every RCEMIP host, and h stays what SAM archived rather
           than becoming a function of the T beside it.
  gate     TABS directly.

MATCHING, in two steps (2026-08-10).

FIRST the host is coarsened in 2 x 2 x 2 blocks -- vertically as well as
horizontally -- before any one-point statistic is taken, to keep the
standard deviations off its own grid scale where numerical artifacts live
(main.tex, one-point statistics). Horizontally that is 100 m -> 200 m,
which is STEAM's spacing because the runs are generated at twice the host's
dx; the factor is computed from the run's own dx attribute rather than
written down, so it follows the domain. Vertically it halves the level
count, 255 -> 127 for TWPICE and 256 -> 128 for GATE, with TWPICE's odd
topmost level dropped.

THEN STEAM is matched to THAT grid by the standing per-level rule: at each
(coarsened) host level, whichever field is finer is block-averaged by the
nearest integer factor bringing the two spacings closest together. The host
is now coarser everywhere, so all the averaging falls on STEAM, over 3
levels near the surface rising to 7 aloft -- against 1 to 4 before the host
was coarsened vertically. The factor used at every level is saved so the
figure can state it.

Comparison levels are the coarsened host's own, clipped to STEAM's top:
112 of them, where the native grid gave 223.

Cloud fraction is the fraction of cells holding at least 0.01 g/kg of
condensate, thresholded AFTER coarsening -- the coarsened cell value is what
the resolved field says is there.

SAM fields. h is c_p times the archived MSE, which SAM stores in K and
non-frozen; nothing is reconstructed. qt is qv + qc + qi, excluding the
precipitating species (QR/QS/QG are archived but deliberately left out, the
same convention make_input_profiles.py uses for the driving profiles). There
is no QT field. TABS there is, for both cases, since 2026-08-10 -- TWPICE's
was generated onto Expansion that morning and copied in.

Note that SAM's MSE and this comparison's h do not use the same latent heat:
SAM's is 2.5104e6 against steam's 2.5e6, worth +0.19 K equivalent at the
surface and decaying with qv. h is carried through as c_p * MSE regardless,
so the host's own energy variable is what is reported; T is the archived
TABS. Neither is derived from the other.

Comparison levels are the host's own, clipped to STEAM's 20 km top.

Usage: python compute_gigales_stats.py [case ...]   (default: twpice gate)
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

from common import (VARS, STD_VARS, CLOUD_KGKG, PDF_LEVELS, coarsen_factor,
                    coarsen_xyz, match_factors, reduce_source)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # hydrodynamic-comparison/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
RUNS = REPO / "runs" / "hydro"
HOST = REPO / "data" / "twpice"
OUT = OUTPUT / "gigales_stats.npz"

sys.path.insert(0, str(REPO))
from make_input_profiles import (_twpice_field, read_var,   # noqa: E402
                                 liquid_fraction)

SETS = ("c002", "c005", "c017")
N_MEMBERS = 5
CASES = ("twpice", "gate")
HOST_DX = 100.0                # both SAM cases, native
SNAPSHOT = "0000003450"
GATE_FILE = "GATE_IDEAL_S_2048x2048x256_100m_2s_2048_0000041400.nc"   # 23 h


def twpice_fields(xy_coarsen):
    """TWPICE h, qt, T, qc, qi on STEAM's grid, each (nx, ny, nz).

    Each 2048^2 x 255 field is 4.3 GB, so they are read and coarsened one at a
    time. All five files are (y, x, z), the MSE included despite its dimension
    NAMES saying otherwise -- see make_input_profiles._twpice_field, which
    carried the wrong flag for MSE until 2026-08-10. Nothing in this file's
    output moved when it was corrected: every statistic here is per-level and
    permutation-invariant.

    Everything is cast to float64 at the moment of reading, before any
    arithmetic touches it. MSE is K-scale, where a float32 accumulator drifts
    (the standing gotcha), and doing it on read rather than at each reduction
    means a reduction added later cannot reintroduce the problem.
    """
    z = coarsen_xyz(read_var(HOST / f"TWPICE_LPT_3D_QV_{SNAPSHOT}.nc", "z"),
                    xy_coarsen)

    def load(name, y_first, scale):
        a = _twpice_field(HOST / f"TWPICE_LPT_3D_{name}_{SNAPSHOT}.nc",
                          name, y_first).astype(np.float64)
        c = coarsen_xyz(a, xy_coarsen)
        del a
        print(f"  {name} -> {c.shape}", flush=True)
        return c * scale

    h = load("MSE", True, cp)           # SAM stores MSE in K; h = cp * MSE
    qv = load("QV", True, 1e-3)         # SAM mixing ratios are g/kg
    qc = load("QC", True, 1e-3)
    qi = load("QI", True, 1e-3)
    T = load("TABS", True, 1.0)         # K, archived; not inverted from MSE
    qt = qv + qc + qi                   # no precipitating water, by ruling
    del qv

    fields = {"h": h, "qt": qt, "T": T, "qc": qc, "qi": qi}
    # z last, to match STEAM's layout for the level reductions below.
    return z, {k: np.moveaxis(v, 0, -1) for k, v in fields.items()}


def gate_fields(xy_coarsen):
    """GATE h, qt, T, qc, qi on STEAM's grid, each (nx, ny, nz), and its z.

    Read one variable at a time, as for TWPICE. TABS is cast to float64 on
    read (K-scale) and the condensate is partitioned at native resolution
    before coarsening.
    """
    path = REPO / "data" / "gate" / GATE_FILE
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        # Coarsened with the fields, so the gz term below is formed on the
        # grid the fields are on. h is linear in T, z and qv, so building it
        # after the block mean equals block-averaging an h built before.
        z = coarsen_xyz(np.asarray(ds.variables["z"][:], dtype=np.float64),
                        xy_coarsen)
        T = np.asarray(ds.variables["TABS"][0], dtype=np.float64)
        liquid = liquid_fraction(T).astype(np.float32)
        Tc = coarsen_xyz(T, xy_coarsen)
        del T
        qn = np.asarray(ds.variables["QN"][0], dtype=np.float64) * 1e-3
        qc = coarsen_xyz(qn * liquid, xy_coarsen)
        qi = coarsen_xyz(qn * (1.0 - liquid), xy_coarsen)
        del qn, liquid
        qv = coarsen_xyz(np.asarray(ds.variables["QV"][0], dtype=np.float64)
                        * 1e-3, xy_coarsen)
    h = cp * Tc + g * z[:, None, None] + Lv * qv
    qt = qv + qc + qi
    del qv
    print(f"  gate coarsened to {h.shape}", flush=True)
    fields = {"h": h, "qt": qt, "T": Tc, "qc": qc, "qi": qi}
    return z, {k: np.moveaxis(v, 0, -1) for k, v in fields.items()}


def check_ensemble(case):
    """Every member of every amplitude exists and carries STD_VARS.

    Run BEFORE the host is loaded, which is the whole point of it being a
    pass of its own: the TWPICE host is five 4.3 GB fields and minutes of
    I/O, and finding out afterwards that member 03 has no `p` wastes all of
    it. Opening thirty files to read their variable lists costs nothing --
    no data is touched.

    T comes from compute_diagnostics, and a keeper stripped before
    run_gigales_simulations.py's KEEP list grew does not have it. That is
    not recoverable from the keeper: the working file it was stripped from
    is gone, so the member has to be regenerated.
    """
    for set_tag in SETS:
        for m in range(N_MEMBERS):
            path = RUNS / f"{case}_{set_tag}_m{m:02d}.nc"
            if not path.exists():
                raise SystemExit(
                    f"{path.name} not found -- run "
                    f"run_gigales_simulations.py {case} {set_tag} first")
            with netCDF4.Dataset(path) as ds:
                missing = [v for v in STD_VARS if v not in ds.variables]
            if missing:
                raise SystemExit(
                    f"{path.name} has no {missing}; that keeper predates the "
                    f"current KEEP list and nothing backfills it. Delete the "
                    f"{case} keepers and rerun run_gigales_simulations.py")


def open_members(case, set_tag):
    """The ensemble's keepers for one configuration, and their z axis.

    Existence and contents are check_ensemble's job and it has already run;
    what is checked here is the one thing that needs every member open at
    once, that they share a vertical grid.
    """
    handles, z_axis = [], None
    for m in range(N_MEMBERS):
        path = RUNS / f"{case}_{set_tag}_m{m:02d}.nc"
        ds = netCDF4.Dataset(path)
        ds.set_auto_mask(False)
        z = ds.variables["z"][:].astype(np.float64)
        if z_axis is None:
            z_axis = z
        elif z.shape != z_axis.shape or not np.allclose(z, z_axis, rtol=0,
                                                        atol=1e-6):
            for h in handles:
                h.close()
            ds.close()
            raise SystemExit(
                f"{path.name}: z axis differs from member 00; these cannot "
                f"share one set of comparison levels")
        handles.append(ds)
    return handles, z_axis


def window(z_axis, z_target, n):
    """First index of the n-level window centred on z_target.

    The same arithmetic as common.level_slice, applied to a file rather than
    to a field already in memory, so the ensemble lands on exactly the
    levels the host side was reduced onto.
    """
    i0 = int(np.argmin(np.abs(z_axis - z_target))) - n // 2
    return min(max(i0, 0), z_axis.size - n)


def reduce_ensemble(case, set_tag, z_levels, factors, out):
    """Pooled and per-member statistics, and the pooled PDF samples.

    At each level the five members' horizontal planes are stacked into one
    (member, y, x) array. Everything follows from that stack:

      pooled      np.std of the whole stack -- member and horizontal axes
                  together, 5 * 1024^2 cells as one population. This is the
                  line the profile figure draws.
      per member  stack.std(axis=(1, 2)), five numbers. Only their spread is
                  used, for the figure's envelope over individual runs.
      PDFs        at the two PDF levels the stack is kept whole. The
                  histogram flattens it, so pooling the members is exactly
                  five times the samples -- the same way the RCEMIP host
                  side pools its three snapshots.

    The standard deviations are over STD_VARS, the PDF slices over the four
    in VARS: T is on the profile figure only, and keeping its planes at the
    PDF levels would be 40 MB for nothing.

    Read a level at a time across members, which is what the keepers' one
    level per chunk is for. A level's stack is 42 MB in float64.
    """
    handles, z_axis = open_members(case, set_tag)
    nlev = z_levels.size
    pooled = {v: np.empty(nlev) for v in STD_VARS}
    per_member = {v: np.empty((N_MEMBERS, nlev)) for v in STD_VARS}
    cf_pooled = np.empty(nlev)
    cf_member = np.empty((N_MEMBERS, nlev))
    pdf_at = {int(np.argmin(np.abs(z_levels - z))): z for z in PDF_LEVELS}
    slices = {}

    try:
        for j, (z, n) in enumerate(zip(z_levels, factors)):
            i0 = window(z_axis, z, n)
            planes = {}
            for v in STD_VARS:
                stack = np.stack([
                    ds.variables[v][:, :, i0:i0 + n].mean(axis=-1,
                                                          dtype=np.float64)
                    for ds in handles])
                planes[v] = stack
                pooled[v][j] = stack.std()
                per_member[v][:, j] = stack.std(axis=(1, 2))
            cond = planes["qc"] + planes["qi"]
            cloudy = cond >= CLOUD_KGKG
            cf_pooled[j] = cloudy.mean()
            cf_member[:, j] = cloudy.mean(axis=(1, 2))
            if j in pdf_at:
                tag = f"{pdf_at[j] / 1000:.0f}km"
                for v in VARS:
                    slices[f"{v}_{tag}"] = planes[v].astype(np.float32)
    finally:
        for ds in handles:
            ds.close()

    for v in STD_VARS:
        out[f"std_{v}_{case}_{set_tag}"] = pooled[v]
        out[f"std_{v}_{case}_{set_tag}_members"] = per_member[v]
    out[f"cf_{case}_{set_tag}"] = cf_pooled
    out[f"cf_{case}_{set_tag}_members"] = cf_member
    for k, a in slices.items():
        out[f"pdf_{k}_{case}_{set_tag}"] = a
    print(f"  {case} steam {set_tag}: {N_MEMBERS} members pooled",
          flush=True)


HOST_LOADER = {"twpice": twpice_fields, "gate": gate_fields}


def host_z(case, xy_coarsen):
    """The host's COARSENED z axis, read without touching the 3-D fields.

    Coarsened by the same factor as the loaders apply, because this is what
    fixes the comparison levels: if it were the native axis, the levels
    would be asked of a field that no longer has them.
    """
    if case == "twpice":
        path = HOST / f"TWPICE_LPT_3D_QV_{SNAPSHOT}.nc"
    else:
        path = REPO / "data" / "gate" / GATE_FILE
    return coarsen_xyz(np.asarray(read_var(path, "z"), dtype=np.float64),
                       xy_coarsen)


def do_case(case, out):
    # Before anything expensive: the whole ensemble is there and carries
    # what will be asked of it. Thirty variable lists, no data read.
    check_ensemble(case)
    # The z axis alone fixes the comparison levels and the coarsening
    # factors, so read the grids before loading anything large. Sources are
    # then loaded, reduced and freed one at a time -- holding a host and both
    # STEAM sets at once would be ~40 GB.
    with netCDF4.Dataset(RUNS / f"{case}_{SETS[0]}_m00.nc") as ds:
        z_steam0 = ds.variables["z"][:].astype(np.float64)
        steam_dx = float(ds.dx)
    xy_coarsen = coarsen_factor(steam_dx, HOST_DX)
    z_h = host_z(case, xy_coarsen)
    z_levels = z_h[z_h <= z_steam0[-1]]
    n_steam, n_host = match_factors(z_levels, z_steam0)
    out[f"{case}_z"] = z_levels
    out[f"{case}_n_steam"] = n_steam
    out[f"{case}_n_host"] = n_host
    out[f"{case}_dx"] = steam_dx
    out[f"{case}_xy_coarsen"] = xy_coarsen
    print(f"{case}: {z_levels.size} levels to {z_levels[-1]:.0f} m; "
          f"host {xy_coarsen}x{xy_coarsen}x{xy_coarsen} coarsened to "
          f"{steam_dx:.0f} m; "
          f"STEAM coarsened by {sorted(set(n_steam.tolist()))}, "
          f"host by {sorted(set(n_host.tolist()))}", flush=True)

    z_host_axis, host = HOST_LOADER[case](xy_coarsen)
    std, cf, slices = reduce_source(z_host_axis, host, z_levels, n_host)
    del host
    for v in STD_VARS:
        out[f"std_{v}_{case}_host"] = std[v]
    out[f"cf_{case}_host"] = cf
    for k, a in slices.items():
        out[f"pdf_{k}_{case}_host"] = a
    print(f"  {case} host reduced", flush=True)

    for tag in SETS:
        reduce_ensemble(case, tag, z_levels, n_steam, out)


def main():
    cases = sys.argv[1:] or list(CASES)
    for case in cases:
        if case not in CASES:
            raise SystemExit(f"unknown case {case!r} (have {list(CASES)})")
    out = {"cases": np.array(cases), "sets": np.array(SETS),
           "n_members": N_MEMBERS, "cloud_kgkg": CLOUD_KGKG,
           "pdf_levels": np.array(PDF_LEVELS)}
    for case in cases:
        do_case(case, out)

    OUTPUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
