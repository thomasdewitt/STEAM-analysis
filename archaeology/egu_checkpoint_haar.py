#!/usr/bin/env python3
"""Git archaeology (2026-07-27): does the EGU-era checkpoint scale?

Re-creates a 2048^2 sim with the EGU checkpoint code (turbulon-model
ac16b35, 2026-05-01 -- "temporary /=2 normalization fudge for EGU"),
then computes the single-level horizontal Haar fluctuation, mirroring
make_square_level_haar.py. Thomas: the May code produced dead-on
Haar/structure functions at s=1, so if this looks right the flat-xi(1)
bug was introduced somewhere in the ~90 commits since.

Config matches the EGU production runs (STEAM/generate.py @ 0cd78ed):
4000 km domain, outer scale = L/2 = 2000 km, constant 30 m spheroscale,
canonical anisotropy, s=1, dyadic classes -- but at 2048^2 (dx 1.95 km)
instead of 1024^2, per Thomas's request. Profile substitution: the
EGU-era Dropsonde_extrap profile file (data/rcemip_profiles.nc) no
longer exists on disk, so icon_lem snap0 is used; soft-clamp h bounds
widened accordingly (the old 400*cp ceiling sits BELOW icon_lem's max).

MUST run with the EGU worktree venv:
    ~/code-and-data/turbulon-egu/.venv/bin/python archaeology/egu_checkpoint_haar.py [seed]

Writes runs/archaeology/egu_ac16b35_seed{N:03d}.nc,
stats/egu_checkpoint_haar.npz and figs/archaeology/egu_haar_level.png.
"""

import sys
import time
from pathlib import Path

import numpy as np

import steam
from steam.simulate import simulate

HERE = Path(__file__).resolve().parent.parent   # turbulon-analysis
EGU_REPO = Path.home() / "code-and-data" / "turbulon-egu"
RUNS = HERE / "runs" / "archaeology"
LEVEL_M = 7000.0

# EGU production config (0cd78ed), grid doubled to 2048^2.
NX = NY = 2048
DOMAIN_WIDTH = 4_000_000.0
DOMAIN_HEIGHT = 20_000.0
OUTER_SCALE = DOMAIN_WIDTH / 2
SPHEROSCALE_CONST = 30.0
CP = 1004.0


def check_checkpoint():
    """Refuse to run against anything but the EGU worktree's steam."""
    src = Path(steam.__file__).resolve()
    if EGU_REPO not in src.parents:
        raise RuntimeError(
            f"steam imported from {src}; run this with "
            f"{EGU_REPO}/.venv/bin/python so the ac16b35 code is used")
    import subprocess
    head = subprocess.run(["git", "-C", str(EGU_REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"steam from {src.parent}  (worktree HEAD {head})")
    return head


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    head = check_checkpoint()
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"egu_ac16b35_seed{seed:03d}.nc"
    if out.exists():
        out.unlink()

    src = np.load(HERE / "stats" / "icon_lem_snap0.npz")
    z = src["z_profile"]
    assert float(z[0]) == 0.0 and np.allclose(np.diff(z), 50.0)

    t0 = time.monotonic()
    simulate(
        h_profile=src["h_profile"], qt_profile=src["qt_profile"],
        nx=NX, ny=NY,
        dx=DOMAIN_WIDTH / NX, dy=DOMAIN_WIDTH / NY,
        outer_scale=OUTER_SCALE,
        spheroscale=np.full(z.size, SPHEROSCALE_CONST),
        domain_height=DOMAIN_HEIGHT,
        profile_dz=50.0,
        output_path=out,
        sparsity_factors=(1, 1, 1),
        n_scale_classes_per_dyad=1,
        surface_pressure=float(src["surface_pressure"]),
        seed=seed,
        h_min=250.0 * CP,
        h_max=420.0 * CP,           # widened: icon_lem max 411 kJ/kg > old 400*cp
        qt_min=0.0, qt_max=0.03,
        anisotropy="canonical",
        compress=False,
    )
    print(f"simulate: {(time.monotonic() - t0) / 60:.1f} min -> {out.name}")

    # ── Single-level horizontal Haar (mirrors make_square_level_haar.py) ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import netCDF4
    import scaleinvariance as si

    with netCDF4.Dataset(out) as ds:
        ds.set_auto_mask(False)
        zf = ds.variables["z"][:]
        iz = int(np.argmin(np.abs(zf - LEVEL_M)))
        levels = {name: np.asarray(ds.variables[name][:, :, iz], dtype=np.float32)
                  for name in ("h", "qt")}
        dx_out = float(ds.dx)
        z_used = float(zf[iz])

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
        "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    npz = {"level_m": z_used, "dx": dx_out, "head": head, "seed": seed}
    for (name, field), ax in zip(levels.items(), axes):
        lags, F = si.haar_fluctuation(field, order=1.0, axis=0, periodic=True)
        npz[f"{name}_lags"], npz[f"{name}_F"] = lags, F
        ax.loglog(lags * dx_out / 1000, F, color="#1764ab", lw=1.3,
                  label=f"ac16b35 seed {seed}")
        # H = 0.5 reference (this checkpoint's hurst_horizontal), pegged
        # to the central value, spanning ~2 orders.
        mid = len(lags) // 2
        span = 10.0 ** 0.9
        xs = np.array([lags[mid] / span, lags[mid] * span]) * dx_out / 1000
        x_mid = lags[mid] * dx_out / 1000
        ax.loglog(xs, F[mid] * (xs / x_mid) ** 0.5, color="0.4", lw=0.9, ls="-.")
        ax.annotate("$H=0.5$", (xs[1], F[mid] * span ** 0.5), fontsize=6.5,
                    color="0.4", xytext=(2, -2), textcoords="offset points")
        ax.set(xlabel="lag [km]", ylabel="$\\hat{M}_1(\\ell)$", title=name)
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"EGU checkpoint (ac16b35), 2048$^2$ @ 4000 km, "
                 f"z = {z_used:.0f} m", fontsize=10)
    fig.tight_layout()
    figdir = HERE / "figs" / "archaeology"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "egu_haar_level.png")
    np.savez(HERE / "stats" / "egu_checkpoint_haar.npz", **npz)
    print(f"wrote figs/archaeology/egu_haar_level.png (z = {z_used:.0f} m)")


if __name__ == "__main__":
    main()
