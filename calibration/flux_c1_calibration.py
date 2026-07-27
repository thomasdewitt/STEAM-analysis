#!/usr/bin/env python3
"""Re-fit STEAM's flux generator scale c against realized multifractal C1.

Resurrected 2026-07-27 from the deleted pre-restructure script (git
e7fbaaf^:calibration/flux_c1_calibration.py) and adapted to the current
canonical model (audit item C20): flux substeps are gone -- the cascade
density knob is n_scale_classes_per_dyad, and the k ladder itself carries
the density (adjacent-class ratio 2**(1/n_c)). Sweeping n_c in {1, 2}
doubles as a check of the n_c**(-1/alpha) density compensation.

Method (unchanged): the flux cascade converges to a universal-multifractal
measure with K(q) = C1/(alpha-1) * (q^alpha - q). Measure K(2) by
horizontal box coarse-graining, all z levels and boxes entering each moment
estimate jointly::

    <F_lambda**2> ~ lambda**(-K(2)),   C1 = K(2) * (alpha-1) / (2^alpha - 2).

Fit C1 = A * c^alpha through the origin on low-clip runs (clip <= 2%), and
report a free log-log exponent as a check -- the alpha-stable collapse
predicts exponent ~ alpha = 1.8 (this is also the empirical test of the
c-inside-the-exponent form, audit item B11 discussion 2026-07-27).

Runs on CPU (simulate_flux_only is numpy-internal). Writes
calibration/flux_c1_calibration.json and calibration/figs/.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from steam.simulate import _compute_all_grids, simulate_flux_only, FLUX_ALPHA

DEFAULT_C_VALUES = (0.03, 0.05, 0.1, 0.2, 0.4, 0.6)
DEFAULT_N_CLASSES_PER_DYAD = (1, 2)
DEFAULT_BOX_SIZES = (2, 4, 8, 16, 32, 64)
MAX_CLIP_FRACTION_FOR_FIT = 0.02


def _csv_numbers(text: str, cast=float) -> tuple:
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def build_root_grids(
    nx: int,
    ny: int,
    domain_height: float,
    outer_scale: float,
    finest_scale: float,
    spheroscale: float,
    n_scale_classes_per_dyad: int,
):
    """Construct unpadded root-simulation grids with the n_c-dense k ladder."""
    n_octaves = math.log2(outer_scale / finest_scale)
    if not math.isclose(n_octaves, round(n_octaves), abs_tol=1e-12):
        raise ValueError("outer_scale / finest_scale must be an exact power of two")
    if nx % int(outer_scale) or ny % int(outer_scale):
        raise ValueError("nx and ny extents must be integer multiples of outer_scale")

    n_classes = int(round(n_octaves)) * n_scale_classes_per_dyad + 1
    k_values = outer_scale / 2.0 ** (
        np.arange(n_classes) / n_scale_classes_per_dyad
    )
    z_profile = np.arange(int(math.ceil(domain_height)) + 1, dtype=np.float64)
    spheroscale_profile = np.full_like(z_profile, spheroscale)
    return _compute_all_grids(
        k_values,
        inner_extent_x=float(nx),
        inner_extent_y=float(ny),
        inner_height=float(domain_height),
        sparsity_factors=(1, 1, 1),
        spheroscale_profile=spheroscale_profile,
        z_profile=z_profile,
    )


def horizontal_box_moments(field: np.ndarray, box_sizes: tuple[int, ...]):
    """Return joint horizontal-box/z-level second moments for each box size."""
    if field.ndim != 3:
        raise ValueError(f"expected a 3-D flux field, got shape {field.shape}")
    nx, ny, nz = field.shape
    moments = []
    for box in box_sizes:
        if box < 1 or nx % box or ny % box:
            raise ValueError(
                f"box size {box} must divide both horizontal dimensions {(nx, ny)}"
            )
        coarse = field.reshape(nx // box, box, ny // box, box, nz).mean(
            axis=(1, 3), dtype=np.float64,
        )
        moments.append(float(np.mean(np.square(coarse), dtype=np.float64)))
    return np.asarray(moments, dtype=np.float64)


def estimate_c1(field: np.ndarray, box_sizes: tuple[int, ...], alpha: float):
    """Estimate K(2) and multifractal C1 from box-coarse-grained moments."""
    boxes = np.asarray(box_sizes, dtype=np.float64)
    moments = horizontal_box_moments(field, box_sizes)
    slope, intercept = np.polyfit(np.log(boxes), np.log(moments), 1)
    fitted = np.exp(intercept + slope * np.log(boxes))
    residual = np.log(moments) - np.log(fitted)
    ss_res = float(np.sum(residual**2))
    centered = np.log(moments) - float(np.mean(np.log(moments)))
    ss_tot = float(np.sum(centered**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    k2 = -float(slope)
    return {
        "K2": k2,
        "C1": k2 * (alpha - 1.0) / (2.0**alpha - 2.0),
        "fit_r_squared": r_squared,
        "box_sizes": list(box_sizes),
        "moments_q2": moments.tolist(),
        "fitted_moments_q2": fitted.tolist(),
    }


def fit_amplitude(c_values: np.ndarray, c1_values: np.ndarray, alpha: float):
    """Fit C1 = A * c^alpha through the origin (the alpha-stable collapse)."""
    x = np.power(c_values, alpha)
    y = np.asarray(c1_values)
    coefficient = float(np.dot(x, y) / np.dot(x, x))
    fitted = coefficient * x
    residuals = y - fitted
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    dof = max(1, len(y) - 1)
    coefficient_se = math.sqrt((ss_res / dof) / float(np.dot(x, x)))
    return {
        "A": coefficient,
        "A_standard_error": coefficient_se,
        "r_squared": r_squared,
        "n_realizations": len(y),
        "fitted_c1": fitted.tolist(),
    }


def fit_collapse_exponent(c_values: np.ndarray, c1_values: np.ndarray):
    """Free log-log fit C1 = A*c^p; the alpha-stable collapse has p ~ alpha."""
    x = np.log(c_values)
    y = np.log(c1_values)
    exponent, log_amplitude = np.polyfit(x, y, 1)
    fitted = log_amplitude + exponent * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "exponent_on_c": float(exponent),
        "amplitude": float(np.exp(log_amplitude)),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
    }


def _git_revision(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def make_plot(summary: dict, path: Path):
    results = summary["results"]
    alpha = summary["flux_alpha"]
    n_values = sorted({item["n_scale_classes_per_dyad"] for item in results})
    n_realizations = len(results[0]["realizations"])

    fig, axes = plt.subplots(
        1, len(n_values), figsize=(4.6 * len(n_values), 3.8),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = ("#1764ab", "#2a9d8f", "#e76f51")
    x_line = np.geomspace(
        min(item["c_alpha"] for item in results) / 1.25,
        max(item["c_alpha"] for item in results) * 1.25,
        300,
    )

    for ax, n_c, color in zip(axes, n_values, colors):
        selected = [
            item for item in results if item["n_scale_classes_per_dyad"] == n_c
        ]
        for item in selected:
            x_i = item["c_alpha"]
            ax.scatter(
                np.full(len(item["realizations"]), x_i),
                [run["C1"] for run in item["realizations"]],
                s=20, color=color, alpha=0.32, zorder=2,
            )
        ax.errorbar(
            [item["c_alpha"] for item in selected],
            [item["C1_mean"] for item in selected],
            yerr=[item["C1_standard_deviation"] for item in selected],
            fmt="o", color=color, capsize=3,
            label=rf"mean $\pm$ 1 s.d. ($N={n_realizations}$)", zorder=4,
        )
        A_panel = summary["fits_by_n_c"][str(n_c)]["amplitude_fit"]["A"]
        ax.plot(
            x_line, A_panel * x_line,
            color="0.15", lw=1.8,
            label=rf"low-clip fit: $A={A_panel:.3f}$",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$n_c={n_c}$")
        ax.set_xlabel(rf"flux generator scale $c^{{\alpha}}$ ($\alpha={alpha}$)")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(r"realized $C_1$")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-values", default=",".join(map(str, DEFAULT_C_VALUES)))
    parser.add_argument(
        "--n-classes-per-dyad",
        default=",".join(map(str, DEFAULT_N_CLASSES_PER_DYAD)),
    )
    parser.add_argument("--realizations", type=int, default=4)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--domain-height", type=float, default=64.0)
    parser.add_argument("--outer-scale", type=float, default=256.0)
    parser.add_argument("--finest-scale", type=float, default=2.0)
    parser.add_argument("--spheroscale", type=float, default=2.0)
    parser.add_argument("--box-sizes", default=",".join(map(str, DEFAULT_BOX_SIZES)))
    parser.add_argument("--base-seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path, default=here)
    return parser.parse_args()


def main():
    args = parse_args()
    c_values = _csv_numbers(args.c_values, float)
    n_c_values = _csv_numbers(args.n_classes_per_dyad, int)
    box_sizes = _csv_numbers(args.box_sizes, int)
    if args.realizations < 2:
        raise ValueError("at least two realizations are required to measure spread")
    if len(box_sizes) < 5:
        raise ValueError("use at least five box sizes to span four scaling octaves")
    if any(n < 1 for n in n_c_values):
        raise ValueError("n_scale_classes_per_dyad values must be positive integers")

    seed_sequences = np.random.SeedSequence(args.base_seed).spawn(args.realizations)
    seeds = [
        int(seq.generate_state(1, dtype=np.uint64)[0]) for seq in seed_sequences
    ]

    started = time.perf_counter()
    results = []
    fit_c = []
    fit_c1 = []
    for n_c in n_c_values:
        grids = build_root_grids(
            args.nx, args.ny, args.domain_height,
            args.outer_scale, args.finest_scale, args.spheroscale, n_c,
        )
        final_shape = tuple(
            int(values[-1]) for values in (grids["nx"], grids["ny"], grids["nz"])
        )
        if any(final_shape[axis] % box for box in box_sizes for axis in (0, 1)):
            raise ValueError(
                f"box sizes {box_sizes} do not divide final shape {final_shape}"
            )
        for c in c_values:
            realizations = []
            for seed in seeds:
                flux, diagnostics = simulate_flux_only(
                    grids,
                    seed=seed,
                    flux_noise_scale=c,
                    n_scale_classes_per_dyad=n_c,
                )
                run = estimate_c1(flux, box_sizes, FLUX_ALPHA)
                run["clip_fraction"] = diagnostics["clip_fraction"]
                run["seed"] = seed
                realizations.append(run)
            c1_samples = np.array([run["C1"] for run in realizations])
            clip_fractions = np.array([run["clip_fraction"] for run in realizations])
            low_clip = bool(np.all(clip_fractions <= MAX_CLIP_FRACTION_FOR_FIT))
            item = {
                "c": c,
                "c_alpha": c**FLUX_ALPHA,
                "n_scale_classes_per_dyad": n_c,
                "C1_mean": float(c1_samples.mean()),
                "C1_standard_deviation": float(c1_samples.std(ddof=1)),
                "clip_fraction_max": float(clip_fractions.max()),
                "low_clip": low_clip,
                "realizations": realizations,
            }
            results.append(item)
            if low_clip:
                fit_c.extend([c] * len(realizations))
                fit_c1.extend(c1_samples.tolist())
            print(
                f"n_c={n_c} c={c}: C1 = {item['C1_mean']:.4f} "
                f"+/- {item['C1_standard_deviation']:.4f} "
                f"(max clip {item['clip_fraction_max']:.3%}"
                f"{', in fit' if low_clip else ', EXCLUDED'})",
                flush=True,
            )

    # Fit per class density: pooling n_c groups is invalid if the
    # n_c^(-1/alpha) density compensation is not exactly invariant (it is
    # not -- see fits_by_n_c and the cross-density ratios in the output).
    fits_by_n_c = {}
    for n_c in n_c_values:
        group = [
            item for item in results
            if item["n_scale_classes_per_dyad"] == n_c and item["low_clip"]
        ]
        group_c = np.asarray([
            item["c"] for item in group for _ in item["realizations"]
        ])
        group_c1 = np.asarray([
            run["C1"] for item in group for run in item["realizations"]
        ])
        if group_c.size == 0:
            raise ValueError(
                f"n_c={n_c}: no runs passed the low-clip cut "
                f"(clip <= {MAX_CLIP_FRACTION_FOR_FIT:.0%}); "
                "add smaller c values to the ladder"
            )
        fits_by_n_c[n_c] = {
            "amplitude_fit": fit_amplitude(group_c, group_c1, FLUX_ALPHA),
            "collapse_exponent_fit": fit_collapse_exponent(group_c, group_c1),
            "n_low_clip_points": int(group_c.size),
        }

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "steam_revision": _git_revision(
            Path.home() / "code-and-data" / "turbulon-model"
        ),
        "flux_alpha": FLUX_ALPHA,
        "config": {
            "nx": args.nx, "ny": args.ny,
            "domain_height": args.domain_height,
            "outer_scale": args.outer_scale,
            "finest_scale": args.finest_scale,
            "spheroscale": args.spheroscale,
            "box_sizes": list(box_sizes),
            "realizations": args.realizations,
            "base_seed": args.base_seed,
            "max_clip_fraction_for_fit": MAX_CLIP_FRACTION_FOR_FIT,
        },
        "fits_by_n_c": {str(k): v for k, v in fits_by_n_c.items()},
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "flux_c1_calibration.json", "w") as f:
        json.dump(summary, f, indent=1)
    make_plot(summary, out_dir / "figs" / "flux_c1_calibration.png")

    print()
    for n_c, fits in fits_by_n_c.items():
        A = fits["amplitude_fit"]["A"]
        free = fits["collapse_exponent_fit"]
        print(
            f"n_c={n_c}: C1 = {A:.3f} * c^{FLUX_ALPHA} "
            f"(SE {fits['amplitude_fit']['A_standard_error']:.3f}, "
            f"R^2 {fits['amplitude_fit']['r_squared']:.4f}); "
            f"free exponent {free['exponent_on_c']:.3f} "
            f"(predicts {FLUX_ALPHA}, R^2 {free['r_squared']:.4f})"
        )
    base_n_c = min(fits_by_n_c)
    A_base = fits_by_n_c[base_n_c]["amplitude_fit"]["A"]
    for n_c in sorted(fits_by_n_c):
        if n_c != base_n_c:
            ratio = fits_by_n_c[n_c]["amplitude_fit"]["A"] / A_base
            print(
                f"density invariance check A(n_c={n_c})/A(n_c={base_n_c}) = "
                f"{ratio:.3f} (exact compensation predicts 1; "
                f"log2 ratio {math.log2(ratio):+.3f} per doubling-equivalent)"
            )
    print(f"\nProduction mapping (n_c={base_n_c}):")
    for target in (0.05, 0.1):
        print(f"c for C1 = {target}: {(target / A_base) ** (1 / FLUX_ALPHA):.4f}")
    print(f"Elapsed: {summary['elapsed_seconds']:.0f} s")


if __name__ == "__main__":
    main()
