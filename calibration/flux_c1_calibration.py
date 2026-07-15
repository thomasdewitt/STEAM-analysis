#!/usr/bin/env python3
"""Calibrate STEAM's flux innovation scale c against realized lognormal C1.

The script uses STEAM's flux-only cascade path and measures K(2) by horizontal
box coarse-graining at fixed z::

    <F_lambda**2> ~ lambda**(-K(2)),   C1 = K(2) / 2.

All z levels and horizontal boxes enter each moment estimate together before
the log-log regression. This avoids averaging nonlinear per-level fits.
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
import torch

from steam.simulate import _compute_all_grids, simulate_flux_only


DEFAULT_C_VALUES = (0.1, 0.2, 0.4, 0.6)
DEFAULT_BOX_SIZES = (2, 4, 8, 16, 32, 64)
REFERENCE_KAPPA = 0.6
POSITIVITY_DIAGNOSTIC_C = 0.5


def _csv_numbers(text: str, cast=float) -> tuple:
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def build_root_grids(
    nx: int,
    ny: int,
    domain_height: float,
    outer_scale: float,
    finest_scale: float,
    spheroscale: float,
):
    """Construct the same unpadded, dyadic grids used by a root simulation."""
    n_steps = math.log2(outer_scale / finest_scale)
    if not math.isclose(n_steps, round(n_steps), abs_tol=1e-12):
        raise ValueError("outer_scale / finest_scale must be an exact power of two")
    if nx % int(outer_scale) or ny % int(outer_scale):
        raise ValueError("nx and ny extents must be integer multiples of outer_scale")

    k_values = outer_scale / 2.0 ** np.arange(int(round(n_steps)) + 1)
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


def estimate_c1(field: np.ndarray, box_sizes: tuple[int, ...]):
    """Estimate K(2) and lognormal C1 from box-coarse-grained moments."""
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
        "C1": k2 / 2.0,
        "fit_r_squared": r_squared,
        "box_sizes": list(box_sizes),
        "moments_q2": moments.tolist(),
        "fitted_moments_q2": fitted.tolist(),
    }


def fit_kappa(c_values: np.ndarray, c1_values: np.ndarray):
    """Fit C1 = kappa*c^2/(2 ln 2) through the physical origin."""
    x = np.square(c_values)
    y = np.asarray(c1_values)
    coefficient = float(np.dot(x, y) / np.dot(x, x))
    fitted = coefficient * x
    residuals = y - fitted
    ss_res = float(np.dot(residuals, residuals))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    dof = max(1, len(y) - 1)
    coefficient_se = math.sqrt((ss_res / dof) / float(np.dot(x, x)))
    factor = 2.0 * math.log(2.0)
    return {
        "kappa": factor * coefficient,
        "kappa_standard_error": factor * coefficient_se,
        "r_squared": r_squared,
        "fitted_c1": fitted.tolist(),
    }


def fit_collapse_exponent(c_values: np.ndarray, c1_values: np.ndarray):
    """Free log-log fit C1 = A*(c^2)^p; ideal c^2 collapse has p=1."""
    x = np.log(np.square(c_values))
    y = np.log(c1_values)
    exponent, log_amplitude = np.polyfit(x, y, 1)
    fitted = log_amplitude + exponent * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "exponent_on_c_squared": float(exponent),
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
    c = np.asarray([item["c"] for item in results])
    x = c**2
    means = np.asarray([item["C1_mean"] for item in results])
    spread = np.asarray([item["C1_standard_deviation"] for item in results])
    kappa = summary["kappa_fit"]["kappa"]
    n_realizations = len(results[0]["realizations"])

    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for item in results:
        x_i = item["c"] ** 2
        ax.scatter(
            np.full(len(item["realizations"]), x_i),
            [run["C1"] for run in item["realizations"]],
            s=22, color="0.55", alpha=0.55, zorder=2,
        )
    ax.errorbar(
        x, means, yerr=spread, fmt="o", color="#1764ab", capsize=3,
        label=rf"STEAM mean $\pm$ 1 s.d. ($N={n_realizations}$)", zorder=4,
    )

    x_line = np.geomspace(x.min() / 1.25, x.max() * 1.25, 300)
    ax.plot(
        x_line, kappa * x_line / (2 * np.log(2)), color="#d1495b", lw=2,
        label=rf"through-origin fit: $\kappa={kappa:.3f}$",
    )
    ax.plot(
        x_line, REFERENCE_KAPPA * x_line / (2 * np.log(2)),
        color="0.25", lw=1.3, ls="--",
        label=rf"packing prediction: $\kappa={REFERENCE_KAPPA:.1f}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"flux innovation variance scale $c^2$")
    ax.set_ylabel(r"realized $C_1 = K(2)/2$")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-values", default=",".join(map(str, DEFAULT_C_VALUES)))
    parser.add_argument("--realizations", type=int, default=4)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--domain-height", type=float, default=64.0)
    parser.add_argument("--outer-scale", type=float, default=256.0)
    parser.add_argument("--finest-scale", type=float, default=2.0)
    parser.add_argument("--spheroscale", type=float, default=2.0)
    parser.add_argument("--box-sizes", default=",".join(map(str, DEFAULT_BOX_SIZES)))
    parser.add_argument("--base-seed", type=int, default=20260715)
    parser.add_argument("--output-dir", type=Path, default=here)
    return parser.parse_args()


def main():
    args = parse_args()
    c_values = _csv_numbers(args.c_values, float)
    box_sizes = _csv_numbers(args.box_sizes, int)
    if args.realizations < 2:
        raise ValueError("at least two realizations are required to measure spread")
    if len(box_sizes) < 5:
        raise ValueError("use at least five box sizes to span four scaling octaves")

    grids = build_root_grids(
        args.nx, args.ny, args.domain_height,
        args.outer_scale, args.finest_scale, args.spheroscale,
    )
    final_shape = tuple(
        int(values[-1]) for values in (grids["nx"], grids["ny"], grids["nz"])
    )
    if any(final_shape[axis] % box for box in box_sizes for axis in (0, 1)):
        raise ValueError(f"box sizes {box_sizes} do not divide final shape {final_shape}")

    n_calibration_runs = len(c_values) * args.realizations
    seed_sequences = np.random.SeedSequence(args.base_seed).spawn(
        n_calibration_runs + args.realizations,
    )
    seeds = [int(seq.generate_state(1, dtype=np.uint64)[0]) for seq in seed_sequences]

    started = time.perf_counter()
    results = []
    all_c = []
    all_c1 = []
    run_index = 0
    for c in c_values:
        realizations = []
        for realization in range(args.realizations):
            seed = seeds[run_index]
            run_index += 1
            run_started = time.perf_counter()
            flux, clip = simulate_flux_only(
                grids,
                seed=seed,
                flux_noise_scale=c,
                min_distance_to_ground=1,
                zero_bottom=True,
                zero_top=True,
            )
            estimate = estimate_c1(flux, box_sizes)
            runtime = time.perf_counter() - run_started
            run = {
                "realization": realization,
                "seed": seed,
                **estimate,
                "clip_fraction": clip["clip_fraction"],
                "final_zero_fraction": clip["final_zero_fraction"],
                "clip_steps": clip["steps"],
                "runtime_seconds": runtime,
                "flux_horizontal_mean_max_abs_error": float(
                    np.max(np.abs(flux.mean(axis=(0, 1)) - 1.0))
                ),
            }
            realizations.append(run)
            all_c.append(c)
            all_c1.append(estimate["C1"])
            print(
                f"c={c:.2f} realization={realization + 1}/{args.realizations} "
                f"C1={estimate['C1']:.5f} clip={clip['clip_fraction']:.3%} "
                f"time={runtime:.1f}s",
                flush=True,
            )

        c1 = np.asarray([run["C1"] for run in realizations])
        clip_fractions = np.asarray([run["clip_fraction"] for run in realizations])
        results.append({
            "c": c,
            "c_squared": c**2,
            "C1_mean": float(c1.mean()),
            "C1_standard_deviation": float(c1.std(ddof=1)),
            "C1_min": float(c1.min()),
            "C1_max": float(c1.max()),
            "effective_kappa_from_mean_C1": float(
                2.0 * math.log(2.0) * c1.mean() / c**2
            ),
            "clip_fraction_mean": float(clip_fractions.mean()),
            "clip_fraction_standard_deviation": float(clip_fractions.std(ddof=1)),
            "realizations": realizations,
        })

    kappa = fit_kappa(np.asarray(all_c), np.asarray(all_c1))
    mean_c1 = np.asarray([item["C1_mean"] for item in results])
    collapse = fit_collapse_exponent(np.asarray(c_values), mean_c1)

    positivity_runs = []
    for realization in range(args.realizations):
        seed = seeds[n_calibration_runs + realization]
        run_started = time.perf_counter()
        _flux, clip = simulate_flux_only(
            grids,
            seed=seed,
            flux_noise_scale=POSITIVITY_DIAGNOSTIC_C,
            min_distance_to_ground=1,
            zero_bottom=True,
            zero_top=True,
        )
        positivity_runs.append({
            "realization": realization,
            "seed": seed,
            "clip_fraction": clip["clip_fraction"],
            "final_zero_fraction": clip["final_zero_fraction"],
            "clip_steps": clip["steps"],
            "runtime_seconds": time.perf_counter() - run_started,
        })
        print(
            f"c={POSITIVITY_DIAGNOSTIC_C:.2f} positivity diagnostic "
            f"{realization + 1}/{args.realizations} "
            f"clip={clip['clip_fraction']:.3%}",
            flush=True,
        )

    positivity_clip = np.asarray([run["clip_fraction"] for run in positivity_runs])
    positivity_final_zero = np.asarray(
        [run["final_zero_fraction"] for run in positivity_runs]
    )
    total_runtime = time.perf_counter() - started

    model_repo = Path(__file__).resolve().parents[2] / "turbulon-model"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_git_revision": _git_revision(model_repo),
        "method": {
            "description": (
                "Horizontal non-overlapping box averages at each z; all boxes "
                "and z levels enter <F_lambda^2> jointly; ordinary least-squares "
                "fit of log moment versus log box size; C1=K(2)/2."
            ),
            "box_sizes": list(box_sizes),
            "scaling_octaves": math.log2(max(box_sizes) / min(box_sizes)),
        },
        "grid": {
            "final_shape": final_shape,
            "k_values": np.asarray(grids["k"]).tolist(),
            "outer_scale": args.outer_scale,
            "finest_scale": args.finest_scale,
            "spheroscale": args.spheroscale,
            "domain_height": args.domain_height,
            "sparsity_factors": [1, 1, 1],
            "min_distance_to_ground": 1,
        },
        "execution": {
            "backend": "CPU; NumPy fields with STEAM's Torch CPU FFT convolution",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available_but_unused": torch.cuda.is_available(),
            "cuda_device": gpu_name,
            "total_runtime_seconds": total_runtime,
        },
        "reference_kappa": REFERENCE_KAPPA,
        "results": results,
        "kappa_fit": kappa,
        "collapse_loglog_fit": collapse,
        "positivity_diagnostic": {
            "c": POSITIVITY_DIAGNOSTIC_C,
            "fraction_definition": (
                "Number of pre-clip negative point-updates divided by the total "
                "number of point-updates across all dyadic classes."
            ),
            "clip_fraction_mean": float(positivity_clip.mean()),
            "clip_fraction_standard_deviation": float(positivity_clip.std(ddof=1)),
            "final_zero_fraction_mean": float(positivity_final_zero.mean()),
            "final_zero_fraction_standard_deviation": float(
                positivity_final_zero.std(ddof=1)
            ),
            "realizations": positivity_runs,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "flux_c1_calibration.json"
    figure_path = args.output_dir / "figs" / "flux_c1_collapse.pdf"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    make_plot(summary, figure_path)
    print(f"kappa={kappa['kappa']:.5f} +/- {kappa['kappa_standard_error']:.5f}")
    print(
        "collapse exponent on c^2="
        f"{collapse['exponent_on_c_squared']:.4f}, R^2={collapse['r_squared']:.5f}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
