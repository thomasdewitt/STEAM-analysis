"""
intermittency_viz.py

Visualization of intermittency character: STEAM vs SAM vs a synthetic FIF
multifractal.

For each target variable (q_t, MSE):

1. Horizontal profile visualization — 5 separate figures per variable,
   each a 3-row (STEAM / SAM / FIF) stack of thin black lines drawn from
   an independent realization (different seed/timestep/draw). Axes are
   stripped entirely (no ticks, no spines, no x-axis). STEAM and FIF are
   shown as their first 1024 samples stretched to full panel width; SAM
   is shown as 2048 samples. Each profile is individually demeaned.

2. K(q) intermittency function — empirical points for STEAM and SAM on
   one plot, with their theoretical Lovejoy-Schertzer fits
     K(q) = (C1/(alpha-1)) * (q^alpha - q).

3. FIF multifractal of size 8192, generated with STEAM's fitted C1 and H
   (from K_empirical), alpha fixed to 2, used as the middle profile above.

Everything is horizontal-direction within the 5–6 km altitude band, matching
the existing horizontal SF comparison.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import scaleinvariance as si
si.set_backend('torch')
si.set_device('cuda')
si.set_numerical_precision('float64')

FIGURES_DIR = _ROOT / 'Figures'
FIGURES_DIR.mkdir(exist_ok=True)

ALT_MIN = 5000.0
ALT_MAX = 6000.0
PROFILE_ALT_M = 5500.0
N_REALIZATIONS = 5                  # rows per dataset column
STEAM_LEN = 1024                    # samples shown for STEAM
SAM_LEN   = 2048                    # samples shown for SAM
FIF_LEN   = 1024                    # samples shown for FIF (after coarsening)
FIF_DISPLAY_N = 2048                # coarsened FIF length before truncation
FIF_GEN_SIZE = 16384                # generate FIF at higher resolution...
FIF_COARSEN  = FIF_GEN_SIZE // FIF_DISPLAY_N   # ...then box-average down
FIF_ALPHA_FORCED = 2.0
Q_VALUES = np.arange(0.1, 2.51, 0.1)

# Fit window (km) used both for the H/K-empirical fit and reported info
FIT_MIN_KM = 1.0
FIT_MAX_KM = 3.0

STEAM_COLOR = '#bc4b3d'     # clay red
SAM_COLOR   = '#3e6d7a'     # deep teal


# ── Loaders (matching the horizontal SF script) ───────────────────────

def _steam_seed_files():
    return [f.resolve() for f in
            sorted((_ROOT / 'STEAM' / 'data').glob('nested_refine_seed*.nc'))]


def _load_steam(variable):
    from utils.steam_loader import load_steam_variable_interpolated
    z, data, dx = load_steam_variable_interpolated(
        variable, group='strips', seed_files=_steam_seed_files(),
        min_height=ALT_MIN, max_height=ALT_MAX)
    return z, data.astype(np.float64, copy=False), dx


def _load_sam(variable):
    from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
    z, data, dx = load_sam_twpice_variable_interpolated(
        variable, min_height=ALT_MIN, max_height=ALT_MAX)
    return z, data.astype(np.float64, copy=False), dx


def _sam_qt():
    z, qv, dx = _load_sam('QV')
    _, qc, _  = _load_sam('QC')
    _, qi, _  = _load_sam('QI')
    return z, qv + qc + qi, dx


# ── Helpers ───────────────────────────────────────────────────────────

def _km_lag_window(lo_km, hi_km, dx_m):
    lo = max(2, int(round(lo_km * 1e3 / dx_m)))
    hi = max(lo + 1, int(round(hi_km * 1e3 / dx_m)))
    return lo, hi


def _pick_profiles(data_4d, z, target_alt_m, n_picks, length):
    """Return up to n_picks 1D demeaned profiles of given length.

    data_4d has shape (n_ens, axis1, axis2, nz); axis1 is the full-density
    horizontal axis for both STEAM and SAM. Realizations are drawn from
    the combined (axis0, axis2) pool of independent spatial lines —
    axis0 is the ensemble/time axis and axis2 is the strided-horizontal
    axis (different rows in the horizontal plane). If the pool is still
    too small, the remainder comes from non-overlapping windows along
    axis 1 of the first (i, j) pair.
    """
    k = int(np.argmin(np.abs(z - target_alt_m)))
    n_ens, n_spatial, n_ax2 = (data_4d.shape[0],
                                data_4d.shape[1],
                                data_4d.shape[2])
    pairs = [(i, j) for i in range(n_ens) for j in range(n_ax2)]
    sel = np.linspace(0, len(pairs) - 1, n_picks).round().astype(int)
    sel = list(dict.fromkeys(sel.tolist()))
    profiles = []
    for s in sel:
        i, j = pairs[s]
        v = np.asarray(data_4d[i, :length, j, k], dtype=np.float64)
        profiles.append(v - v.mean())
    start = length
    i0, j0 = pairs[0]
    while len(profiles) < n_picks and start + length <= n_spatial:
        v = np.asarray(data_4d[i0, start:start + length, j0, k],
                       dtype=np.float64)
        profiles.append(v - v.mean())
        start += length
    return profiles


def _to_numpy(x):
    from scaleinvariance.backend import to_numpy
    return np.asarray(to_numpy(x))


# ── Per-variable driver ───────────────────────────────────────────────

def _run(variable_label, steam_var, sam_loader, file_tag):
    print(f'\n=== {variable_label} ===')

    z_s, steam_data, dx_s = _load_steam(steam_var)
    z_a, sam_data,  dx_a = sam_loader()
    print(f'  STEAM: shape {steam_data.shape}, dx={dx_s:.1f} m')
    print(f'  SAM:   shape {sam_data.shape}, dx={dx_a:.1f} m')

    # K(q) on axis=1 (horizontal) with per-dataset lag window for 1–3 km
    min_s, max_s = _km_lag_window(FIT_MIN_KM, FIT_MAX_KM, dx_s)
    min_a, max_a = _km_lag_window(FIT_MIN_KM, FIT_MAX_KM, dx_a)
    print(f'  STEAM lag window (lags): {min_s}–{max_s}')
    print(f'  SAM   lag window (lags): {min_a}–{max_a}')

    q_s, K_s_emp, H_s, C1_s, alpha_s = si.K_empirical(
        steam_data, q_values=Q_VALUES, axis=1,
        min_sep=min_s, max_sep=max_s,
        scaling_method='structure_function')
    q_a, K_a_emp, H_a, C1_a, alpha_a = si.K_empirical(
        sam_data, q_values=Q_VALUES, axis=1,
        min_sep=min_a, max_sep=max_a,
        scaling_method='structure_function')
    q_s, K_s_emp = _to_numpy(q_s), _to_numpy(K_s_emp)
    q_a, K_a_emp = _to_numpy(q_a), _to_numpy(K_a_emp)
    print(f'  STEAM fit: H={H_s:.4f}, C1={C1_s:.4f}, alpha={alpha_s:.4f}')
    print(f'  SAM   fit: H={H_a:.4f}, C1={C1_a:.4f}, alpha={alpha_a:.4f}')

    # FIF: 5 independent draws at FIF_GEN_SIZE with STEAM C1/H and α=2,
    # coarsened to FIF_DISPLAY_N and truncated to FIF_LEN.
    fif_profiles = []
    for _ in range(N_REALIZATIONS):
        ff = np.asarray(si.FIF_1D(FIF_GEN_SIZE, alpha=FIF_ALPHA_FORCED,
                                   C1=float(C1_s), H=float(H_s)))
        ff = ff.reshape(-1, FIF_COARSEN).mean(axis=1)[:FIF_LEN]
        fif_profiles.append(ff - ff.mean())
    print(f'  FIF: {N_REALIZATIONS} draws at gen={FIF_GEN_SIZE}, '
          f'coarsen={FIF_COARSEN}× → {FIF_DISPLAY_N}, shown {FIF_LEN}; '
          f'α={FIF_ALPHA_FORCED}, C1={C1_s:.4f}, H={H_s:.4f}')

    # ── Plot 1: 5 separate (STEAM, SAM, FIF) stacks ──────────────────
    steam_profiles = _pick_profiles(steam_data, z_s, PROFILE_ALT_M,
                                     N_REALIZATIONS, STEAM_LEN)
    sam_profiles   = _pick_profiles(sam_data,   z_a, PROFILE_ALT_M,
                                     N_REALIZATIONS, SAM_LEN)
    print(f'  profile sets: STEAM={len(steam_profiles)} '
          f'(len {STEAM_LEN}), SAM={len(sam_profiles)} (len {SAM_LEN}), '
          f'FIF={len(fif_profiles)} (len {FIF_LEN})')

    n_sets = min(N_REALIZATIONS, len(steam_profiles),
                 len(sam_profiles), len(fif_profiles))
    for r in range(n_sets):
        rows = [('STEAM', steam_profiles[r]),
                ('SAM',   sam_profiles[r]),
                ('FIF',   fif_profiles[r])]
        fig, axes = plt.subplots(3, 1, figsize=(8, 4.5))
        for ax, (label, prof) in zip(axes, rows):
            ax.plot(prof, '-', color='black', lw=0.25)
            ax.set_xlim(0, len(prof) - 1)
            ax.text(0.01, 0.88, label, transform=ax.transAxes,
                    fontsize=11, color='black', family='monospace')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ('top', 'right', 'left', 'bottom'):
                ax.spines[s].set_visible(False)
            ax.grid(False)
        plt.tight_layout()
        out_a = FIGURES_DIR / f'intermittency_profiles_{file_tag}_{r+1}.pdf'
        plt.savefig(out_a, transparent=True)
        plt.close(fig)
        print(f'  saved {out_a}')

    # ── Plot 2: K(q) comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    q_fine = np.linspace(Q_VALUES.min(), Q_VALUES.max(), 200)
    ax.plot(q_s, K_s_emp, 'o', color=STEAM_COLOR, ms=4,
            label='STEAM empirical')
    ax.plot(q_fine, si.K_analytic(q_fine, C1_s, alpha_s),
            '-', color=STEAM_COLOR, lw=1.5,
            label=f'STEAM fit: C₁={C1_s:.3f}, α={alpha_s:.2f}')
    ax.plot(q_a, K_a_emp, 's', color=SAM_COLOR, ms=4,
            label='SAM empirical')
    ax.plot(q_fine, si.K_analytic(q_fine, C1_a, alpha_a),
            '-', color=SAM_COLOR, lw=1.5,
            label=f'SAM fit: C₁={C1_a:.3f}, α={alpha_a:.2f}')
    ax.axhline(0, color='#888', lw=0.5)
    ax.set_xlabel('Order $q$')
    ax.set_ylabel('$K(q)$')
    ax.set_title(f'{variable_label} intermittency K(q), horizontal, '
                 f'{ALT_MIN/1e3:.0f}–{ALT_MAX/1e3:.0f} km')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(False)
    plt.tight_layout()
    out_b = FIGURES_DIR / f'intermittency_Kq_{file_tag}.pdf'
    plt.savefig(out_b, transparent=True)
    plt.close(fig)
    print(f'  saved {out_b}')


def main():
    print(f'backend={si.get_backend()}, device={si.get_device()}, '
          f'precision={si.get_numerical_precision()}')
    print(f'Altitude band: {ALT_MIN:.0f}–{ALT_MAX:.0f} m, '
          f'profile slice @ {PROFILE_ALT_M:.0f} m')
    print(f'K_empirical fit window: {FIT_MIN_KM}–{FIT_MAX_KM} km')

    _run('q_t', 'qt', _sam_qt, 'qt')
    _run('MSE', 'h', lambda: _load_sam('MSE'), 'MSE')


if __name__ == '__main__':
    main()
