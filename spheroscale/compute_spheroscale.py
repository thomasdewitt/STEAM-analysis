import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Haar scale (n points) ────────────────────────────────────────────────────
# Number of points used in each Haar fluctuation (must be even; n/2 pts per side).
# With n=4: left avg = (phi[x-1.5dx] + phi[x-0.5dx])/2,
#           right avg = (phi[x+0.5dx] + phi[x+1.5dx])/2.
N_HAAR = 4

# ── Scaling exponents ────────────────────────────────────────────────────────
# Horizontal (Kolmogorov):  Delta_phi_h ~ l^(1/3)
# Vertical   (Bolgiano-O):  Delta_phi_v ~ l^(3/5)
H_HORIZ = 1.0 / 3.0
H_VERT  = 3.0 / 5.0
# H_HORIZ = .52
# H_VERT = .85

# Spheroscale exponent: l_s = (phi_v/phi_h)^(1/(H_h - H_v))
#   = (phi_v/phi_h)^(1/(1/3 - 3/5)) = (phi_v/phi_h)^(-15/4)
_LS_EXP = 1.0 / (H_HORIZ - H_VERT)   # = -15/4


def load_data(dataset, variable, experiment=None):
    if dataset == 'SAM_TWPICE':
        from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
        return load_sam_twpice_variable_interpolated(variable)
    elif dataset == 'SAM_RCEMIP':
        from utils.sam_rcemip_loader import load_sam_rcemip_variable_interpolated
        return load_sam_rcemip_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'CM1':
        from utils.cm1_loader import load_cm1_variable_interpolated
        return load_cm1_variable_interpolated(variable, experiment=experiment)
    elif dataset == 'STEAM':
        from utils.steam_loader import load_steam_variable_interpolated
        return load_steam_variable_interpolated(variable)
    elif dataset == 'dropsonde':
        raise ValueError(
            "dropsonde dataset has no horizontal dimension; spheroscale requires both h and v.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def compute_spheroscale_field(data, z, dx, n=N_HAAR):
    """Compute spheroscale at every valid grid location.

    The Haar fluctuation with n points (n/2 per side) centred at boundary x:
        Delta_phi = (1/n) |[phi(x-0.5dx) + ... + phi(x-(n/2-0.5)dx)]
                          - [phi(x+0.5dx) + ... + phi(x+(n/2-0.5)dx)]|

    Horizontal boundaries use periodic wrapping (all nx positions valid).
    Vertical boundaries use mode='valid' (n/2 levels dropped at each end).

    Parameters
    ----------
    data : ndarray, shape (..., nx, ny, nz)
        Field values. Last axis is vertical (z).
    z : 1D array, length nz
        Heights in metres (must be uniformly spaced).
    dx : float
        Horizontal grid spacing in metres.
    n : int
        Haar scale — number of points (even integer). Default N_HAAR.

    Returns
    -------
    l_s_m : ndarray, shape (..., nx, ny, nz-(n-1))
        Spheroscale in metres.
    z_bd : 1D array, length nz-(n-1)
        Heights (m) of the z-boundary midpoints for the last axis of l_s_m.
    """
    half = n // 2
    dz = float(np.median(np.diff(z)))
    nx, ny, nz = data.shape[-3], data.shape[-2], data.shape[-1]

    # ── Horizontal Haar along x (axis -3) — periodic ─────────────────────────
    left_h  = sum(np.roll(data,  half-1-k, axis=-3) for k in range(half))
    right_h = sum(np.roll(data, -(k+1),    axis=-3) for k in range(half))
    delta_h = np.abs(left_h - right_h) / n          # (..., nx, ny, nz)

    # ── Vertical Haar along z (axis -1) — mode='valid' ───────────────────────
    left_v  = sum(data[..., k : nz-(n-1-k)] for k in range(half))
    right_v = sum(data[..., half+k : nz-((half-1)-k)] for k in range(half))
    delta_v = np.abs(left_v - right_v) / n          # (..., nx, ny, nz-(n-1))

    # ── Fluxes ────────────────────────────────────────────────────────────────
    phi_h = delta_h / (n * dx) ** H_HORIZ           # (..., nx, ny, nz)
    phi_v = delta_v / (n * dz) ** H_VERT            # (..., nx, ny, nz-(n-1))

    # ── Align: trim phi_h in z to match phi_v z-boundary positions ───────────
    nzbd = nz - (n - 1)
    phi_h_a = phi_h[..., half-1 : half-1+nzbd]     # (..., nx, ny, nzbd)
    phi_v_a = phi_v                                  # (..., nx, ny, nzbd)

    # ── Spheroscale: l_s = (phi_v / phi_h)^(1/(H_h - H_v)) ──────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(phi_h_a > 0, phi_v_a / phi_h_a, np.nan)
        l_s_m = np.where(ratio > 0, ratio ** _LS_EXP, np.nan)

    z_bd = np.array([(z[half - 1 + j] + z[half + j]) / 2.0 for j in range(nzbd)])

    return l_s_m, z_bd


def load_and_compute(dataset, variable, experiment=None,
                     alt_min=None, alt_max=None, n_haar=N_HAAR):
    """Load a dataset and return flattened (l_s_m, z) arrays ready for plotting."""
    z, data, dx = load_data(dataset, variable, experiment)
    if dx is None:
        raise ValueError(f"Dataset {dataset!r} returns no horizontal spacing.")

    print(f"Loaded {dataset} {variable}, shape: {data.shape}")

    if alt_min is not None or alt_max is not None:
        z_min = alt_min if alt_min is not None else float(z.min())
        z_max = alt_max if alt_max is not None else float(z.max())
        zmask = (z >= z_min) & (z <= z_max)
        data  = data[..., zmask]
        z     = z[zmask]
        print(f"  Altitude subset: {z_min:.0f}–{z_max:.0f} m, shape now: {data.shape}")

    print(f"  Computing spheroscale field (n={n_haar})...")
    l_s_m, z_bd = compute_spheroscale_field(data, z, dx, n=n_haar)

    z_arr    = np.broadcast_to(z_bd, l_s_m.shape).ravel()
    l_s_flat = l_s_m.ravel()
    valid    = np.isfinite(l_s_flat) & (l_s_flat > 0)

    l_s_v = l_s_flat[valid]
    z_v   = z_arr[valid]
    print(f"  Valid points: {valid.sum():,} / {valid.size:,}")
    print(f"  l_s range:    {l_s_v.min():.3g} – {l_s_v.max():.3g} m")
    print(f"  l_s mean:     {l_s_v.mean():.3g} m")

    return l_s_v, z_v, z_bd
