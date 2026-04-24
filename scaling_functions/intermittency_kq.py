import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Configuration ──
DATASET = 'STEAM'               # 'SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM', or 'dropsonde'
EXPERIMENT = 'RCE_large300'     # SAM_RCEMIP/CM1 only: 'RCE_large300' or 'RCE_small_les300'
VARIABLE = 'qt'
# SAM_TWPICE:              QV, QT, QC, QI, MSE, TABS, U, V, W, PP
# SAM_RCEMIP RCE_large300: hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, QV, tntr, tntrs, tntrl
# SAM_RCEMIP RCE_small_les300: U, V, W, PP, QRAD, TABS, QV, QN, QP, LQRAD, SQRAD
# CM1:                     hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, tntr, tntrs, tntrl
# STEAM:                   qv, qt, qc, qi, h, T, p
# dropsonde:               q, ta, theta, rh, u, v, p, wspd, wdir
METHOD = 'structure_function'   # 'haar' or 'structure_function'
DIRECTION = 'x'                 # 'x' (horizontal) or 'z' (vertical)
STEAM_GROUP = 'strips'          # STEAM only: 'parent', 'strips', 'cubes', or explicit path
ALT_MIN = 4000                  # Altitude subsetting in meters (None = use all)
ALT_MAX = 5000
MIN_SEP = 4
MAX_SEP = 64
ORDERS = np.arange(0.25, 3.25, 0.25)

# ── Load data ──
if DATASET == 'SAM_TWPICE':
    from utils.sam_twpice_loader import load_sam_twpice_variable_interpolated
    z, data, dx = load_sam_twpice_variable_interpolated(VARIABLE)
elif DATASET == 'SAM_RCEMIP':
    from utils.sam_rcemip_loader import load_sam_rcemip_variable_interpolated
    z, data, dx = load_sam_rcemip_variable_interpolated(VARIABLE, experiment=EXPERIMENT)
elif DATASET == 'CM1':
    from utils.cm1_loader import load_cm1_variable_interpolated
    z, data, dx = load_cm1_variable_interpolated(VARIABLE, experiment=EXPERIMENT)
elif DATASET == 'STEAM':
    from utils.steam_loader import load_steam_variable_interpolated
    z, data, dx = load_steam_variable_interpolated(VARIABLE, group=STEAM_GROUP)
elif DATASET == 'dropsonde':
    from utils.dropsonde_loader import load_dropsonde_variable_interpolated
    z, data, dx = load_dropsonde_variable_interpolated(VARIABLE)
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

from config import get_unit_label
unit = get_unit_label(VARIABLE)

print(f"Loaded {DATASET} {VARIABLE}, shape: {data.shape}")

# ── Subset altitude ──
if ALT_MIN is not None or ALT_MAX is not None:
    z_min = ALT_MIN if ALT_MIN is not None else z.min()
    z_max = ALT_MAX if ALT_MAX is not None else z.max()
    zmask = (z >= z_min) & (z <= z_max)
    data = data[..., zmask]
    z = z[zmask]
    print(f"  Altitude subset: {z_min:.0f}–{z_max:.0f} m, shape now: {data.shape}")

# ── Select analysis axis ──
if DIRECTION == 'z':
    analysis_axis = -1
elif DIRECTION == 'x':
    analysis_axis = 1
else:
    raise ValueError(f"Unknown direction: {DIRECTION}")

# ── Compute xi(q) for each order ──
from scaleinvariance import structure_function_analysis, haar_fluctuation_analysis, K as K_func
from scipy.optimize import curve_fit

ALPHA = 2  # Assumed Levy stability parameter for K(q) fit

xi = np.zeros(len(ORDERS))
for i, q in enumerate(ORDERS):
    if METHOD == 'structure_function':
        lags, vals = structure_function_analysis(data, order=q, axis=analysis_axis,
                                                  lags='powers of 1.05')
    elif METHOD == 'haar':
        lags, vals = haar_fluctuation_analysis(data, order=q, axis=analysis_axis,
                                                lags='powers of 1.05', nan_behavior='ignore')
    else:
        raise ValueError(f"Unknown method: {METHOD}")

    # Fit log-log slope within [MIN_SEP, MAX_SEP]
    mask = (lags >= MIN_SEP) & (lags <= MAX_SEP) & (vals > 0)
    if mask.sum() < 2:
        xi[i] = np.nan
        continue
    log_lags = np.log(lags[mask])
    log_vals = np.log(vals[mask])
    slope, _ = np.polyfit(log_lags, log_vals, 1)
    xi[i] = slope
    print(f"  q={q:.2f}: xi={slope:.4f}")

# ── Compute K(q) ──
# H(1) = xi(1), so q*H(1) is the linear (monofractal) expectation
idx_1 = np.argmin(np.abs(ORDERS - 1.0))
H1 = xi[idx_1]
Kq = ORDERS * H1 - xi

print(f"\nH(q=1) = {H1:.4f}")

# ── Fit K(q) ──
valid = ~np.isnan(Kq)
popt, _ = curve_fit(lambda q, C1: K_func(q, C1, ALPHA), ORDERS[valid], Kq[valid], p0=0.1)
C1_fit = popt[0]
Kq_fit = K_func(ORDERS, C1_fit, ALPHA)
print(f"Fitted C1 = {C1_fit:.4f} (alpha = {ALPHA})")

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: xi(q) with linear reference
ax1.plot(ORDERS, xi, 'o-', color='#2171b5', lw=1.5, ms=4, label=r'$\xi(q)$')
ax1.plot(ORDERS, ORDERS * H1, '--', color='gray', lw=1, label=f'linear: $qH_1$ (H={H1:.3f})')
ax1.set_xlabel('Order $q$')
ax1.set_ylabel(r'$\xi(q)$')
ax1.set_title(f'{DATASET} {VARIABLE} ({DIRECTION})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: K(q)
ax2.plot(ORDERS, Kq, 'o', color='#cb181d', ms=5, label='Data')
ax2.plot(ORDERS, Kq_fit, '-', color='#cb181d', lw=1.5,
         label=rf'Fit: $C_1$={C1_fit:.3f}, $\alpha$={ALPHA}')
ax2.set_xlabel('Order $q$')
ax2.set_ylabel(r'$K(q)$')
ax2.set_title(r'$K(q) = qH_1 - \xi(q)$')
ax2.axhline(0, color='gray', lw=0.5)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

method_label = 'Haar' if METHOD == 'haar' else 'SF'
alt_str = ''
if ALT_MIN is not None or ALT_MAX is not None:
    lo = f'{(ALT_MIN or z.min())/1e3:.1f}'
    hi = f'{(ALT_MAX or z.max())/1e3:.1f}'
    alt_str = f', {lo}–{hi} km'
fig.suptitle(f'{DATASET} {VARIABLE} Intermittency ({method_label}, {DIRECTION}-direction{alt_str})', y=1.02)
plt.tight_layout()

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'
alt_tag = ''
if ALT_MIN is not None or ALT_MAX is not None:
    alt_tag = f'_{int(ALT_MIN or 0)}-{int(ALT_MAX or 99999)}m'
plt.savefig(FIGURES_DIR / f'Kq_{METHOD}_{DATASET}_{VARIABLE}_{DIRECTION}{alt_tag}.pdf', transparent=True)
plt.show()
