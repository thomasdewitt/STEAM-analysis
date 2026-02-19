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
METHOD = 'haar'   # 'haar' or 'structure_function'
DIRECTION = 'z'                 # 'x' or 'z'
ALT_MIN = 100                  # Altitude subsetting in meters (None = use all)
ALT_MAX = 6000
ORDERS = [1]
FIT_MIN = 4                     # Fit range in lags (inclusive)
FIT_MAX = 64

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
    z, data, dx = load_steam_variable_interpolated(VARIABLE)
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

vert_spacing = np.median(np.diff(z))

# ── Select analysis axis and spacing ──
if DIRECTION == 'z':
    analysis_axis = -1
    spacing = vert_spacing  # meters per lag
elif DIRECTION == 'x':
    if dx is None:
        raise ValueError("No horizontal spacing available for this dataset")
    analysis_axis = 1
    spacing = dx  # meters per lag
else:
    raise ValueError(f"Unknown direction: {DIRECTION}")

# ── Compute structure functions ──
from scaleinvariance import structure_function_analysis, haar_fluctuation_analysis

colors = ['#4292c6', '#2171b5', '#08519c', '#e6550d', '#a63603', '#cb181d']
H_results = {}

fig, ax = plt.subplots(figsize=(7, 5))

for i, q in enumerate(ORDERS):
    if METHOD == 'structure_function':
        lags, vals = structure_function_analysis(data, order=q, axis=analysis_axis,
                                                  lags='powers of 1.05')
    elif METHOD == 'haar':
        lags, vals = haar_fluctuation_analysis(data, order=q, axis=analysis_axis,
                                                lags='powers of 1.05')
    else:
        raise ValueError(f"Unknown method: {METHOD}")

    scales = lags * spacing / 1e3  # km

    color = colors[i % len(colors)]
    ax.loglog(scales, vals, '-', color=color, lw=1.5, label=f'q={q:.1f}')

    # Fit H over [FIT_MIN, FIT_MAX] and plot dashed fit line
    fitmask = (lags >= FIT_MIN) & (lags <= FIT_MAX) & (vals > 0)
    if fitmask.sum() >= 2:
        log_l = np.log(lags[fitmask])
        log_v = np.log(vals[fitmask])
        slope, intercept = np.polyfit(log_l, log_v, 1)
        H = slope / q if q != 0 else np.nan

        fit_lags = lags[fitmask]
        fit_vals = np.exp(intercept) * fit_lags**slope
        fit_scales = fit_lags * spacing / 1e3
        ax.loglog(fit_scales, fit_vals, '--', color=color, lw=1, alpha=0.7)
        H_results[q] = H
        print(f"  q={q:.1f}: xi={slope:.4f}, H={H:.4f}")

ax.set_xlabel('Scale (km)')
method_label = 'Haar Fluctuation' if METHOD == 'haar' else 'Structure Function'
ax.set_ylabel(f'{method_label} ({unit})')

alt_str = ''
if ALT_MIN is not None or ALT_MAX is not None:
    lo = f'{(ALT_MIN or z.min())/1e3:.1f}'
    hi = f'{(ALT_MAX or z.max())/1e3:.1f}'
    alt_str = f', {lo}–{hi} km'
h1_str = ''
if 1 in H_results:
    h1_str = f', H(1)={H_results[1]:.3f}'
ax.set_title(f'{DATASET} {VARIABLE} {method_label} ({DIRECTION}{alt_str}{h1_str})')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'
alt_tag = ''
if ALT_MIN is not None or ALT_MAX is not None:
    alt_tag = f'_{int(ALT_MIN or 0)}-{int(ALT_MAX or 99999)}m'
plt.savefig(FIGURES_DIR / f'{METHOD}_{DATASET}_{VARIABLE}_{DIRECTION}{alt_tag}_orders.pdf', transparent=True)
plt.show()
