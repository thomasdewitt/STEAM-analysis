import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Configuration ──
DATASET = 'STEAM'               # 'SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM', or 'dropsonde'
EXPERIMENT = 'RCE_large300'         # SAM_RCEMIP/CM1 only: 'RCE_large300' or 'RCE_small_les300'
VARIABLE = 'qt'
# SAM_TWPICE:              QV, QT, QC, QI, MSE, TABS, U, V, W, PP
# SAM_RCEMIP RCE_large300: hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, QV, tntr, tntrs, tntrl
# SAM_RCEMIP RCE_small_les300: U, V, W, PP, QRAD, TABS, QV, QN, QP, LQRAD, SQRAD
# CM1:                     hus, ta, ua, va, wa, pa, clw, cli, plw, pli, hur, tntr, tntrs, tntrl
# STEAM:                   qv, qt, qc, qi, h, T, p
# dropsonde:               q, ta, theta, rh, u, v, p, wspd, wdir
METHOD = 'haar'                # 'haar' or 'structure_function'
LOG_VARIABLE = False
DIFF_BEFORE_ANALYSIS = False    # Take vertical differences before scaling analysis

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

vert_spacing = np.median(np.diff(z))
print(f"Loaded {DATASET} {VARIABLE}, shape: {data.shape}, dz={vert_spacing:.1f}m")

if LOG_VARIABLE:
    data = np.log10(data)

if DIFF_BEFORE_ANALYSIS:
    data = np.diff(data, axis=-1)

# ── Compute scaling for 4D volume and mean profile ──
mean_profile = np.nanmean(data, axis=(0, 1, 2))

if METHOD == 'haar':
    from scaleinvariance import haar_fluctuation_analysis
    lags_4d, vals_4d = haar_fluctuation_analysis(data, axis=3, lags='powers of 1.05', nan_behavior='ignore')
    lags_prof, vals_prof = haar_fluctuation_analysis(mean_profile, axis=0, lags='powers of 1.05', nan_behavior='ignore')
    ylabel = f'Haar Fluctuation ({unit})'
elif METHOD == 'structure_function':
    from scaleinvariance import structure_function_analysis
    lags_4d, vals_4d = structure_function_analysis(data, axis=3, lags='powers of 1.05')
    lags_prof, vals_prof = structure_function_analysis(mean_profile, axis=0, lags='powers of 1.05')
    ylabel = f'Structure Function $S_1(r)$ ({unit})'
else:
    raise ValueError(f"Unknown method: {METHOD}")

# ── Plot ──
scales_4d = lags_4d * vert_spacing / 1e3   # km
scales_prof = lags_prof * vert_spacing / 1e3

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(scales_4d, vals_4d, '-', color='#2171b5', lw=1.5, label='4D Volume')
ax.loglog(scales_prof, vals_prof, '-', color='#cb181d', lw=1.5, label='Mean Profile')

# Reference slopes (in km)
ref_x = np.array([0.4, 5.0])
idx_4d = np.argmin(np.abs(scales_4d - 0.4))
ref_y075 = vals_4d[idx_4d] * 1.5 * (ref_x / ref_x[0])**0.6
ax.loglog(ref_x, ref_y075, '--', color='gray', lw=1, label='slope 0.6')

idx_prof = np.argmin(np.abs(scales_prof - 0.4))
ref_y1 = vals_prof[idx_prof] * 0.7 * (ref_x / ref_x[0])**1.0
ax.loglog(ref_x, ref_y1, ':', color='gray', lw=1, label='slope 1')

method_label = 'Haar' if METHOD == 'haar' else 'Structure Function'
ax.set_xlabel('Scale (km)')
ax.set_ylabel(ylabel)
ax.set_title(f'{DATASET} {VARIABLE} {method_label}: 4D Volume vs Mean Profile')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'
plt.savefig(FIGURES_DIR / f'{METHOD}_{DATASET}_{VARIABLE}_4d_vs_profile.pdf', transparent=True)
plt.show()
