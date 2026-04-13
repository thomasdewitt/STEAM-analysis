"""Run STEAM simulations and write output to STEAM/data/.

Edit the parameters block below, then:
    python STEAM/steam_simulate.py
or run the full pipeline:
    bash STEAM/run_steam.sh
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, '/Users/thomas/code-and-data/turbulon-model')

import xarray as xr
from steam import simulate, compute_diagnostics

# ── Simulation parameters ──────────────────────────────────────────────────────
PROFILE_DATASET  = 'Dropsonde_extrap'   # one of: CM1_small, ICON_small, SAM_large, SAM_small, Dropsonde, Dropsonde_extrap
NSEEDS           = 2

# Domain
NX, NY           = 512,512      # horizontal grid cells
DX, DY           = 5000.0, 5000.0  # m
OUTER_SCALE      = DX * 256       # m  (must be power-of-2 × DX)
SPHEROSCALE      = 10.0        # m
DOMAIN_HEIGHT    = 20000.0       # m
SPARSITY_FACTORS = (1,1,1)
SURFACE_PRESSURE = 101325.0      # Pa
max_h=400 * 1004
min_h=250 * 1004
min_qt=0
max_qt=30/1000
n_size_classes = 30
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR  = Path(__file__).resolve().parent / 'data'
PROFILES  = Path(__file__).resolve().parent.parent / 'data' / 'rcemip_profiles.nc'

print(f'Horizontal outer scale: {OUTER_SCALE/1000:.0f}km')
print(f'Vertical outer scale:   {SPHEROSCALE * (OUTER_SCALE/SPHEROSCALE)**.555/1000:.1f}km')
print(f'Domain width:           {NX*DX/1000:.0f}km')
# print(f'Final nz:               {DOMAIN_HEIGHT / (2 * SPHEROSCALE * (DX/SPHEROSCALE)**.555):.0f}')

def main():
    # Load mean profiles
    ds = xr.open_dataset(PROFILES)
    prof = ds.sel(dataset=PROFILE_DATASET)
    h_profile  = prof['mse'].values   # J/kg
    qt_profile = prof['qt'].values    # kg/kg

    heights = ds['height'].values
    profile_dz = float(heights[1] - heights[0])

    mask = heights <= DOMAIN_HEIGHT
    heights    = heights[mask]
    h_profile  = h_profile[mask]
    qt_profile = qt_profile[mask]
    # import numpy as np
    # h_profile = 350e3 - 20e3 * (np.arange(len(h_profile))*(profile_dz) / DOMAIN_HEIGHT)

    for i in range(NSEEDS):
        seed = 1 + i
        out_path = DATA_DIR / f'steam_{PROFILE_DATASET}_seed_{seed:03d}.nc'
        print(f'\n=== Seed {seed} → {out_path} ===')
        simulate(
            h_profile=h_profile,
            qt_profile=qt_profile,
            nx=NX, ny=NY,
            dx=DX, dy=DY,
            outer_scale=OUTER_SCALE,
            spheroscale=SPHEROSCALE,
            domain_height=DOMAIN_HEIGHT,
            profile_dz=profile_dz,
            output_path=out_path,
            sparsity_factors=SPARSITY_FACTORS,
            surface_pressure=SURFACE_PRESSURE,
            seed=seed,
            h_max=max_h,
            h_min=min_h,
            qt_min=min_qt,
            qt_max=max_qt,
            n_size_classes=n_size_classes

        )
        compute_diagnostics(out_path)
        print(f'Done: {out_path}')


if __name__ == '__main__':
    main()
