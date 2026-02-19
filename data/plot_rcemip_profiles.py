#!/usr/bin/env python3
"""
Plot mean MSE and qt profiles from rcemip_profiles.nc.

Two figures, each with two subplots:
    left  – full profile (0–33 km)
    right – lower-troposphere zoom (0–5 km)

Saved to Figures/rcemip_profiles_{mse,qt}.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

HERE   = Path(__file__).resolve().parent
FIGS   = HERE.parent / 'Figures'
NCFILE = HERE / 'rcemip_profiles.nc'

# ── Colors (following project conventions) ────────────────────────────────────
COLORS = {
    'CM1_small':  '#238b45',   # green  – matches existing CM1 color
    'ICON_small': '#2171b5',   # blue
    'SAM_large':  '#d94801',   # orange
    'SAM_small':  '#6a51a3',   # purple – matches existing SAM_RCEMIP color
    'Dropsonde':  '#252525',   # near-black (observational reference)
}

LABELS = {
    'CM1_small':  'CM1 (small)',
    'ICON_small': 'ICON (small)',
    'SAM_large':  'SAM (large)',
    'SAM_small':  'SAM (small)',
    'Dropsonde':  'Dropsonde',
}

ZOOM_TOP = 5.0   # km


def _plot_var(ds, varname, xlabel_full, xlabel_zoom, xscale_full, xscale_zoom, fname):
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=False)
    ax_full, ax_zoom = axes

    z_km = ds['height'].values / 1e3

    for name in ds['dataset'].values:
        profile = ds[varname].sel(dataset=name).values * (xscale_full if True else 1)
        profile_scaled_full = ds[varname].sel(dataset=name).values * xscale_full
        profile_scaled_zoom = ds[varname].sel(dataset=name).values * xscale_zoom

        color = COLORS[name]
        label = LABELS[name]
        lw    = 2.2 if name == 'Dropsonde' else 1.8
        ls    = '--' if name == 'Dropsonde' else '-'

        # full profile
        mask_full = z_km <= 33.5
        ax_full.plot(profile_scaled_full[mask_full], z_km[mask_full],
                     color=color, lw=lw, ls=ls, label=label)

        # lower-trop zoom
        mask_zoom = z_km <= ZOOM_TOP
        ax_zoom.plot(profile_scaled_zoom[mask_zoom], z_km[mask_zoom],
                     color=color, lw=lw, ls=ls, label=label)

    for ax in axes:
        ax.set_ylabel('Height (km)')
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)

    ax_full.set_xlabel(xlabel_full)
    ax_full.set_title('Full profile')
    ax_full.set_ylim(0, 33)

    ax_zoom.set_xlabel(xlabel_zoom)
    ax_zoom.set_title(f'Lower troposphere (0–{ZOOM_TOP:.0f} km)')
    ax_zoom.set_ylim(0, ZOOM_TOP)
    ax_zoom.legend(fontsize=8, framealpha=0.7, loc='upper right')

    fig.tight_layout()
    out = FIGS / fname
    fig.savefig(out, transparent=True)
    print(f'Saved → {out}')
    plt.close(fig)


def main():
    ds = xr.open_dataset(NCFILE)

    # MSE: display in kJ/kg
    _plot_var(
        ds, 'mse',
        xlabel_full='MSE (kJ kg⁻¹)',
        xlabel_zoom='MSE (kJ kg⁻¹)',
        xscale_full=1e-3,
        xscale_zoom=1e-3,
        fname='rcemip_profiles_mse.pdf',
    )

    # qt: display in g/kg
    _plot_var(
        ds, 'qt',
        xlabel_full='qt (g kg⁻¹)',
        xlabel_zoom='qt (g kg⁻¹)',
        xscale_full=1e3,
        xscale_zoom=1e3,
        fname='rcemip_profiles_qt.pdf',
    )

    ds.close()


if __name__ == '__main__':
    main()
