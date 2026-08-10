#!/usr/bin/env python3
"""Three albedo fields side by side: SAM-TWPICE, SAM-GATE and one STEAM square.

The picture behind the fractal metrics -- the same two-stream visual albedo
the masks are cut from (albedo.py), drawn as a continuous field rather than
thresholded, so the reader sees the cloud scenes the exponents are measured
on. Nothing is drawn but the three fields: no axes, no frames, no colorbar,
no titles. Panel order is TWPICE, GATE, STEAM.

The two SAM cases are the same snapshots compute_sam_fractal.py reads, at
native 2048^2 x 100 m; the STEAM panel is the first member of the c017
square campaign, 2048^2 at 1 km. The domains differ by a factor of ten in
extent and the panels are not to a common scale -- each is its own field,
whole.

Computing the SAM optical depths means reading two 2048^2 x 256 condensate
fields per case, so the three albedo planes are cached in
output/albedo_fields.npz. Pass --refresh to rebuild it.

Output is a rasterized PDF: the images are ~2048 px across at 600 dpi, so
the panels are near their native resolution and the file stays small.

Usage: python plot_albedo_fields.py [--refresh]
"""

import sys
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from albedo import albedo

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                 # fractal-analysis/
REPO = BASE.parent
OUTPUT = BASE / "output"
FIGS = BASE / "figs"
CACHE = OUTPUT / "albedo_fields.npz"
OUT = FIGS / "albedo_fields"

# The STEAM panel: first seed of the c017 square campaign.
STEAM_RUN = REPO / "runs" / "square" / "sq1km_c017_m00.nc"

PANELS = ("twpice", "gate", "steam")

DPI = 600
PANEL_IN = 3.4                     # 3.4 in x 600 dpi = 2040 px, ~native
GAP_IN = 0.10

# Deep ocean blue for clear sky through to white at albedo 1. Perceptually
# monotone in lightness, so the field reads as cloud over water rather than
# as a two-colour map.
CLOUD_CMAP = LinearSegmentedColormap.from_list("cloud", [
    (0.00, "#061a3c"),
    (0.18, "#123f74"),
    (0.38, "#2f74ac"),
    (0.60, "#79b0d6"),
    (0.80, "#c6ddee"),
    (1.00, "#ffffff"),
])


def steam_tau():
    """Stored column optical depth of the first c017 member."""
    if not STEAM_RUN.exists():
        raise SystemExit(f"{STEAM_RUN} not found -- run the square campaign")
    with netCDF4.Dataset(STEAM_RUN) as ds:
        ds.set_auto_mask(False)
        return np.asarray(ds.groups["parent"].variables["tau"][:],
                          dtype=np.float32)


def build():
    """The three albedo planes, computed from scratch."""
    # Imported here rather than at module scope: it loads cloudyview by path
    # and pulls in objscale, neither of which the plotting needs on a cache
    # hit.
    from compute_sam_fractal import twpice_tau, gate_tau

    fields = {}
    for name, tau_of in (("twpice", twpice_tau), ("gate", gate_tau),
                         ("steam", steam_tau)):
        print(f"  {name} ...", flush=True)
        tau = np.asarray(tau_of(), dtype=np.float64)
        if not np.all(np.isfinite(tau)):
            raise SystemExit(f"non-finite tau for {name}")
        fields[name] = albedo(tau).astype(np.float32)
        print(f"    {fields[name].shape}, mean albedo "
              f"{fields[name].mean():.3f}", flush=True)
    return fields


def load(refresh=False):
    if not refresh and CACHE.exists():
        d = np.load(CACHE)
        if all(k in d for k in PANELS):
            return {k: d[k] for k in PANELS}
        print(f"{CACHE.name} is missing a panel -- rebuilding", flush=True)
    fields = build()
    OUTPUT.mkdir(exist_ok=True)
    np.savez(CACHE, **fields)
    print(f"cached {CACHE.name}", flush=True)
    return fields


def main():
    refresh = "--refresh" in sys.argv[1:]
    fields = load(refresh=refresh)

    width = 3 * PANEL_IN + 2 * GAP_IN
    fig = plt.figure(figsize=(width, PANEL_IN))
    for i, name in enumerate(PANELS):
        left = i * (PANEL_IN + GAP_IN) / width
        ax = fig.add_axes([left, 0.0, PANEL_IN / width, 1.0])
        ax.imshow(fields[name], origin="lower", cmap=CLOUD_CMAP,
                  vmin=0.0, vmax=1.0, interpolation="nearest",
                  rasterized=True)
        ax.set_axis_off()

    FIGS.mkdir(exist_ok=True)
    fig.savefig(f"{OUT}.pdf", dpi=DPI)
    fig.savefig(f"{OUT}.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {OUT.name}.pdf and {OUT.name}.png")


if __name__ == "__main__":
    main()
