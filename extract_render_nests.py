#!/usr/bin/env python3
"""Lift the render-nest groups out of a production square into their own file.

run_render_nests.py writes nests A and B as groups inside the parent
square's .nc. The squares and their strip nests are scratch -- deleted
after the figures are built (Thomas, 2026-07-31) -- but the render nests
are a kept deliverable, so they have to be copied out first.

Writes runs/render_nests_<stem>.nc with render_a and render_b at the top
level, carrying each group's dimensions, variables and attributes plus
the parent's global attributes.

Usage: python extract_render_nests.py parent.nc [parent.nc ...]
"""

import sys
from pathlib import Path

import netCDF4

GROUPS = ("render_a", "render_b")


def copy_group(src, dst):
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    for name, dim in src.dimensions.items():
        dst.createDimension(name, None if dim.isunlimited() else len(dim))
    for name, var in src.variables.items():
        filters = var.filters() or {}
        out = dst.createVariable(
            name, var.dtype, var.dimensions,
            chunksizes=var.chunking() if var.chunking() != "contiguous" else None,
            compression=("blosc_zstd" if filters.get("blosc") else None),
            complevel=1)
        out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        out[...] = var[...]


for arg in sys.argv[1:]:
    parent = Path(arg)
    out_path = parent.parent / f"render_nests_{parent.stem}.nc"
    with netCDF4.Dataset(parent) as src, \
            netCDF4.Dataset(out_path, "w") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        dst.source_parent = parent.name
        for name in GROUPS:
            copy_group(src.groups["refinements"].groups[name],
                       dst.createGroup(name))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)",
          flush=True)
