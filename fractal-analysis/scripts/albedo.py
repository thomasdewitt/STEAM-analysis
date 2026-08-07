#!/usr/bin/env python3
"""Two-stream visual albedo, and the cloud thresholds taken from it.

The convention is cloudyview's, not a second one invented here: the
conservative-scattering two-stream reflectance for an overhead sun,

    A = tau / (tau + 2/(1 - g)),    g = 0.85

(`cloudyview/glimpse.py`, `TWO_STREAM_G`). Unlike beam opacity
1 - exp(-tau), which saturates by tau ~ 4, this keeps contrast between
cirrus and deep cores, which is why the renders use it.

Masks are defined on albedo rather than optical depth so the exponents are
directly comparable to the satellite retrievals of DeWitt et al. (2026),
which threshold reflectance.

A is monotonic in tau, so `A > R` and `tau > tau_for_albedo(R)` select
exactly the same cells. The code thresholds the stored optical depth for
that reason -- it is the same mask without materializing a second field.
"""

TWO_STREAM_G = 0.85
_DENOM = 2.0 / (1.0 - TWO_STREAM_G)        # 13.33 at g = 0.85

# The paper's three cloud thresholds, matching the observational set.
ALBEDO_THRESHOLDS = (0.1, 0.2, 0.3)


def albedo(tau):
    """Two-stream visual albedo of a column optical depth."""
    return tau / (tau + _DENOM)


def tau_for_albedo(R):
    """Column optical depth at which the two-stream albedo reaches R."""
    return _DENOM * R / (1.0 - R)


def threshold_tag(R):
    """Stable key/label fragment for one threshold, e.g. 0.2 -> 'R020'."""
    return f"R{round(R * 100):03d}"
