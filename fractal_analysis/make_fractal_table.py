#!/usr/bin/env python3
"""Assemble the cloud-geometry comparison table in LaTeX.

One section per albedo threshold. Within each, the MODIS retrieval sits at
the top, ruled off from the four simulated cases below it (the two SAM LES
and STEAM at both flux amplitudes).

An italic entry is one where objscale warned about the fit -- too few
populated size bins, or too narrow a range of scales. A bold entry is the
simulated case closest to MODIS in that column, along with any case within
0.01 of it -- the table's own display precision, below which a difference
is not something to declare a winner on. Bold is only used where MODIS
reports the metric at all. Every SAM size
distribution earns one, because a single snapshot per case simply does not
carry the range; the dimensions, which are measured per object, do not.
Reading those numbers as measured would be a mistake, and italics say so on
the page rather than in a log.

MODIS is recomputed here rather than quoted. compute_modis_fractal.py runs
the 72 granules behind DeWitt et al. (2026), "Toward less subjective metrics
for quantifying the shape and organization of clouds", ACP 26, 6951-6971,
through the same objscale calls as the simulations, using a loader written
from the file specification rather than ported from that paper's code. Two
things follow. The retrieval and the models now differ only in where the
mask came from, not in how it was measured; and the cloud fraction and area
exponent that paper does not report are available, so no cell is empty.

The published Table 1 is reproduced to a mean absolute difference of 0.011
over the nine cells it reports, which is the check that the pipeline here is
measuring what that paper measured. PUBLISHED below keeps those values for
that comparison; they are not what the table prints.

Usage: python make_fractal_table.py
"""

from pathlib import Path

import numpy as np

from albedo import ALBEDO_THRESHOLDS, threshold_tag

HERE = Path(__file__).resolve().parent
OUT = HERE / "fractal_table.tex"

# metric key -> column header
COLUMNS = (
    ("cover", r"CF"),
    ("D_e", r"$D_e$"),
    ("D_f", r"$D_f$"),
    ("tau_area", r"$\tau_{\mathrm{area}}$"),
    ("tau_per", r"$\tau_{\mathrm{per}}$"),
)

# DeWitt et al. (2026) Table 1, by reflectance threshold. None = not
# reported there. Kept only so main() can print the reproduction check; the
# table prints the recomputed values.
PUBLISHED = {
    0.1: {"cover": None, "D_e": 1.77, "D_f": 1.38,
          "tau_area": None, "tau_per": 1.26},
    0.2: {"cover": None, "D_e": 1.72, "D_f": 1.38,
          "tau_area": None, "tau_per": 1.29},
    0.3: {"cover": None, "D_e": 1.71, "D_f": 1.39,
          "tau_area": None, "tau_per": 1.34},
}

# Which solar-zenith convention the retrieval is read under. False leaves
# the L1B reflectance as stored, rho*cos(theta_0); True divides the cosine
# out. False is the convention of the published table -- the reproduction
# check confirms it, matching to 0.011 against 0.017 corrected -- so it is
# what makes this table continuous with that paper. The argument for True is
# that the model masks are an overhead-sun albedo, and it is not a weak one;
# what settles it in practice is that the exponents barely move either way
# (D_e by 0.03 at most) while cloud cover moves by a factor of two, so the
# comparison the table is making does not rest on the choice.
MODIS_SOLAR_CORRECTION = False

# Entries this close to the best one are bolded alongside it; 0.01 is the
# table's own display precision.
TIE_TOLERANCE = 0.01

# label -> (npz file, key prefix within it)
CASES = (
    ("SAM-GATE", "sam_fractal_metrics.npz", "gate"),
    ("SAM-TWPICE", "sam_fractal_metrics.npz", "twpice"),
    (r"STEAM $c = 0.05$", "fractal_metrics_C1small.npz", None),
    (r"STEAM $c = 0.17$", "fractal_metrics_C1large.npz", None),
)


def cell(value, warned=False, bold=False):
    """One entry: two decimals, italic if flagged, bold if closest to MODIS."""
    if value is None or not np.isfinite(value):
        return "--"
    text = f"{value:.2f}"
    if warned:
        text = rf"\textit{{{text}}}"
    if bold:
        text = rf"\textbf{{{text}}}"
    return text


def modis_file():
    tag = "_sza" if MODIS_SOLAR_CORRECTION else ""
    return f"modis_fractal_metrics{tag}.npz"


def entry_from(data, tag):
    """{metric: (value, warned)} for one threshold of one npz."""
    out = {}
    for key, _ in COLUMNS:
        full = f"{tag}_{key}"
        out[key] = (float(data[full]) if full in data else None,
                    bool(data.get(f"{tag}_warn_{key}", False)))
    return out


def case_values(loaded, R):
    """{label: {metric: (value, warned)}} for every simulated case."""
    out = {}
    for label, filename, prefix in CASES:
        data = loaded[filename]
        tag = (threshold_tag(R) if prefix is None
               else f"{prefix}_{threshold_tag(R)}")
        entry = {}
        for key, _ in COLUMNS:
            full = f"{tag}_{key}"
            value = float(data[full]) if full in data else None
            entry[key] = (value, bool(data.get(f"{tag}_warn_{key}", False)))
        out[label] = entry
    return out


def closest_to_modis(values, reference):
    """{metric: {labels}} of the cases nearest the retrieval.

    Every case within TIE_TOLERANCE of the best one is marked, not just the
    single winner: at two decimals a 0.01 separation is the smallest
    difference the table can even show, and bolding one of two entries that
    differ by that much states a verdict the numbers do not support.

    Distances are taken on the ROUNDED values, so the rule can be checked
    against the printed page rather than against a file the reader does not
    have.

    Every column now has a reference, since the retrieval is recomputed
    rather than quoted and so carries a cloud fraction and an area exponent
    too. A flagged fit can still win and keeps its italics, so the reader
    sees both facts at once.
    """
    best = {}
    for key, _ in COLUMNS:
        target = reference[key][0]
        if target is None or not np.isfinite(target):
            continue
        distances = {label: abs(round(v, 2) - round(target, 2))
                     for label, entry in values.items()
                     for v, _ in [entry[key]]
                     if v is not None and np.isfinite(v)}
        if not distances:
            continue
        nearest = min(distances.values())
        best[key] = {label for label, d in distances.items()
                     if d <= nearest + TIE_TOLERANCE + 1e-9}
    return best


def main():
    loaded = {}
    for _, filename, _ in CASES:
        if filename not in loaded:
            path = HERE / filename
            if not path.exists():
                raise SystemExit(
                    f"{filename} not found -- run compute_fractal_metrics.py "
                    f"(once per set) and compute_sam_fractal.py first")
            loaded[filename] = dict(np.load(path, allow_pickle=False))

    modis_path = HERE / modis_file()
    if not modis_path.exists():
        raise SystemExit(
            f"{modis_path.name} not found -- run compute_modis_fractal.py"
            + (" --sza" if MODIS_SOLAR_CORRECTION else "") + " first")
    modis_data = dict(np.load(modis_path, allow_pickle=False))

    lines = [
        r"% Generated by make_fractal_table.py -- do not edit by hand.",
        r"\begin{table*}[t]",
        r"\caption{Cloud geometry against reflectance threshold $R$. "
        r"$D_e$ is the ensemble (correlation) fractal dimension, $D_f$ the "
        r"individual fractal dimension ($D_i$ in DeWitt et al., 2026), and "
        r"$\tau_{\mathrm{area}}$, $\tau_{\mathrm{per}}$ the area and "
        r"nested-perimeter size-distribution exponents ($\beta$ in that "
        r"paper). CF is cloud fraction. The MODIS row is recomputed here "
        r"from the 72 granules of that study, through the same estimators "
        r"as the simulations, and restricted to sensor zenith angles below "
        r"$60^\circ$: beyond that a pixel is several km across and views "
        r"cloud sides as much as cloud tops, which the per-pixel footprints "
        r"passed to the estimators cannot repair. That restriction is a "
        r"deliberate difference from the published analysis, and is the "
        r"likeliest source of the residual disagreement with it; the nine "
        r"values that paper reports are nonetheless reproduced to a mean "
        r"absolute difference of 0.011. "
        r"Italic entries are fits objscale flagged as resting on too few "
        r"size bins or too narrow a range of scales; bold marks the "
        r"simulated case closest to the retrieval in each column, and any "
        r"case within 0.01 of it.}",
        r"\label{tab:cloud geometry}",
        r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}",
        r"\tophline",
        " & " + " & ".join(head for _, head in COLUMNS) + r" \\",
        r"\middlehline",
    ]

    for i, R in enumerate(ALBEDO_THRESHOLDS):
        if i:
            lines.append(r"\middlehline")
        lines.append(rf"\multicolumn{{{len(COLUMNS) + 1}}}{{l}}"
                     rf"{{\textbf{{$R > {R:g}$}}}} \\")
        modis = entry_from(modis_data, threshold_tag(R))
        lines.append("MODIS & "
                     + " & ".join(cell(*modis[key]) for key, _ in COLUMNS)
                     + r" \\")
        lines.append(rf"\cline{{1-{len(COLUMNS) + 1}}}")
        values = case_values(loaded, R)
        best = closest_to_modis(values, modis)
        for label, _, _ in CASES:
            cells = [cell(*values[label][key], bold=(label in best.get(key, ())))
                     for key, _ in COLUMNS]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomhline", r"\end{tabular}",
              r"\belowtable{}", r"\end{table*}"]

    text = "\n".join(lines) + "\n"
    OUT.write_text(text)
    print(text)

    # The reproduction check. Printed rather than written into the table:
    # it is evidence that the pipeline measures what the published one
    # measured, not a result about clouds.
    deltas = []
    print(f"reproduction of DeWitt et al. (2026) Table 1, from "
          f"{modis_path.name}:")
    for R in ALBEDO_THRESHOLDS:
        entry = entry_from(modis_data, threshold_tag(R))
        for key, head in COLUMNS:
            published = PUBLISHED[R][key]
            if published is None:
                continue
            here = entry[key][0]
            deltas.append(abs(here - published))
            print(f"  R>{R:g}  {head:<22} published {published:.2f}   "
                  f"here {here:.3f}   delta {here - published:+.3f}")
    print(f"  mean |delta| over {len(deltas)} reported values: "
          f"{np.mean(deltas):.3f}")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
