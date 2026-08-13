#!/usr/bin/env bash
set -euo pipefail

RUNS="$HOME/code-and-data/turbulon-analysis/runs/small-domain"
OUT="$HOME/code-and-data/turbulon-analysis/small-domain/figs"

# The two blocks get their own directories because they share a run:
# small_c002_s0030.nc is rendered in both, at different sizes, and witness
# names its output from the file stem alone -- one directory and the second
# render silently replaces the first.
SPHERO_DIR="$OUT/witness_spheroscale"
INTER_DIR="$OUT/witness_intermittency"

# Which set tag goes in which third of the intermittency composite.
# Default is increasing left to right; reorder these three lines to change it.
OVERLAY_LEFT=c002
OVERLAY_MID=c005
OVERLAY_RIGHT=c017

# Width in pixels of the black rule drawn over each seam of that composite.
SEAM_WIDTH=6

mkdir -p "$SPHERO_DIR" "$INTER_DIR"

# Spheroscale comparison
witness "$RUNS/small_c002_s0030.nc" --output "$SPHERO_DIR" --size 1920 700 --group parent --nest-group nest --camera-position 0.82854244016 -0.685508341159 -0.77 --camera-azimuth 59.4 --camera-elevation 10 --fov 100 --sun-azimuth 20 --sun-elevation 55 --periodic --gamma 1.66 --white-point 15 --contrast 1 --haze 1;
witness "$RUNS/small_c002_s0010.nc" --output "$SPHERO_DIR" --size 1920 700 --group parent --nest-group nest --camera-position 0.82854244016 -0.685508341159 -0.77 --camera-azimuth 59.4 --camera-elevation 10 --fov 100 --sun-azimuth 20 --sun-elevation 55 --periodic --gamma 1.66 --white-point 15 --contrast 1 --haze 1;
witness "$RUNS/small_c002_s0100.nc" --output "$SPHERO_DIR" --size 1920 700 --group parent --nest-group nest --camera-position 0.82854244016 -0.685508341159 -0.77 --camera-azimuth 59.4 --camera-elevation 10 --fov 100 --sun-azimuth 20 --sun-elevation 55 --periodic --gamma 1.66 --white-point 15 --contrast 1 --haze 1;
# Intermittency comparison
witness "$RUNS/small_c002_s0030.nc" --output "$INTER_DIR" --size 1920 1080 --group parent --nest-group nest --camera-position 0.853042805498 -0.363875335448 -0.987484355444 --camera-azimuth 79.8 --camera-elevation 30 --fov 100 --sun-azimuth 20 --sun-elevation 55 --periodic --gamma 1.66 --white-point 15 --contrast 1 --haze 1
witness "$RUNS/small_c005_s0030.nc" --output "$INTER_DIR" --size 1920 1080 --group parent --nest-group nest --camera-position 0.853042805498 -0.363875335448 -0.987484355444 --camera-azimuth 79.8 --camera-elevation 30 --fov 100 --sun-azimuth 20 --sun-elevation 55 --periodic --gamma 1.66 --white-point 15 --contrast 1 --haze 1
witness "$RUNS/small_c017_s0030.nc" --output "$INTER_DIR" --size 1920 1080 --group parent --nest-group nest --camera-position 0.853042805498 -0.363875335448 -0.987484355444 --camera-azimuth 79.8 --camera-elevation 30 --fov 100 --sun-azimuth 20 --sun-elevation 55 --periodic --gamma 1.66 --white-point 15 --contrast 1 --haze 1
# Real image comparison
#witness "$RUNS/small_c002_s0030_center_nest.nc" --output "$OUT" --size 1920 1080 --group parent --nest-group nest --camera-position -0.176105149115 -0.00938793029458 -0.987484355444 --camera-azimuth 81.48 --camera-elevation 29.6 --fov 100 --sun-azimuth 287 --sun-elevation 61 --gamma 1.66 --white-point 15 --contrast 1 --haze 1

# --- Composites ---------------------------------------------------------
# Both read the PNGs witness just wrote, and fail on a missing one rather
# than composing whatever happens to be on disk.

require() {
    for f in "$@"; do
        [ -f "$f" ] || { echo "missing render: $f" >&2; exit 1; }
    done
}

# The three spheroscales, stacked with the spheroscale increasing downward.
sphero_stack() {
    local imgs=(
        "$SPHERO_DIR/witness_small_c002_s0010.png"
        "$SPHERO_DIR/witness_small_c002_s0030.png"
        "$SPHERO_DIR/witness_small_c002_s0100.png"
    )
    require "${imgs[@]}"
    magick "${imgs[@]}" -append "$OUT/witness_spheroscale_stack.png"
    echo "  Saved: $OUT/witness_spheroscale_stack.png"
}

# One frame per intermittency, cut into vertical thirds and reassembled:
# left third from OVERLAY_LEFT, middle from OVERLAY_MID, right from
# OVERLAY_RIGHT. The three share a camera, so the strips line up at the seams.
intermittency_thirds() {
    local left="$INTER_DIR/witness_small_${OVERLAY_LEFT}_s0030.png"
    local mid="$INTER_DIR/witness_small_${OVERLAY_MID}_s0030.png"
    local right="$INTER_DIR/witness_small_${OVERLAY_RIGHT}_s0030.png"
    require "$left" "$mid" "$right"

    local w h
    w=$(magick identify -format "%w" "$left")
    h=$(magick identify -format "%h" "$left")

    # Integer thirds; the right strip absorbs the remainder, so the widths
    # always sum back to w.
    local x1=$((w / 3))
    local x2=$((2 * w / 3))

    # The seam lines are drawn over the appended image rather than spliced
    # between the strips, so the composite keeps the width of a single frame
    # and every strip stays at the x it was cut from.
    local half=$((SEAM_WIDTH / 2))

    magick \
        \( "$left"  -crop "${x1}x${h}+0+0"            +repage \) \
        \( "$mid"   -crop "$((x2 - x1))x${h}+${x1}+0" +repage \) \
        \( "$right" -crop "$((w - x2))x${h}+${x2}+0"  +repage \) \
        +append \
        -fill black -stroke none \
        -draw "rectangle $((x1 - half)),0 $((x1 - half + SEAM_WIDTH - 1)),$((h - 1))" \
        -draw "rectangle $((x2 - half)),0 $((x2 - half + SEAM_WIDTH - 1)),$((h - 1))" \
        "$OUT/witness_intermittency_thirds.png"
    echo "  Saved: $OUT/witness_intermittency_thirds.png"
}

sphero_stack
intermittency_thirds
