#!/usr/bin/env bash
# STEAM end-to-end pipeline.
# Edit STEAM/steam_simulate.py to change simulation parameters.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source ~/miniforge3/etc/profile.d/conda.sh

rm -f "$REPO/STEAM/data/"*.nc
# 1. Run simulation (all seeds, all config in steam_simulate.py)
conda activate main
python "$REPO/STEAM/steam_simulate.py"

# 2. Glimpse the most recently written output file
LAST_FILE=$(ls -t "$REPO/STEAM/data/"steam_*.nc 2>/dev/null | head -1)
if [ -n "$LAST_FILE" ]; then
    conda activate cloud-vis
    glimpse "$LAST_FILE" --output '/Users/thomas/code-and-data/turbulon-analysis/Figures'
    conda activate main
fi

# 3. Plot profiles for all output files in STEAM/data/
DATA_DIR="$REPO/STEAM/data"
PATTERN="steam_*.nc"
for VAR in h qt cloud_fraction; do
    python "$REPO/profiles/plot_profiles.py" \
        --dataset STEAM \
        --variable "$VAR" \
        --steam_data_dir "$DATA_DIR" \
        --steam_file_pattern "$PATTERN" \
        --no_show
done

python /Users/thomas/code-and-data/turbulon-analysis/scaling_functions/plot_structure_functions.py --dataset STEAM --alt_min 3000 --alt_max 12000 --fit_min 4 --fit_max 32 --method haar --variable h --no_show
python /Users/thomas/code-and-data/turbulon-analysis/scaling_functions/plot_structure_functions.py --dataset STEAM --alt_min 3000 --alt_max 12000 --fit_min 4 --fit_max 32 --method haar --variable qt --no_show
python /Users/thomas/code-and-data/turbulon-analysis/scaling_functions/compare_scaling_4d_vs_profile.py --dataset STEAM --variable h --no_show
python /Users/thomas/code-and-data/turbulon-analysis/scaling_functions/compare_scaling_4d_vs_profile.py --dataset STEAM --variable qt --no_show

echo "Done — figures in $REPO/Figures/"
