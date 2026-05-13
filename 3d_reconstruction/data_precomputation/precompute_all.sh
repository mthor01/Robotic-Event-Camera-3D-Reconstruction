#!/usr/bin/env bash
# Run the full precompute pipeline in order:
#   1. project_realsense_to_event.py
#   2. precompute_spatial_mask.py
#   3. precompute_voxels.py
#   4. precompute_pose_depth.py
#
# Usage:
#   ./precompute_all.sh                        # process all objects in data_root
#   ./precompute_all.sh --data_dir data/real/bottle
#   ./precompute_all.sh --data_dir data/real/bottle data/real/cube
#   ./precompute_all.sh --data_root data/real  # (default)

set -e

# ---- Parse arguments -------------------------------------------------------
DATA_ARGS=()   # will contain either --data_root ... or --data_dir ...

i=1
while [[ $i -le $# ]]; do
    arg="${!i}"
    if [[ "$arg" == "--data_dir" ]]; then
        DATA_ARGS+=("--data_dir")
        i=$((i + 1))
        # Collect all values until the next flag or end
        while [[ $i -le $# && "${!i}" != --* ]]; do
            DATA_ARGS+=("${!i}")
            i=$((i + 1))
        done
    elif [[ "$arg" == "--data_root" ]]; then
        i=$((i + 1))
        DATA_ARGS+=("--data_root" "${!i}")
        i=$((i + 1))
    else
        echo "Unknown argument: $arg"
        echo "Usage: $0 [--data_dir DIR [DIR ...] | --data_root DIR]"
        exit 1
    fi
done

# Default to data/real if nothing was specified
if [[ ${#DATA_ARGS[@]} -eq 0 ]]; then
    DATA_ARGS=("--data_root" "data/real")
fi

# Change to the 3d_reconstruction root so that relative data paths (data/real/...)
# resolve correctly.  The individual scripts are invoked by sub-path.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================================"
echo " precompute_all.sh"
echo " Data args: ${DATA_ARGS[*]}"
echo "============================================================"

run_step() {
    local script="$1"
    shift
    echo ""
    echo "------------------------------------------------------------"
    echo " Running: python3 data_precomputation/$script --overwrite ${DATA_ARGS[*]} $*"
    echo "------------------------------------------------------------"
    python3 "data_precomputation/$script" --overwrite "${DATA_ARGS[@]}" "$@"
}

run_step project_realsense_to_event.py
run_step precompute_spatial_mask.py
run_step precompute_voxels.py

echo ""
echo "============================================================"
echo " All precompute steps finished successfully."
echo "============================================================"
