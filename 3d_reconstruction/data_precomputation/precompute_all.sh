#!/usr/bin/env bash
# Run the full precompute pipeline in order:
#   1. project_realsense_to_event.py
#   2. precompute_table_plane.py
#   3. precompute_voxels.py
#
# Usage:
#   ./precompute_all.sh                                         # all objects, default settings
#   ./precompute_all.sh --data_dir data/real/bottle             # single object
#   ./precompute_all.sh --data_dir data/real/bottle data/real/cube
#   ./precompute_all.sh --data_root data/real                   # explicit root (default)
#   ./precompute_all.sh --data_root data/real --crop_then_resize # 960x720 crop -> 320x240
#
# Per-step flags:
#   Use --project, --table, --voxel as section markers.
#   Flags before any marker are forwarded to ALL steps.
#   Flags after a marker are forwarded only to that step.
#
#   Examples:
#     ./precompute_all.sh --data_dir data/real/bottle \
#         --voxel --num_bins 7 --float16
#
#     ./precompute_all.sh \
#         --project --resize_h 260 --resize_w 346 \
#         --table  --table_z 0.02 \
#         --voxel  --num_bins 7 --workers 2

set -euo pipefail

# ---- Parse arguments -------------------------------------------------------
DATA_ARGS=()      # --data_root / --data_dir  (shared, not forwarded to scripts directly)
ALL_ARGS=()       # forwarded to every script
PROJECT_ARGS=()   # forwarded only to project_realsense_to_event.py
TABLE_ARGS=()     # forwarded only to precompute_table_plane.py
VOXEL_ARGS=()     # forwarded only to precompute_voxels.py

# current_target tracks which array receives the next argument
current_target="ALL"

i=1
while [[ $i -le $# ]]; do
    arg="${!i}"
    case "$arg" in
        --data_dir)
            current_target="ALL"  # reset section on data args
            DATA_ARGS+=("--data_dir")
            i=$((i + 1))
            while [[ $i -le $# && "${!i}" != --* ]]; do
                DATA_ARGS+=("${!i}")
                i=$((i + 1))
            done
            ;;
        --data_root)
            current_target="ALL"
            i=$((i + 1))
            DATA_ARGS+=("--data_root" "${!i}")
            i=$((i + 1))
            ;;
        --project)
            current_target="PROJECT"
            i=$((i + 1))
            ;;
        --table)
            current_target="TABLE"
            i=$((i + 1))
            ;;
        --voxel)
            current_target="VOXEL"
            i=$((i + 1))
            ;;
        *)
            case "$current_target" in
                PROJECT) PROJECT_ARGS+=("$arg") ;;
                TABLE)   TABLE_ARGS+=("$arg") ;;
                VOXEL)   VOXEL_ARGS+=("$arg") ;;
                *)       ALL_ARGS+=("$arg") ;;
            esac
            i=$((i + 1))
            ;;
    esac
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
echo " Data args:    ${DATA_ARGS[*]}"
[[ ${#ALL_ARGS[@]}     -gt 0 ]] && echo " All steps:    ${ALL_ARGS[*]}"
[[ ${#PROJECT_ARGS[@]} -gt 0 ]] && echo " --project:    ${PROJECT_ARGS[*]}"
[[ ${#TABLE_ARGS[@]}   -gt 0 ]] && echo " --table:      ${TABLE_ARGS[*]}"
[[ ${#VOXEL_ARGS[@]}   -gt 0 ]] && echo " --voxel:      ${VOXEL_ARGS[*]}"
echo "============================================================"

run_step() {
    local script="$1"
    shift
    # "$@" contains the per-step extra args passed by the caller
    echo ""
    echo "------------------------------------------------------------"
    echo " Running: python3 data_precomputation/$script --overwrite ${DATA_ARGS[*]} ${ALL_ARGS[*]} $*"
    echo "------------------------------------------------------------"
    python3 "data_precomputation/$script" --overwrite "${DATA_ARGS[@]}" "${ALL_ARGS[@]}" "$@"
}

run_step project_realsense_to_event.py "${PROJECT_ARGS[@]}"
run_step precompute_table_plane.py     "${TABLE_ARGS[@]}"
run_step precompute_voxels.py          "${VOXEL_ARGS[@]}"

echo ""
echo "============================================================"
echo " All precompute steps finished successfully."
echo "============================================================"
