#!/usr/bin/env bash
# Run the full precompute pipeline in order:
#   1. project_realsense_to_event.py
#   2. precompute_table_plane.py
#   3. precompute_voxels.py
#
# Each step writes its outputs into the sequence directory and overwrites
# existing ones (--overwrite is always passed).
#
# Usage (relative paths are relative to the repository root):
#   ./precompute_all.sh                                          # every sequence below data/Event_and_Depth (default)
#   ./precompute_all.sh --data_dir data/Event_and_Depth/eval/20  # one sequence
#   ./precompute_all.sh --data_dir data/Event_and_Depth/train/21 data/Event_and_Depth/eval/20
#   ./precompute_all.sh --data_root data/my_dataset              # every sequence below another root
#
# Per-step flags:
#   Use --project, --table, --voxel as section markers.
#   Flags before any marker are forwarded to ALL steps, so they must be
#   accepted by all three scripts.
#   Flags after a marker are forwarded only to that step.
#   Run a script with --help to list its flags.
#
#   Examples:
#     # Store voxels as float16, like the published dataset
#     ./precompute_all.sh --voxel --float16
#
#     ./precompute_all.sh --data_dir data/Event_and_Depth/eval/20 \
#         --project --no_rgb --workers 8 \
#         --table   --debug \
#         --voxel   --float16 --workers 2

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

# Default to data/Event_and_Depth if nothing was specified
if [[ ${#DATA_ARGS[@]} -eq 0 ]]; then
    DATA_ARGS=("--data_root" "data/Event_and_Depth")
fi

# Change to the repository root so that relative data paths (data/...)
# resolve correctly.  The individual scripts are invoked by sub-path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
