#!/usr/bin/env bash
# Run depth-model evaluation locally inside the training-and-reconstruction image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration: edit paths, labels, examples, and resource-related settings.
# Checkpoint and label arrays must have the same number of entries.
# ---------------------------------------------------------------------------
DATA_DIR="../data/new_2/eval"
COMPARISON_NAME="final_comparison_2"
WORKERS=8
BATCH_SIZE=15

CHECKPOINTS=(
    "checkpoints/unet_table/Single-View_U-Net.pth"
    "checkpoints/unet_table/Multi-View_U-Net.pth"
    "checkpoints/mvs/best_l1_best_run_feat_pyramid_128_32_no_1x1.pth"
)
CHECKPOINT_LABELS=(
    "Single-view U-Net"
    "Multi-view U-Net"
    "MVS-based Model"
)
EXAMPLE_SEQUENCES=(1)
EXAMPLE_FRAMES=(500)

if [[ ${#CHECKPOINTS[@]} -ne ${#CHECKPOINT_LABELS[@]} ]]; then
    echo "CHECKPOINTS and CHECKPOINT_LABELS must contain the same number of entries" >&2
    exit 2
fi
if [[ ${#EXAMPLE_SEQUENCES[@]} -ne ${#EXAMPLE_FRAMES[@]} ]]; then
    echo "EXAMPLE_SEQUENCES and EXAMPLE_FRAMES must contain the same number of entries" >&2
    exit 2
fi

ARGS=(
    --checkpoint "${CHECKPOINTS[@]}"
    --checkpoint_label "${CHECKPOINT_LABELS[@]}"
    --data_dir "$DATA_DIR"
    --comparison_name "$COMPARISON_NAME"
    --workers "$WORKERS"
    --batch_size "$BATCH_SIZE"
    --spatial_mask_offset_max 0.02
    --spatial_mask_offset_steps 9
    --fast_mode 1
    --example_sequence "${EXAMPLE_SEQUENCES[@]}"
    --example_frame "${EXAMPLE_FRAMES[@]}"
    --allow_unbalanced_pose_views
)

exec python3 -u evaluation.py "${ARGS[@]}" "$@"
