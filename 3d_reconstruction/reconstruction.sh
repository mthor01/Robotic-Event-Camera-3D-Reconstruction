#!/usr/bin/env bash
# Run TSDF reconstruction locally inside the training-and-reconstruction image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration: edit dataset, checkpoint, TSDF, and output settings here.
# ---------------------------------------------------------------------------
DATA_DIR="data/new_2/eval"
CHECKPOINTS=(
    "training/checkpoints/unet_table/Single-View_U-Net.pth"
    "training/checkpoints/unet_table/Multi-View_U-Net.pth"
    "training/checkpoints/multiview/best_l1_best_run_feat_pyramid_128_32_no_1x1.pth"
)

ARGS=(
    --data_dir "$DATA_DIR"
    --checkpoint "${CHECKPOINTS[@]}"
    --mesh_frame_count 200
    --tsdf_confidence_levels 8
    --tsdf_min_confidence 0.2
    --uncertainty_weighted_tsdf
    --compare_uncertainty_tsdf
    --allow_fewer_pose_views
    --save_largest_connected_surface
)

exec python3 -u reconstruction.py "${ARGS[@]}" "$@"
