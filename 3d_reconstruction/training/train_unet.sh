#!/usr/bin/env bash
# Run U-Net training locally inside the training-and-reconstruction image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration: edit these values for your machine, dataset, and experiment.
# ---------------------------------------------------------------------------
RUN_NAME="baseline_9_views_final"
DATA_DIR="../data/new_2"
BATCH_SIZE=14
WORKERS=8

ARGS=(
    --name "$RUN_NAME"
    --data_dir "$DATA_DIR"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --model_scale 2.0
    --num_views 9
    --pose_view_selection
    --pose_move_threshold 0.05
    --pose_channels
    --lr 1e-4
    --weight_decay 0.0005
    --ema_decay 0.9995
    --lambda_grad 0.1
    --lambda_normal 0.1
    --allow_fewer_pose_views
    --allow_unbalanced_pose_views
)

exec python3 -u train_unet.py "${ARGS[@]}" "$@"
