#!/usr/bin/env bash
# Run MVS training locally inside the training-and-reconstruction image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration: edit these values for your machine, dataset, and experiment.
# ---------------------------------------------------------------------------
RUN_NAME="after_master_test"
DATA_DIR="../data/new_2"
BATCH_SIZE=14
WORKERS=8

ARGS=(
    --name "$RUN_NAME"
    --data_dir "$DATA_DIR"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --allow_fewer_pose_views
    --allow_unbalanced_pose_views

    --feature_channels 128
    --middle_feature_channels 64
    --fine_feature_channels 32

    --fpn_dropout 0.10
    --reference_dropout 0.10
    --hourglass_dropout 0.10
    --drop_path_rate 0.10

    --cost_channels 64
    --reference_channels 16
    --coarse_hourglass_levels 3
    --middle_hourglass_levels 2
    --fine_hourglass_levels 1

    --coarse_depths 32
    --middle_depths 16
    --fine_depths 8
    --fine_window_min 0.02
    --fine_window_max 0.08
    --learned_fine_window

    --num_views 9
    --pose_view_selection
    --pose_move_threshold 0.05
    --no_pose_noise

    --refiner_channels 64
    --refiner_max_residual_m 0.02

    --uncertainty
    --confidence_abs_tolerance 0.01
    --confidence_rel_tolerance 0.0

    --lambda_normal 0.1
    --lambda_grad 0.1

    --optimizer adamw
    --lr 0.0001
    --weight_decay 0.0005
    --lr_scheduler cosine
    --min_lr 0.000001
    --warmup_epochs 0
    --epochs 50
    --ema_decay 0.9995
)

# Extra command-line arguments are appended, for example:
#   ./train_mvs.sh --epochs 5 --name local_smoke_test
exec python3 -u train_mvs.py "${ARGS[@]}" "$@"
