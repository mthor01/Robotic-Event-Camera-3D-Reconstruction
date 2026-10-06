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

# The values below are the defaults of train_mvs.py, i.e. the configuration
# of the thesis model; they are listed here so they are easy to change.
ARGS=(
    --name "$RUN_NAME"
    --data_dir "$DATA_DIR"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --allow_fewer_pose_views

    --feature_channels 128
    --cost_channels 64
    --reference_channels 16
    --coarse_hourglass_levels 3
    --middle_hourglass_levels 2
    --fine_hourglass_levels 1
    --refiner_channels 64
    --refiner_max_residual_m 0.01

    --fpn_dropout 0.10
    --reference_dropout 0.10
    --hourglass_dropout 0.10
    --drop_path_rate 0.10

    --coarse_depths 32
    --middle_depths 16
    --fine_depths 8
    --fine_window_min 0.02
    --fine_window_max 0.08

    --num_views 9
    --pose_move_threshold 0.05

    --uncertainty
    --confidence_abs_tolerance 0.01
    --confidence_rel_tolerance 0.0

    --lambda_normal 0.1
    --lambda_grad 0.1

    --lr 0.0001
    --weight_decay 0.0005
    --min_lr 0.000001
    --epochs 50
    --ema_decay 0.9995
)

# Extra command-line arguments are appended, for example:
#   ./train_mvs.sh --epochs 5 --name local_smoke_test
exec python3 -u train_mvs.py "${ARGS[@]}" "$@"
