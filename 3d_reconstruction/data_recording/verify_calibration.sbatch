#!/bin/bash
#SBATCH --job-name="verify_calibration"
#SBATCH --partition=L40Sday
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/mthormann/robot_and_record/3d_reconstruction/logs/%J.out
#SBATCH --error=/home/mthormann/robot_and_record/3d_reconstruction/logs/%J.err

set -euo pipefail

PROJECT_DIR=/home/mthormann/robot_and_record/3d_reconstruction
CONTAINER=/home/mthormann/robot_and_record/docker_installation/training_and_reconstruction/meta_and_pytorch.sif

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

EXTRA_ARGS=(
    # --no-plots
    # --max-cols 6
)

singularity exec --nv "$CONTAINER" \
    python3 data_recording/verify_calibration.py \
    --data-dir camera_data \
    "${EXTRA_ARGS[@]}"
