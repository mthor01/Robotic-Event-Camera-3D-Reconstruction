#!/bin/bash
#SBATCH --job-name="voxel_activity_speed"
#SBATCH --partition=day
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/mthormann/robot_and_record/3d_reconstruction/viz_and_tests/logs/%x_%j.out
#SBATCH --error=/home/mthormann/robot_and_record/3d_reconstruction/viz_and_tests/logs/%x_%j.err

set -euo pipefail

REPOSITORY_ROOT="/home/mthormann/robot_and_record"
SCRIPT_DIR="${REPOSITORY_ROOT}/3d_reconstruction/viz_and_tests"
IMAGE="${REPOSITORY_ROOT}/docker_installation/training_and_reconstruction/meta_and_pytorch.sif"

mkdir -p "${SCRIPT_DIR}/logs"

# Plot configuration. Change --sequence here before submitting the job.
ARGS=(
    --sequence data/new_2/eval/20
    --smooth-frames 3
    # --output viz_and_tests/plots/voxel_activity_ee_speed.png
)

cd "${REPOSITORY_ROOT}/3d_reconstruction"
singularity exec "${IMAGE}" \
    python3 "${SCRIPT_DIR}/plot_voxel_activity_ee_speed.py" "${ARGS[@]}" "$@"
