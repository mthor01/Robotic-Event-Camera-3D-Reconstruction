#!/usr/bin/env bash
#SBATCH --job-name="extract_dataset"
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/mthormann/robot_and_record/logs/%x_%j.out
#SBATCH --error=/home/mthormann/robot_and_record/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/home/mthormann/robot_and_record
cd "$PROJECT_DIR"

# Configuration: edit these paths or append command-line overrides to sbatch.
SOURCE="data/new_2"
DATASET_OUTPUT="data/dataset"
CORE_OUTPUT="data/core_dataset"

python3 data_precomputation/extract_dataset.py \
    --source "$SOURCE" \
    --dataset-output "$DATASET_OUTPUT" \
    --core-output "$CORE_OUTPUT" \
    "$@"
