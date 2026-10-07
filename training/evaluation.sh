#!/usr/bin/env bash
# Run depth-model evaluation locally inside the training-and-reconstruction image.
# Results are written to training/evaluation_results/. Append --fast_mode N to
# evaluate only every N-th frame for a quick check.

set -euo pipefail

# Relative paths below and on the command line are relative to training/,
# where this script lives.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration: edit paths, examples, and resource-related settings.
# ---------------------------------------------------------------------------
DATA_DIR="../data/eval"
CHECKPOINT="checkpoints/mvs/MVS.pth"
WORKERS=8
BATCH_SIZE=15

# Frames shown in selected_frames_overview.png, paired by position. A sequence
# is given by its directory name (e.g. 20) or by its displayed index
# (1 = first sequence in sorted order); frames are original frame indices.
EXAMPLE_SEQUENCES=(1)
EXAMPLE_FRAMES=(500)

if [[ ${#EXAMPLE_SEQUENCES[@]} -ne ${#EXAMPLE_FRAMES[@]} ]]; then
    echo "EXAMPLE_SEQUENCES and EXAMPLE_FRAMES must contain the same number of entries" >&2
    exit 2
fi

ARGS=(
    --checkpoint "$CHECKPOINT"
    --data_dir "$DATA_DIR"
    --workers "$WORKERS"
    --batch_size "$BATCH_SIZE"
    --example_sequence "${EXAMPLE_SEQUENCES[@]}"
    --example_frame "${EXAMPLE_FRAMES[@]}"
    --allow_fewer_pose_views
)

exec python3 -u evaluation.py "${ARGS[@]}" "$@"
