#!/usr/bin/env bash
# Run depth-model evaluation locally inside the training-and-reconstruction image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration: edit paths, examples, and resource-related settings.
# ---------------------------------------------------------------------------
DATA_DIR="../data/Event_and_Depth/eval"
CHECKPOINT="checkpoints/mvs/MVS.pth"
WORKERS=8
BATCH_SIZE=15

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
    --fast_mode 100
    --example_sequence "${EXAMPLE_SEQUENCES[@]}"
    --example_frame "${EXAMPLE_FRAMES[@]}"
    --allow_unbalanced_pose_views
)

exec python3 -u evaluation.py "${ARGS[@]}" "$@"
