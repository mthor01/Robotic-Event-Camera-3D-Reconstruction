#!/usr/bin/env bash
# Run TSDF reconstruction locally inside the training-and-reconstruction image.
#
# By default only the ground-truth mesh and the confidence-weighted predicted
# mesh are fused. Optional outputs are enabled by appending flags to ARGS or to
# the command line, for example:
#   ./reconstruction.sh --compare_uncertainty_tsdf         # also the uniform mesh
#   ./reconstruction.sh --save_largest_connected_surface   # *_largest_component.obj
# Results are written to reconstruction_results/<checkpoint name>/.

set -euo pipefail

# Relative paths below and on the command line are relative to the repository
# root, where this script lives.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration: edit dataset, checkpoint, and TSDF settings here.
# ---------------------------------------------------------------------------
DATA_DIR="data/Event_and_Depth/eval"
CHECKPOINT="training/checkpoints/mvs/MVS.pth"

ARGS=(
    --data_dir "$DATA_DIR"
    --checkpoint "$CHECKPOINT"
    --mesh_frame_count 200        # evenly spaced frames fused per sequence
    --tsdf_confidence_levels 8    # quantization of the confidence weights
    --tsdf_min_confidence 0.2     # pixels below this confidence are not fused
    --allow_fewer_pose_views
)

exec python3 -u reconstruction.py "${ARGS[@]}" "$@"
