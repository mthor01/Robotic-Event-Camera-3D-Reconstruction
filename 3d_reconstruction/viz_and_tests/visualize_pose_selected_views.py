#!/usr/bin/env python3
"""Export the views selected for one multiview training target.

The script uses ``MultiViewTableDataset`` directly, so pose-based view
selection and input loading are identical to multiview training.  It writes
one event image and one table-plane image for every selected view, plus the
ground-truth depth image of the target frame.

Example:
    python3 viz_and_tests/visualize_pose_selected_views.py data/new/train/22 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


_HERE = Path(__file__).resolve().parent
_RECONSTRUCTION_ROOT = _HERE.parent
_TRAINING_DIR = _RECONSTRUCTION_ROOT / "training"
sys.path.insert(0, str(_TRAINING_DIR))

import multiview as mv  # noqa: E402


def _resolve_sequence_path(path: Path) -> Path:
    """Resolve relative paths against the 3d_reconstruction directory."""
    if path.is_absolute():
        return path.resolve()
    return (_RECONSTRUCTION_ROOT / path).resolve()


def _event_activity(voxels: np.ndarray) -> np.ndarray:
    """Match the event visualization used by multiview training debugging."""
    activity = np.abs(voxels).sum(axis=0).astype(np.float32)
    low = float(activity.min())
    high = float(activity.max())
    if high - low < 1e-8:
        return np.zeros_like(activity, dtype=np.float32)
    return np.clip((activity - low) / (high - low), 0.0, 1.0)


def _save_gray(path: Path, image: np.ndarray) -> None:
    plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=1.0)


def _save_depth(path: Path, depth_m: np.ndarray, valid: np.ndarray) -> None:
    """Save metric depth with a fixed training-range colour scale."""
    normalized = np.clip(
        (depth_m - mv.DEPTH_MIN) / (mv.D_MAX - mv.DEPTH_MIN),
        0.0,
        1.0,
    )
    rgba = plt.get_cmap("turbo")(normalized)
    rgba[~valid] = (0.0, 0.0, 0.0, 1.0)
    plt.imsave(path, rgba)


def _find_dataset_item(dataset: mv.MultiViewTableDataset, target_frame: int) -> int:
    matches = np.flatnonzero(dataset.valid_indices == target_frame)
    if len(matches) == 1:
        return int(matches[0])

    if not 0 <= target_frame < dataset.n_frames:
        raise ValueError(
            f"Target frame {target_frame} is outside the available range "
            f"[0, {dataset.n_frames - 1}]."
        )

    valid = dataset.valid_indices
    nearest = valid[np.argsort(np.abs(valid - target_frame))[:5]]
    suggestions = ", ".join(str(int(frame)) for frame in nearest)
    raise ValueError(
        f"Target frame {target_frame} does not have enough pose-separated "
        f"neighbours for {dataset.num_views} views at a "
        f"{dataset.pose_move_threshold:g} m threshold. Nearest valid target "
        f"frames: {suggestions}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save separate event and table-prior images for the pose-selected "
            "views used by multiview training, plus target ground-truth depth."
        )
    )
    parser.add_argument("sequence_path", type=Path, help="Recording sequence path")
    parser.add_argument("target_frame", type=int, help="Target frame index")
    parser.add_argument(
        "--num-views",
        type=int,
        default=9,
        help="Odd number of views, matching --num_views in training (default: 9)",
    )
    parser.add_argument(
        "--pose-move-threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum camera-centre motion between selected views in metres, "
            "matching training (default: 0.05)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default, results are written below "
            "viz_and_tests/plots/pose_selected_views/."
        ),
    )
    args = parser.parse_args()

    sequence_path = _resolve_sequence_path(args.sequence_path)
    if not sequence_path.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_path}")

    if args.output_dir is None:
        output_dir = (
            _HERE
            / "plots"
            / "pose_selected_views"
            / f"{sequence_path.name}_target_{args.target_frame:06d}"
        )
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = mv.MultiViewTableDataset(
        sequence_path,
        mv._load_event_calibration(),
        num_views=args.num_views,
        pose_view_selection=True,
        pose_move_threshold=args.pose_move_threshold,
        fill_invalid=False,
    )
    dataset_item = _find_dataset_item(dataset, args.target_frame)
    sample = dataset[dataset_item]

    inputs = sample["imgs"].numpy()
    view_ids = sample["view_ids"].numpy()
    target_depth = sample["dep_t"].squeeze(0).numpy()
    target_valid = sample["mask_t"].squeeze(0).numpy() > 0.5

    manifest_lines = [
        f"sequence: {sequence_path}",
        f"target_frame: {args.target_frame}",
        f"num_views: {args.num_views}",
        f"pose_move_threshold_m: {args.pose_move_threshold:g}",
        "",
        "view_index,role,frame_id",
    ]

    for view_index, frame_id_raw in enumerate(view_ids):
        frame_id = int(frame_id_raw)
        role = "target" if view_index == 0 else "source"
        stem = f"view_{view_index:02d}_{role}_frame_{frame_id:06d}"

        event_image = _event_activity(inputs[view_index, : mv.NUM_BINS])
        table_prior = inputs[view_index, mv.NUM_BINS]
        _save_gray(output_dir / f"{stem}_events.png", event_image)
        _save_depth(
            output_dir / f"{stem}_table_prior.png",
            table_prior * (mv.D_MAX - mv.DEPTH_MIN) + mv.DEPTH_MIN,
            np.ones_like(table_prior, dtype=bool),
        )
        manifest_lines.append(f"{view_index},{role},{frame_id}")

    _save_depth(
        output_dir / f"target_frame_{args.target_frame:06d}_ground_truth_depth.png",
        target_depth,
        target_valid,
    )
    (output_dir / "selection.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Selected frame IDs: {', '.join(str(int(i)) for i in view_ids)}")
    print(f"Wrote {2 * len(view_ids) + 1} images to: {output_dir}")
    print(f"Selection manifest: {output_dir / 'selection.txt'}")


if __name__ == "__main__":
    main()
