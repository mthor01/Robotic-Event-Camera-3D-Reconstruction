#!/usr/bin/env python3
"""Export the views selected for one multiview training target.

The script uses ``MultiViewTableDataset`` directly, so pose-based view
selection and input loading are identical to multiview training. It writes
undilated and dilated event images, one table-plane image, and the original
RealSense RGB image for every selected view into separate directories, plus
the ground-truth depth image of the target frame.

Example:
    python3 viz_and_tests/visualize_pose_selected_views.py \
        data/new_2/train/22 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py  # noqa: E402
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


def _dilate_image(image: np.ndarray, radius: int) -> np.ndarray:
    """Apply square maximum-filter dilation without adding a dependency."""
    if radius <= 0:
        return image
    height, width = image.shape
    padded = np.pad(image, radius, mode="constant", constant_values=0.0)
    dilated = np.zeros_like(image)
    kernel_size = 2 * radius + 1
    for y_offset in range(kernel_size):
        for x_offset in range(kernel_size):
            dilated = np.maximum(
                dilated,
                padded[
                    y_offset:y_offset + height,
                    x_offset:x_offset + width,
                ],
            )
    return dilated


def _save_gray(path: Path, image: np.ndarray) -> None:
    plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=1.0)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    """Save an original RGB frame without spatial transformation."""
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an RGB(A) image, received shape {image.shape}.")
    plt.imsave(path, image)


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


def _clear_view_images(*directories: Path) -> None:
    """Remove generated view images so stale legacy filenames do not remain."""
    for directory in directories:
        for path in directory.glob("view_*.png"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save undilated and dilated event images, table-prior images, "
            "and original RGB images for the pose-selected views used by "
            "multiview training, plus target ground-truth depth."
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
    pose_layout_group = parser.add_mutually_exclusive_group()
    pose_layout_group.add_argument(
        "--allow-fewer-pose-views",
        "--allow_fewer_pose_views",
        "--allow-unbalanced-pose-views",
        "--allow_unbalanced_pose_views",
        dest="allow_unbalanced_pose_views",
        action="store_true",
        help=(
            "Keep boundary targets and represent unavailable source slots by "
            "black images, matching multiview training's default."
        ),
    )
    pose_layout_group.add_argument(
        "--strict-balanced-pose-views",
        "--strict_balanced_pose_views",
        dest="allow_unbalanced_pose_views",
        action="store_false",
        help="Require four valid sources before and after a nine-view target.",
    )
    parser.set_defaults(allow_unbalanced_pose_views=True)
    parser.add_argument(
        "--dilate-events",
        "--dilate_events",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--event-dilation-radius",
        "--event_dilation_radius",
        type=int,
        default=1,
        help=(
            "Neighbourhood radius for images in events_dilated. A radius of "
            "1 uses a 3x3 neighbourhood (default: 1)."
        ),
    )
    parser.add_argument(
        "--binary-events",
        "--binary_events",
        action="store_true",
        help=(
            "Render every positive event-activity pixel as pure white while "
            "leaving zero-activity pixels black."
        ),
    )
    args = parser.parse_args()

    if args.event_dilation_radius < 1:
        parser.error("--event-dilation-radius must be at least 1")

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
    undilated_event_output_dir = output_dir / "events_undilated"
    dilated_event_output_dir = output_dir / "events_dilated"
    table_output_dir = output_dir / "table_plane_priors"
    rgb_output_dir = output_dir / "rgb"
    undilated_event_output_dir.mkdir(parents=True, exist_ok=True)
    dilated_event_output_dir.mkdir(parents=True, exist_ok=True)
    table_output_dir.mkdir(parents=True, exist_ok=True)
    rgb_output_dir.mkdir(parents=True, exist_ok=True)
    _clear_view_images(
        output_dir / "events",
        undilated_event_output_dir,
        dilated_event_output_dir,
        table_output_dir,
        rgb_output_dir,
    )

    dataset = mv.MultiViewTableDataset(
        sequence_path,
        mv._load_event_calibration(),
        num_views=args.num_views,
        pose_view_selection=True,
        pose_move_threshold=args.pose_move_threshold,
        allow_unbalanced_pose_views=args.allow_unbalanced_pose_views,
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
        f"allow_unbalanced_pose_views: {args.allow_unbalanced_pose_views}",
        "intrinsics_transform: center_crop_resize",
        f"event_dilation_radius: {args.event_dilation_radius}",
        f"binary_events: {args.binary_events}",
        "",
        "view_index,role,frame_id",
    ]

    realsense_path = sequence_path / "hdf5" / "realsense.h5"
    if not realsense_path.is_file():
        raise FileNotFoundError(f"RealSense recording not found: {realsense_path}")

    with h5py.File(realsense_path, "r") as realsense_file:
        if "rgb" not in realsense_file:
            raise KeyError(f"Dataset 'rgb' not found in {realsense_path}")
        rgb_frames = realsense_file["rgb"]
        if rgb_frames.ndim != 4 or rgb_frames.shape[-1] not in (3, 4):
            raise ValueError(
                f"Expected 'rgb' to have shape (N,H,W,3/4), received "
                f"{rgb_frames.shape}."
            )

        for view_index, frame_id_raw in enumerate(view_ids):
            frame_id = int(frame_id_raw)
            role = "target" if view_index == 0 else "source"
            filename = f"view_{view_index}.png"

            undilated_event_image = _event_activity(
                inputs[view_index, : mv.NUM_BINS]
            )
            if args.binary_events:
                undilated_event_image = (
                    undilated_event_image > 0.0
                ).astype(np.float32)
            dilated_event_image = _dilate_image(
                undilated_event_image,
                args.event_dilation_radius,
            )
            table_prior = inputs[view_index, mv.NUM_BINS]

            if frame_id >= len(rgb_frames):
                raise IndexError(
                    f"Selected frame {frame_id} is outside the RGB dataset "
                    f"range [0, {len(rgb_frames) - 1}]."
                )
            if frame_id >= 0:
                rgb_image = rgb_frames[frame_id]
            else:
                rgb_image = np.zeros(rgb_frames.shape[1:], dtype=rgb_frames.dtype)

            _save_gray(
                undilated_event_output_dir / filename,
                undilated_event_image,
            )
            _save_gray(
                dilated_event_output_dir / filename,
                dilated_event_image,
            )
            _save_depth(
                table_output_dir / filename,
                table_prior * (mv.D_MAX - mv.DEPTH_MIN) + mv.DEPTH_MIN,
                np.full_like(table_prior, frame_id >= 0, dtype=bool),
            )
            _save_rgb(rgb_output_dir / filename, rgb_image)
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
    print(
        f"Wrote {len(view_ids)} undilated event images to: "
        f"{undilated_event_output_dir}"
    )
    print(
        f"Wrote {len(view_ids)} dilated event images to: "
        f"{dilated_event_output_dir}"
    )
    print(f"Wrote {len(view_ids)} table-plane images to: {table_output_dir}")
    print(f"Wrote {len(view_ids)} original RGB images to: {rgb_output_dir}")
    print(f"Wrote target ground-truth depth to: {output_dir}")
    print(f"Selection manifest: {output_dir / 'selection.txt'}")


if __name__ == "__main__":
    main()
