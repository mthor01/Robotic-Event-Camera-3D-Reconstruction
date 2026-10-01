#!/usr/bin/env python3
"""Plot frame-aligned voxel activity, end-effector speed, and trigger spacing.

Event activity for frame ``i`` is the cumulative absolute activity across all
voxel bins and pixels::

    activity[i] = sum(abs(voxels[i]))

The activity series is min-max normalized to [0, 1]. End-effector speed is
computed from the high-rate transforms in ``hdf5/raw_poses.h5`` and sampled at
the camera-frame timestamps. Raw pose timestamps receive the exact correction
used by ``data_recording.rec_data.assign_poses_to_frames``::

    corrected = t_recv_ns - transport_delay_ns + pose_time_offset_ms

If raw poses are unavailable, the script falls back to the frame-assigned
transforms in ``hdf5/poses.h5``.

Example:
    python3 viz_and_tests/plot_voxel_activity_ee_speed.py \
        --sequence data/new_1/train/22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_HERE = Path(__file__).resolve().parent
_RECONSTRUCTION_ROOT = _HERE.parent


def resolve_sequence(path: Path) -> Path:
    """Resolve an absolute path, cwd-relative path, or 3d_reconstruction path."""
    candidates = [path.expanduser()]
    if not path.is_absolute():
        candidates.append(_RECONSTRUCTION_ROOT / path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Sequence directory not found: {path} "
        f"(also tried {_RECONSTRUCTION_ROOT / path})"
    )


def load_voxel_activity(voxel_path: Path) -> np.ndarray:
    """Return per-frame cumulative absolute voxel activity without loading all frames."""
    if not voxel_path.is_file():
        raise FileNotFoundError(f"Missing voxel file: {voxel_path}")
    with h5py.File(voxel_path, "r") as handle:
        if "voxels" not in handle:
            raise KeyError(f"{voxel_path} does not contain 'voxels'")
        voxels = handle["voxels"]
        if voxels.ndim != 4:
            raise ValueError(
                f"Expected voxels with shape (frames, bins, H, W), got {voxels.shape}"
            )
        activity = np.empty(voxels.shape[0], dtype=np.float64)
        for frame_index in range(voxels.shape[0]):
            frame = voxels[frame_index].astype(np.float64)
            activity[frame_index] = np.abs(frame).sum(dtype=np.float64)
    return activity


def load_trigger_intervals_ms(voxel_path: Path, frame_count: int) -> np.ndarray:
    """Return trigger-to-trigger intervals aligned to their later frame.

    Entry ``i`` is ``trigger[i] - trigger[i - 1]`` in milliseconds. The first
    entry is NaN because there is no preceding trigger for frame zero.
    """
    with h5py.File(voxel_path, "r") as handle:
        if "hw_trigger_times_us" not in handle:
            raise KeyError(
                f"{voxel_path} does not contain 'hw_trigger_times_us'; "
                "regenerate voxels with hardware-trigger timestamps"
            )
        trigger_us = handle["hw_trigger_times_us"][:frame_count].astype(np.float64)

    intervals_ms = np.full(len(trigger_us), np.nan, dtype=np.float64)
    if len(trigger_us) > 1:
        intervals_ms[1:] = np.diff(trigger_us) / 1e3
    return intervals_ms


def normalize_activity(activity: np.ndarray) -> np.ndarray:
    """Min-max normalize finite activity values to [0, 1]."""
    normalized = np.full(activity.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(activity)
    if not finite.any():
        return normalized
    minimum = float(activity[finite].min())
    maximum = float(activity[finite].max())
    if maximum <= minimum:
        normalized[finite] = 0.0
    else:
        normalized[finite] = (activity[finite] - minimum) / (maximum - minimum)
    return normalized


def load_frame_timestamps_ns(
    sequence_dir: Path, frame_count: int
) -> tuple[np.ndarray, bool]:
    """Load frame timestamps in ns and report whether they use the system clock."""
    realsense_path = sequence_dir / "hdf5" / "realsense.h5"
    if realsense_path.is_file():
        with h5py.File(realsense_path, "r") as handle:
            if "t_global_ms" in handle:
                values = handle["t_global_ms"][:frame_count].astype(np.float64)
                return values * 1e6, True
            if "t_hw_as_sys_ns" in handle:
                values = handle["t_hw_as_sys_ns"][:frame_count].astype(np.float64)
                return values, True
            if "t_sys_ns" in handle:
                values = handle["t_sys_ns"][:frame_count].astype(np.float64)
                return values, True

    fps = 30.0
    metadata_path = sequence_dir / "hdf5" / "metadata.h5"
    if metadata_path.is_file():
        with h5py.File(metadata_path, "r") as handle:
            fps = float(handle.attrs.get("fps", fps))
    step_ns = 1e9 / max(fps, 1e-6)
    return np.arange(frame_count, dtype=np.float64) * step_ns, False


def _centered_motion_speeds(
    timestamps_ns: np.ndarray, transforms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centered translational and rotational speeds at each sample."""
    sample_count = min(len(timestamps_ns), len(transforms))
    translational = np.full(sample_count, np.nan, dtype=np.float64)
    rotational = np.full(sample_count, np.nan, dtype=np.float64)
    if sample_count < 2:
        return translational, rotational
    timestamps_ns = timestamps_ns[:sample_count].astype(np.float64)
    transforms = transforms[:sample_count].astype(np.float64)
    left = np.maximum(np.arange(sample_count) - 1, 0)
    right = np.minimum(np.arange(sample_count) + 1, sample_count - 1)
    elapsed_s = (timestamps_ns[right] - timestamps_ns[left]) / 1e9
    distance = np.linalg.norm(
        transforms[right, :3, 3] - transforms[left, :3, 3], axis=1
    )
    relative_rotation = np.einsum(
        "nij,njk->nik",
        np.transpose(transforms[left, :3, :3], (0, 2, 1)),
        transforms[right, :3, :3],
    )
    cosine = np.clip(
        (np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    angle_rad = np.arccos(cosine)
    valid = np.isfinite(elapsed_s) & (elapsed_s > 0.0)
    translational[valid & np.isfinite(distance)] = (
        distance[valid & np.isfinite(distance)]
        / elapsed_s[valid & np.isfinite(distance)]
    )
    rotational[valid & np.isfinite(angle_rad)] = (
        angle_rad[valid & np.isfinite(angle_rad)]
        / elapsed_s[valid & np.isfinite(angle_rad)]
    )
    return translational, rotational


def _load_frame_assigned_speeds(
    sequence_dir: Path, frame_timestamps_ns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback motion speeds from poses already assigned to camera frames."""
    poses_path = sequence_dir / "hdf5" / "poses.h5"
    if not poses_path.is_file():
        raise FileNotFoundError(f"Missing pose file: {poses_path}")
    with h5py.File(poses_path, "r") as handle:
        if "ee_T" not in handle:
            raise KeyError(f"{poses_path} does not contain 'ee_T'")
        transforms = handle["ee_T"][:].astype(np.float64)

    frame_count = min(len(transforms), len(frame_timestamps_ns))
    return _centered_motion_speeds(
        frame_timestamps_ns[:frame_count], transforms[:frame_count]
    )


def load_ee_speeds(
    sequence_dir: Path, frame_count: int
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Sample high-rate translation and rotation speed at voxel frame times.

    Returns speed, a source description, and the median distance in milliseconds
    between each frame timestamp and its nearest corrected raw-pose timestamp.
    """
    frame_timestamps_ns, system_clock = load_frame_timestamps_ns(
        sequence_dir, frame_count
    )
    raw_path = sequence_dir / "hdf5" / "raw_poses.h5"
    metadata_path = sequence_dir / "hdf5" / "metadata.h5"
    if not raw_path.is_file() or not system_clock:
        speeds = _load_frame_assigned_speeds(sequence_dir, frame_timestamps_ns)
        reason = "raw poses unavailable" if not raw_path.is_file() else "no system-clock frame timestamps"
        return *speeds, f"frame-assigned poses ({reason})", float("nan")

    with h5py.File(raw_path, "r") as handle:
        if "t_recv_ns" not in handle or "ee_T" not in handle:
            speeds = _load_frame_assigned_speeds(sequence_dir, frame_timestamps_ns)
            return *speeds, "frame-assigned poses (raw data incomplete)", float("nan")
        raw_timestamps_ns = handle["t_recv_ns"][:].astype(np.int64)
        raw_transforms = handle["ee_T"][:].astype(np.float64)

    transport_delay_ns = 0
    pose_time_offset_ms = 0.0
    if metadata_path.is_file():
        with h5py.File(metadata_path, "r") as handle:
            transport_delay_ns = int(handle.attrs.get("transport_delay_ns", 0))
            pose_time_offset_ms = float(handle.attrs.get("pose_time_offset_ms", 0.0))
    corrected_timestamps_ns = (
        raw_timestamps_ns.astype(np.float64)
        - transport_delay_ns
        + pose_time_offset_ms * 1e6
    )

    sample_count = min(len(corrected_timestamps_ns), len(raw_transforms))
    corrected_timestamps_ns = corrected_timestamps_ns[:sample_count]
    raw_transforms = raw_transforms[:sample_count]
    order = np.argsort(corrected_timestamps_ns)
    corrected_timestamps_ns = corrected_timestamps_ns[order]
    raw_transforms = raw_transforms[order]
    unique = np.concatenate(
        ([True], np.diff(corrected_timestamps_ns) > 0)
    ) if sample_count else np.empty(0, dtype=bool)
    corrected_timestamps_ns = corrected_timestamps_ns[unique]
    raw_transforms = raw_transforms[unique]
    if len(corrected_timestamps_ns) < 2:
        speeds = _load_frame_assigned_speeds(sequence_dir, frame_timestamps_ns)
        return *speeds, "frame-assigned poses (insufficient raw samples)", float("nan")

    raw_translational, raw_rotational = _centered_motion_speeds(
        corrected_timestamps_ns, raw_transforms
    )

    def interpolate(raw_values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(raw_values) & np.isfinite(corrected_timestamps_ns)
        sampled = np.full(len(frame_timestamps_ns), np.nan, dtype=np.float64)
        if finite.sum() >= 2:
            inside = (
                (frame_timestamps_ns >= corrected_timestamps_ns[finite][0])
                & (frame_timestamps_ns <= corrected_timestamps_ns[finite][-1])
            )
            sampled[inside] = np.interp(
                frame_timestamps_ns[inside],
                corrected_timestamps_ns[finite],
                raw_values[finite],
            )
        return sampled

    translational = interpolate(raw_translational)
    rotational = interpolate(raw_rotational)

    indices = np.searchsorted(corrected_timestamps_ns, frame_timestamps_ns)
    indices = np.clip(indices, 0, max(len(corrected_timestamps_ns) - 1, 0))
    left = np.maximum(indices - 1, 0)
    use_left = (
        np.abs(corrected_timestamps_ns[left] - frame_timestamps_ns)
        < np.abs(corrected_timestamps_ns[indices] - frame_timestamps_ns)
    )
    indices[use_left] = left[use_left]
    nearest_offset_ms = np.abs(
        corrected_timestamps_ns[indices] - frame_timestamps_ns
    ) / 1e6
    median_offset_ms = float(np.median(nearest_offset_ms))
    return (
        translational,
        rotational,
        "high-rate raw poses synchronized with recorder correction",
        median_offset_ms,
    )


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average that ignores non-finite samples."""
    window = min(window, len(values))
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64)
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(np.float64), kernel, mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def create_plot(
    sequence_dir: Path,
    activity: np.ndarray,
    translational_speed: np.ndarray,
    rotational_speed: np.ndarray,
    trigger_intervals_ms: np.ndarray,
    output_path: Path,
    smooth_frames: int,
) -> None:
    frame_count = min(
        len(activity), len(translational_speed), len(rotational_speed),
        len(trigger_intervals_ms),
    )
    if frame_count == 0:
        raise RuntimeError(f"No common voxel/pose frames in {sequence_dir}")
    frames = np.arange(frame_count)
    activity = activity[:frame_count]
    translational_speed = translational_speed[:frame_count]
    rotational_speed = rotational_speed[:frame_count]
    trigger_intervals_ms = trigger_intervals_ms[:frame_count]

    figure, (activity_axis, trigger_axis) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    translation_axis = activity_axis.twinx()
    rotation_axis = activity_axis.twinx()
    rotation_axis.spines["right"].set_position(("axes", 1.09))
    rotation_axis.spines["right"].set_visible(True)

    activity_line = activity_axis.plot(
        frames,
        activity,
        color="#D1495B",
        linewidth=1.4,
        label="Normalized cumulative voxel activity",
    )[0]
    translation_line = translation_axis.plot(
        frames,
        translational_speed,
        color="#2867B2",
        linewidth=1.4,
        label="EE translational speed",
    )[0]
    rotation_line = rotation_axis.plot(
        frames,
        rotational_speed,
        color="#198754",
        linewidth=1.4,
        label="EE rotational speed",
    )[0]

    activity_axis.set_ylabel("Normalized event activity [0, 1]", color="#D1495B")
    activity_axis.tick_params(axis="y", colors="#D1495B")
    activity_axis.set_ylim(-0.02, 1.02)
    activity_axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    translation_axis.set_ylabel("EE translational speed [m/s]", color="#2867B2")
    rotation_axis.set_ylabel("EE rotational speed [rad/s]", color="#198754")
    translation_axis.tick_params(axis="y", colors="#2867B2")
    rotation_axis.tick_params(axis="y", colors="#198754")
    activity_axis.spines["left"].set_color("#D1495B")
    translation_axis.spines["right"].set_color("#2867B2")
    rotation_axis.spines["right"].set_color("#198754")
    activity_axis.set_xlim(0, max(frame_count - 1, 1))
    activity_axis.legend(
        [activity_line, translation_line, rotation_line],
        [
            activity_line.get_label(),
            translation_line.get_label(),
            rotation_line.get_label(),
        ],
        loc="upper right",
    )

    trigger_axis.plot(
        frames,
        trigger_intervals_ms,
        color="#6F42C1",
        linewidth=1.2,
        marker=".",
        markersize=2.5,
        label="Time since previous hardware trigger",
    )
    finite_intervals = trigger_intervals_ms[np.isfinite(trigger_intervals_ms)]
    if finite_intervals.size:
        median_interval = float(np.median(finite_intervals))
        trigger_axis.axhline(
            median_interval,
            color="#6F42C1",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=f"Median: {median_interval:.3f} ms",
        )
    trigger_axis.set_xlabel("Frame number")
    trigger_axis.set_ylabel("Trigger interval [ms]", color="#6F42C1")
    trigger_axis.tick_params(axis="y", colors="#6F42C1")
    trigger_axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    trigger_axis.legend(loc="upper right")

    smoothing_text = (
        "unsmoothed" if smooth_frames == 1 else f"{smooth_frames}-frame moving average"
    )
    figure.suptitle(
        f"Voxel activity and end-effector motion — {sequence_dir.name} "
        f"({smoothing_text})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative normalized voxel activity and EE speed by frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequence",
        "--sequence-dir",
        type=Path,
        required=True,
        help="Sequence containing events/voxels_cam0.h5 and hdf5/poses.h5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG; defaults to <sequence>/pose_plots/voxel_activity_ee_speed.png.",
    )
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=1,
        help="Centered moving-average width; 1 disables smoothing.",
    )
    args = parser.parse_args()
    if args.smooth_frames < 1:
        parser.error("--smooth-frames must be at least 1")

    sequence_dir = resolve_sequence(args.sequence)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else sequence_dir / "pose_plots" / "voxel_activity_ee_speed.png"
    )
    raw_activity = load_voxel_activity(
        sequence_dir / "events" / "voxels_cam0.h5"
    )
    voxel_path = sequence_dir / "events" / "voxels_cam0.h5"
    trigger_intervals_ms = load_trigger_intervals_ms(voxel_path, len(raw_activity))
    activity = normalize_activity(raw_activity)
    translational_speed, rotational_speed, speed_source, median_pose_offset_ms = load_ee_speeds(
        sequence_dir, len(activity)
    )
    common_frames = min(
        len(activity), len(translational_speed), len(rotational_speed)
    )
    activity_plot = smooth(activity[:common_frames], args.smooth_frames)
    translational_plot = smooth(
        translational_speed[:common_frames], args.smooth_frames
    )
    rotational_plot = smooth(rotational_speed[:common_frames], args.smooth_frames)
    create_plot(
        sequence_dir,
        activity_plot,
        translational_plot,
        rotational_plot,
        trigger_intervals_ms[:common_frames],
        output_path,
        args.smooth_frames,
    )

    print(f"Sequence: {sequence_dir}")
    print(
        f"Voxel frames: {len(activity)}, synchronized motion samples: "
        f"{len(translational_speed)}"
    )
    print(f"Plotted common frames: {common_frames}")
    print(f"Speed source: {speed_source}")
    finite_trigger_intervals = trigger_intervals_ms[np.isfinite(trigger_intervals_ms)]
    if finite_trigger_intervals.size:
        print(
            "Trigger interval [ms]: "
            f"median={np.median(finite_trigger_intervals):.3f}, "
            f"min={np.min(finite_trigger_intervals):.3f}, "
            f"max={np.max(finite_trigger_intervals):.3f}"
        )
    if np.isfinite(median_pose_offset_ms):
        print(f"Median frame-to-nearest-raw-pose offset: {median_pose_offset_ms:.3f} ms")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
