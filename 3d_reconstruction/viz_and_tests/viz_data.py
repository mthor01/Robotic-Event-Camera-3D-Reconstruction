#!/usr/bin/env python3
"""Visualize recorded data and the RandomHemisphereAgent sampling volume.

The script loads the active values from ``franka_pipeline/config_defaults.py``
and draws the intersection of the hollow upper hemisphere with the sampler's
base-distance and minimum-height constraints. It can show either Monte Carlo
samples or every pose recorded below ``--data_dir``. Robot IK, self-collision,
and environment-collision checks are not represented. Optionally, a single
recording sequence can be rendered as a 3D calibrated-camera trajectory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _HERE.parents[1]
_RECONSTRUCTION_ROOT = _REPOSITORY_ROOT / "3d_reconstruction"
_FRANKA_PIPELINE_ROOT = _REPOSITORY_ROOT / "franka_pipeline"
sys.path.insert(0, str(_FRANKA_PIPELINE_ROOT))
sys.path.insert(0, str(_RECONSTRUCTION_ROOT))

import config_defaults as cfg  # noqa: E402
from config import DEPTH_MIN, D_MAX  # noqa: E402


@dataclass(frozen=True)
class HemisphereConfig:
    target: np.ndarray
    center: np.ndarray
    outer_radius: float
    inner_radius: float
    base_min_radius: float
    base_max_radius: float
    min_z: float
    num_poses: int
    lock_rotation_horizontal: bool


def load_current_config() -> HemisphereConfig:
    """Load the same position-sampling values used by the current agent config."""
    target = np.array([cfg.TARGET_X, cfg.TARGET_Y, cfg.TARGET_Z], dtype=np.float64)
    center = target.copy()
    center[2] += float(cfg.CENTER_Z_OFFSET)
    config = HemisphereConfig(
        target=target,
        center=center,
        outer_radius=float(cfg.SPHERE_RADIUS),
        inner_radius=float(cfg.INNER_RADIUS),
        base_min_radius=float(cfg.BASE_EXCLUSION_RADIUS),
        base_max_radius=float(cfg.BASE_MAX_RADIUS),
        min_z=float(cfg.MIN_Z_HEIGHT),
        num_poses=int(cfg.NUM_POSES),
        lock_rotation_horizontal=bool(cfg.LOCK_ROTATION_HORIZONTAL),
    )
    if not 0.0 <= config.inner_radius < config.outer_radius:
        raise ValueError("Expected 0 <= INNER_RADIUS < SPHERE_RADIUS")
    if not 0.0 <= config.base_min_radius < config.base_max_radius:
        raise ValueError("Expected 0 <= BASE_EXCLUSION_RADIUS < BASE_MAX_RADIUS")
    return config


def sample_hollow_hemisphere(
    config: HemisphereConfig,
    count: int,
    seed: int,
) -> np.ndarray:
    """Use the same uniform hollow-volume distribution as the agent."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    phi = np.arccos(rng.uniform(0.0, 1.0, count))
    radius = rng.uniform(
        config.inner_radius**3,
        config.outer_radius**3,
        count,
    ) ** (1.0 / 3.0)
    sin_phi = np.sin(phi)
    offsets = np.column_stack(
        (
            radius * sin_phi * np.cos(theta),
            radius * sin_phi * np.sin(theta),
            radius * np.cos(phi),
        )
    )
    return config.center[None, :] + offsets


def accepted_position_mask(
    points: np.ndarray,
    config: HemisphereConfig,
) -> np.ndarray:
    """Return positions inside the agent's complete configured sampling region."""
    base_xy_radius = np.linalg.norm(points[:, :2], axis=1)
    radius_from_center = np.linalg.norm(points - config.center[None, :], axis=1)
    return (
        (radius_from_center >= config.inner_radius)
        & (radius_from_center <= config.outer_radius)
        & (points[:, 2] >= config.center[2])
        & (points[:, 2] >= config.min_z)
        & (base_xy_radius >= config.base_min_radius)
        & (base_xy_radius <= config.base_max_radius)
    )


def _load_camera_calibration() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration_dir = _REPOSITORY_ROOT / "3d_reconstruction" / "camera_data"
    T_rgb_from_ee = np.load(calibration_dir / "T_rgb_from_ee.npz")["T"].astype(
        np.float64
    )
    T_event_from_rgb = np.load(
        calibration_dir / "T_event_from_rgb.npz"
    )["T"].astype(np.float64)
    T_color_from_depth = np.load(
        calibration_dir / "T_color_from_depth.npz"
    )["T"].astype(np.float64)
    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee
    T_depth_from_ee = np.linalg.inv(T_color_from_depth) @ T_rgb_from_ee
    return T_rgb_from_ee, T_event_from_ee, T_depth_from_ee


def camera_positions_from_sampled_ee(
    ee_positions: np.ndarray,
    config: HemisphereConfig,
    seed: int,
    camera: str,
) -> np.ndarray:
    """Convert sampled EE positions to calibrated camera centers.

    EE orientations are reconstructed with the same depth-camera look-at
    convention used by RandomHemisphereAgent.
    """
    _, T_event_from_ee, T_depth_from_ee = _load_camera_calibration()
    if camera == "event":
        T_camera_from_ee = T_event_from_ee
    elif camera == "depth":
        T_camera_from_ee = T_depth_from_ee
    else:
        raise ValueError(f"Unsupported camera: {camera}")

    # Reconstruct the EE orientation chosen by RandomHemisphereAgent. It points
    # the depth-camera optical axis at the target, accounting for its extrinsic.
    direction = config.target[None, :] - ee_positions
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    if config.lock_rotation_horizontal:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_axis = np.cross(direction, world_up[None, :])
        degenerate = np.linalg.norm(x_axis, axis=1) < 1e-8
        x_axis[degenerate] = np.array([0.0, 1.0, 0.0])
    else:
        rng = np.random.default_rng(seed + 1)
        random_axis = rng.standard_normal(direction.shape)
        x_axis = random_axis - (
            np.sum(random_axis * direction, axis=1, keepdims=True) * direction
        )
        degenerate = np.linalg.norm(x_axis, axis=1) < 1e-8
        fallback = np.array([0.0, 1.0, 0.0])
        x_axis[degenerate] = fallback - (
            np.sum(fallback * direction[degenerate], axis=1, keepdims=True)
            * direction[degenerate]
        )
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    y_axis = np.cross(direction, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
    R_base_from_depth = np.stack((x_axis, y_axis, direction), axis=2)

    R_base_from_ee = np.einsum(
        "nij,jk->nik", R_base_from_depth, T_depth_from_ee[:3, :3]
    )

    camera_origin_in_ee = np.linalg.inv(T_camera_from_ee)[:3, 3]
    camera_offset_in_base = np.einsum(
        "nij,j->ni", R_base_from_ee, camera_origin_in_ee
    )
    return ee_positions + camera_offset_in_base


def _resolve_data_dir(path: Path) -> Path:
    """Resolve direct paths and paths relative to 3d_reconstruction."""
    if path.expanduser().is_dir():
        return path.expanduser().resolve()
    reconstruction_relative = (
        _REPOSITORY_ROOT / "3d_reconstruction" / path.expanduser()
    )
    if reconstruction_relative.is_dir():
        return reconstruction_relative.resolve()
    raise FileNotFoundError(
        f"Data directory does not exist: {path} "
        f"(also tried {reconstruction_relative})"
    )


def _recording_pose_files(data_dir: Path) -> list[Path]:
    """Find pose files, restricting dataset roots to train and eval splits."""
    direct_pose_file = data_dir / "hdf5" / "poses.h5"
    if direct_pose_file.is_file():
        return [direct_pose_file]

    split_dirs = [
        data_dir / split_name
        for split_name in ("train", "eval")
        if (data_dir / split_name).is_dir()
    ]
    if split_dirs:
        return sorted(
            pose_file
            for split_dir in split_dirs
            for pose_file in split_dir.rglob("hdf5/poses.h5")
        )

    # Preserve support for passing a split directory such as data/new_2/train.
    return sorted(data_dir.rglob("hdf5/poses.h5"))


def load_recorded_ee_poses(
    data_dir: Path,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Recursively load every recorded ``ee_T`` pose below a data directory."""
    import h5py

    pose_files = _recording_pose_files(data_dir)
    if not pose_files:
        raise FileNotFoundError(f"No hdf5/poses.h5 files found below {data_dir}")

    transforms = []
    sequence_counts: list[tuple[str, int]] = []
    for pose_file in pose_files:
        with h5py.File(pose_file, "r") as handle:
            if "ee_T" not in handle:
                raise KeyError(f"{pose_file} does not contain dataset 'ee_T'")
            ee_T = handle["ee_T"][:].astype(np.float64)
        if ee_T.ndim != 3 or ee_T.shape[1:] != (4, 4):
            raise ValueError(
                f"Expected ee_T with shape (N, 4, 4) in {pose_file}, "
                f"got {ee_T.shape}"
            )
        finite = np.isfinite(ee_T).all(axis=(1, 2))
        ee_T = ee_T[finite]
        if len(ee_T) == 0:
            continue
        transforms.append(ee_T)
        sequence_name = str(pose_file.parent.parent.relative_to(data_dir))
        sequence_counts.append((sequence_name, len(ee_T)))
    if not transforms:
        raise RuntimeError(f"No finite recorded EE poses found below {data_dir}")
    return np.concatenate(transforms, axis=0), sequence_counts


def load_recorded_translational_speeds(
    data_dir: Path,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    """Load EE speeds and diagnostics describing rejected speed samples."""
    import h5py

    resolved_dir = _resolve_data_dir(data_dir)
    pose_files = _recording_pose_files(resolved_dir)
    all_speeds: list[np.ndarray] = []
    diagnostics: list[dict[str, int | str]] = []
    for pose_file in pose_files:
        sequence_dir = pose_file.parent.parent
        sequence_name = str(sequence_dir.relative_to(resolved_dir))
        with h5py.File(pose_file, "r") as handle:
            positions = handle["ee_T"][:, :3, 3].astype(np.float64)

        timestamps_s = None
        timestamp_source = "generated from metadata FPS"
        realsense_path = sequence_dir / "hdf5" / "realsense.h5"
        if realsense_path.is_file():
            with h5py.File(realsense_path, "r") as handle:
                if "t_global_ms" in handle:
                    timestamps_s = handle["t_global_ms"][:].astype(np.float64) / 1e3
                    timestamp_source = "realsense.h5:t_global_ms"
                elif "t_hw_as_sys_ns" in handle:
                    timestamps_s = handle["t_hw_as_sys_ns"][:].astype(np.float64) / 1e9
                    timestamp_source = "realsense.h5:t_hw_as_sys_ns"
                elif "t_sys_ns" in handle:
                    timestamps_s = handle["t_sys_ns"][:].astype(np.float64) / 1e9
                    timestamp_source = "realsense.h5:t_sys_ns"

        if timestamps_s is None:
            fps = 30.0
            metadata_path = sequence_dir / "hdf5" / "metadata.h5"
            if metadata_path.is_file():
                with h5py.File(metadata_path, "r") as handle:
                    fps = float(handle.attrs.get("fps", fps))
            timestamps_s = np.arange(len(positions), dtype=np.float64) / max(fps, 1e-6)

        pose_count = len(positions)
        timestamp_count = len(timestamps_s)
        count = min(pose_count, timestamp_count)
        diagnostic: dict[str, int | str] = {
            "sequence": sequence_name,
            "timestamp_source": timestamp_source,
            "pose_count": pose_count,
            "timestamp_count": timestamp_count,
            "aligned_count": count,
            "poses_without_timestamp": max(0, pose_count - timestamp_count),
            "unused_timestamps": max(0, timestamp_count - pose_count),
            "invalid_dt": 0,
            "nonfinite_speed": 0,
            "valid_speed_count": 0,
        }
        if count < 2:
            diagnostics.append(diagnostic)
            continue
        positions = positions[:count]
        timestamps_s = timestamps_s[:count]
        left = np.maximum(np.arange(count) - 1, 0)
        right = np.minimum(np.arange(count) + 1, count - 1)
        dt = timestamps_s[right] - timestamps_s[left]
        displacement = positions[right] - positions[left]
        valid = np.isfinite(dt) & (dt > 1e-9)
        speed = np.linalg.norm(displacement[valid], axis=1) / dt[valid]
        finite_speed = np.isfinite(speed)
        usable_speed = speed[finite_speed]
        diagnostic["invalid_dt"] = int(count - valid.sum())
        diagnostic["nonfinite_speed"] = int((~finite_speed).sum())
        diagnostic["valid_speed_count"] = len(usable_speed)
        diagnostics.append(diagnostic)
        all_speeds.append(usable_speed)

    speeds = (
        np.concatenate(all_speeds)
        if all_speeds
        else np.empty(0, dtype=np.float64)
    )
    return speeds, diagnostics


def create_speed_distribution_plot(speeds_m_s: np.ndarray, output_path: Path) -> None:
    """Plot the distribution of recorded EE translational speed."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    speed = np.asarray(speeds_m_s, dtype=np.float64)
    speed = speed[np.isfinite(speed) & (speed >= 0.0)]
    if speed.size == 0:
        return

    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        x=speed,
        ax=axis,
        color="#2878B5",
        linewidth=2.6,
        fill=False,
        cut=0,
    )
    median, p95, p99 = np.percentile(speed, [50, 95, 99])
    for value, label, color in (
        (median, "Median", "#E07A1F"),
        (p95, "95th Percentile", "#C33C54"),
        (p99, "99th Percentile", "#3A923A"),
    ):
        axis.axvline(value, color=color, linestyle="--", linewidth=1.8,
                     label=f"{label}: {value:.3f} m/s")
    axis.set_xlabel("End-effector translational speed [m/s]")
    axis.set_ylabel("Density")
    axis.set_title("Recorded End-Effector Speed Distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def camera_positions_from_recorded_ee(
    T_base_from_ee: np.ndarray,
    camera: str,
) -> np.ndarray:
    """Transform recorded EE poses to calibrated camera centers."""
    return camera_transforms_from_recorded_ee(T_base_from_ee, camera)[:, :3, 3]


def camera_transforms_from_recorded_ee(
    T_base_from_ee: np.ndarray,
    camera: str,
) -> np.ndarray:
    """Transform recorded EE poses to calibrated base-from-camera poses."""
    _, T_event_from_ee, T_depth_from_ee = _load_camera_calibration()
    if camera == "event":
        T_camera_from_ee = T_event_from_ee
    elif camera == "depth":
        T_camera_from_ee = T_depth_from_ee
    else:
        raise ValueError(f"Unsupported camera: {camera}")
    T_ee_from_camera = np.linalg.inv(T_camera_from_ee)
    return np.matmul(T_base_from_ee, T_ee_from_camera)


def load_sequence_ee_poses(sequence_dir: Path) -> np.ndarray:
    """Load the ordered EE poses from exactly one recording sequence."""
    resolved_dir = _resolve_data_dir(sequence_dir)
    pose_file = resolved_dir / "hdf5" / "poses.h5"
    if not pose_file.is_file():
        raise FileNotFoundError(
            f"--sequence-dir must identify one sequence containing "
            f"hdf5/poses.h5: {resolved_dir}"
        )
    transforms, _ = load_recorded_ee_poses(resolved_dir)
    return transforms


def validate_samples(
    points: np.ndarray,
    valid: np.ndarray,
    config: HemisphereConfig,
) -> None:
    """Assert that sampling and filtering agree with the configured bounds."""
    tolerance = 1e-10
    distance_from_center = np.linalg.norm(points - config.center[None, :], axis=1)
    base_xy_radius = np.linalg.norm(points[:, :2], axis=1)
    assert np.all(distance_from_center >= config.inner_radius - tolerance)
    assert np.all(distance_from_center <= config.outer_radius + tolerance)
    assert np.all(points[:, 2] >= config.center[2] - tolerance)
    assert np.any(valid), "The current constraints leave no sampled valid positions"
    assert np.all(points[valid, 2] >= config.min_z - tolerance)
    assert np.all(base_xy_radius[valid] >= config.base_min_radius - tolerance)
    assert np.all(base_xy_radius[valid] <= config.base_max_radius + tolerance)
    assert np.array_equal(valid, accepted_position_mask(points, config))


def create_plot(
    displayed_points: np.ndarray,
    valid: np.ndarray,
    config: HemisphereConfig,
    output_path: Path,
    position_label: str,
    source_mode: str,
    source_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Circle
    from matplotlib.ticker import FuncFormatter

    sns.set_theme(
        context="talk",
        style="whitegrid",
        font_scale=0.9,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD2D9",
            "grid.color": "#DCE1E6",
            "grid.alpha": 0.65,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.labelcolor": "black",
        },
    )
    colors = {
        "inside": "#168AAD",
        "outside": "#AEB8C2",
        "recorded": "#377EB8",
        "target": "#E63946",
        "boundary": "#F4A261",
        "center": "#6D597A",
    }
    centimetres = FuncFormatter(lambda value, _: f"{100.0 * value:g}")

    accepted = displayed_points[valid]
    rejected = displayed_points[~valid]
    if source_mode == "recorded":
        # Recorded mode intentionally renders every loaded pose.
        accepted_draw = accepted
        rejected_draw = rejected
    else:
        # Cap only the rendered Monte Carlo cloud; all candidates are validated.
        accepted_draw = accepted[:: max(1, len(accepted) // 40000)]
        rejected_draw = rejected[:: max(1, len(rejected) // 25000)]

    figure = plt.figure(figsize=(16, 8.5))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.12, 1.0),
        wspace=0.16,
    )
    axis_side = figure.add_subplot(grid[0, 0])
    axis_xy = figure.add_subplot(grid[0, 1])

    if source_mode == "recorded":
        recorded_label = {
            "event-camera": "Recorded Event Camera Poses",
            "depth-camera": "Recorded Depth Camera Poses",
            "EE": "Recorded EE Poses",
        }[position_label]
        sns.scatterplot(
            x=displayed_points[:, 1], y=displayed_points[:, 2],
            s=8, alpha=0.30, color=colors["recorded"], linewidth=0,
            rasterized=True, label=recorded_label, ax=axis_side,
        )
    else:
        sns.scatterplot(
            x=rejected_draw[:, 1], y=rejected_draw[:, 2],
            s=6, alpha=0.10, color=colors["outside"], linewidth=0,
            rasterized=True, label="Outside Sampler Region", ax=axis_side,
        )
        sns.scatterplot(
            x=accepted_draw[:, 1], y=accepted_draw[:, 2],
            s=8, alpha=0.34, color=colors["inside"], linewidth=0,
            rasterized=True, label="Inside Sampler Region", ax=axis_side,
        )
    axis_side.scatter(
        config.target[1], config.target[2], marker="*", s=360,
        color=colors["target"], edgecolor="white", linewidth=1.4,
        label="Target", zorder=10,
    )
    if not np.allclose(config.center, config.target):
        axis_side.scatter(
            config.center[1], config.center[2], marker="x", s=90,
            color=colors["center"], label="Hemisphere Center", zorder=9,
        )
    arc_angle = np.linspace(0.0, np.pi, 240)
    axis_side.plot(
        config.center[1] + config.outer_radius * np.cos(arc_angle),
        config.center[2] + config.outer_radius * np.sin(arc_angle),
        color=colors["boundary"], linewidth=1.6, alpha=0.95,
        label="Outer EE Hemisphere",
    )
    side_y_low = min(displayed_points[:, 1].min(),
                     config.center[1] - config.outer_radius)
    side_y_high = max(displayed_points[:, 1].max(),
                      config.center[1] + config.outer_radius)
    side_y_margin = max(0.03, 0.04 * (side_y_high - side_y_low))
    side_z_low = min(displayed_points[:, 2].min(), config.center[2]) - 0.03
    side_z_high = max(
        displayed_points[:, 2].max(),
        config.center[2] + config.outer_radius,
    ) + 0.30
    axis_side.set_xlim(side_y_low - side_y_margin, side_y_high + side_y_margin)
    axis_side.set_ylim(side_z_low, side_z_high)
    axis_side.set_title("Side view", pad=16, fontweight="semibold")
    axis_side.set_xlabel("Y [cm]")
    axis_side.set_ylabel("Z [cm]")
    axis_side.xaxis.set_major_formatter(centimetres)
    axis_side.yaxis.set_major_formatter(centimetres)
    axis_side.set_aspect("equal", adjustable="box")
    axis_side.grid(True, linestyle="--", linewidth=0.8, alpha=0.55)
    sns.despine(ax=axis_side, offset=4)

    if source_mode == "recorded":
        sns.scatterplot(
            x=displayed_points[:, 0], y=displayed_points[:, 1],
            s=8, alpha=0.30, color=colors["recorded"], linewidth=0,
            rasterized=True, label=recorded_label, ax=axis_xy,
        )
    else:
        sns.scatterplot(
            x=rejected_draw[:, 0], y=rejected_draw[:, 1],
            s=6, alpha=0.10, color=colors["outside"], linewidth=0,
            rasterized=True, label="Outside Sampler Region", ax=axis_xy,
        )
        sns.scatterplot(
            x=accepted_draw[:, 0], y=accepted_draw[:, 1],
            s=8, alpha=0.34, color=colors["inside"], linewidth=0,
            rasterized=True, label="Inside Sampler Region", ax=axis_xy,
        )
    axis_xy.scatter(
        config.target[0], config.target[1], marker="*", s=360,
        color=colors["target"], edgecolor="white", linewidth=0.8,
        label="Target", zorder=8,
    )
    for radius, linestyle, label in (
        (config.base_min_radius, "--", "Base Exclusion Radius"),
        (config.base_max_radius, "-", "Base Maximum Radius"),
    ):
        axis_xy.add_patch(
            Circle((0.0, 0.0), radius, fill=False, linestyle=linestyle,
                   linewidth=1.5, color=colors["target"], label=label)
        )
    axis_xy.add_patch(
        Circle(
            config.center[:2], config.outer_radius, fill=False,
            linewidth=1.6, alpha=0.95, color=colors["boundary"],
            label="Outer EE Hemisphere Projection",
        )
    )
    axis_xy.set_title("Top view", pad=16, fontweight="semibold")
    axis_xy.set_xlabel("X [cm]")
    axis_xy.set_ylabel("Y [cm]")
    axis_xy.xaxis.set_major_formatter(centimetres)
    axis_xy.yaxis.set_major_formatter(centimetres)
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.grid(True, linestyle="--", linewidth=0.8, alpha=0.55)
    sns.despine(ax=axis_xy, offset=4)

    for plot_axis in (axis_side, axis_xy):
        automatic_legend = plot_axis.get_legend()
        if automatic_legend is not None:
            automatic_legend.remove()

    figure.suptitle(
        "Random-Hemisphere Pose Coverage",
        fontsize=20,
        fontweight="bold",
        color="black",
        y=0.975,
    )
    figure.tight_layout(rect=(0.015, 0.02, 0.985, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_recording_path_plot(
    T_base_from_camera: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    camera_name: str,
    sequence_name: str,
) -> None:
    """Draw one ordered camera trajectory and sampled optical viewing axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="talk",
        style="whitegrid",
        font_scale=0.9,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD2D9",
            "grid.color": "#DCE1E6",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.labelcolor": "black",
        },
    )

    positions_cm = 100.0 * T_base_from_camera[:, :3, 3]
    target_cm = 100.0 * target
    viewing_axes = T_base_from_camera[:, :3, 2].copy()
    viewing_axis_norms = np.linalg.norm(viewing_axes, axis=1, keepdims=True)
    if np.any(viewing_axis_norms < 1e-8):
        raise ValueError("A recorded camera pose has an invalid viewing axis")
    viewing_axes /= viewing_axis_norms
    figure = plt.figure(figsize=(12.5, 10))
    axis = figure.add_subplot(111, projection="3d")
    # Keep the sequence endpoints in front of the trajectory. Axes3D normally
    # recomputes artist ordering from depth and can therefore ignore zorder.
    axis.computed_zorder = False
    axis.plot(
        positions_cm[:, 0], positions_cm[:, 1], positions_cm[:, 2],
        color="#355F8D", linewidth=1.5, alpha=0.72, zorder=2,
    )
    axis.scatter(
        positions_cm[:, 0], positions_cm[:, 1], positions_cm[:, 2],
        color="#377EB8", s=22, alpha=0.9,
        linewidth=0, depthshade=False, zorder=3,
    )
    axis.scatter(
        *positions_cm[0], color="#2A9D8F", s=140, marker="o",
        edgecolor="white", linewidth=1.2, depthshade=False,
        label="Sequence Start", zorder=20,
    )
    axis.scatter(
        *positions_cm[-1], color="#9B2226", s=155, marker="X",
        edgecolor="white", linewidth=1.2, depthshade=False,
        label="Sequence End", zorder=21,
    )
    axis.scatter(
        *target_cm, color="#E63946", s=360, marker="*",
        edgecolor="white", linewidth=1.3, label="Target", zorder=8,
    )

    # Show enough orientations to make changes visible without hiding the path.
    axis_count = min(24, len(positions_cm))
    sampled = np.unique(
        np.linspace(0, len(positions_cm) - 1, axis_count, dtype=int)
    )
    bounds = np.vstack((positions_cm, target_cm[None, :]))
    span = np.ptp(bounds, axis=0)
    arrow_length_cm = float(np.clip(0.12 * max(span.max(), 1.0), 3.0, 8.0))
    axis.quiver(
        positions_cm[sampled, 0],
        positions_cm[sampled, 1],
        positions_cm[sampled, 2],
        viewing_axes[sampled, 0],
        viewing_axes[sampled, 1],
        viewing_axes[sampled, 2],
        length=arrow_length_cm,
        normalize=True,
        color="#F4A261",
        linewidth=1.35,
        arrow_length_ratio=0.24,
        alpha=0.95,
        zorder=6,
    )

    # Equal data scaling prevents the path and viewing directions from being
    # distorted by Matplotlib's default 3D box proportions.
    center = target_cm
    half_range = max(
        1.10 * float(np.abs(bounds - center[None, :]).max()),
        5.0,
    )
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    z_upper = max(
        center[2] + half_range,
        1.10 * float(bounds[:, 2].max()),
    )
    axis.set_zlim(0.0, z_upper)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X [cm]", labelpad=12)
    axis.set_ylabel("Y [cm]", labelpad=12)
    axis.set_zlabel("Z [cm]", labelpad=12)
    axis.xaxis.pane.set_facecolor("white")
    axis.yaxis.pane.set_facecolor("white")
    axis.zaxis.pane.set_facecolor("white")
    axis.view_init(elev=24, azim=32)
    axis.set_title(
        "Example Recording Path",
        pad=24,
        fontsize=19,
        fontweight="bold",
        color="black",
    )
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_sensor_modalities_plot(
    sequence_dir: Path,
    output_stem: Path,
    frame_index: int | None,
    sequence_name: str,
) -> tuple[Path, Path, int]:
    """Render aligned RGB, depth, events, and event-frame depth in a 2x2 grid."""
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    hdf5_dir = sequence_dir / "hdf5"
    realsense_path = hdf5_dir / "realsense.h5"
    voxels_path = sequence_dir / "events" / "voxels_cam0.h5"
    projected_depth_path = hdf5_dir / "depth_in_event_frame.h5"
    required_paths = (realsense_path, voxels_path, projected_depth_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The sensor overview requires these missing files: "
            + ", ".join(missing)
        )

    with h5py.File(realsense_path, "r") as realsense_file, \
         h5py.File(voxels_path, "r") as voxels_file, \
         h5py.File(projected_depth_path, "r") as projected_depth_file:
        for file_path, handle, key in (
            (realsense_path, realsense_file, "rgb"),
            (realsense_path, realsense_file, "depth"),
            (voxels_path, voxels_file, "voxels"),
            (projected_depth_path, projected_depth_file, "depth"),
        ):
            if key not in handle:
                raise KeyError(f"{file_path} does not contain dataset {key!r}")

        common_frames = min(
            len(realsense_file["rgb"]),
            len(realsense_file["depth"]),
            len(voxels_file["voxels"]),
            len(projected_depth_file["depth"]),
        )
        if common_frames == 0:
            raise RuntimeError(f"No common sensor frames found in {sequence_dir}")
        selected_frame = common_frames // 2 if frame_index is None else frame_index
        if not 0 <= selected_frame < common_frames:
            raise IndexError(
                f"--frame-index {selected_frame} is outside the common frame "
                f"range 0..{common_frames - 1} for {sequence_dir}"
            )

        rgb = realsense_file["rgb"][selected_frame]
        raw_depth = realsense_file["depth"][selected_frame].astype(np.float32)
        voxel = voxels_file["voxels"][selected_frame].astype(np.float32)
        projected_depth = projected_depth_file["depth"][selected_frame].astype(
            np.float32
        )

    depth_scale_path = (
        _REPOSITORY_ROOT / "3d_reconstruction" / "camera_data" / "depth_scale.npz"
    )
    depth_scale = float(np.load(depth_scale_path)["scale"])
    raw_depth *= depth_scale

    # The voxel grid has already undergone the exact crop/resize transform used
    # by the model. Collapse its temporal bins into a grayscale activity image.
    events = np.abs(voxel).sum(axis=0)
    event_min = float(events.min())
    event_max = float(events.max())
    if event_max > event_min:
        events = (events - event_min) / (event_max - event_min)
    else:
        events = np.zeros_like(events)

    if events.shape != projected_depth.shape:
        raise ValueError(
            f"Voxel image shape {events.shape} does not match projected depth "
            f"shape {projected_depth.shape} in {sequence_dir}. Regenerate both "
            "with the same crop/resize mode."
        )
    # Use the original, un-dilated event support in both event-frame panels.
    overlay_event_mask = events > 0.0
    event_display = overlay_event_mask.astype(np.float32)
    # Event pixels are opaque white; zero-event pixels are transparent, so the
    # event layer cannot darken the projected-depth colours.
    event_highlight = np.ones((*events.shape, 4), dtype=np.float32)
    event_highlight[..., 3] = event_display

    sns.set_theme(
        context="talk",
        style="white",
        font_scale=0.9,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "black",
            "text.color": "black",
            "axes.titlecolor": "black",
        },
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 9), constrained_layout=True,
    )
    figure.set_constrained_layout_pads(
        w_pad=0.04,
        h_pad=0.08,
        wspace=0.04,
        hspace=0.16,
    )
    panels = (
        (axes[0, 0], rgb, "RGB", None, {}),
        (
            axes[0, 1],
            np.ma.masked_less_equal(raw_depth, 0.0),
            "Depth",
            "turbo",
            {"vmin": 0.05, "vmax": 1.0},
        ),
        (
            axes[1, 0],
            event_display,
            "Events",
            "gray",
            {"vmin": 0.0, "vmax": 1.0},
        ),
    )
    for axis, image, title, color_map, image_kwargs in panels:
        axis.imshow(image, cmap=color_map, **image_kwargs)
        axis.set_title(title, pad=10, fontweight="semibold")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    overlay_axis = axes[1, 1]
    overlay_axis.imshow(
        np.ma.masked_less_equal(projected_depth, 0.0),
        cmap="turbo",
        vmin=0.05,
        vmax=1.0,
    )
    overlay_axis.imshow(event_highlight)
    overlay_axis.set_title(
        "Depth in event frame with events",
        pad=10,
        fontweight="semibold",
    )
    overlay_axis.set_xticks([])
    overlay_axis.set_yticks([])
    for spine in overlay_axis.spines.values():
        spine.set_visible(False)

    output_path = output_stem.with_name(
        f"{output_stem.stem}_frame_{selected_frame:06d}_modalities.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # Export the bottom-right panel separately with the same target transform
    # used during training: cap metric depth at D_MAX, then scale the training
    # interval [DEPTH_MIN, D_MAX] to [0, 1]. Invalid depth remains black and
    # event pixels remain opaque white, matching the modalities overview.
    valid_depth = projected_depth > 0.0
    training_depth = np.minimum(projected_depth, D_MAX)
    training_depth = np.clip(
        (training_depth - DEPTH_MIN) / (D_MAX - DEPTH_MIN),
        0.0,
        1.0,
    )
    depth_event_image = matplotlib.colormaps["turbo"](training_depth)
    depth_event_image[~valid_depth, :3] = 0.0
    depth_event_image[~valid_depth, 3] = 1.0
    depth_event_image[overlay_event_mask, :3] = 1.0
    depth_event_image[overlay_event_mask, 3] = 1.0

    training_depth_path = output_stem.with_name(
        f"{output_stem.stem}_frame_{selected_frame:06d}"
        "_depth_events_training_scaled.png"
    )
    plt.imsave(training_depth_path, depth_event_image)
    return output_path, training_depth_path, selected_frame


def write_summary(
    path: Path,
    ee_points: np.ndarray,
    displayed_points: np.ndarray,
    valid: np.ndarray,
    config: HemisphereConfig,
    position_label: str,
    source_label: str,
    sequence_counts: list[tuple[str, int]],
    speeds_m_s: np.ndarray,
    speed_diagnostics: list[dict[str, int | str]],
) -> None:
    accepted_ee = ee_points[valid]
    accepted_displayed = displayed_points[valid]

    def bounds_text(values: np.ndarray, operation: str) -> str:
        if len(values) == 0:
            return "n/a"
        bound = values.min(axis=0) if operation == "min" else values.max(axis=0)
        return str(bound.tolist())

    lines = [
        "RandomHemisphereAgent sampler-area check",
        f"Config source: {_FRANKA_PIPELINE_ROOT / 'config_defaults.py'}",
        f"Pose source: {source_label}",
        "",
        f"Target XYZ [m]: {config.target.tolist()}",
        f"Hemisphere center XYZ [m]: {config.center.tolist()}",
        f"Inner/outer radius [m]: {config.inner_radius} / {config.outer_radius}",
        f"Allowed base XY radius [m]: {config.base_min_radius} / {config.base_max_radius}",
        f"Minimum Z [m]: {config.min_z}",
        f"Configured poses per run: {config.num_poses}",
        f"Displayed position type: {position_label}",
        "",
        f"Loaded/generated positions: {len(ee_points)}",
        f"Positions inside sampler region: {int(valid.sum())}",
        f"Fraction inside sampler region: {100.0 * float(valid.mean()):.3f}%",
        f"Valid EE XYZ minimum [m]: {bounds_text(accepted_ee, 'min')}",
        f"Valid EE XYZ maximum [m]: {bounds_text(accepted_ee, 'max')}",
        f"Displayed XYZ minimum [m]: {bounds_text(accepted_displayed, 'min')}",
        f"Displayed XYZ maximum [m]: {bounds_text(accepted_displayed, 'max')}",
        "",
        "Scope: sampler position constraints only; IK/collision feasibility is not tested.",
    ]
    if sequence_counts:
        sequence_lengths = np.asarray(
            [count for _, count in sequence_counts], dtype=np.int64
        )
        lines.extend(
            [
                "",
                f"Recorded sequences: {len(sequence_counts)}",
                (
                    "Sequence length range [pose frames]: "
                    f"{int(sequence_lengths.min())}--{int(sequence_lengths.max())}"
                ),
            ]
        )
        lines.extend(
            f"  {sequence_name}: {count} poses"
            for sequence_name, count in sequence_counts
        )
    finite_speeds = np.asarray(speeds_m_s, dtype=np.float64)
    finite_speeds = finite_speeds[
        np.isfinite(finite_speeds) & (finite_speeds >= 0.0)
    ]
    if finite_speeds.size:
        lines.extend(
            [
                "",
                f"Highest measured EE translational speed [m/s]: {finite_speeds.max():.6f}",
            ]
        )
    if speed_diagnostics:
        total_pose_samples = sum(int(row["pose_count"]) for row in speed_diagnostics)
        total_aligned = sum(int(row["aligned_count"]) for row in speed_diagnostics)
        total_without_timestamp = sum(
            int(row["poses_without_timestamp"]) for row in speed_diagnostics
        )
        total_unused_timestamps = sum(
            int(row["unused_timestamps"]) for row in speed_diagnostics
        )
        total_invalid_dt = sum(int(row["invalid_dt"]) for row in speed_diagnostics)
        total_nonfinite_speed = sum(
            int(row["nonfinite_speed"]) for row in speed_diagnostics
        )
        total_valid_speeds = sum(
            int(row["valid_speed_count"]) for row in speed_diagnostics
        )
        lines.extend(
            [
                "",
                "EE speed sample diagnostics",
                f"Pose samples read for speed calculation: {total_pose_samples}",
                f"Pose/timestamp pairs after length alignment: {total_aligned}",
                f"Poses without a corresponding timestamp: {total_without_timestamp}",
                f"Extra timestamps without a corresponding pose: {total_unused_timestamps}",
                f"Rejected pairs with non-finite or non-positive time difference: {total_invalid_dt}",
                f"Rejected pairs producing non-finite speed: {total_nonfinite_speed}",
                f"Valid speed samples used by the plot: {total_valid_speeds}",
            ]
        )
        problematic = [
            row
            for row in speed_diagnostics
            if any(
                int(row[key]) > 0
                for key in (
                    "poses_without_timestamp",
                    "unused_timestamps",
                    "invalid_dt",
                    "nonfinite_speed",
                )
            )
        ]
        if problematic:
            lines.append("Sequences with rejected or unpaired speed samples:")
            for row in problematic:
                lines.append(
                    "  "
                    f"{row['sequence']}: poses={row['pose_count']}, "
                    f"timestamps={row['timestamp_count']} "
                    f"({row['timestamp_source']}), aligned={row['aligned_count']}, "
                    f"missing_timestamp={row['poses_without_timestamp']}, "
                    f"extra_timestamp={row['unused_timestamps']}, "
                    f"invalid_dt={row['invalid_dt']}, "
                    f"nonfinite_speed={row['nonfinite_speed']}, "
                    f"used={row['valid_speed_count']}"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pose coverage and optional recording trajectories."
    )
    parser.add_argument(
        "--samples", type=int, default=120_000,
        help="Number of Monte Carlo candidates used to show the volume.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=(
            "Load recorded hdf5/poses.h5 files instead of generating Monte "
            "Carlo samples. Dataset roots are restricted to their train/ and "
            "eval/ subdirectories; direct sequence and split paths remain "
            "supported. Paths such as data/new are resolved relative to "
            "3d_reconstruction."
        ),
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help=(
            "Additionally render the ordered camera path from one sequence. "
            "The directory must directly contain hdf5/poses.h5. Relative paths "
            "are resolved against 3d_reconstruction."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help=(
            "Frame shown in the four-panel sensor overview generated by "
            "--sequence-dir. The middle common frame is used by default."
        ),
    )
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--event-camera-positions",
        action="store_true",
        help=(
            "Convert EE poses to calibrated event-camera centers before plotting. "
            "Sampling validity remains defined by the EE constraints."
        ),
    )
    camera_group.add_argument(
        "--depth-camera-positions",
        action="store_true",
        help=(
            "Convert EE poses to calibrated depth-camera centers before plotting. "
            "This is the camera used by RandomHemisphereAgent's look-at rotation."
        ),
    )
    parser.add_argument(
        "--view-camera-z",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option; ignored because the side view is "
            "now a direct 2D Y-Z projection."
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary", type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.frame_index is not None and args.sequence_dir is None:
        parser.error("--frame-index requires --sequence-dir")

    config = load_current_config()
    if args.data_dir is not None:
        data_dir = _resolve_data_dir(args.data_dir)
        ee_transforms, sequence_counts = load_recorded_ee_poses(data_dir)
        ee_points = ee_transforms[:, :3, 3]
        valid = accepted_position_mask(ee_points, config)
        source_mode = "recorded"
        source_label = f"{len(ee_points)} recorded poses ({data_dir.name})"
        source_stem = f"recorded_{data_dir.name}"
        if args.event_camera_positions:
            displayed_points = camera_positions_from_recorded_ee(
                ee_transforms, camera="event"
            )
            position_label = "event-camera"
            default_stem = f"random_hemisphere_{source_stem}_event_camera"
        elif args.depth_camera_positions:
            displayed_points = camera_positions_from_recorded_ee(
                ee_transforms, camera="depth"
            )
            position_label = "depth-camera"
            default_stem = f"random_hemisphere_{source_stem}_depth_camera"
        else:
            displayed_points = ee_points
            position_label = "EE"
            default_stem = f"random_hemisphere_{source_stem}_ee"
    else:
        ee_points = sample_hollow_hemisphere(config, args.samples, args.seed)
        valid = accepted_position_mask(ee_points, config)
        validate_samples(ee_points, valid, config)
        sequence_counts = []
        source_mode = "simulated"
        source_label = f"{len(ee_points)} Monte Carlo candidates"
        if args.event_camera_positions:
            displayed_points = camera_positions_from_sampled_ee(
                ee_points, config, args.seed, camera="event"
            )
            position_label = "event-camera"
            default_stem = "random_hemisphere_event_camera_valid_area"
        elif args.depth_camera_positions:
            displayed_points = camera_positions_from_sampled_ee(
                ee_points, config, args.seed, camera="depth"
            )
            position_label = "depth-camera"
            default_stem = "random_hemisphere_depth_camera_valid_area"
        else:
            displayed_points = ee_points
            position_label = "EE"
            default_stem = "random_hemisphere_valid_area"
    output_path = (
        args.output.resolve()
        if args.output is not None
        else _HERE / "plots" / f"{default_stem}.png"
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else _HERE / "plots" / f"{default_stem}.txt"
    )
    speed_distribution_path = None
    recorded_speeds = np.empty(0, dtype=np.float64)
    speed_diagnostics: list[dict[str, int | str]] = []
    speed_source_dir = args.data_dir or args.sequence_dir
    if speed_source_dir is not None:
        recorded_speeds, speed_diagnostics = load_recorded_translational_speeds(
            speed_source_dir
        )
        if recorded_speeds.size:
            speed_distribution_path = output_path.with_name(
                f"{output_path.stem}_speed_distribution.png"
            )
            create_speed_distribution_plot(
                recorded_speeds,
                speed_distribution_path,
            )
    create_plot(
        displayed_points,
        valid,
        config,
        output_path,
        position_label,
        source_mode,
        source_label,
    )
    write_summary(
        summary_path,
        ee_points,
        displayed_points,
        valid,
        config,
        position_label,
        source_label,
        sequence_counts,
        recorded_speeds,
        speed_diagnostics,
    )

    sequence_output_path = None
    sensor_overview_path = None
    training_depth_overlay_path = None
    if args.sequence_dir is not None:
        resolved_sequence_dir = _resolve_data_dir(args.sequence_dir)
        sequence_ee_transforms = load_sequence_ee_poses(resolved_sequence_dir)
        sequence_camera = "depth" if args.depth_camera_positions else "event"
        sequence_camera_transforms = camera_transforms_from_recorded_ee(
            sequence_ee_transforms, camera=sequence_camera
        )
        data_root = _REPOSITORY_ROOT / "3d_reconstruction" / "data"
        try:
            sequence_relative_path = resolved_sequence_dir.relative_to(data_root)
        except ValueError:
            sequence_relative_path = Path(resolved_sequence_dir.name)
        sequence_name = str(sequence_relative_path)
        sequence_stem = "_".join(sequence_relative_path.parts)
        sequence_output_path = output_path.with_name(
            f"{output_path.stem}_{sequence_stem}_recording_path_3d.png"
        )
        create_recording_path_plot(
            sequence_camera_transforms,
            config.target,
            sequence_output_path,
            sequence_camera,
            sequence_name,
        )
        sensor_output_stem = output_path.with_name(
            f"{output_path.stem}_{sequence_stem}"
        )
        sensor_overview_path, training_depth_overlay_path, _ = create_sensor_modalities_plot(
            resolved_sequence_dir,
            sensor_output_stem,
            args.frame_index,
            sequence_name,
        )

    print(f"Pose source: {source_label}")
    print(
        f"Inside sampler region: {int(valid.sum())} "
        f"({100.0 * float(valid.mean()):.2f}%)"
    )
    print(f"Displayed positions: {position_label}")
    print(f"Plot written to: {output_path}")
    print(f"Summary written to: {summary_path}")
    if speed_distribution_path is not None:
        print(f"Speed distribution written to: {speed_distribution_path}")
    if sequence_output_path is not None:
        print(f"3D recording path written to: {sequence_output_path}")
    if sensor_overview_path is not None:
        print(f"Sensor overview written to: {sensor_overview_path}")
    if training_depth_overlay_path is not None:
        print(
            "Training-scaled depth/events image written to: "
            f"{training_depth_overlay_path}"
        )


if __name__ == "__main__":
    main()
