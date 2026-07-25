#!/usr/bin/env python3
"""Visualize the position volume accepted by RandomHemisphereAgent.

The script loads the active values from ``franka_pipeline/config_defaults.py``
and draws the intersection of the hollow upper hemisphere with the sampler's
base-distance and minimum-height constraints. It can show either Monte Carlo
samples or every pose recorded below ``--data_dir``. Robot IK, self-collision,
and environment-collision checks are not represented.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _HERE.parents[1]
_FRANKA_PIPELINE_ROOT = _REPOSITORY_ROOT / "franka_pipeline"
sys.path.insert(0, str(_FRANKA_PIPELINE_ROOT))

import config_defaults as cfg  # noqa: E402


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


def load_recorded_ee_poses(
    data_dir: Path,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Recursively load every recorded ``ee_T`` pose below a data directory."""
    import h5py

    direct_pose_file = data_dir / "hdf5" / "poses.h5"
    pose_files = (
        [direct_pose_file]
        if direct_pose_file.is_file()
        else sorted(data_dir.rglob("hdf5/poses.h5"))
    )
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


def camera_positions_from_recorded_ee(
    T_base_from_ee: np.ndarray,
    camera: str,
) -> np.ndarray:
    """Transform recorded EE poses to calibrated camera centers."""
    _, T_event_from_ee, T_depth_from_ee = _load_camera_calibration()
    if camera == "event":
        T_camera_from_ee = T_event_from_ee
    elif camera == "depth":
        T_camera_from_ee = T_depth_from_ee
    else:
        raise ValueError(f"Unsupported camera: {camera}")
    T_ee_from_camera = np.linalg.inv(T_camera_from_ee)
    T_base_from_camera = np.matmul(T_base_from_ee, T_ee_from_camera)
    return T_base_from_camera[:, :3, 3]


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
            "figure.facecolor": "#F7F8FA",
            "axes.facecolor": "#FBFCFD",
            "axes.edgecolor": "#CBD2D9",
            "grid.color": "#DCE1E6",
            "grid.alpha": 0.65,
        },
    )
    colors = {
        "inside": "#168AAD",
        "outside": "#AEB8C2",
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

    sns.scatterplot(
        x=rejected_draw[:, 1], y=rejected_draw[:, 2],
        s=6, alpha=0.10, color=colors["outside"], linewidth=0,
        rasterized=True, label="outside sampler region", ax=axis_side,
    )
    sns.scatterplot(
        x=accepted_draw[:, 1], y=accepted_draw[:, 2],
        s=8, alpha=0.34, color=colors["inside"], linewidth=0,
        rasterized=True, label="inside sampler region", ax=axis_side,
    )
    axis_side.scatter(
        config.target[1], config.target[2], marker="*", s=300,
        color=colors["target"], edgecolor="white", linewidth=1.4,
        label="look-at target", zorder=10,
    )
    if not np.allclose(config.center, config.target):
        axis_side.scatter(
            config.center[1], config.center[2], marker="x", s=90,
            color=colors["center"], label="hemisphere center", zorder=9,
        )
    arc_angle = np.linspace(0.0, np.pi, 240)
    for radius, alpha, label in (
        (config.outer_radius, 0.95, "outer EE hemisphere"),
        (config.inner_radius, 0.65, "inner EE hemisphere"),
    ):
        axis_side.plot(
            config.center[1] + radius * np.cos(arc_angle),
            config.center[2] + radius * np.sin(arc_angle),
            color=colors["boundary"], linewidth=1.6, alpha=alpha,
            label=label,
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
    axis_side.legend(
        loc="upper center", bbox_to_anchor=(0.56, 0.97),
        fontsize=8, frameon=True, framealpha=0.94,
        edgecolor="#D5DAE0",
    )

    sns.scatterplot(
        x=rejected_draw[:, 0], y=rejected_draw[:, 1],
        s=6, alpha=0.10, color=colors["outside"], linewidth=0,
        rasterized=True, label="outside sampler region", ax=axis_xy,
    )
    sns.scatterplot(
        x=accepted_draw[:, 0], y=accepted_draw[:, 1],
        s=8, alpha=0.34, color=colors["inside"], linewidth=0,
        rasterized=True, label="inside sampler region", ax=axis_xy,
    )
    axis_xy.scatter(
        config.target[0], config.target[1], marker="*", s=150,
        color=colors["target"], edgecolor="white", linewidth=0.8,
        label="look-at target", zorder=8,
    )
    for radius, linestyle, label in (
        (config.base_min_radius, "--", "base exclusion radius"),
        (config.base_max_radius, "-", "base maximum radius"),
    ):
        axis_xy.add_patch(
            Circle((0.0, 0.0), radius, fill=False, linestyle=linestyle,
                   linewidth=1.5, color=colors["target"], label=label)
        )
    axis_xy.add_patch(
        Circle(
            config.center[:2], config.outer_radius, fill=False,
            linewidth=1.6, color=colors["boundary"],
            label="EE hemisphere projection",
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
    axis_xy.legend(
        loc="best", fontsize=8, frameon=True, framealpha=0.94,
        edgecolor="#D5DAE0",
    )

    valid_fraction = 100.0 * float(valid.mean())
    figure.suptitle(
        f"Random-Hemisphere Pose Coverage · {position_label}",
        fontsize=20,
        fontweight="bold",
        color="#1F2933",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        f"{source_label}  ·  {valid_fraction:.1f}% inside sampler region  ·  "
        f"shell {100.0 * config.inner_radius:.0f}–"
        f"{100.0 * config.outer_radius:.0f} cm",
        ha="center",
        color="#52606D",
        fontsize=11,
    )
    figure.text(
        0.5,
        0.025,
        "Region membership is evaluated in EE coordinates. In camera-position modes, "
        "points are transformed through calibration; the orange and red boundaries "
        "remain EE constraints.",
        ha="center",
        color="#66788A",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.015, 0.07, 0.985, 0.89))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    path: Path,
    ee_points: np.ndarray,
    displayed_points: np.ndarray,
    valid: np.ndarray,
    config: HemisphereConfig,
    position_label: str,
    source_label: str,
    sequence_counts: list[tuple[str, int]],
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
        lines.extend(["", f"Recorded sequences: {len(sequence_counts)}"])
        lines.extend(
            f"  {sequence_name}: {count} poses"
            for sequence_name, count in sequence_counts
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize RandomHemisphereAgent's valid position-sampling area."
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
            "Load all recorded hdf5/poses.h5 files recursively below this "
            "directory instead of generating Monte Carlo samples. Paths such as "
            "data/new are also resolved relative to 3d_reconstruction."
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
    )

    print(f"Pose source: {source_label}")
    print(
        f"Inside sampler region: {int(valid.sum())} "
        f"({100.0 * float(valid.mean()):.2f}%)"
    )
    print(f"Displayed positions: {position_label}")
    print(f"Plot written to: {output_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
