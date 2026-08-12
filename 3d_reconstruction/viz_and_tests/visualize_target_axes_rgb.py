#!/usr/bin/env python3
"""Draw a base-aligned coordinate system in synchronized RGB and depth frames.

The selected frame is random unless ``--frame-index`` is supplied. The target
origin is expressed in the robot base frame; its axes retain the orientation of
the base coordinate system and are only translated to the target point.

Example:
    python3 viz_and_tests/visualize_target_axes_rgb.py \
        --data-dir data/real/train/lego_1

    python3 viz_and_tests/visualize_target_axes_rgb.py \
        --data-dir data/real/train/lego_1 --frame-index 250 \
        --axis-length 0.05 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from config import (
    DEPTH_VIZ_MAX,
    DEPTH_VIZ_MIN,
    SPATIAL_TARGET_X,
    SPATIAL_TARGET_Y,
    SPATIAL_TARGET_Z,
)


AXIS_COLORS_BGR = (
    (0, 0, 255),    # X: red
    (0, 180, 0),    # Y: green
    (255, 0, 0),    # Z: blue
)
AXIS_LABELS = ("X", "Y", "Z")


def load_rgb_calibration(calib_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsics_path = calib_dir / "rs_rgb_intrinsics.npz"
    transform_path = calib_dir / "T_rgb_from_ee.npz"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"Missing RGB intrinsics: {intrinsics_path}")
    if not transform_path.is_file():
        raise FileNotFoundError(f"Missing hand-eye transform: {transform_path}")

    intrinsics = np.load(intrinsics_path)
    K = intrinsics["camera_matrix"].astype(np.float64)
    distortion = intrinsics.get("dist_coeffs")
    if distortion is None:
        distortion = intrinsics.get("distortion_coefficients")
    if distortion is None:
        distortion = np.zeros(5, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)

    T_rgb_from_ee = np.load(transform_path)["T"].astype(np.float64)
    return K, distortion, T_rgb_from_ee


def load_depth_calibration(
    calib_dir: Path,
    T_rgb_from_ee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load depth intrinsics, EE-to-depth transform, and metric depth scale."""
    intrinsics_path = calib_dir / "rs_depth_intrinsics.npz"
    extrinsics_path = calib_dir / "T_color_from_depth.npz"
    scale_path = calib_dir / "depth_scale.npz"
    for path in (intrinsics_path, extrinsics_path, scale_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing depth calibration file: {path}")

    intrinsics = np.load(intrinsics_path)
    K = intrinsics["camera_matrix"].astype(np.float64)
    distortion = intrinsics.get("dist_coeffs")
    if distortion is None:
        distortion = intrinsics.get("distortion_coefficients")
    if distortion is None:
        distortion = np.zeros(5, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)

    T_color_from_depth = np.load(extrinsics_path)["T"].astype(np.float64)
    T_depth_from_ee = np.linalg.inv(T_color_from_depth) @ T_rgb_from_ee
    depth_scale = float(np.load(scale_path)["scale"])
    return K, distortion, T_depth_from_ee, depth_scale


def project_target_axes(
    T_base_from_ee: np.ndarray,
    T_camera_from_ee: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
    target_base: np.ndarray,
    axis_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected origin/endpoints and their camera-frame depths."""
    points_base = np.vstack(
        (
            target_base,
            target_base + np.array([axis_length, 0.0, 0.0]),
            target_base + np.array([0.0, axis_length, 0.0]),
            target_base + np.array([0.0, 0.0, axis_length]),
        )
    )

    T_camera_from_base = T_camera_from_ee @ np.linalg.inv(T_base_from_ee)
    points_base_h = np.column_stack((points_base, np.ones(len(points_base))))
    points_rgb = (T_camera_from_base @ points_base_h.T).T[:, :3]

    projected, _ = cv2.projectPoints(
        points_rgb,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        distortion,
    )
    return projected.reshape(-1, 2), points_rgb[:, 2]


def draw_axes_bgr(
    image_bgr: np.ndarray,
    projected: np.ndarray,
    depths: np.ndarray,
    thickness: int,
) -> np.ndarray:
    """Draw visible base-frame axes on a BGR image."""
    output = image_bgr.copy()
    if depths[0] <= 0:
        raise ValueError("The target point is behind the RGB camera in this frame")

    origin = tuple(np.rint(projected[0]).astype(int))
    cv2.circle(output, origin, max(3, thickness + 1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(output, origin, max(3, thickness + 1), (0, 0, 0), 1, cv2.LINE_AA)

    for axis_index, (color, label) in enumerate(zip(AXIS_COLORS_BGR, AXIS_LABELS), start=1):
        if depths[axis_index] <= 0:
            continue
        endpoint = tuple(np.rint(projected[axis_index]).astype(int))
        cv2.arrowedLine(
            output,
            origin,
            endpoint,
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.putText(
            output,
            label,
            (endpoint[0] + 5, endpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            max(1, thickness),
            cv2.LINE_AA,
        )
    return output


def colorize_depth(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert raw RealSense depth to a fixed-range TURBO visualization."""
    depth_m = depth_raw.astype(np.float32) * depth_scale
    valid = depth_m > 0
    normalized = np.clip(
        (depth_m - DEPTH_VIZ_MIN) / (DEPTH_VIZ_MAX - DEPTH_VIZ_MIN),
        0.0,
        1.0,
    )
    depth_u8 = np.rint(normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def visualize(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = args.data_dir.resolve()
    rgb_path = data_dir / "hdf5" / "realsense.h5"
    poses_path = data_dir / "hdf5" / "poses.h5"
    for path in (rgb_path, poses_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    calib_dir = args.calib_dir.resolve()
    K_rgb, distortion_rgb, T_rgb_from_ee = load_rgb_calibration(calib_dir)
    K_depth, distortion_depth, T_depth_from_ee, depth_scale = load_depth_calibration(
        calib_dir, T_rgb_from_ee
    )
    with h5py.File(rgb_path, "r") as rgb_file, h5py.File(poses_path, "r") as poses_file:
        if "rgb" not in rgb_file or "depth" not in rgb_file or "ee_T" not in poses_file:
            raise KeyError(
                "Expected datasets 'rgb' and 'depth' in realsense.h5 and "
                "'ee_T' in poses.h5"
            )
        frame_count = min(
            len(rgb_file["rgb"]),
            len(rgb_file["depth"]),
            len(poses_file["ee_T"]),
        )
        if frame_count == 0:
            raise RuntimeError(f"No aligned RGB frames and poses found in {data_dir}")

        if args.frame_index is None:
            frame_index = int(np.random.default_rng(args.seed).integers(frame_count))
        else:
            frame_index = args.frame_index
        if not 0 <= frame_index < frame_count:
            raise IndexError(f"Frame {frame_index} is outside the valid range 0..{frame_count - 1}")

        rgb = rgb_file["rgb"][frame_index]
        depth_raw = rgb_file["depth"][frame_index]
        T_base_from_ee = poses_file["ee_T"][frame_index].astype(np.float64)

    target = np.asarray(args.target, dtype=np.float64)
    projected_rgb, depths_rgb = project_target_axes(
        T_base_from_ee,
        T_rgb_from_ee,
        K_rgb,
        distortion_rgb,
        target,
        args.axis_length,
    )
    projected_depth, depths_depth = project_target_axes(
        T_base_from_ee,
        T_depth_from_ee,
        K_depth,
        distortion_depth,
        target,
        args.axis_length,
    )
    annotated_rgb = draw_axes_bgr(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        projected_rgb,
        depths_rgb,
        args.thickness,
    )
    annotated_depth = draw_axes_bgr(
        colorize_depth(depth_raw, depth_scale),
        projected_depth,
        depths_depth,
        args.thickness,
    )

    output = args.output
    if output is None:
        output = _ROOT / "viz_and_tests" / "plots" / (
            f"target_axes_rgb_{data_dir.name}_frame_{frame_index:05d}.png"
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated_rgb):
        raise OSError(f"Could not write output image: {output}")

    depth_output = args.depth_output
    if depth_output is None:
        if "target_axes_rgb" in output.stem:
            depth_stem = output.stem.replace("target_axes_rgb", "target_axes_depth")
        else:
            depth_stem = output.stem + "_depth"
        depth_output = output.with_name(depth_stem + output.suffix)
    depth_output = depth_output.resolve()
    depth_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(depth_output), annotated_depth):
        raise OSError(f"Could not write depth output image: {depth_output}")

    print(f"Sequence:     {data_dir}")
    print(f"Frame:        {frame_index} / {frame_count - 1}")
    print(f"Target base:  {target.tolist()} m")
    print(f"Axis length:  {args.axis_length:.3f} m")
    print(f"Saved RGB:    {output}")
    print(f"Saved depth:  {depth_output}")

    if args.show:
        cv2.imshow("RGB frame with target coordinate system", annotated_rgb)
        cv2.imshow("Depth frame with target coordinate system", annotated_depth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return output, depth_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select an RGB frame and draw the robot-base coordinate axes "
            "translated to the configured target point."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Sequence directory containing hdf5/realsense.h5 and hdf5/poses.h5.",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=_ROOT / "camera_data",
        help="Calibration directory (default: 3d_reconstruction/camera_data).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame to visualize. Omit to select a random frame.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random frame selection.")
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(SPATIAL_TARGET_X, SPATIAL_TARGET_Y, SPATIAL_TARGET_Z),
        help="Target position in the robot base frame, in metres.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.05,
        help="Length of each axis in metres (default: 0.05).",
    )
    parser.add_argument("--thickness", type=int, default=2, help="Line thickness in pixels.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument(
        "--depth-output",
        type=Path,
        default=None,
        help="Depth output PNG path. By default it is saved beside the RGB output.",
    )
    parser.add_argument("--show", action="store_true", help="Show the annotated image in a window.")
    args = parser.parse_args()
    if args.axis_length <= 0:
        parser.error("--axis-length must be positive")
    if args.thickness <= 0:
        parser.error("--thickness must be positive")
    return args


if __name__ == "__main__":
    visualize(parse_args())
