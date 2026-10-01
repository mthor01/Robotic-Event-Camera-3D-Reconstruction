#!/usr/bin/env python3
"""Visualize the spatial evaluation cube in an RGB and a depth frame.

Examples:
    python3 viz_and_tests/visualize_spatial_mask_cube.py \
        --sequence lego_1 --frame 250

    python3 viz_and_tests/visualize_spatial_mask_cube.py \
        --sequence data/real/validation/lego_1 --farthest-frame --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    PREPROCESS_CROP_HW,
    PREPROCESS_RESIZE_HW,
    DATA_ROOT,
    DEPTH_VIZ_MAX,
    DEPTH_VIZ_MIN,
    SPATIAL_CUBE_SIDE,
    SPATIAL_TARGET_X,
    SPATIAL_TARGET_Y,
    SPATIAL_TARGET_Z,
)
from helpers import depth_cube_mask, transform_intrinsics  # noqa: E402


CUBE_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
CUBE_COLOR_BGR = (255, 80, 30)
MASK_COLOR_BGR = np.array([255, 80, 30], dtype=np.float32)
BOUNDARY_COLOR_BGR = np.array([42, 95, 232], dtype=np.uint8)
SPATIAL_MASK_VERTICAL_SHIFT_M = 0.005
RAISED_CUBE_BOTTOM_Z_M = 0.01 + SPATIAL_MASK_VERTICAL_SHIFT_M
OUTSIDE_OVERLAY_RGB = np.array([76, 114, 176], dtype=np.float32) / 255.0
REGION_COLORS_RGB = {
    "outside": np.asarray([35, 39, 47], dtype=np.uint8),
    "inside_excluded": np.asarray([150, 155, 165], dtype=np.uint8),
    "non_boundary": np.asarray([43, 116, 189], dtype=np.uint8),
    "boundary": np.asarray([232, 95, 42], dtype=np.uint8),
}


def resolve_sequence(sequence: str, data_root: Path) -> Path:
    """Resolve either a sequence path or a sequence name below the data root."""
    supplied = Path(sequence).expanduser()
    if supplied.is_dir():
        return supplied.resolve()

    root = data_root.expanduser().resolve()
    direct = root / sequence
    if direct.is_dir():
        return direct

    matches = sorted(
        path for path in root.glob(f"*/{sequence}")
        if path.is_dir()
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find sequence '{sequence}' as a path or below {root}"
        )
    if len(matches) > 1:
        choices = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Sequence name '{sequence}' is ambiguous; pass its path instead. "
            f"Matches: {choices}"
        )
    return matches[0].resolve()


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera intrinsics: {path}")
    values = np.load(path)
    K = values["camera_matrix"].astype(np.float64)
    distortion = values.get("dist_coeffs")
    if distortion is None:
        distortion = values.get("distortion_coefficients")
    if distortion is None:
        distortion = np.zeros(5, dtype=np.float64)
    return K, np.asarray(distortion, dtype=np.float64).reshape(-1)


def load_calibration(
    calib_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    K_rgb, distortion_rgb = load_intrinsics(calib_dir / "rs_rgb_intrinsics.npz")
    K_depth, distortion_depth = load_intrinsics(calib_dir / "rs_depth_intrinsics.npz")
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_color_from_depth = np.load(calib_dir / "T_color_from_depth.npz")["T"].astype(np.float64)
    T_depth_from_ee = np.linalg.inv(T_color_from_depth) @ T_rgb_from_ee
    depth_scale = float(np.load(calib_dir / "depth_scale.npz")["scale"])
    return (
        K_rgb,
        distortion_rgb,
        T_rgb_from_ee,
        K_depth,
        distortion_depth,
        T_depth_from_ee,
        depth_scale,
    )


def load_event_calibration(
    calib_dir: Path,
    depth_file: h5py.File,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the event pose transform and intrinsics matching stored depth."""
    event = np.load(calib_dir / "event_intrinsics.npz")
    K_native = event["camera_matrix"].astype(np.float64)
    native_w, native_h = (int(value) for value in event["image_size"])
    resize_hw = PREPROCESS_RESIZE_HW
    crop_hw = PREPROCESS_CROP_HW
    expected_depth_hw = tuple(int(value) for value in depth_file["depth"].shape[-2:])
    if expected_depth_hw != tuple(resize_hw):
        raise RuntimeError(
            f"Stored event-frame depth resolution {expected_depth_hw} does not "
            f"match the configured output resolution {tuple(resize_hw)}"
        )
    K_event = transform_intrinsics(
        K_native,
        (native_h, native_w),
        resize_hw,
        crop_hw,
    )
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float64)
    return K_event, T_event_from_rgb @ T_rgb_from_ee


def cube_corners(center: np.ndarray, side: float) -> np.ndarray:
    half = side / 2.0
    return np.asarray(
        [center + half * np.array([x, y, z])
         for z in (-1.0, 1.0)
         for y in (-1.0, 1.0)
         for x in (-1.0, 1.0)],
        dtype=np.float64,
    )


def project_points(
    points_base: np.ndarray,
    T_base_from_ee: np.ndarray,
    T_camera_from_ee: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T_camera_from_base = T_camera_from_ee @ np.linalg.inv(T_base_from_ee)
    homogeneous = np.column_stack((points_base, np.ones(len(points_base))))
    points_camera = (T_camera_from_base @ homogeneous.T).T[:, :3]
    projected, _ = cv2.projectPoints(
        points_camera,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        distortion,
    )
    return projected.reshape(-1, 2), points_camera[:, 2]


def draw_cube(
    image: np.ndarray,
    projected: np.ndarray,
    depths: np.ndarray,
    thickness: int,
) -> np.ndarray:
    output = image.copy()
    for first, second in CUBE_EDGES:
        if depths[first] <= 0.0 or depths[second] <= 0.0:
            continue
        point_a = tuple(np.rint(projected[first]).astype(int))
        point_b = tuple(np.rint(projected[second]).astype(int))
        cv2.line(output, point_a, point_b, CUBE_COLOR_BGR, thickness, cv2.LINE_AA)
    for point, depth in zip(projected, depths):
        if depth > 0.0:
            cv2.circle(
                output,
                tuple(np.rint(point).astype(int)),
                max(2, thickness),
                CUBE_COLOR_BGR,
                -1,
                cv2.LINE_AA,
            )
    return output


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.clip(
        (depth_m - DEPTH_VIZ_MIN) / (DEPTH_VIZ_MAX - DEPTH_VIZ_MIN),
        0.0,
        1.0,
    )
    colored = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored[~valid] = 0
    return colored


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    output = image.copy()
    blended = (
        (1.0 - alpha) * output[mask].astype(np.float32)
        + alpha * MASK_COLOR_BGR
    )
    output[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, CUBE_COLOR_BGR, 2, cv2.LINE_AA)
    return output


def boundary_mask(domain: np.ndarray, dilation_px: int) -> np.ndarray:
    """Match evaluation.py's inside, four-connected, dilated boundary."""
    domain_u8 = domain.astype(np.uint8)
    eroded = cv2.erode(
        domain_u8,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    edge = domain & ~eroded.astype(bool)
    if dilation_px > 0:
        kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool)
    return edge & domain


def black_outside(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(image)
    output[mask] = image[mask]
    return output


def reproject_depth_mask_to_rgb(
    depth_m: np.ndarray,
    depth_mask: np.ndarray,
    K_depth: np.ndarray,
    T_rgb_from_depth: np.ndarray,
    K_rgb: np.ndarray,
    distortion_rgb: np.ndarray,
    rgb_shape: tuple[int, int],
) -> np.ndarray:
    """Project measured depth pixels accepted by the cube into the RGB image."""
    ys, xs = np.nonzero(depth_mask)
    rgb_mask = np.zeros(rgb_shape, dtype=np.uint8)
    if xs.size == 0:
        return rgb_mask.astype(bool)

    z = depth_m[ys, xs].astype(np.float64)
    points_depth = np.column_stack(
        (
            (xs.astype(np.float64) - K_depth[0, 2]) * z / K_depth[0, 0],
            (ys.astype(np.float64) - K_depth[1, 2]) * z / K_depth[1, 1],
            z,
            np.ones_like(z),
        )
    )
    points_rgb = (T_rgb_from_depth @ points_depth.T).T[:, :3]
    visible = points_rgb[:, 2] > 0.0
    if not np.any(visible):
        return rgb_mask.astype(bool)
    projected, _ = cv2.projectPoints(
        points_rgb[visible],
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K_rgb,
        distortion_rgb,
    )
    pixels = np.rint(projected.reshape(-1, 2)).astype(np.int32)
    height, width = rgb_shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[inside]
    rgb_mask[pixels[:, 1], pixels[:, 0]] = 1
    # Splat sparse depth samples into a continuous display mask without
    # changing the underlying cube-membership test.
    kernel = np.ones((3, 3), dtype=np.uint8)
    rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel)
    rgb_mask = cv2.dilate(rgb_mask, kernel, iterations=1)
    return rgb_mask.astype(bool)


def visualize(args: argparse.Namespace) -> Path:
    sequence_dir = resolve_sequence(args.sequence, args.data_root)
    depth_path = sequence_dir / "hdf5" / "depth_in_event_frame.h5"
    poses_path = sequence_dir / "hdf5" / "poses.h5"
    for path in (depth_path, poses_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    center = np.asarray(args.center, dtype=np.float64)
    center[2] += SPATIAL_MASK_VERTICAL_SHIFT_M
    with h5py.File(depth_path, "r") as depths, \
            h5py.File(poses_path, "r") as poses:
        K_event, T_event_from_ee = load_event_calibration(
            args.calib_dir.resolve(), depths
        )
        frame_count = min(len(depths["depth"]), len(poses["ee_T"]))
        if args.farthest_frame:
            T_base_from_ee_all = poses["ee_T"][:frame_count].astype(np.float64)
            T_ee_from_event = np.linalg.inv(T_event_from_ee)
            camera_positions = (
                T_base_from_ee_all @ T_ee_from_event[None, :, :]
            )[:, :3, 3]
            distances = np.linalg.norm(camera_positions - center[None, :], axis=1)
            distances[~np.isfinite(distances)] = -np.inf
            if not np.any(np.isfinite(distances)):
                raise RuntimeError("No valid camera poses were found in the sequence")
            frame_index = int(np.argmax(distances))
            selected_distance = float(distances[frame_index])
        else:
            frame_index = args.frame
            selected_distance = None
        if not 0 <= frame_index < frame_count:
            raise IndexError(f"Frame {frame_index} is outside the valid range 0..{frame_count - 1}")
        depth_m = depths["depth"][frame_index].astype(np.float32)
        T_base_from_ee = poses["ee_T"][frame_index].astype(np.float64)

    T_event_from_base = T_event_from_ee @ np.linalg.inv(T_base_from_ee)
    mask = depth_cube_mask(
        depth_m,
        T_event_from_base,
        K_event,
        center,
        args.cube_side / 2.0,
    )
    raised_center = np.array(
        [center[0], center[1], RAISED_CUBE_BOTTOM_Z_M + args.cube_side / 2.0],
        dtype=np.float64,
    )
    raised_mask = depth_cube_mask(
        depth_m,
        T_event_from_base,
        K_event,
        raised_center,
        args.cube_side / 2.0,
    )
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
    raised_domain = valid_depth & raised_mask
    raised_boundary = boundary_mask(raised_domain, args.boundary_dilation)

    raised_non_boundary = raised_domain & ~raised_boundary
    regions = np.empty((*depth_m.shape, 3), dtype=np.uint8)
    regions[:] = REGION_COLORS_RGB["outside"]
    regions[raised_mask] = REGION_COLORS_RGB["inside_excluded"]
    regions[raised_non_boundary] = REGION_COLORS_RGB["non_boundary"]
    regions[raised_boundary] = REGION_COLORS_RGB["boundary"]

    figure, axes = plt.subplots(2, 2, figsize=(8, 6.5))
    axes = axes.ravel()
    outside_overlay = np.zeros((*depth_m.shape, 4), dtype=np.float32)
    outside_overlay[..., :3] = OUTSIDE_OVERLAY_RGB

    axes[0].imshow(
        depth_m,
        cmap="turbo",
        vmin=DEPTH_VIZ_MIN,
        vmax=DEPTH_VIZ_MAX,
    )
    axes[0].set_title("Depth", fontsize=11)

    for axis, spatial_mask, title in (
        (axes[1], mask, "Workspace Cube"),
        (axes[2], raised_mask, "Raised Cube"),
    ):
        axis.imshow(
            depth_m,
            cmap="turbo",
            vmin=DEPTH_VIZ_MIN,
            vmax=DEPTH_VIZ_MAX,
        )
        overlay = outside_overlay.copy()
        overlay[..., 3] = np.where(spatial_mask, 0.0, 0.72)
        axis.imshow(overlay)
        axis.set_title(title, fontsize=11)

    axes[3].imshow(regions)
    axes[3].set_title("Raised-Cube Boundary Regions", fontsize=11)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.94,
        bottom=0.01,
        wspace=0.04,
        hspace=0.12,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"spatial_mask_cube_{sequence_dir.name}_frame_{frame_index:05d}"
    output = output_dir / f"{stem}.png"
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Sequence:          {sequence_dir}")
    print(f"Frame:             {frame_index} / {frame_count - 1}")
    if selected_distance is not None:
        print(f"Selection:         farthest camera pose")
        print(f"Target distance:   {selected_distance:.3f} m")
    print(f"Cube center [m]:   {center.tolist()}")
    print(f"Cube side [m]:     {args.cube_side:.3f}")
    print(f"Masked depth px:   {int(mask.sum())}")
    print(f"Saved:             {output}")

    if args.show:
        image = cv2.imread(str(output))
        cv2.imshow("Spatial mask evaluation regions", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the configured world-frame spatial mask cube in RGB and depth."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Sequence name below the data root, or a path to a sequence directory.",
    )
    frame_selection = parser.add_mutually_exclusive_group(required=True)
    frame_selection.add_argument(
        "--frame", type=int, help="Zero-based frame index."
    )
    frame_selection.add_argument(
        "--farthest-frame",
        action="store_true",
        help="Select the frame whose event-camera position is farthest from the cube center.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_ROOT / DATA_ROOT,
        help="Dataset root searched when --sequence is a name.",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=_ROOT / "camera_data",
        help="RealSense calibration directory.",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(SPATIAL_TARGET_X, SPATIAL_TARGET_Y, SPATIAL_TARGET_Z),
        help="Cube center in the robot-base frame, in metres.",
    )
    parser.add_argument(
        "--cube-side",
        type=float,
        default=SPATIAL_CUBE_SIDE,
        help="Cube side length in metres.",
    )
    parser.add_argument(
        "--boundary-dilation",
        type=int,
        default=2,
        help="Evaluation boundary dilation radius in pixels (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "plots",
        help="Directory for the output PNG file.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output resolution.")
    parser.add_argument("--show", action="store_true", help="Display the output interactively.")
    args = parser.parse_args()
    if args.frame is not None and args.frame < 0:
        parser.error("--frame must be non-negative")
    if args.cube_side <= 0.0:
        parser.error("--cube-side must be positive")
    if args.boundary_dilation < 0:
        parser.error("--boundary-dilation must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


if __name__ == "__main__":
    visualize(parse_args())
