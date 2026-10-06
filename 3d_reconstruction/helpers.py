"""Shared helpers for geometry, datasets, calibration, plotting, and view selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Camera geometry and calibration

INTRINSICS_TRANSFORM = "center_crop_resize"


def transform_intrinsics(
    K: np.ndarray,
    native_hw: tuple[int, int],
    resize_hw: tuple[int, int],
    crop_hw: tuple[int, int],
) -> np.ndarray:
    """Transform native intrinsics for a centered crop followed by resize."""
    native_h, native_w = native_hw
    resize_h, resize_w = resize_hw
    crop_h, crop_w = crop_hw
    if min(native_h, native_w, resize_h, resize_w, crop_h, crop_w) <= 0:
        raise ValueError("Image, crop, and resize dimensions must be positive")
    if crop_h > native_h or crop_w > native_w:
        raise ValueError(
            f"Crop {(crop_h, crop_w)} exceeds native image {(native_h, native_w)}"
        )
    transformed = np.array(K, copy=True)
    transformed[0, 2] -= (native_w - crop_w) // 2
    transformed[1, 2] -= (native_h - crop_h) // 2
    transformed[0, :] *= resize_w / crop_w
    transformed[1, :] *= resize_h / crop_h
    return transformed


def load_event_intrinsics(calib_dir: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load the native event-camera matrix and image size as ``(H, W)``."""
    values = np.load(Path(calib_dir) / "event_intrinsics.npz")
    native_w, native_h = (int(value) for value in values["image_size"])
    K_native = values["camera_matrix"].astype(np.float32, copy=True)
    return K_native, (native_h, native_w)


def load_event_from_ee(calib_dir: Path) -> np.ndarray:
    """Load the composed end-effector-to-event-camera transform."""
    calib_dir = Path(calib_dir)
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float32)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float32)
    return T_event_from_rgb @ T_rgb_from_ee


def load_event_calibration(calib_dir: Path) -> dict[str, np.ndarray | tuple[int, int]]:
    """Load event intrinsics, native resolution, and hand-eye calibration."""
    K_native, native_hw = load_event_intrinsics(calib_dir)
    return {
        "K_native": K_native,
        "native_hw": native_hw,
        "T_event_from_ee": load_event_from_ee(calib_dir),
    }


def camera_centers_world(T_camera_from_world: np.ndarray) -> np.ndarray:
    """Return camera centers in world coordinates for batched world-to-camera poses."""
    rotation = T_camera_from_world[:, :3, :3]
    translation = T_camera_from_world[:, :3, 3]
    return -np.einsum(
        "nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation
    ).astype(np.float32)


def depth_pixel_rays(K: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return one normalized ``[(u-cx)/fx, (v-cy)/fy, 1]`` ray per pixel."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    rays = np.stack(
        ((uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)), axis=-1
    )
    return rays.reshape(-1, 3)


# Dataset layout

REQUIRED_SEQUENCE_FILES = (
    Path("events/voxels_cam0.h5"),
    Path("hdf5/depth_in_event_frame.h5"),
    Path("hdf5/poses.h5"),
    Path("hdf5/table_plane.h5"),
)


def is_precomputed_sequence(path: Path) -> bool:
    return all((path / relative).exists() for relative in REQUIRED_SEQUENCE_FILES)


def find_precomputed_sequences(root: Path) -> list[Path]:
    """Return ``root`` itself or its valid immediate child sequences."""
    if is_precomputed_sequence(root):
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and is_precomputed_sequence(child)
    )


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    """Resolve a path absolutely, using ``base_dir`` for relative inputs."""
    path = Path(path)
    return path if path.is_absolute() else (base_dir / path).resolve()


# Plotting

def set_3d_axes_equal(axes) -> None:
    """Set equal data scale on all dimensions of a Matplotlib 3-D axes."""
    limits = (axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d())
    ranges = [abs(upper - lower) for lower, upper in limits]
    centers = [(lower + upper) * 0.5 for lower, upper in limits]
    radius = max(ranges) * 0.5
    axes.set_xlim3d(centers[0] - radius, centers[0] + radius)
    axes.set_ylim3d(centers[1] - radius, centers[1] + radius)
    axes.set_zlim3d(centers[2] - radius, centers[2] + radius)


# Training

class ModelEMA:
    """Exponential moving average of parameters and floating-point buffers."""

    def __init__(self, model, decay: float):
        import copy

        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            source = model.state_dict()
            for name, averaged in self.model.state_dict().items():
                current = source[name].detach()
                if averaged.is_floating_point():
                    averaged.mul_(self.decay).add_(
                        current.to(dtype=averaged.dtype), alpha=1.0 - self.decay
                    )
                else:
                    averaged.copy_(current)


# Workspace spatial masks

def points_in_cube(
    points_world: np.ndarray,
    cube_center: np.ndarray,
    cube_half_side: float,
) -> np.ndarray:
    """Return a boolean cube-membership mask for world-frame 3-D points."""
    points = np.asarray(points_world, dtype=np.float64)
    center = np.asarray(cube_center, dtype=np.float64).reshape(3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if cube_half_side <= 0.0:
        raise ValueError("cube_half_side must be positive")
    return np.all(np.abs(points - center[None, :]) <= cube_half_side, axis=1)


def depth_to_world_points(
    depth_m: np.ndarray,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backproject valid depth pixels and return ``(points_world, ys, xs)``."""
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float64), ys, xs
    z = depth[ys, xs]
    x = (xs.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (ys.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    points_cam = np.stack((x, y, z, np.ones_like(z)), axis=0)
    points_world = (
        np.linalg.inv(np.asarray(T_cam_from_world, dtype=np.float64)) @ points_cam
    )[:3].T
    return points_world, ys, xs


def depth_cube_mask(
    depth_m: np.ndarray,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
    cube_center: np.ndarray,
    cube_half_side: float,
) -> np.ndarray:
    """Return pixels whose measured 3-D points lie inside the world-frame cube."""
    depth = np.asarray(depth_m)
    mask = np.zeros(depth.shape, dtype=bool)
    points_world, ys, xs = depth_to_world_points(depth, T_cam_from_world, K)
    if len(points_world):
        inside = points_in_cube(points_world, cube_center, cube_half_side)
        mask[ys[inside], xs[inside]] = True
    return mask


def depth_cube_masks_for_bottom_offsets(
    depth_m: np.ndarray,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
    target_x: float,
    target_y: float,
    cube_half_side: float,
    bottom_z_offsets_m: np.ndarray,
) -> list[np.ndarray]:
    """Generate depth masks for cubes whose lower Z planes vary by offset."""
    depth = np.asarray(depth_m)
    points_world, ys, xs = depth_to_world_points(depth, T_cam_from_world, K)
    masks = [np.zeros(depth.shape, dtype=bool) for _ in bottom_z_offsets_m]
    for mask, bottom_z in zip(masks, bottom_z_offsets_m):
        center = np.array(
            [target_x, target_y, float(bottom_z) + cube_half_side],
            dtype=np.float64,
        )
        if len(points_world):
            inside = points_in_cube(points_world, center, cube_half_side)
            mask[ys[inside], xs[inside]] = True
    return masks


# Multiview source selection

def fixed_source_offsets(num_views: int, view_interval: int) -> list[int]:
    """Return alternating past/future offsets for fixed-interval view selection."""
    offsets: list[int] = []
    distance = 1
    while len(offsets) < num_views - 1:
        offsets.append(-distance * view_interval)
        if len(offsets) < num_views - 1:
            offsets.append(distance * view_interval)
        distance += 1
    return offsets


def pose_neighbours(
    camera_centers: np.ndarray,
    target_idx: int,
    direction: int,
    count: int,
    move_threshold: float,
) -> list[int]:
    """Return geometrically spaced neighbours in one temporal direction."""
    neighbours: list[int] = []
    anchor = target_idx
    cursor = target_idx + direction
    while 0 <= cursor < len(camera_centers) and len(neighbours) < count:
        moved = np.linalg.norm(camera_centers[cursor] - camera_centers[anchor])
        if moved >= move_threshold:
            neighbours.append(cursor)
            anchor = cursor
        cursor += direction
    return neighbours


def select_pose_views(
    camera_centers: np.ndarray,
    target_idx: int,
    num_views: int,
    move_threshold: float,
    allow_unbalanced: bool = True,
) -> list[int] | None:
    """Select target/source views separated by physical camera movement."""
    if num_views < 1 or num_views % 2 != 1:
        raise ValueError("pose-based selection requires a positive odd num_views")
    if num_views == 1:
        return [target_idx]
    per_direction = (num_views - 1) // 2
    before = pose_neighbours(
        camera_centers, target_idx, -1, per_direction, move_threshold
    )
    after = pose_neighbours(
        camera_centers, target_idx, 1, per_direction, move_threshold
    )
    if not allow_unbalanced:
        if len(before) < per_direction or len(after) < per_direction:
            return None
        return [target_idx, *before[:per_direction], *after[:per_direction]]
    before_slots = [*before, *([-1] * (per_direction - len(before)))]
    after_slots = [*after, *([-1] * (per_direction - len(after)))]
    return [target_idx, *before_slots, *after_slots]


def build_pose_view_ids(
    camera_centers: np.ndarray,
    num_views: int,
    move_threshold: float,
    allow_unbalanced: bool = True,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """Build source tuples and valid target indices for a complete sequence."""
    view_ids: dict[int, list[int]] = {}
    valid: list[int] = []
    for target_idx in range(len(camera_centers)):
        selected = select_pose_views(
            camera_centers, target_idx, num_views, move_threshold, allow_unbalanced
        )
        if selected is not None:
            view_ids[target_idx] = selected
            valid.append(target_idx)
    return view_ids, np.asarray(valid, dtype=np.int64)


def pose_layout_counts(view_ids: dict[int, list[int]]) -> dict[str, int]:
    """Count balanced, asymmetric, and fully one-sided target tuples."""
    counts = {
        "balanced": 0,
        "asymmetric": 0,
        "one_sided": 0,
        "reference_only": 0,
    }
    for target_idx, selected in view_ids.items():
        before = sum(0 <= source_idx < target_idx for source_idx in selected[1:])
        after = sum(source_idx > target_idx for source_idx in selected[1:])
        if before == 0 and after == 0:
            counts["reference_only"] += 1
        elif before == after:
            counts["balanced"] += 1
        elif before == 0 or after == 0:
            counts["one_sided"] += 1
        else:
            counts["asymmetric"] += 1
    return counts
