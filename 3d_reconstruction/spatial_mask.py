"""Shared axis-aligned workspace-cube masking utilities.

The cube is expressed in the robot base/world frame. Evaluation uses depth
maps to classify pixels by their reconstructed 3-D position; reconstruction
uses the same membership test to crop mesh vertices.
"""

from __future__ import annotations

import numpy as np


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
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
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
