"""Utility functions for the franka_pipeline."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# Transformation Matrix Utilities
# =============================================================================


def pos_rot_to_transformation_matrix(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Create a 4x4 transformation matrix from position and rotation.

    Args:
        pos: Translation vector of shape (3,) as [x, y, z].
        rot: Quaternion rotation of shape (4,) as [x, y, z, w].

    Returns:
        4x4 homogeneous transformation matrix.
    """
    R = Rotation.from_quat(rot).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def transformation_matrix_to_pos_rot(
    t_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and rotation from a 4x4 transformation matrix.

    Args:
        t_mat: 4x4 homogeneous transformation matrix.

    Returns:
        Tuple of (pos, rot) where pos is [x, y, z] translation vector
        and rot is [x, y, z, w] quaternion rotation.
    """
    pos = t_mat[:3, 3].copy()
    R = t_mat[:3, :3]
    rot = Rotation.from_matrix(R).as_quat()
    return pos, rot


def t_R_from_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract translation vector and rotation matrix from transformation matrix.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        Tuple of (t, R) where t is (3,) translation vector and R is (3,3) rotation matrix.
    """
    return T[:3, 3].copy(), T[:3, :3].copy()


def T_from_t_R(t: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Create transformation matrix from translation vector and rotation matrix.

    Args:
        t: Translation vector of shape (3,) or (3, 1).
        R: Rotation matrix of shape (3, 3).

    Returns:
        4x4 homogeneous transformation matrix.

    Raises:
        ValueError: If t cannot be flattened to shape (3,).
    """
    t_flat = t.flatten()
    if t_flat.shape != (3,):
        raise ValueError(f"t must flatten to shape (3,), got {t_flat.shape}")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t_flat
    return T


# =============================================================================
# Image Processing Utilities
# =============================================================================


def crop_and_resize(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """Crop image to target aspect ratio and resize.

    Crops the image to match the target aspect ratio by removing borders,
    then resizes to the target dimensions without distortion.

    Args:
        image: Input image array (HWC format).
        target_width: Desired output width. If None, set equal to target_height.
        target_height: Desired output height. If None, set equal to target_width.

    Returns:
        Cropped and resized image.

    Raises:
        ValueError: If neither target dimension is provided.
    """
    if target_width is None and target_height is None:
        raise ValueError("At least one target dimension must be provided.")

    # Default to square output if one dimension missing
    if target_width is None:
        target_width = target_height
    if target_height is None:
        target_height = target_width

    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    orig_aspect = w / h

    # Determine cropping dimensions to match target aspect ratio
    if orig_aspect > target_aspect:
        # Original is wider: crop width
        new_width = int(h * target_aspect)
        new_height = h
        x_start = (w - new_width) // 2
        y_start = 0
    elif orig_aspect < target_aspect:
        # Original is taller: crop height
        new_width = w
        new_height = int(w / target_aspect)
        x_start = 0
        y_start = (h - new_height) // 2
    else:
        # Same aspect ratio, no crop needed
        new_width, new_height = w, h
        x_start, y_start = 0, 0

    cropped = image[y_start : y_start + new_height, x_start : x_start + new_width]
    resized = cv2.resize(
        cropped, (target_width, target_height), interpolation=cv2.INTER_AREA
    )

    return resized
