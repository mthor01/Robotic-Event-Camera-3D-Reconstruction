"""Shared crop-then-resize geometry for preprocessing and camera intrinsics."""

from __future__ import annotations

import numpy as np

INTRINSICS_TRANSFORM = "center_crop_resize"


def transform_intrinsics(
    K: np.ndarray,
    native_hw: tuple[int, int],
    resize_hw: tuple[int, int],
    crop_hw: tuple[int, int],
) -> np.ndarray:
    """Center-crop native image coordinates, then resize the intrinsics."""
    native_h, native_w = native_hw
    resize_h, resize_w = resize_hw
    crop_h, crop_w = crop_hw
    out = K.copy()

    if crop_h > native_h or crop_w > native_w:
        raise ValueError(
            f"Crop {(crop_h, crop_w)} exceeds native image {(native_h, native_w)}"
        )
    y0 = (native_h - crop_h) // 2
    x0 = (native_w - crop_w) // 2
    out[0, 2] -= x0
    out[1, 2] -= y0
    out[0, :] *= resize_w / crop_w
    out[1, :] *= resize_h / crop_h
    return out
