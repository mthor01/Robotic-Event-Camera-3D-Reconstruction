"""Shared geometry for the two supported spatial preprocessing modes."""

from __future__ import annotations

import numpy as np

RESIZE_THEN_CROP = "resize_center_crop"
CROP_THEN_RESIZE = "center_crop_resize"


def transform_intrinsics(
    K: np.ndarray,
    native_hw: tuple[int, int],
    resize_hw: tuple[int, int],
    crop_hw: tuple[int, int],
    crop_then_resize: bool = False,
) -> np.ndarray:
    """Apply the selected centered crop/resize geometry to intrinsics."""
    native_h, native_w = native_hw
    resize_h, resize_w = resize_hw
    crop_h, crop_w = crop_hw
    out = K.copy()

    if crop_then_resize:
        y0 = (native_h - crop_h) // 2
        x0 = (native_w - crop_w) // 2
        out[0, 2] -= x0
        out[1, 2] -= y0
        out[0, :] *= resize_w / crop_w
        out[1, :] *= resize_h / crop_h
    else:
        out[0, :] *= resize_w / native_w
        out[1, :] *= resize_h / native_h
        out[0, 2] -= (resize_w - crop_w) // 2
        out[1, 2] -= (resize_h - crop_h) // 2
    return out


def transform_name(crop_then_resize: bool) -> str:
    return CROP_THEN_RESIZE if crop_then_resize else RESIZE_THEN_CROP
