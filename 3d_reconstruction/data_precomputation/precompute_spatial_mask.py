#!/usr/bin/env python3
"""
Precompute spatial masks that restrict depth supervision to a cube around
a fixed target point in the robot base frame.

For each depth frame, every pixel with valid depth is unprojected to 3-D
in the robot base frame and tested against a configurable axis-aligned cube
(default 32 cm side) centred on SPATIAL_TARGET_* from config.py.

Two mask modes are supported:
  Binary (default)  — uint8 {0, 1}: 1 inside the cube, 0 outside.
  Soft (--soft_mask) — float32 [0, 1]: pixels more than 2 cm above the cube's
                       lowest Z plane get 1.0; pixels in the bottom 2 cm are
                       scaled linearly from 0 to 1.  Pixels outside the cube
                       are 0 in both modes.

Optional white masking (--mask_white):
    Combines the spatial cube mask with a white-pixel mask. White pixels are detected
    from RGB values: a pixel is white when all three channels are above a threshold
    (--white_threshold, default 245 on 0..255 scale). Only pixels valid in both masks
    are kept (AND combination for binary, multiplication for soft).

RGB FoV masking (always enabled):
    In the event-camera frame, pixels whose 3-D points project outside the RGB
    camera image bounds are removed from the final mask.

The mask is computed in the **event-camera image plane** (same H×W as
depth_in_event_frame.h5) so it can be applied directly during training or
visualization without additional alignment.  An optional native-resolution
version (mask_depth) is also stored when realsense.h5 is present.

Outputs (per recording):
    hdf5/spatial_mask.h5   — dataset "mask"       (N, H, W) uint8 or float32
                           — dataset "mask_depth"  (N, dH, dW) same dtype
                             (only when hdf5/realsense.h5 exists)
    videos/spatial_mask.mp4  — (optional, with --save_videos)
    debug/spatial_mask_debug.png — (optional, with --debug)

Usage:
    python3 data_precomputation/precompute_spatial_mask.py --mask_white --debug --save_videos --data_dir data/real/lego_1 --overwrite
"""

import argparse
from pathlib import Path
from typing import List

import cv2
import h5py
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


from config import (
    CALIB_DIR as _CALIB_DIR,
    DATA_ROOT as _DATA_ROOT,
    SPATIAL_CUBE_SIDE,
    SPATIAL_TARGET_X,
    SPATIAL_TARGET_Y,
    SPATIAL_TARGET_Z,
    DEPTH_VIZ_MIN,
    DEPTH_VIZ_MAX,
)

_CFG_ROOT = _Path(__file__).resolve().parent.parent
CALIB_DIR = _CFG_ROOT / _CALIB_DIR
DATA_ROOT  = _CFG_ROOT / _DATA_ROOT

# Default target point — imported from config
DEFAULT_TARGET_X = SPATIAL_TARGET_X
DEFAULT_TARGET_Y = SPATIAL_TARGET_Y
DEFAULT_TARGET_Z = SPATIAL_TARGET_Z


def load_calibration(calib_dir: Path) -> dict:
    """Load calibration needed for event-pixel and depth-pixel → base-frame unprojection."""
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"]       # (4,4)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"] # (4,4)
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K_event = ev["camera_matrix"]      # (3,3)
    dist_event = ev["dist_coeffs"].ravel()  # (5,)
    ev_size = ev["image_size"]         # [W, H]
    rgb_intr = np.load(calib_dir / "rs_rgb_intrinsics.npz")
    K_rgb = rgb_intr["camera_matrix"]
    rgb_size = rgb_intr["image_size"]

    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee
    # inv(T_event_from_ee) brings event-cam → ee → base when composed with ee_T
    T_ee_from_event = np.linalg.inv(T_event_from_ee)
    T_rgb_from_event = np.linalg.inv(T_event_from_rgb)

    # depth camera calibration (for native-resolution mask stored in mask_depth)
    T_color_from_depth = np.load(calib_dir / "T_color_from_depth.npz")["T"]  # (4,4)
    depth_intr = np.load(calib_dir / "rs_depth_intrinsics.npz")
    K_depth    = depth_intr["camera_matrix"]   # (3,3)
    depth_size = depth_intr["image_size"]       # [W, H]
    depth_scale = float(np.load(calib_dir / "depth_scale.npz")["scale"])
    T_ee_from_depth = np.linalg.inv(T_rgb_from_ee) @ T_color_from_depth

    return {
        "T_ee_from_event": T_ee_from_event,
        "T_rgb_from_event": T_rgb_from_event,
        "K_event": K_event,
        "K_rgb": K_rgb,
        "dist_event": dist_event,
        "ev_w": int(ev_size[0]),
        "ev_h": int(ev_size[1]),
        "rgb_w": int(rgb_size[0]),
        "rgb_h": int(rgb_size[1]),
        "T_ee_from_depth": T_ee_from_depth,
        "K_depth": K_depth,
        "depth_w": int(depth_size[0]),
        "depth_h": int(depth_size[1]),
        "depth_scale": depth_scale,
    }


def build_event_rays(
    K_event: np.ndarray,
    h: int,
    w: int,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Pre-compute normalised ray directions for every event pixel.

    Returns (h*w, 3) where each row is K_event_inv @ [u, v, 1].
    """
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    if dist_coeffs is not None:
        # depth_in_event_frame/rgb_in_event_frame are rendered in distorted event
        # coordinates (cv2.projectPoints with dist coeffs). To unproject correctly,
        # first undistort pixel coordinates to normalized camera coordinates.
        pts = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2).astype(np.float64)
        und = cv2.undistortPoints(pts, K_event.astype(np.float64), dist_coeffs.astype(np.float64))
        x = und[:, 0, 0]
        y = und[:, 0, 1]
        rays = np.stack([x, y, np.ones_like(x)], axis=1)
        return rays

    K_inv = np.linalg.inv(K_event)
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (h, w, 3)
    rays = (K_inv @ pixels.reshape(-1, 3).T).T               # (h*w, 3)
    return rays


def compute_white_mask(
    rgb: np.ndarray,
    white_threshold: float = 70.0,
    soft: bool = False,
) -> np.ndarray:
    """Compute mask that excludes white pixels using RGB channels.

    Args:
        rgb:             (H, W, 3) RGB image.
        white_threshold: Per-channel threshold. If <= 1.0 it is interpreted as a
                         normalized threshold and scaled by 255.
        soft:            If False, returns uint8 {0, 1}.
                         If True, returns float32 [0, 1] (for compatibility with soft masks).

    Returns:
        (H, W) uint8 or float32 mask: 1 (or 1.0) for non-white pixels, 0 (or 0.0) for white pixels.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

    thr = float(white_threshold)
    if thr <= 1.0:
        thr = thr * 255.0

    rgb_f = rgb.astype(np.float32)
    # A pixel is white only if ALL channels are above the threshold.
    white_pixels = np.all(rgb_f >= thr, axis=2)

    if soft:
        return (~white_pixels).astype(np.float32)
    else:
        return (~white_pixels).astype(np.uint8)


def compute_rgb_fov_mask_event(
    depth_event: np.ndarray,
    rays_event: np.ndarray,
    T_rgb_from_event: np.ndarray,
    K_rgb: np.ndarray,
    rgb_h: int,
    rgb_w: int,
    fov_shrink_px: int = 2,
    soft: bool = False,
) -> np.ndarray:
    """Geometric FoV mask in event frame with optional inward border shrink."""
    H, W = depth_event.shape
    flat_depth = depth_event.ravel().astype(np.float64)
    valid = flat_depth > 0.0

    mask_flat = np.zeros(H * W, dtype=np.float32 if soft else np.uint8)
    if not np.any(valid):
        return mask_flat.reshape(H, W)

    pts_event = rays_event[valid] * flat_depth[valid, None]
    R = T_rgb_from_event[:3, :3]
    t = T_rgb_from_event[:3, 3:4]
    pts_rgb = (R @ pts_event.T + t).T

    z = pts_rgb[:, 2]
    in_front = z > 1e-6

    u = np.full_like(z, -1.0, dtype=np.float64)
    v = np.full_like(z, -1.0, dtype=np.float64)
    u[in_front] = K_rgb[0, 0] * (pts_rgb[in_front, 0] / z[in_front]) + K_rgb[0, 2]
    v[in_front] = K_rgb[1, 1] * (pts_rgb[in_front, 1] / z[in_front]) + K_rgb[1, 2]

    b = float(max(0, int(fov_shrink_px)))
    inside = in_front & (u >= b) & (u < float(rgb_w) - b) & (v >= b) & (v < float(rgb_h) - b)
    valid_indices = np.nonzero(valid)[0]
    if soft:
        mask_flat[valid_indices[inside]] = 1.0
    else:
        mask_flat[valid_indices[inside]] = 1
    return mask_flat.reshape(H, W)


def combine_masks(
    spatial_mask: np.ndarray,
    white_mask: np.ndarray,
    soft: bool = False,
) -> np.ndarray:
    """Combine spatial and white masks element-wise.

    For soft masks (float32): multiplies the masks.
    For binary masks (uint8): applies bitwise AND.

    Args:
        spatial_mask: Spatial cube mask (H, W).
        white_mask:   White pixel mask (H, W).
        soft:         If False, uses bitwise AND; if True, uses multiplication.

    Returns:
        (H, W) combined mask, same dtype as inputs.
    """
    if soft:
        return (spatial_mask.astype(np.float32) * white_mask.astype(np.float32)).astype(np.float32)
    else:
        return (spatial_mask & white_mask).astype(np.uint8)


def denoise_mask(
    mask: np.ndarray,
    soft: bool = False,
    min_component_area: int = 30,
    morph_kernel: int = 3,
    morph_iters: int = 1,
    soft_threshold: float = 0.5,
) -> np.ndarray:
    """Remove scattered noise from mask using morphology + small-component filtering."""
    if morph_kernel < 1:
        morph_kernel = 1
    if morph_kernel % 2 == 0:
        morph_kernel += 1
    morph_iters = max(0, int(morph_iters))
    min_component_area = max(0, int(min_component_area))

    if soft:
        base = (mask.astype(np.float32) >= float(soft_threshold)).astype(np.uint8)
    else:
        base = (mask > 0).astype(np.uint8)

    if morph_iters > 0 and morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, kernel, iterations=morph_iters)

    if min_component_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
        keep = np.zeros_like(base, dtype=np.uint8)
        for comp_id in range(1, n):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area >= min_component_area:
                keep[labels == comp_id] = 1
        base = keep

    if soft:
        return (mask.astype(np.float32) * base.astype(np.float32)).astype(np.float32)
    return base.astype(np.uint8)


def compute_spatial_mask(
    depth: np.ndarray,
    ee_T: np.ndarray,
    T_ee_from_event: np.ndarray,
    rays: np.ndarray,
    half_side: float,
    target_point: np.ndarray,
    soft: bool = False,
) -> np.ndarray:
    """Compute spatial mask: binary (uint8) or soft/continuous (float32).

    Args:
        depth:            (H, W) float32 metres, projected into event frame.
        ee_T:             (4, 4) T_base_from_ee for this frame.
        T_ee_from_event:  (4, 4) static transform (inv of T_event_from_ee).
        rays:             (H*W, 3) normalised ray directions in event frame.
        half_side:        Half the cube side length in metres.
        target_point:     (3,) cube centre in robot base frame.
        soft:             If False (default), returns uint8 {0, 1}.
                          If True, returns float32 where pixels outside the
                          cube are 0, pixels more than 2 cm above the cube
                          bottom (in Z) are 1, and pixels within the bottom
                          2 cm are scaled linearly 0 → 1.

    Returns:
        (H, W) uint8 mask if soft=False, float32 mask if soft=True.
    """
    H, W = depth.shape
    flat_depth = depth.ravel().astype(np.float64)
    valid = flat_depth > 0.0

    mask_flat = np.zeros(H * W, dtype=np.float32 if soft else np.uint8)
    if not np.any(valid):
        return mask_flat.reshape(H, W)

    # Unproject valid pixels to 3-D in event-camera frame
    pts_event = rays[valid] * flat_depth[valid, None]  # (M, 3)

    # Transform to base frame: base = ee_T @ (T_ee_from_event @ pts_event)
    T_base_from_event = ee_T @ T_ee_from_event
    R = T_base_from_event[:3, :3]
    t = T_base_from_event[:3, 3:4]
    pts_base = (R @ pts_event.T + t).T  # (M, 3)

    # Check cube membership: |pt - target_point| <= half_side for all axes
    diff = np.abs(pts_base - target_point[None, :])  # (M, 3)
    inside = np.all(diff <= half_side, axis=1)  # (M,)

    valid_indices = np.nonzero(valid)[0]
    if soft:
        # Linear ramp over the bottom 2 cm of the cube (in Z / robot-up axis).
        # height_in_cube = distance above the cube's lowest Z plane.
        cube_bottom_z = target_point[2] - half_side
        height = pts_base[inside, 2] - cube_bottom_z  # (M_inside,)
        values = np.clip(height / 0.02, 0.0, 1.0).astype(np.float32)
        mask_flat[valid_indices[inside]] = values
    else:
        mask_flat[valid_indices[inside]] = 1
    return mask_flat.reshape(H, W)


def save_debug_png(seq_dir: Path, depth_frames: list, mask_frames: list) -> None:
    """Save a side-by-side depth / mask contact sheet for 10 evenly-spaced frames.

    Layout per row: [colourised depth | green overlay for masked pixels]
    Supports both binary masks (uint8 {0,1}) and soft masks (float32 [0,1]).
    Output: debug/spatial_mask_debug.png
    """
    n = len(depth_frames)
    indices = np.linspace(0, n - 1, min(10, n), dtype=int)
    rows = []
    for i in indices:
        depth = depth_frames[i]
        mask  = mask_frames[i]

        # Left panel: colourised depth using configured depth range
        d_norm = np.clip((depth - DEPTH_VIZ_MIN) / (DEPTH_VIZ_MAX - DEPTH_VIZ_MIN), 0, 1)
        d_u8   = (d_norm * 255).astype(np.uint8)
        left   = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
        left[depth == 0] = 0

        # Right panel: mask overlaid on grey depth.
        # Binary masks paint fully green where mask==1; soft masks use mask value
        # as alpha so partially-included pixels remain visible.
        grey = cv2.cvtColor(d_u8, cv2.COLOR_GRAY2BGR)
        right = grey.copy()
        alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
        if np.any(alpha > 0):
            green = np.zeros_like(right, dtype=np.float32)
            green[:, :, 1] = 220.0
            right_f = right.astype(np.float32)
            a = (0.5 * alpha)[:, :, None]
            right = np.clip(right_f * (1.0 - a) + green * a, 0.0, 255.0).astype(np.uint8)

        # Label with frame index
        H, W = depth.shape
        for panel in (left, right):
            cv2.putText(panel, f"frame {i}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        row = np.hstack([left, right])
        rows.append(row)

    sheet = np.vstack(rows)
    out_dir = seq_dir / "debug"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "spatial_mask_debug.png"
    cv2.imwrite(str(out_path), sheet)
    print(f"  [{seq_dir.name}] debug PNG → {out_path}")


def process_sequence(
    seq_dir: Path,
    calib: dict,
    cube_side: float,
    target_point: np.ndarray,
    overwrite: bool = False,
    save_videos: bool = False,
    debug: bool = False,
    soft_mask: bool = False,
    mask_white: bool = False,
    white_threshold: float = 245.0,
    fov_shrink_px: int = 2,
    denoise: bool = True,
    denoise_min_area: int = 30,
    denoise_kernel: int = 3,
    denoise_iters: int = 1,
    denoise_soft_threshold: float = 0.5,
) -> dict:
    """Precompute spatial mask for one recording directory."""
    result = {"name": seq_dir.name, "success": False, "n_frames": 0, "error": None}

    depth_h5_path = seq_dir / "hdf5" / "depth_in_event_frame.h5"
    if not depth_h5_path.exists():
        result["error"] = "depth_in_event_frame.h5 not found (run project_realsense_to_event.py first)"
        return result

    poses_path = seq_dir / "hdf5" / "poses.h5"
    if not poses_path.exists():
        result["error"] = "poses.h5 not found"
        return result

    out_path = seq_dir / "hdf5" / "spatial_mask.h5"

    # Load poses
    with h5py.File(poses_path, "r") as pf:
        ee_T_all = pf["ee_T"][:]  # (N, 4, 4)

    with h5py.File(depth_h5_path, "r") as df:
        n_depth = df["depth"].shape[0]
        H = df["depth"].shape[1]
        W = df["depth"].shape[2]
        # If depth was stored at a down-scaled resolution (by project_realsense_to_event.py),
        # we need to scale K_event to match, otherwise unprojected rays will be wrong.
        native_ev_h = int(df.attrs.get("native_ev_h", calib["ev_h"]))
        native_ev_w = int(df.attrs.get("native_ev_w", calib["ev_w"]))
        _resize_h   = int(df.attrs.get("resize_h",   native_ev_h))
        _resize_w   = int(df.attrs.get("resize_w",   native_ev_w))
        _crop_h     = int(df.attrs.get("crop_h",     H))
        _crop_w     = int(df.attrs.get("crop_w",     W))

    # Scale K_event from native resolution to the stored depth resolution.
    K_event = calib["K_event"].copy().astype(np.float64)
    if H != native_ev_h or W != native_ev_w:
        K_event[0] *= _resize_w / native_ev_w   # fx, cx scale by resize
        K_event[1] *= _resize_h / native_ev_h   # fy, cy scale by resize
        K_event[0, 2] -= (_resize_w - _crop_w) / 2   # cx shift by crop
        K_event[1, 2] -= (_resize_h - _crop_h) / 2   # cy shift by crop

    n_frames = min(n_depth, len(ee_T_all))
    if n_depth != len(ee_T_all):
        print(f"  [{seq_dir.name}] WARNING: {n_depth} depth frames != {len(ee_T_all)} poses; using min({n_frames})")

    # Skip if already done
    if not overwrite and out_path.exists():
        with h5py.File(out_path, "r") as ef:
            if ef["mask"].shape[0] >= n_frames:
                result["success"] = True
                result["n_frames"] = n_frames
                result["error"] = "Already computed (use --overwrite)"
                return result

    # Build rays for event camera at the stored depth resolution
    rays = build_event_rays(K_event, H, W, dist_coeffs=calib["dist_event"])
    half_side = cube_side / 2.0

    # Event-frame RGB (projected to event camera) for RGB-based white masking
    rgb_event_h5_path = seq_dir / "hdf5" / "rgb_in_event_frame.h5"
    has_rgb_event = rgb_event_h5_path.exists()

    # Depth-frame mask: compute natively at depth resolution from realsense.h5
    realsense_h5_path = seq_dir / "hdf5" / "realsense.h5"
    has_realsense = realsense_h5_path.exists()
    if has_realsense:
        dH, dW = calib["depth_h"], calib["depth_w"]
        rays_depth = build_event_rays(calib["K_depth"], dH, dW)
    else:
        dH = dW = 0
        rays_depth = None

    if mask_white:
        if not has_rgb_event:
            result["error"] = "mask_white requires hdf5/rgb_in_event_frame.h5"
            return result
        if not has_realsense:
            result["error"] = "mask_white requires hdf5/realsense.h5 with rgb dataset"
            return result

    # Process frames
    rs_file = h5py.File(realsense_h5_path, "r") if has_realsense else None
    rgb_event_file = h5py.File(rgb_event_h5_path, "r") if has_rgb_event else None
    try:
        with h5py.File(depth_h5_path, "r") as df, \
             h5py.File(out_path, "w") as of:
            mask_dtype = np.float32 if soft_mask else np.uint8
            mask_ds = of.create_dataset(
                "mask", shape=(n_frames, H, W), dtype=mask_dtype,
                chunks=(1, H, W), compression="gzip", compression_opts=4,
            )
            if has_realsense:
                mask_depth_ds = of.create_dataset(
                    "mask_depth", shape=(n_frames, dH, dW), dtype=mask_dtype,
                    chunks=(1, dH, dW), compression="gzip", compression_opts=4,
                )
            of.attrs["cube_side_m"] = cube_side
            of.attrs["target_point"] = target_point
            of.attrs["soft_mask"] = soft_mask
            of.attrs["mask_white"] = mask_white
            of.attrs["mask_rgb_fov_event"] = bool(mask_white)
            of.attrs["denoise"] = bool(denoise)
            of.attrs["denoise_min_area"] = int(denoise_min_area)
            of.attrs["denoise_kernel"] = int(denoise_kernel)
            of.attrs["denoise_iters"] = int(denoise_iters)
            of.attrs["denoise_soft_threshold"] = float(denoise_soft_threshold)
            if mask_white:
                of.attrs["white_threshold"] = white_threshold
                of.attrs["white_mask_mode"] = "rgb_all_channels"
                of.attrs["fov_shrink_px"] = int(fov_shrink_px)
            if soft_mask:
                desc = f"Soft spatial mask: float32, linear ramp 0→1 over bottom 2 cm of {cube_side:.2f}m cube"
                if mask_white:
                    desc += " + RGB FoV mask in event frame"
                if mask_white:
                    desc += f" + RGB white masking (all channels >= {white_threshold})"
                of.attrs["description"] = desc
            else:
                desc = f"Spatial mask: 1 inside {cube_side:.2f}m cube"
                if mask_white:
                    desc += " + RGB FoV mask in event frame"
                if mask_white:
                    desc += f" + RGB white masking (all channels >= {white_threshold})"
                of.attrs["description"] = desc

            masks = np.empty((n_frames, H, W), dtype=mask_dtype)
            debug_depths: list = [] if debug else None  # type: ignore
            debug_masks:  list = [] if debug else None  # type: ignore
            rs_depth_ds = rs_file["depth"] if rs_file is not None else None
            rs_rgb_ds = rs_file["rgb"] if (rs_file is not None and "rgb" in rs_file) else None
            rgb_event_ds = rgb_event_file["rgb"] if (rgb_event_file is not None and "rgb" in rgb_event_file) else None
            if mask_white and rgb_event_ds is None:
                raise RuntimeError("mask_white requires 'rgb' dataset in hdf5/rgb_in_event_frame.h5")
            if mask_white and rs_rgb_ds is None:
                raise RuntimeError("mask_white requires 'rgb' dataset in hdf5/realsense.h5")
            for i in tqdm(range(n_frames), desc=f"  {seq_dir.name}", leave=False):
                depth = df["depth"][i].astype(np.float32)
                m = compute_spatial_mask(
                    depth, ee_T_all[i], calib["T_ee_from_event"], rays, half_side,
                    target_point, soft=soft_mask,
                )
                # Combine with white mask if requested
                if mask_white:
                    # Geometric RGB FoV mask, conservatively shrunken at borders.
                    rgb_fov = compute_rgb_fov_mask_event(
                        depth,
                        rays,
                        calib["T_rgb_from_event"],
                        calib["K_rgb"],
                        calib["rgb_h"],
                        calib["rgb_w"],
                        fov_shrink_px=fov_shrink_px,
                        soft=soft_mask,
                    )
                    m = combine_masks(m, rgb_fov, soft=soft_mask)
                    if i >= rgb_event_ds.shape[0]:
                        raise RuntimeError(
                            f"mask_white requires rgb_in_event_frame.h5/rgb with at least {n_frames} frames; "
                            f"got {rgb_event_ds.shape[0]}"
                        )
                    w = compute_white_mask(rgb_event_ds[i], white_threshold, soft=soft_mask)
                    m = combine_masks(m, w, soft=soft_mask)
                if denoise:
                    m = denoise_mask(
                        m,
                        soft=soft_mask,
                        min_component_area=denoise_min_area,
                        morph_kernel=denoise_kernel,
                        morph_iters=denoise_iters,
                        soft_threshold=denoise_soft_threshold,
                    )
                mask_ds[i] = m
                masks[i] = m
                if debug:
                    debug_depths.append(depth)
                    debug_masks.append(m)

                if rs_depth_ds is not None:
                    depth_raw = rs_depth_ds[i].astype(np.float32) * calib["depth_scale"]
                    md = compute_spatial_mask(
                        depth_raw, ee_T_all[i], calib["T_ee_from_depth"], rays_depth,
                        half_side, target_point, soft=soft_mask,
                    )
                    # Combine with white mask if requested
                    if mask_white:
                        if i >= rs_rgb_ds.shape[0]:
                            raise RuntimeError(
                                f"mask_white requires realsense.h5/rgb with at least {n_frames} frames; "
                                f"got {rs_rgb_ds.shape[0]}"
                            )
                        wd = compute_white_mask(rs_rgb_ds[i], white_threshold, soft=soft_mask)
                        md = combine_masks(md, wd, soft=soft_mask)
                    if denoise:
                        md = denoise_mask(
                            md,
                            soft=soft_mask,
                            min_component_area=denoise_min_area,
                            morph_kernel=denoise_kernel,
                            morph_iters=denoise_iters,
                            soft_threshold=denoise_soft_threshold,
                        )
                    mask_depth_ds[i] = md
    finally:
        if rgb_event_file is not None:
            rgb_event_file.close()
        if rs_file is not None:
            rs_file.close()

    if debug:
        save_debug_png(seq_dir, debug_depths, debug_masks)

    if save_videos:
        video_dir = seq_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / "spatial_mask.mp4"
        fps = 30
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
            isColor=False,
        )
        for i in range(n_frames):
            writer.write((masks[i] * 255).astype(np.uint8))
        writer.release()
        result["video_path"] = video_path

    result["success"] = True
    result["n_frames"] = n_frames
    return result


def find_sequences(data_root: Path) -> List[Path]:
    """Find recordings that have both projected depth and poses."""
    seqs = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "hdf5" / "depth_in_event_frame.h5").exists() and \
           (d / "hdf5" / "poses.h5").exists():
            seqs.append(d)
    return seqs


def main():
    parser = argparse.ArgumentParser(
        description="Precompute spatial masks (cube around target point) for depth training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", nargs="+", type=str, default=None,
                        help="Specific recording directory(ies)")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                        help="Root directory containing recording subdirs")
    parser.add_argument("--calib_dir", type=str, default=str(CALIB_DIR),
                        help="Path to camera_data/ calibration directory")
    parser.add_argument("--cube_side", type=float, default=SPATIAL_CUBE_SIDE,
                        help="Cube side length in metres")
    parser.add_argument("--target_x", type=float, default=DEFAULT_TARGET_X,
                        help="Target point X in robot base frame (metres)")
    parser.add_argument("--target_y", type=float, default=DEFAULT_TARGET_Y,
                        help="Target point Y in robot base frame (metres)")
    parser.add_argument("--target_z", type=float, default=DEFAULT_TARGET_Z,
                        help="Target point Z in robot base frame (metres)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing spatial_mask.h5 files")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save a video of the spatial mask to videos/spatial_mask.mp4")
    parser.add_argument("--debug", action="store_true",
                        help="Save a debug PNG (depth + mask overlay, 10 frames) to debug/spatial_mask_debug.png")
    parser.add_argument("--soft_mask", action="store_true",
                        help="Produce a float32 soft mask instead of binary uint8: "
                             "pixels more than 2 cm above the cube bottom (in Z) get 1.0, "
                             "pixels within the bottom 2 cm are scaled linearly 0→1")
    parser.add_argument("--mask_white", action="store_true",
                        help="Additionally mask out white (far/invalid) pixels by combining "
                             "spatial mask with white mask. Only pixels valid in both masks are kept.")
    parser.add_argument("--white_threshold", type=float, default=245.0,
                        help="RGB white threshold per channel. A pixel is considered white if "
                            "R,G,B are all >= threshold. If threshold <= 1.0, it is interpreted "
                            "as normalized and scaled by 255.")
    parser.add_argument("--fov_shrink_px", type=int, default=2,
                        help="Shrink RGB FoV mask inward by this many pixels on each border "
                             "(applied only when --mask_white is enabled).")
    parser.add_argument("--disable_denoise", action="store_true",
                        help="Disable post-mask denoising (morphological cleanup + small-component removal).")
    parser.add_argument("--denoise_min_area", type=int, default=30,
                        help="Remove connected components smaller than this many pixels.")
    parser.add_argument("--denoise_kernel", type=int, default=3,
                        help="Morphological opening kernel size (odd integer >= 1).")
    parser.add_argument("--denoise_iters", type=int, default=1,
                        help="Number of morphological opening iterations.")
    parser.add_argument("--denoise_soft_threshold", type=float, default=0.5,
                        help="Threshold used to binarize soft masks before denoising.")
    args = parser.parse_args()

    calib = load_calibration(Path(args.calib_dir))
    target_point = np.array([args.target_x, args.target_y, args.target_z])

    if args.data_dir:
        dirs = [Path(d) for d in args.data_dir]
    else:
        dirs = find_sequences(Path(args.data_root))
        if not dirs:
            print(f"No valid sequences found under {args.data_root}")
            return

    print(f"Processing {len(dirs)} recording(s), cube side = {args.cube_side:.2f} m, "
          f"target = ({target_point[0]:.3f}, {target_point[1]:.3f}, {target_point[2]:.3f})")

    for d in dirs:
        result = process_sequence(
            d,
            calib,
            args.cube_side,
            target_point,
            overwrite=args.overwrite,
            save_videos=args.save_videos,
            debug=args.debug,
            soft_mask=args.soft_mask,
            mask_white=args.mask_white,
            white_threshold=args.white_threshold,
            fov_shrink_px=args.fov_shrink_px,
            denoise=not args.disable_denoise,
            denoise_min_area=args.denoise_min_area,
            denoise_kernel=args.denoise_kernel,
            denoise_iters=args.denoise_iters,
            denoise_soft_threshold=args.denoise_soft_threshold,
        )
        status = "OK" if result["success"] else "FAIL"
        msg = f"  [{result['name']}] {status}"
        if result["n_frames"]:
            msg += f" ({result['n_frames']} frames)"
        if result["error"]:
            msg += f" — {result['error']}"
        print(msg)

    print("\nDone.")


if __name__ == "__main__":
    main()
