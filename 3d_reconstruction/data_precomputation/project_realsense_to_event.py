#!/usr/bin/env python3
"""
Project RealSense depth and RGB into the event camera frame using calibration data.

Depth projection (per depth frame in realsense.h5):
  1. Unproject every valid depth pixel to 3-D in the depth-camera frame.
  2. Transform the 3-D points into the event-camera frame (T_event_from_depth).
  3. Re-project into the event image using event intrinsics + distortion.
  4. Scatter the depth values onto the event-resolution grid.

RGB projection (per RGB frame in realsense.h5):
  Uses the same depth-based 3-D points (from step 1-2 above) to carry the
  corresponding RGB colour from the RealSense colour sensor into the event
  frame.  Each 3-D point is looked up in the RGB image via the RGB intrinsics
  & T_color_from_depth, and its colour is scattered to the event pixel.

Outputs (per recording):
  hdf5/depth_in_event_frame.h5   – (N, EH, EW) float32 depth in metres
  hdf5/rgb_in_event_frame.h5     – (N, EH, EW, 3) uint8 RGB
  videos/depth_in_event_frame.mp4 – colorised overlay for verification
  videos/rgb_in_event_frame.mp4   – RGB overlay for verification

Usage:
    # Process a single recording
    python3 project_realsense_to_event.py --data_dir data/real/1

    # Process all recordings under data/real/
    python3 project_realsense_to_event.py --data_root data/real

    # Depth only (skip RGB projection)
    python3 project_realsense_to_event.py --data_root data/real --no_rgb

    # Overwrite existing outputs
    python3 project_realsense_to_event.py --data_root data/real --overwrite

    # Custom calibration directory
    python3 project_realsense_to_event.py --data_root data/real --calib_dir camera_data
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import sys
from pathlib import Path as _Path_cfg
sys.path.insert(0, str(_Path_cfg(__file__).resolve().parent.parent))

from config import (
    CALIB_DIR as _CALIB_DIR,
    DATA_ROOT as _DATA_ROOT,
    FPS as DEFAULT_FPS,
    DEPTH_BLEED_RADIUS,
    TRAIN_RESIZE_HW,
    TRAIN_CROP_HW,
    CROP_THEN_RESIZE_CROP_HW,
    CROP_THEN_RESIZE_HW,
    DEPTH_VIZ_MIN,
    DEPTH_VIZ_MAX,
)

_CFG_ROOT = _Path_cfg(__file__).resolve().parent.parent
CALIB_DIR = _CFG_ROOT / _CALIB_DIR
DATA_ROOT  = _CFG_ROOT / _DATA_ROOT

# ─── module-level FPS (overridden by --fps CLI arg) ───────────────
FPS = DEFAULT_FPS


def _resize_crop(frame: np.ndarray, resize_hw, crop_hw, crop_then_resize=False) -> np.ndarray:
    """Apply either supported crop/resize order to depth or RGB.

    For depth frames the input must already be fully gap-filled at native
    resolution.  We then track valid coverage (non-zero pixels) through the
    same bilinear resize so that any output pixel whose bilinear footprint
    overlaps even one zero/invalid input pixel is zeroed out.  This prevents
    ghost depth values at the boundary of the valid region.
    RGB frames are bilinearly resized as-is.
    """
    import torch
    import torch.nn.functional as F
    is_rgb = frame.ndim == 3
    t = torch.from_numpy(frame.astype(np.float32))

    def center_crop(x, hw):
        if hw is None:
            return x
        ch, cw = hw
        y0 = (x.shape[-2] - ch) // 2
        x0 = (x.shape[-1] - cw) // 2
        return x[..., y0:y0 + ch, x0:x0 + cw]

    if is_rgb:
        t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        if crop_then_resize:
            t = center_crop(t, crop_hw)
            if resize_hw is not None:
                t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
        else:
            if resize_hw is not None:
                t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
            t = center_crop(t, crop_hw)
        return t.squeeze(0).permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    else:
        # Depth: propagate a validity mask through the same transform so that
        # bilinear blending with zeros never creates ghost non-zero depths.
        valid = (t > 0).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = t.unsqueeze(0).unsqueeze(0)                    # (1,1,H,W)
        if crop_then_resize:
            t = center_crop(t, crop_hw)
            valid = center_crop(valid, crop_hw)
            if resize_hw is not None:
                t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
                valid = F.interpolate(valid, size=resize_hw, mode="bilinear", align_corners=False)
        else:
            if resize_hw is not None:
                t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
                valid = F.interpolate(valid, size=resize_hw, mode="bilinear", align_corners=False)
            t = center_crop(t, crop_hw)
            valid = center_crop(valid, crop_hw)
        out = t.squeeze(0).squeeze(0).numpy()
        # Zero any pixel where the bilinear footprint was not 100 % valid.
        out[valid.squeeze(0).squeeze(0).numpy() < 1.0] = 0.0
        return out

def load_calibration(calib_dir: Path) -> dict:
    """Load all calibration data needed for depth/RGB → event projection."""
    T_event_from_depth = np.load(calib_dir / "T_event_from_depth.npz")["T"]  # (4,4)

    ev = np.load(calib_dir / "event_intrinsics.npz")
    K_event = ev["camera_matrix"]      # (3,3)
    dist_event = ev["dist_coeffs"]     # (1,5) or (5,)
    ev_size = ev["image_size"]         # [W, H]

    dep = np.load(calib_dir / "rs_depth_intrinsics.npz")
    K_depth = dep["camera_matrix"]     # (3,3)
    dep_size = dep["image_size"]       # [W, H]

    ds = np.load(calib_dir / "depth_scale.npz")
    depth_scale = float(ds["scale"])   # uint16 → metres

    # RGB intrinsics + depth-to-colour transform (for RGB projection)
    rgb = np.load(calib_dir / "rs_rgb_intrinsics.npz")
    K_rgb = rgb["camera_matrix"]       # (3,3)
    rgb_size = rgb["image_size"]       # [W, H]

    T_color_from_depth = np.load(calib_dir / "T_color_from_depth.npz")["T"]  # (4,4)

    return {
        "T_event_from_depth": T_event_from_depth,
        "K_event": K_event,
        "dist_event": dist_event.ravel(),
        "ev_w": int(ev_size[0]),
        "ev_h": int(ev_size[1]),
        "K_depth": K_depth,
        "dep_w": int(dep_size[0]),
        "dep_h": int(dep_size[1]),
        "depth_scale": depth_scale,
        "K_rgb": K_rgb,
        "rgb_w": int(rgb_size[0]),
        "rgb_h": int(rgb_size[1]),
        "T_color_from_depth": T_color_from_depth,
    }


def build_depth_pixel_grid(K_depth: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Pre-compute normalised ray directions for every depth pixel.

    Returns (h*w, 3) array where each row is [(u-cx)/fx, (v-cy)/fy, 1].
    """
    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (h, w)
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    return rays.reshape(-1, 3)


def _fill_small_depth_gaps_nearest(
    depth: np.ndarray,
    max_distance_px: float = 2.0,
) -> np.ndarray:
    """Fill small projection holes from the spatially nearest valid pixel.

    A minimum-depth morphological fill makes every ambiguous boundary hole
    foreground, visibly expanding near objects.  Nearest-neighbour filling is
    symmetric with respect to depth and is therefore a closer depth analogue
    of the local gap filling used for projected RGB.  The distance limit keeps
    genuinely unobserved image regions invalid.
    """
    invalid = depth <= 0
    if not np.any(invalid) or np.all(invalid):
        return depth

    # distanceTransform expects non-zero pixels to be measured to the nearest
    # zero pixel.  DIST_LABEL_PIXEL assigns a unique label to each valid source
    # pixel, allowing its depth to be copied into nearby holes.
    distances, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    valid_labels = labels[~invalid]
    valid_depths = depth[~invalid]
    label_to_depth = np.zeros(int(labels.max()) + 1, dtype=depth.dtype)
    label_to_depth[valid_labels] = valid_depths

    fill = invalid & (distances <= max_distance_px)
    result = depth.copy()
    result[fill] = label_to_depth[labels[fill]]
    return result


def project_depth_frame(
    depth_u16: np.ndarray,
    rays: np.ndarray,
    R: np.ndarray,
    t_vec: np.ndarray,
    K_event: np.ndarray,
    dist_event: np.ndarray,
    ev_h: int,
    ev_w: int,
    depth_scale: float,
    bleed_correction: bool = True,
) -> np.ndarray:
    """
    Project a single depth frame into the event camera image.

    Returns (ev_h, ev_w) float32 depth in metres; 0 where no data.
    """
    depth_m = depth_u16.astype(np.float64).ravel() * depth_scale
    valid = depth_m > 0.0
    if not np.any(valid):
        return np.zeros((ev_h, ev_w), dtype=np.float32)

    # Unproject valid pixels to 3-D (depth frame)
    pts_depth = rays[valid] * depth_m[valid, None]  # (M, 3)

    # Transform to event frame
    pts_event = (R @ pts_depth.T + t_vec).T          # (M, 3)

    # Keep only points in front of the event camera
    in_front = pts_event[:, 2] > 0.0
    pts_event = pts_event[in_front]
    z_event = pts_event[:, 2].copy()

    # Project with distortion using cv2
    rvec = cv2.Rodrigues(np.eye(3, dtype=np.float64))[0]  # identity rotation
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts_2d, _ = cv2.projectPoints(
        pts_event, rvec, tvec, K_event, dist_event
    )
    pts_2d = pts_2d.reshape(-1, 2)

    # Round to nearest pixel and scatter
    pu = np.round(pts_2d[:, 0]).astype(np.int32)
    pv = np.round(pts_2d[:, 1]).astype(np.int32)

    in_bounds = (pu >= 0) & (pu < ev_w) & (pv >= 0) & (pv < ev_h)
    pu = pu[in_bounds]
    pv = pv[in_bounds]
    z_event = z_event[in_bounds]

    depth_out = np.zeros((ev_h, ev_w), dtype=np.float32)

    # For overlapping projections keep the nearest depth (z-buffer)
    # Process in order of decreasing depth so nearer overwrites farther
    order = np.argsort(-z_event)
    depth_out[pv[order], pu[order]] = z_event[order].astype(np.float32)

    # Parallax bleed correction: a scattered background point that lands on a
    # foreground pixel appears as an isolated high-depth value surrounded by
    # low-depth values.  Replace it with the nearest valid neighbour depth.
    # SENTINEL substitutes for 0/invalid so erode gives the true nearest-valid
    # minimum instead of propagating zeros.
    # Keep BLEED_RADIUS small (1 = 3×3) so only immediately adjacent foreground
    # triggers a correction; larger radii turn into a foreground halo.
    SENTINEL = np.float32(1e6)
    if bleed_correction:
        BLEED_THRESHOLD_M = np.float32(0.05)   # pixel must be >5 cm farther than neighbour
        bleed_kernel = np.ones((2 * DEPTH_BLEED_RADIUS + 1, 2 * DEPTH_BLEED_RADIUS + 1), dtype=np.uint8)
        depth_temp = depth_out.copy()
        depth_temp[depth_temp == 0] = SENTINEL
        local_min = cv2.erode(depth_temp, bleed_kernel)
        local_min[local_min >= SENTINEL * 0.9] = np.float32(0.0)
        bleed_mask = (
            (depth_out > 0) &
            (local_min > 0) &
            (depth_out > local_min + BLEED_THRESHOLD_M)
        )
        depth_out[bleed_mask] = local_min[bleed_mask]

    # Fill only the small holes caused by scattering the lower-resolution depth
    # grid into the event image.  Copying the spatially nearest sample avoids
    # the foreground expansion caused by the previous local-minimum fill.
    depth_out = _fill_small_depth_gaps_nearest(depth_out, max_distance_px=2.0)

    return depth_out


def colorise_depth(depth: np.ndarray, min_m: float = DEPTH_VIZ_MIN, max_m: float = DEPTH_VIZ_MAX) -> np.ndarray:
    """Convert depth (float32, metres) to a colour image for visualisation."""
    d = np.clip((depth - min_m) / (max_m - min_m), 0, 1)
    d_u8 = (d * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    # Set invalid (0) pixels to black
    coloured[depth == 0] = 0
    return coloured


def project_rgb_frame(
    depth_u16: np.ndarray,
    rgb_u8: np.ndarray,
    rays: np.ndarray,
    R_ev: np.ndarray,
    t_ev: np.ndarray,
    T_color_from_depth: np.ndarray,
    K_rgb: np.ndarray,
    K_event: np.ndarray,
    dist_event: np.ndarray,
    ev_h: int,
    ev_w: int,
    rgb_h: int,
    rgb_w: int,
    depth_scale: float,
) -> np.ndarray:
    """
    Project a single RGB frame into the event camera image using depth.

    Strategy: unproject depth pixels to 3-D (depth frame) → look up their
    colour in the RGB image via T_color_from_depth + K_rgb → scatter colour
    to event pixels via T_event_from_depth + K_event.

    Returns (ev_h, ev_w, 3) uint8 RGB image; black where no data.
    """
    depth_m = depth_u16.astype(np.float64).ravel() * depth_scale
    valid = depth_m > 0.0
    if not np.any(valid):
        return np.zeros((ev_h, ev_w, 3), dtype=np.uint8)

    # Unproject valid depth pixels to 3-D (depth frame)
    pts_depth = rays[valid] * depth_m[valid, None]  # (M, 3)

    # --- Look up colours from RGB image ---
    R_col = T_color_from_depth[:3, :3]
    t_col = T_color_from_depth[:3, 3:4]
    pts_color = (R_col @ pts_depth.T + t_col).T  # (M, 3) in colour frame
    # Project to RGB pixel coords (no distortion — RealSense colour is rectified)
    z_col = pts_color[:, 2]
    col_valid = z_col > 0
    uv_rgb = np.zeros((len(pts_depth), 2), dtype=np.float64)
    uv_rgb[col_valid, 0] = K_rgb[0, 0] * pts_color[col_valid, 0] / z_col[col_valid] + K_rgb[0, 2]
    uv_rgb[col_valid, 1] = K_rgb[1, 1] * pts_color[col_valid, 1] / z_col[col_valid] + K_rgb[1, 2]
    ru = np.round(uv_rgb[:, 0]).astype(np.int32)
    rv = np.round(uv_rgb[:, 1]).astype(np.int32)
    rgb_in_bounds = col_valid & (ru >= 0) & (ru < rgb_w) & (rv >= 0) & (rv < rgb_h)

    # Sample colours (for out-of-bounds pixels, default to black)
    colours = np.zeros((len(pts_depth), 3), dtype=np.uint8)
    colours[rgb_in_bounds] = rgb_u8[rv[rgb_in_bounds], ru[rgb_in_bounds]]

    # --- Project 3-D points into event image ---
    pts_event = (R_ev @ pts_depth.T + t_ev).T  # (M, 3)
    in_front = pts_event[:, 2] > 0.0
    pts_event_f = pts_event[in_front]
    colours_f = colours[in_front]
    z_event = pts_event_f[:, 2].copy()

    rvec = cv2.Rodrigues(np.eye(3, dtype=np.float64))[0]
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts_2d, _ = cv2.projectPoints(pts_event_f, rvec, tvec, K_event, dist_event)
    pts_2d = pts_2d.reshape(-1, 2)

    pu = np.round(pts_2d[:, 0]).astype(np.int32)
    pv = np.round(pts_2d[:, 1]).astype(np.int32)
    in_bounds = (pu >= 0) & (pu < ev_w) & (pv >= 0) & (pv < ev_h)
    pu = pu[in_bounds]
    pv = pv[in_bounds]
    z_event = z_event[in_bounds]
    colours_f = colours_f[in_bounds]

    rgb_out = np.zeros((ev_h, ev_w, 3), dtype=np.uint8)

    # Z-buffer: nearest colour wins (sort by decreasing depth, nearer overwrites)
    order = np.argsort(-z_event)
    rgb_out[pv[order], pu[order]] = colours_f[order]

    # Fill small gaps with dilation (same strategy as depth)
    for c in range(3):
        ch = rgb_out[:, :, c]
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(ch, kernel, iterations=1)
        ch[ch == 0] = dilated[ch == 0]

    return rgb_out


def process_recording(seq_dir: Path, calib: dict, project_rgb: bool = True,
                      save_videos: bool = False, workers: int = 4,
                      bleed_correction: bool = True, overwrite: bool = False,
                      resize_hw=None, crop_hw=None,
                      use_voxels: bool = False,
                      crop_then_resize: bool = False) -> None:
    """Project all depth (and optionally RGB) frames for one recording directory."""
    rs_h5_path = seq_dir / "hdf5" / "realsense.h5"
    if not rs_h5_path.exists():
        print(f"[skip] No realsense.h5 in {seq_dir}")
        return

    out_depth_h5 = seq_dir / "hdf5" / "depth_in_event_frame.h5"
    out_depth_vid = seq_dir / "videos" / "depth_in_event_frame.mp4"
    out_rgb_h5 = seq_dir / "hdf5" / "rgb_in_event_frame.h5"
    out_rgb_vid = seq_dir / "videos" / "rgb_in_event_frame.mp4"

    if not overwrite and out_depth_h5.exists():
        print(f"[skip] {seq_dir.name} — depth_in_event_frame.h5 already exists (use --overwrite)")
        return

    if save_videos:
        (seq_dir / "videos").mkdir(parents=True, exist_ok=True)

    # Extract calibration
    T = calib["T_event_from_depth"]
    R = T[:3, :3]
    t_vec = T[:3, 3:4]
    K_event = calib["K_event"]
    dist_event = calib["dist_event"]
    ev_h, ev_w = calib["ev_h"], calib["ev_w"]
    K_depth = calib["K_depth"]
    dep_h, dep_w = calib["dep_h"], calib["dep_w"]
    depth_scale = calib["depth_scale"]

    K_rgb = calib["K_rgb"]
    rgb_h, rgb_w = calib["rgb_h"], calib["rgb_w"]
    T_color_from_depth = calib["T_color_from_depth"]

    out_h, out_w = ev_h, ev_w
    if crop_then_resize:
        if crop_hw is not None:
            out_h, out_w = crop_hw
        if resize_hw is not None:
            out_h, out_w = resize_hw
    else:
        if resize_hw is not None:
            out_h, out_w = resize_hw
        if crop_hw is not None:
            out_h, out_w = crop_hw

    # Pre-compute depth pixel rays
    rays = build_depth_pixel_grid(K_depth, dep_h, dep_w)

    # Open source HDF5
    with h5py.File(rs_h5_path, "r") as rs_h5:
        depth_ds = rs_h5["depth"]
        N = depth_ds.shape[0]

        has_rgb_src = project_rgb and ("rgb" in rs_h5)
        rgb_ds = rs_h5["rgb"] if has_rgb_src else None
        if project_rgb and not has_rgb_src:
            print(f"  [{seq_dir.name}] Warning: --rgb requested but no 'rgb' dataset in realsense.h5")

        print(f"[{seq_dir.name}] Projecting {N} frames ({dep_w}x{dep_h}) → event frame ({ev_w}x{ev_h})"
              + (f" → stored ({out_w}x{out_h})" if (out_h, out_w) != (ev_h, ev_w) else "")
              + (f" + RGB" if has_rgb_src else ""))

        # Video writers (optional)
        depth_video = None
        rgb_video = None
        if save_videos:
            depth_video = cv2.VideoWriter(
                str(out_depth_vid), cv2.VideoWriter_fourcc(*"mp4v"),
                FPS, (out_w, out_h), isColor=True,
            )
            if has_rgb_src:
                rgb_video = cv2.VideoWriter(
                    str(out_rgb_vid), cv2.VideoWriter_fourcc(*"mp4v"),
                    FPS, (out_w, out_h), isColor=True,
                )

        # Event frames (or all voxel bins) for overlay
        ev_h5_path     = seq_dir / "hdf5"   / "events_cam0.h5"
        voxel_h5_path  = seq_dir / "events" / "voxels_cam0.h5"
        if use_voxels and voxel_h5_path.exists():
            ev_h5 = h5py.File(voxel_h5_path, "r")
            _ev_source = "voxels"
        elif ev_h5_path.exists():
            ev_h5 = h5py.File(ev_h5_path, "r")
            _ev_source = "frames"
        else:
            ev_h5 = None
            _ev_source = None
        if use_voxels and not voxel_h5_path.exists():
            print(f"  [{seq_dir.name}] Warning: --use_voxels requested but "
                  f"events/voxels_cam0.h5 not found; falling back to event frames")

        # Output HDF5 files
        out_dh5 = h5py.File(out_depth_h5, "w")
        depth_out_ds = out_dh5.create_dataset(
            "depth", shape=(N, out_h, out_w), dtype=np.float32,
            chunks=(1, out_h, out_w), compression="gzip", compression_opts=4,
        )
        out_dh5.attrs["description"] = "Depth projected into event camera frame (metres)"
        out_dh5.attrs["source"] = str(rs_h5_path)
        out_dh5.attrs["native_ev_h"] = ev_h
        out_dh5.attrs["native_ev_w"] = ev_w
        out_dh5.attrs["resize_h"] = resize_hw[0] if resize_hw is not None else ev_h
        out_dh5.attrs["resize_w"] = resize_hw[1] if resize_hw is not None else ev_w
        out_dh5.attrs["crop_h"]   = crop_hw[0]   if crop_hw   is not None else (resize_hw[0] if resize_hw else ev_h)
        out_dh5.attrs["crop_w"]   = crop_hw[1]   if crop_hw   is not None else (resize_hw[1] if resize_hw else ev_w)
        out_dh5.attrs["intrinsics_transform"] = (
            "center_crop_resize" if crop_then_resize else "resize_center_crop"
        )

        out_rh5 = None
        rgb_out_ds = None
        if has_rgb_src:
            out_rh5 = h5py.File(out_rgb_h5, "w")
            rgb_out_ds = out_rh5.create_dataset(
                "rgb", shape=(N, out_h, out_w, 3), dtype=np.uint8,
                chunks=(1, out_h, out_w, 3), compression="gzip", compression_opts=4,
            )
            out_rh5.attrs["description"] = "RGB projected into event camera frame"
            out_rh5.attrs["source"] = str(rs_h5_path)
            out_rh5.attrs["intrinsics_transform"] = (
                "center_crop_resize" if crop_then_resize else "resize_center_crop"
            )

        try:
            batch_size = max(1, workers * 4)
            with ThreadPoolExecutor(max_workers=workers) as pool, \
                 tqdm(total=N, desc=f"  {seq_dir.name}", unit="frame") as pbar:
                for batch_start in range(0, N, batch_size):
                    batch_end = min(batch_start + batch_size, N)
                    batch_range = range(batch_start, batch_end)

                    # Sequential reads (HDF5 handles are not thread-safe)
                    batch_depth_u16 = [depth_ds[i] for i in batch_range]
                    batch_rgb_u8 = [rgb_ds[i] for i in batch_range] if has_rgb_src else []
                    if ev_h5 is not None:
                        if _ev_source == "voxels":
                            vox_ds  = ev_h5["voxels"]
                            vox_n   = vox_ds.shape[0]
                            # Spatial metadata for reversing the resize+crop baked into the voxels.
                            # Voxels may be stored at (crop_h x crop_w) after an intermediate
                            # resize to (resize_h x resize_w) from native (native_h x native_w).
                            # Simply stretching crop→out would squash the FOV.  Instead:
                            #   1. embed the cropped frame back into the resize canvas
                            #   2. scale that canvas to (out_h x out_w)
                            _vatts      = dict(vox_ds.attrs)
                            _vox_crop_h = int(vox_ds.shape[2])
                            _vox_crop_w = int(vox_ds.shape[3])
                            _vox_rsz_h  = int(_vatts.get("resize_h", _vox_crop_h))
                            _vox_rsz_w  = int(_vatts.get("resize_w", _vox_crop_w))

                            def _vox_to_gray(v):
                                """Render all voxel bins (n_bins, crop_H, crop_W) → uint8 grayscale at (out_h, out_w).

                                Sums absolute event activity across every bin so all polarities and
                                time-windows contribute.  No-event pixels map to black (0); active
                                pixels are bright in proportion to total activity — much more visible
                                as an overlay than a single mid-bin rendered with a grey offset.
                                """
                                v = v.astype(np.float32)  # (n_bins, H, W)
                                # Clip each bin before summing to prevent a few hot pixels dominating
                                n_bins = v.shape[0]
                                activity = np.sum(np.abs(np.clip(v, -1.0, 1.0)), axis=0)  # (H, W)
                                # Normalise to [0, 1]: max possible is n_bins * 1.0
                                activity = np.clip(activity / n_bins, 0.0, 1.0)
                                gray = (activity * 255).astype(np.uint8)
                                if gray.shape == (out_h, out_w):
                                    return gray
                                # Step 1: undo center-crop → embed in resize canvas
                                if _vox_crop_h != _vox_rsz_h or _vox_crop_w != _vox_rsz_w:
                                    canvas = np.zeros((_vox_rsz_h, _vox_rsz_w), dtype=np.uint8)
                                    y0 = (_vox_rsz_h - _vox_crop_h) // 2
                                    x0 = (_vox_rsz_w - _vox_crop_w) // 2
                                    canvas[y0:y0+_vox_crop_h, x0:x0+_vox_crop_w] = gray
                                    gray = canvas
                                # Step 2: scale resize canvas → output resolution
                                if gray.shape[0] != out_h or gray.shape[1] != out_w:
                                    gray = cv2.resize(gray, (out_w, out_h),
                                                      interpolation=cv2.INTER_LINEAR)
                                return gray

                            batch_ev = [
                                _vox_to_gray(vox_ds[i]) if i < vox_n else None
                                for i in batch_range
                            ]
                        else:
                            ev_data = ev_h5["events/frames"]
                            ev_n = ev_data.shape[0]
                            batch_ev = [ev_data[i] if i < ev_n else None for i in batch_range]
                    else:
                        batch_ev = [None] * len(batch_range)

                    # Parallel projection
                    depth_futures = [
                        pool.submit(
                            project_depth_frame,
                            d, rays, R, t_vec, K_event, dist_event, ev_h, ev_w, depth_scale,
                            bleed_correction,
                        )
                        for d in batch_depth_u16
                    ]
                    rgb_futures = [
                        pool.submit(
                            project_rgb_frame,
                            batch_depth_u16[j], batch_rgb_u8[j], rays, R, t_vec,
                            T_color_from_depth, K_rgb,
                            K_event, dist_event,
                            ev_h, ev_w, rgb_h, rgb_w, depth_scale,
                        )
                        for j in range(len(batch_range))
                    ] if has_rgb_src else []

                    # Sequential writes (HDF5 and VideoWriter are not thread-safe)
                    for j, i in enumerate(batch_range):
                        proj_depth = depth_futures[j].result()
                        if resize_hw is not None or crop_hw is not None:
                            proj_depth = _resize_crop(proj_depth, resize_hw, crop_hw, crop_then_resize)
                        depth_out_ds[i] = proj_depth

                        if save_videos:
                            colour = colorise_depth(proj_depth)
                            ev_gray = batch_ev[j]
                            if ev_gray is not None:
                                if (_ev_source != "voxels" and
                                        (resize_hw is not None or crop_hw is not None)):
                                    ev_gray = _resize_crop(
                                        ev_gray.astype(np.float32), resize_hw, crop_hw, crop_then_resize
                                    ).astype(np.uint8)
                                # Ensure ev_gray matches the output frame size
                                # (voxels may be stored at a different resolution)
                                if ev_gray.shape[0] != out_h or ev_gray.shape[1] != out_w:
                                    ev_gray = cv2.resize(ev_gray, (out_w, out_h),
                                                         interpolation=cv2.INTER_LINEAR)
                                ev_bgr = cv2.cvtColor(ev_gray, cv2.COLOR_GRAY2BGR)
                                mask = proj_depth > 0
                                frame = ev_bgr.copy()
                                frame[mask] = cv2.addWeighted(ev_bgr, 0.4, colour, 0.6, 0)[mask]
                            else:
                                frame = colour
                            depth_video.write(frame)

                        if has_rgb_src:
                            proj_rgb = rgb_futures[j].result()
                            if resize_hw is not None or crop_hw is not None:
                                proj_rgb = _resize_crop(proj_rgb, resize_hw, crop_hw, crop_then_resize)
                            rgb_out_ds[i] = proj_rgb

                            if save_videos:
                                rgb_bgr = cv2.cvtColor(proj_rgb, cv2.COLOR_RGB2BGR)
                                ev_gray = batch_ev[j]
                                if ev_gray is not None:
                                    if (_ev_source != "voxels" and
                                            (resize_hw is not None or crop_hw is not None)):
                                        ev_gray = _resize_crop(
                                            ev_gray.astype(np.float32), resize_hw, crop_hw, crop_then_resize
                                        ).astype(np.uint8)
                                    # Ensure ev_gray matches the output frame size
                                    if ev_gray.shape[0] != out_h or ev_gray.shape[1] != out_w:
                                        ev_gray = cv2.resize(ev_gray, (out_w, out_h),
                                                             interpolation=cv2.INTER_LINEAR)
                                    ev_bgr2 = cv2.cvtColor(ev_gray, cv2.COLOR_GRAY2BGR)
                                    rgb_mask = np.any(proj_rgb > 0, axis=-1)
                                    rgb_frame = ev_bgr2.copy()
                                    rgb_frame[rgb_mask] = cv2.addWeighted(ev_bgr2, 0.3, rgb_bgr, 0.7, 0)[rgb_mask]
                                else:
                                    rgb_frame = rgb_bgr
                                rgb_video.write(rgb_frame)

                    pbar.update(len(batch_range))
        finally:
            if ev_h5 is not None:
                ev_h5.close()
            if depth_video is not None:
                depth_video.release()
            if rgb_video is not None:
                rgb_video.release()
            out_dh5.close()
            if out_rh5 is not None:
                out_rh5.close()

    print(f"  -> {out_depth_h5}")
    if save_videos:
        print(f"  -> {out_depth_vid}")
    if has_rgb_src:
        print(f"  -> {out_rgb_h5}")
        if save_videos:
            print(f"  -> {out_rgb_vid}")


def find_recordings(root: Path) -> list[Path]:
    """Find recording directories at or recursively below ``root``."""
    root = Path(root)
    if (root / "hdf5" / "realsense.h5").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        hdf5_dir.parent
        for hdf5_dir in root.rglob("hdf5")
        if (hdf5_dir / "realsense.h5").is_file()
    )


def main():
    global FPS
    parser = argparse.ArgumentParser(
        description="Project RealSense depth and RGB into the event camera frame",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", nargs="+", type=str, default=None,
        help="Specific recording directory(ies)",
    )
    parser.add_argument(
        "--data_root", type=str, default=str(DATA_ROOT),
        help="Root directory containing recording subdirs",
    )
    parser.add_argument(
        "--calib_dir", type=str, default=str(CALIB_DIR),
        help="Path to camera_data/ calibration directory",
    )
    parser.add_argument(
        "--fps", type=int, default=FPS,
        help="Video output framerate",
    )
    parser.add_argument(
        "--no_rgb", action="store_true",
        help="Skip RGB projection (depth only)",
    )
    parser.add_argument(
        "--save_videos", action="store_true",
        help="Generate MP4 videos (depth_in_event_frame.mp4, rgb_in_event_frame.mp4). "
             "Default: only create HDF5 files.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for frame projection",
    )
    parser.add_argument(
        "--no-bleed-correction", action="store_true",
        help="Disable parallax bleed correction (background depth on foreground pixels)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing depth_in_event_frame.h5 / rgb_in_event_frame.h5 files",
    )
    parser.add_argument("--resize_h", type=int, default=TRAIN_RESIZE_HW[0],
                        help="Resize height before crop (0 = skip resize, default: TRAIN_RESIZE_HW from config)")
    parser.add_argument("--resize_w", type=int, default=TRAIN_RESIZE_HW[1],
                        help="Resize width before crop (0 = skip resize, default: TRAIN_RESIZE_HW from config)")
    parser.add_argument("--crop_h", type=int, default=TRAIN_CROP_HW[0],
                        help="Center-crop height after resize (0 = skip crop, default: TRAIN_CROP_HW from config)")
    parser.add_argument("--crop_w", type=int, default=TRAIN_CROP_HW[1],
                        help="Center-crop width after resize (0 = skip crop, default: TRAIN_CROP_HW from config)")
    parser.add_argument("--no_resize_crop", action="store_true",
                        help="Store depth/RGB at native event-camera resolution without any resize or crop")
    parser.add_argument("--use_voxels", action="store_true",
                        help="Use the middle bin of precomputed voxel grids (events/voxels_cam0.h5) "
                             "instead of event frames for the overlay video. "
                             "Requires precompute_voxels.py to have been run first.")
    parser.add_argument("--crop_then_resize", "--crop-then-resize", action="store_true",
                        help="Center-crop at native resolution, then resize using "
                             "CROP_THEN_RESIZE_* settings from config.py")
    args = parser.parse_args()

    if args.no_resize_crop and args.crop_then_resize:
        parser.error("--no_resize_crop and --crop_then_resize are mutually exclusive")
    if args.no_resize_crop:
        resize_hw = None
        crop_hw   = None
    elif args.crop_then_resize:
        crop_hw = CROP_THEN_RESIZE_CROP_HW
        resize_hw = CROP_THEN_RESIZE_HW
    else:
        resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None
        crop_hw   = (args.crop_h,   args.crop_w)   if args.crop_h   > 0 and args.crop_w   > 0 else None

    if args.crop_then_resize:
        calib_size = load_calibration(Path(args.calib_dir))
        if crop_hw[0] > calib_size["ev_h"] or crop_hw[1] > calib_size["ev_w"]:
            parser.error(
                f"crop {crop_hw} exceeds native event size "
                f"{(calib_size['ev_h'], calib_size['ev_w'])}"
            )

    calib = load_calibration(Path(args.calib_dir))

    if args.data_dir:
        dirs = sorted({
            recording
            for data_dir in args.data_dir
            for recording in find_recordings(Path(data_dir))
        })
    else:
        dirs = find_recordings(Path(args.data_root))
    if not dirs:
        parser.error("No recordings containing hdf5/realsense.h5 were found")

    print(f"Processing {len(dirs)} recording(s)")
    print(f"RGB projection: {'off' if args.no_rgb else 'on'}")
    print(f"Video generation: {'on' if args.save_videos else 'off'}")
    print(f"Overlay source: {'voxel middle bin' if args.use_voxels else 'event frames'}")
    if resize_hw:
        if args.crop_then_resize:
            print(f"Crop → {crop_hw[1]}×{crop_hw[0]} → resize → {resize_hw[1]}×{resize_hw[0]}")
        else:
            print(f"Resize → {resize_hw[1]}×{resize_hw[0]}"
                  + (f" → crop → {crop_hw[1]}×{crop_hw[0]}" if crop_hw else ""))
    FPS = args.fps

    for d in dirs:
        process_recording(
            d, calib,
            project_rgb=not args.no_rgb,
            save_videos=args.save_videos,
            workers=args.workers,
            bleed_correction=not args.no_bleed_correction,
            overwrite=args.overwrite,
            resize_hw=resize_hw,
            crop_hw=crop_hw,
            use_voxels=args.use_voxels,
            crop_then_resize=args.crop_then_resize,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
