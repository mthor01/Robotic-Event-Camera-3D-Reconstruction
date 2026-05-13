#!/usr/bin/env python3
"""
Precompute spatial masks that restrict depth supervision to a cube around
the robot end-effector.

For each depth frame, every pixel with valid depth is unprojected to 3-D
in the robot base frame.  Pixels whose 3-D point falls inside a cube
(default 40 cm side length) centred on the current end-effector position
are marked 1, all others 0.

The mask lives in the **event-camera image plane** (same resolution as
depth_in_event_frame.h5), so it can be applied directly during training
or visualization without additional alignment.

Outputs (per recording):
    hdf5/spatial_mask.h5          — dataset "mask" (N, H, W) uint8 {0, 1}
    videos/spatial_mask.mp4       — (optional, with --save_videos)

Usage:
    python3 precompute_spatial_mask.py --data_root data/real
    python3 precompute_spatial_mask.py --data_dir data/real/box --cube_side 0.4
    python3 precompute_spatial_mask.py --data_root data/real --overwrite
    python3 precompute_spatial_mask.py --data_root data/real --save_videos
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

    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee
    # inv(T_event_from_ee) brings event-cam → ee → base when composed with ee_T
    T_ee_from_event = np.linalg.inv(T_event_from_ee)

    # depth camera calibration (for native-resolution mask stored in mask_depth)
    T_color_from_depth = np.load(calib_dir / "T_color_from_depth.npz")["T"]  # (4,4)
    depth_intr = np.load(calib_dir / "rs_depth_intrinsics.npz")
    K_depth    = depth_intr["camera_matrix"]   # (3,3)
    depth_size = depth_intr["image_size"]       # [W, H]
    depth_scale = float(np.load(calib_dir / "depth_scale.npz")["scale"])
    T_ee_from_depth = np.linalg.inv(T_rgb_from_ee) @ T_color_from_depth

    return {
        "T_ee_from_event": T_ee_from_event,
        "K_event": K_event,
        "dist_event": dist_event,
        "ev_w": int(ev_size[0]),
        "ev_h": int(ev_size[1]),
        "T_ee_from_depth": T_ee_from_depth,
        "K_depth": K_depth,
        "depth_w": int(depth_size[0]),
        "depth_h": int(depth_size[1]),
        "depth_scale": depth_scale,
    }


def build_event_rays(K_event: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pre-compute normalised ray directions for every event pixel.

    Returns (h*w, 3) where each row is K_event_inv @ [u, v, 1].
    """
    K_inv = np.linalg.inv(K_event)
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (h, w, 3)
    rays = (K_inv @ pixels.reshape(-1, 3).T).T               # (h*w, 3)
    return rays


def compute_spatial_mask(
    depth: np.ndarray,
    ee_T: np.ndarray,
    T_ee_from_event: np.ndarray,
    rays: np.ndarray,
    half_side: float,
    target_point: np.ndarray,
) -> np.ndarray:
    """Compute binary mask: 1 for pixels inside the cube, 0 outside.

    Args:
        depth:            (H, W) float32 metres, projected into event frame.
        ee_T:             (4, 4) T_base_from_ee for this frame.
        T_ee_from_event:  (4, 4) static transform (inv of T_event_from_ee).
        rays:             (H*W, 3) normalised ray directions in event frame.
        half_side:        Half the cube side length in metres.
        target_point:     (3,) cube centre in robot base frame.

    Returns:
        (H, W) uint8 mask.
    """
    H, W = depth.shape
    flat_depth = depth.ravel().astype(np.float64)
    valid = flat_depth > 0.0

    mask_flat = np.zeros(H * W, dtype=np.uint8)
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
    mask_flat[valid_indices[inside]] = 1
    return mask_flat.reshape(H, W)


def save_debug_png(seq_dir: Path, depth_frames: list, mask_frames: list) -> None:
    """Save a side-by-side depth / mask contact sheet for 10 evenly-spaced frames.

    Layout per row: [colourised depth | mask overlay on depth]
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

        # Right panel: mask (white=inside) overlaid on grey depth
        grey = cv2.cvtColor(d_u8, cv2.COLOR_GRAY2BGR)
        # Tint inside-cube pixels green
        right = grey.copy()
        overlay = right.copy()
        overlay[mask == 1] = (0, 220, 0)
        right = cv2.addWeighted(right, 0.5, overlay, 0.5, 0)

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
    rays = build_event_rays(K_event, H, W)
    half_side = cube_side / 2.0

    # Depth-frame mask: compute natively at depth resolution from realsense.h5
    realsense_h5_path = seq_dir / "hdf5" / "realsense.h5"
    has_realsense = realsense_h5_path.exists()
    if has_realsense:
        dH, dW = calib["depth_h"], calib["depth_w"]
        rays_depth = build_event_rays(calib["K_depth"], dH, dW)
    else:
        dH = dW = 0
        rays_depth = None

    # Process frames
    rs_file = h5py.File(realsense_h5_path, "r") if has_realsense else None
    try:
        with h5py.File(depth_h5_path, "r") as df, \
             h5py.File(out_path, "w") as of:
            mask_ds = of.create_dataset(
                "mask", shape=(n_frames, H, W), dtype=np.uint8,
                chunks=(1, H, W), compression="gzip", compression_opts=4,
            )
            if has_realsense:
                mask_depth_ds = of.create_dataset(
                    "mask_depth", shape=(n_frames, dH, dW), dtype=np.uint8,
                    chunks=(1, dH, dW), compression="gzip", compression_opts=4,
                )
            of.attrs["cube_side_m"] = cube_side
            of.attrs["target_point"] = target_point
            of.attrs["description"] = (
                f"Spatial mask: 1 inside {cube_side:.2f}m cube around "
                f"target ({target_point[0]:.3f}, {target_point[1]:.3f}, {target_point[2]:.3f}), 0 outside"
            )

            masks = np.empty((n_frames, H, W), dtype=np.uint8)
            debug_depths: list = [] if debug else None  # type: ignore
            debug_masks:  list = [] if debug else None  # type: ignore
            rs_depth_ds = rs_file["depth"] if rs_file is not None else None
            for i in tqdm(range(n_frames), desc=f"  {seq_dir.name}", leave=False):
                depth = df["depth"][i].astype(np.float32)
                m = compute_spatial_mask(
                    depth, ee_T_all[i], calib["T_ee_from_event"], rays, half_side,
                    target_point,
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
                        half_side, target_point,
                    )
                    mask_depth_ds[i] = md
    finally:
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
        result = process_sequence(d, calib, args.cube_side, target_point, overwrite=args.overwrite, save_videos=args.save_videos, debug=args.debug)
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
