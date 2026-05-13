#!/usr/bin/env python3
"""
Project RealSense depth into the RGB camera frame using calibration data.

For each depth frame in realsense.h5:
  1. Unproject every valid depth pixel to 3-D in the depth-camera frame.
  2. Transform the 3-D points into the colour-camera frame (T_color_from_depth).
  3. Re-project into the RGB image using RGB intrinsics (no distortion —
     RealSense colour is rectified).
  4. Scatter the depth values onto the RGB-resolution grid.

Outputs (per recording):
  hdf5/depth_in_rgb_frame.h5    – (N, RGB_H, RGB_W) float32 depth in metres
  videos/depth_in_rgb_frame.mp4 – colorised overlay on the RGB image for verification

Usage:
    # Process a single recording
    python3 project_realsense_depth_to_rgb.py --data_dir data/real/1

    # Process all recordings under data/real/
    python3 project_realsense_depth_to_rgb.py --data_root data/real

    # Overwrite existing outputs
    python3 project_realsense_depth_to_rgb.py --data_root data/real --overwrite
"""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path as _Path_cfg
sys.path.insert(0, str(_Path_cfg(__file__).resolve().parent.parent))

from config import (
    CALIB_DIR as _CALIB_DIR,
    DATA_ROOT as _DATA_ROOT,
    FPS as DEFAULT_FPS,
    DEPTH_BLEED_RADIUS,
)

_CFG_ROOT = _Path_cfg(__file__).resolve().parent.parent
CALIB_DIR = _CFG_ROOT / _CALIB_DIR
DATA_ROOT  = _CFG_ROOT / _DATA_ROOT

# ─── module-level FPS (overridden by --fps CLI arg) ───────────────
FPS = DEFAULT_FPS


def load_calibration(calib_dir: Path) -> dict:
    """Load calibration data needed for depth → RGB projection."""
    dep = np.load(calib_dir / "rs_depth_intrinsics.npz")
    K_depth = dep["camera_matrix"]    # (3,3)
    dep_size = dep["image_size"]      # [W, H]

    ds = np.load(calib_dir / "depth_scale.npz")
    depth_scale = float(ds["scale"])  # uint16 → metres

    rgb = np.load(calib_dir / "rs_rgb_intrinsics.npz")
    K_rgb = rgb["camera_matrix"]      # (3,3)
    rgb_size = rgb["image_size"]      # [W, H]

    T_color_from_depth = np.load(calib_dir / "T_color_from_depth.npz")["T"]  # (4,4)

    return {
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
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    return rays.reshape(-1, 3)


def project_depth_to_rgb_frame(
    depth_u16: np.ndarray,
    rays: np.ndarray,
    R: np.ndarray,
    t_vec: np.ndarray,
    K_rgb: np.ndarray,
    rgb_h: int,
    rgb_w: int,
    depth_scale: float,
    bleed_correction: bool = True,
) -> np.ndarray:
    """
    Project a single depth frame into the RGB camera image.

    The RealSense colour image is rectified, so no distortion correction
    is needed — plain pinhole projection suffices.

    Returns (rgb_h, rgb_w) float32 depth in metres; 0 where no data.
    """
    depth_m = depth_u16.astype(np.float64).ravel() * depth_scale
    valid = depth_m > 0.0
    if not np.any(valid):
        return np.zeros((rgb_h, rgb_w), dtype=np.float32)

    # Unproject valid pixels to 3-D (depth frame)
    pts_depth = rays[valid] * depth_m[valid, None]  # (M, 3)

    # Transform to colour frame
    pts_color = (R @ pts_depth.T + t_vec).T  # (M, 3)

    # Keep only points in front of the colour camera
    in_front = pts_color[:, 2] > 0.0
    pts_color = pts_color[in_front]
    z_color = pts_color[:, 2].copy()

    # Pinhole projection (no distortion)
    fx, fy = K_rgb[0, 0], K_rgb[1, 1]
    cx, cy = K_rgb[0, 2], K_rgb[1, 2]
    pu = np.round(fx * pts_color[:, 0] / z_color + cx).astype(np.int32)
    pv = np.round(fy * pts_color[:, 1] / z_color + cy).astype(np.int32)

    in_bounds = (pu >= 0) & (pu < rgb_w) & (pv >= 0) & (pv < rgb_h)
    pu = pu[in_bounds]
    pv = pv[in_bounds]
    z_color = z_color[in_bounds]

    depth_out = np.zeros((rgb_h, rgb_w), dtype=np.float32)

    # Z-buffer: nearest depth wins (sort by decreasing depth so nearer overwrites)
    order = np.argsort(-z_color)
    depth_out[pv[order], pu[order]] = z_color[order].astype(np.float32)

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

    # Gap fill: fill zero pixels (scatter resolution mismatch) with the nearest
    # (minimum) valid neighbour using erode+sentinel.
    # Do NOT use cv2.dilate (local MAX) — that fills gaps with background (far)
    # depth, which causes a background grid pattern on the foreground object.
    small_kernel = np.ones((3, 3), dtype=np.uint8)
    depth_temp2 = depth_out.copy()
    depth_temp2[depth_temp2 == 0] = SENTINEL
    fill_min = cv2.erode(depth_temp2, small_kernel)
    fill_min[fill_min >= SENTINEL * 0.9] = np.float32(0.0)
    depth_out[depth_out == 0] = fill_min[depth_out == 0]

    return depth_out


def colorise_depth(depth: np.ndarray, max_m: float = 2.0) -> np.ndarray:
    """Convert depth (float32, metres) to a colour image for visualisation."""
    d = np.clip(depth / max_m, 0, 1)
    d_u8 = (d * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    coloured[depth == 0] = 0
    return coloured


def process_recording(
    seq_dir: Path,
    calib: dict,
    overwrite: bool = False,
    workers: int = 4,
    bleed_correction: bool = True,
) -> None:
    """Project all depth frames to the RGB frame for one recording directory."""
    rs_h5_path = seq_dir / "hdf5" / "realsense.h5"
    if not rs_h5_path.exists():
        print(f"[skip] No realsense.h5 in {seq_dir}")
        return

    out_depth_h5 = seq_dir / "hdf5" / "depth_in_rgb_frame.h5"
    out_depth_vid = seq_dir / "videos" / "depth_in_rgb_frame.mp4"

    if not overwrite and out_depth_h5.exists():
        print(f"[skip] {seq_dir.name} — depth_in_rgb_frame.h5 already exists (use --overwrite)")
        return

    (seq_dir / "videos").mkdir(parents=True, exist_ok=True)

    T = calib["T_color_from_depth"]
    R = T[:3, :3]
    t_vec = T[:3, 3:4]
    K_rgb = calib["K_rgb"]
    rgb_h, rgb_w = calib["rgb_h"], calib["rgb_w"]
    K_depth = calib["K_depth"]
    dep_h, dep_w = calib["dep_h"], calib["dep_w"]
    depth_scale = calib["depth_scale"]

    rays = build_depth_pixel_grid(K_depth, dep_h, dep_w)

    with h5py.File(rs_h5_path, "r") as rs_h5:
        depth_ds = rs_h5["depth"]
        N = depth_ds.shape[0]

        has_rgb_src = "rgb" in rs_h5
        rgb_ds = rs_h5["rgb"] if has_rgb_src else None

        print(f"[{seq_dir.name}] Projecting {N} depth frames ({dep_w}x{dep_h}) "
              f"→ RGB frame ({rgb_w}x{rgb_h})")

        depth_video = cv2.VideoWriter(
            str(out_depth_vid), cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (rgb_w, rgb_h), isColor=True,
        )

        out_dh5 = h5py.File(out_depth_h5, "w")
        depth_out_ds = out_dh5.create_dataset(
            "depth", shape=(N, rgb_h, rgb_w), dtype=np.float32,
            chunks=(1, rgb_h, rgb_w), compression="gzip", compression_opts=4,
        )
        out_dh5.attrs["description"] = "Depth projected into RGB camera frame (metres)"
        out_dh5.attrs["source"] = str(rs_h5_path)

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

                    # Parallel projection
                    depth_futures = [
                        pool.submit(
                            project_depth_to_rgb_frame,
                            d, rays, R, t_vec, K_rgb, rgb_h, rgb_w, depth_scale,
                            bleed_correction,
                        )
                        for d in batch_depth_u16
                    ]

                    # Sequential writes
                    for j, i in enumerate(batch_range):
                        proj_depth = depth_futures[j].result()
                        depth_out_ds[i] = proj_depth

                        colour = colorise_depth(proj_depth)
                        if has_rgb_src:
                            # Overlay colourised depth on the actual RGB frame
                            rgb_bgr = cv2.cvtColor(batch_rgb_u8[j], cv2.COLOR_RGB2BGR)
                            mask = proj_depth > 0
                            frame = rgb_bgr.copy()
                            frame[mask] = cv2.addWeighted(rgb_bgr, 0.4, colour, 0.6, 0)[mask]
                        else:
                            frame = colour
                        depth_video.write(frame)

                    pbar.update(len(batch_range))
        finally:
            depth_video.release()
            out_dh5.close()

    print(f"  -> {out_depth_h5}")
    print(f"  -> {out_depth_vid}")


def find_recordings(root: Path) -> list[Path]:
    """Find all recording directories that contain hdf5/realsense.h5."""
    recordings = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "hdf5" / "realsense.h5").exists():
            recordings.append(p)
    return recordings


def main():
    global FPS
    parser = argparse.ArgumentParser(
        description="Project RealSense depth into the RGB camera frame",
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
        "--overwrite", action="store_true",
        help="Overwrite existing depth_in_rgb_frame.h5 files",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for frame projection",
    )
    parser.add_argument(
        "--no-bleed-correction", action="store_true",
        help="Disable parallax bleed correction (background depth on foreground pixels)",
    )
    args = parser.parse_args()

    FPS = args.fps
    calib = load_calibration(Path(args.calib_dir))

    if args.data_dir:
        dirs = [Path(d) for d in args.data_dir]
    else:
        dirs = find_recordings(Path(args.data_root))
        if not dirs:
            print(f"No recordings found under {args.data_root}")
            return

    print(f"Processing {len(dirs)} recording(s)")

    for d in dirs:
        process_recording(
            d, calib,
            overwrite=args.overwrite,
            workers=args.workers,
            bleed_correction=not args.no_bleed_correction,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
