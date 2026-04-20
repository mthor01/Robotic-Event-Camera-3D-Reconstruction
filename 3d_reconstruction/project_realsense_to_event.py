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

from reconstruction_config import (
    CALIB_DIR, DATA_ROOT, FPS as DEFAULT_FPS,
)

# ─── module-level FPS (overridden by --fps CLI arg) ───────────────
FPS = DEFAULT_FPS


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

    # Fill small gaps caused by resolution mismatch between depth and event
    # cameras.  A 3×3 dilation propagates valid depths into 1-pixel holes,
    # then we keep only newly-filled pixels where a hole existed.
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(depth_out, kernel, iterations=1)
    depth_out[depth_out == 0] = dilated[depth_out == 0]

    return depth_out


def colorise_depth(depth: np.ndarray, max_m: float = 2.0) -> np.ndarray:
    """Convert depth (float32, metres) to a colour image for visualisation."""
    d = np.clip(depth / max_m, 0, 1)
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


def process_recording(seq_dir: Path, calib: dict, project_rgb: bool = True, workers: int = 4) -> None:
    """Project all depth (and optionally RGB) frames for one recording directory."""
    rs_h5_path = seq_dir / "hdf5" / "realsense.h5"
    if not rs_h5_path.exists():
        print(f"[skip] No realsense.h5 in {seq_dir}")
        return

    out_depth_h5 = seq_dir / "hdf5" / "depth_in_event_frame.h5"
    out_depth_vid = seq_dir / "videos" / "depth_in_event_frame.mp4"
    out_rgb_h5 = seq_dir / "hdf5" / "rgb_in_event_frame.h5"
    out_rgb_vid = seq_dir / "videos" / "rgb_in_event_frame.mp4"
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
              f"{' + RGB' if has_rgb_src else ''}")

        # Video writers
        depth_video = cv2.VideoWriter(
            str(out_depth_vid), cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (ev_w, ev_h), isColor=True,
        )
        rgb_video = None
        if has_rgb_src:
            rgb_video = cv2.VideoWriter(
                str(out_rgb_vid), cv2.VideoWriter_fourcc(*"mp4v"),
                FPS, (ev_w, ev_h), isColor=True,
            )

        # Event frames for overlay
        ev_h5_path = seq_dir / "hdf5" / "events_cam0.h5"
        has_events = ev_h5_path.exists()
        ev_h5 = h5py.File(ev_h5_path, "r") if has_events else None

        # Output HDF5 files
        out_dh5 = h5py.File(out_depth_h5, "w")
        depth_out_ds = out_dh5.create_dataset(
            "depth", shape=(N, ev_h, ev_w), dtype=np.float32,
            chunks=(1, ev_h, ev_w), compression="gzip", compression_opts=4,
        )
        out_dh5.attrs["description"] = "Depth projected into event camera frame (metres)"
        out_dh5.attrs["source"] = str(rs_h5_path)

        out_rh5 = None
        rgb_out_ds = None
        if has_rgb_src:
            out_rh5 = h5py.File(out_rgb_h5, "w")
            rgb_out_ds = out_rh5.create_dataset(
                "rgb", shape=(N, ev_h, ev_w, 3), dtype=np.uint8,
                chunks=(1, ev_h, ev_w, 3), compression="gzip", compression_opts=4,
            )
            out_rh5.attrs["description"] = "RGB projected into event camera frame"
            out_rh5.attrs["source"] = str(rs_h5_path)

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
                        depth_out_ds[i] = proj_depth

                        colour = colorise_depth(proj_depth)
                        ev_gray = batch_ev[j]
                        if ev_gray is not None:
                            ev_bgr = cv2.cvtColor(ev_gray, cv2.COLOR_GRAY2BGR)
                            mask = proj_depth > 0
                            frame = ev_bgr.copy()
                            frame[mask] = cv2.addWeighted(ev_bgr, 0.4, colour, 0.6, 0)[mask]
                        else:
                            frame = colour
                        depth_video.write(frame)

                        if has_rgb_src:
                            proj_rgb = rgb_futures[j].result()
                            rgb_out_ds[i] = proj_rgb

                            rgb_bgr = cv2.cvtColor(proj_rgb, cv2.COLOR_RGB2BGR)
                            if ev_gray is not None:
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
            depth_video.release()
            if rgb_video is not None:
                rgb_video.release()
            out_dh5.close()
            if out_rh5 is not None:
                out_rh5.close()

    print(f"  -> {out_depth_h5}")
    print(f"  -> {out_depth_vid}")
    if has_rgb_src:
        print(f"  -> {out_rgb_h5}")
        print(f"  -> {out_rgb_vid}")


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
        "--workers", type=int, default=4,
        help="Number of parallel workers for frame projection",
    )
    args = parser.parse_args()

    calib = load_calibration(Path(args.calib_dir))

    if args.data_dir:
        dirs = [Path(d) for d in args.data_dir]
    else:
        dirs = find_recordings(Path(args.data_root))
        if not dirs:
            print(f"No recordings found under {args.data_root}")
            return

    print(f"Processing {len(dirs)} recording(s)")
    print(f"RGB projection: {'off' if args.no_rgb else 'on'}")
    FPS = args.fps

    for d in dirs:
        process_recording(d, calib, project_rgb=not args.no_rgb, workers=args.workers)

    print("\nDone.")


if __name__ == "__main__":
    main()
