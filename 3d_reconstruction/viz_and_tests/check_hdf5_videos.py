"""
Quick HDF5 verification script.

Reads the HDF5 files written by synchronised_recording.py and exports
test MP4 videos so you can visually check the data is correct.

Usage:
    python check_hdf5_videos.py data/real/<object_name>
    python check_hdf5_videos.py data/real/<object_name> --fps 30
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

CALIB_DIR = Path(__file__).parent / "camera_data"


def make_video(frames: np.ndarray, out_path: Path, fps: float, is_color: bool) -> None:
    """Write a numpy array of frames to an MP4 file."""
    N, H, W = frames.shape[:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H), isColor=is_color)
    for i in range(N):
        writer.write(frames[i])
    writer.release()
    print(f"  -> {out_path}  ({N} frames, {W}x{H})")


def check_realsense(hdf5_dir: Path, out_dir: Path, fps: float) -> None:
    h5_path = hdf5_dir / "realsense.h5"
    if not h5_path.exists():
        print(f"[RealSense] realsense.h5 not found at {h5_path}")
        return

    print(f"\n[RealSense] Reading {h5_path}")
    with h5py.File(h5_path, "r") as f:
        depth = f["depth"][:]        # (N, H, W) uint16
        rgb   = f["rgb"][:]          # (N, H, W, 3) uint8
        t_sys = f["t_sys_ns"][:]     # (N,) int64
        t_hw  = f["t_hw_ms"][:]      # (N,) float64

    N = len(depth)
    print(f"  depth shape : {depth.shape}  dtype={depth.dtype}")
    print(f"  rgb shape   : {rgb.shape}  dtype={rgb.dtype}")
    print(f"  frames      : {N}")
    if N > 1:
        duration_s = (t_sys[-1] - t_sys[0]) / 1e9
        actual_fps = (N - 1) / duration_s if duration_s > 0 else 0
        print(f"  duration    : {duration_s:.2f} s  ({actual_fps:.1f} fps measured)")
    hw_range_ms = t_hw[-1] - t_hw[0] if N > 1 else 0
    print(f"  hw tstamp range: {hw_range_ms:.1f} ms")

    # Depth colorised
    depth_color = np.stack([
        cv2.applyColorMap(cv2.convertScaleAbs(depth[i], alpha=0.03), cv2.COLORMAP_TURBO)
        for i in range(N)
    ])
    make_video(depth_color, out_dir / "check_realsense_depth.mp4", fps, is_color=True)

    # RGB
    make_video(rgb, out_dir / "check_realsense_rgb.mp4", fps, is_color=True)


def check_events(hdf5_dir: Path, out_dir: Path, fps: float, num_cameras: int) -> None:
    for cam_idx in range(num_cameras):
        h5_path = hdf5_dir / f"events_cam{cam_idx}.h5"
        if not h5_path.exists():
            print(f"\n[Events cam{cam_idx}] {h5_path} not found, skipping")
            continue

        print(f"\n[Events cam{cam_idx}] Reading {h5_path}")
        with h5py.File(h5_path, "r") as f:
            frames    = f["events/frames"][:]        # (N, H, W) uint8
            t_start   = f["events/t_ev_start_us"][:] # (N,) int64
            t_end     = f["events/t_ev_end_us"][:]   # (N,) int64

        N = len(frames)
        print(f"  frames shape: {frames.shape}  dtype={frames.dtype}")
        print(f"  frames      : {N}")
        if N > 0:
            span_us = int(t_end[-1]) - int(t_start[0])
            print(f"  event time span: {span_us / 1e6:.2f} s")

        # Grayscale → 3-channel for VideoWriter
        frames_bgr = np.stack([
            cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR) for i in range(N)
        ])
        make_video(frames_bgr, out_dir / f"check_events_cam{cam_idx}.mp4", fps, is_color=True)


def check_poses(hdf5_dir: Path) -> None:
    h5_path = hdf5_dir / "poses.h5"
    if not h5_path.exists():
        print(f"\n[Poses] poses.h5 not found at {h5_path}")
        return

    print(f"\n[Poses] Reading {h5_path}")
    with h5py.File(h5_path, "r") as f:
        ee_T      = f["ee_T"][:]              # (N, 4, 4)
        joints    = f["joint_positions"][:]   # (N, 7)
        gripper_q = f["gripper_q"][:]         # (N,)
        offset_ms = f["nearest_offset_ms"][:] # (N,)

    N = len(ee_T)
    print(f"  ee_T shape      : {ee_T.shape}")
    print(f"  joint_positions : {joints.shape}")
    print(f"  gripper_q range : [{gripper_q.min():.4f}, {gripper_q.max():.4f}]")
    print(f"  offset_ms  median={np.median(offset_ms):.1f}  max={np.max(offset_ms):.1f}")
    # Sanity: last row of each transform should be [0,0,0,1]
    last_rows = ee_T[:, 3, :]
    bad = np.any(np.abs(last_rows - [0, 0, 0, 1]) > 1e-3, axis=1).sum()
    if bad:
        print(f"  WARNING: {bad}/{N} transforms have malformed last row")
    else:
        print(f"  Transform sanity check: OK (all last rows are [0,0,0,1])")


def check_metadata(hdf5_dir: Path) -> None:
    h5_path = hdf5_dir / "metadata.h5"
    if not h5_path.exists():
        print(f"\n[Metadata] metadata.h5 not found at {h5_path}")
        return

    print(f"\n[Metadata] Reading {h5_path}")
    with h5py.File(h5_path, "r") as f:
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")


def make_rgb_on_events_video(
    hdf5_dir: Path, out_dir: Path, fps: float, calib_dir: Path,
) -> None:
    """Project RGB colours onto event frames via depth-based 3-D reprojection."""
    rs_path = hdf5_dir / "realsense.h5"
    ev_path = hdf5_dir / "events_cam0.h5"
    if not rs_path.exists() or not ev_path.exists():
        print("[RGB-on-Events] Need both realsense.h5 and events_cam0.h5, skipping")
        return

    # ── load calibration ───────────────────────────────────────────
    T_event_from_depth = np.load(calib_dir / "T_event_from_depth.npz")["T"]
    T_color_from_depth = np.load(calib_dir / "T_color_from_depth.npz")["T"]

    ev_cal = np.load(calib_dir / "event_intrinsics.npz")
    K_event = ev_cal["camera_matrix"]
    dist_event = ev_cal["dist_coeffs"].ravel()
    ev_w, ev_h = int(ev_cal["image_size"][0]), int(ev_cal["image_size"][1])

    dep_cal = np.load(calib_dir / "rs_depth_intrinsics.npz")
    K_depth = dep_cal["camera_matrix"]
    dep_w, dep_h = int(dep_cal["image_size"][0]), int(dep_cal["image_size"][1])

    rgb_cal = np.load(calib_dir / "rs_rgb_intrinsics.npz")
    K_rgb = rgb_cal["camera_matrix"]
    dist_rgb = rgb_cal["dist_coeffs"].ravel()

    depth_scale = float(np.load(calib_dir / "depth_scale.npz")["scale"])

    # Pre-compute depth pixel ray directions
    fx_d, fy_d = K_depth[0, 0], K_depth[1, 1]
    cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]
    uu, vv = np.meshgrid(np.arange(dep_w, dtype=np.float64),
                         np.arange(dep_h, dtype=np.float64))
    rays = np.stack([(uu - cx_d) / fx_d, (vv - cy_d) / fy_d,
                     np.ones_like(uu)], axis=-1).reshape(-1, 3)

    R_ed = T_event_from_depth[:3, :3]
    t_ed = T_event_from_depth[:3, 3:4]
    R_cd = T_color_from_depth[:3, :3]
    t_cd = T_color_from_depth[:3, 3:4]

    rvec_id = cv2.Rodrigues(np.eye(3, dtype=np.float64))[0]
    tvec_zero = np.zeros((3, 1), dtype=np.float64)

    print(f"\n[RGB-on-Events] Projecting RGB onto event frames ({ev_w}x{ev_h})")

    with h5py.File(rs_path, "r") as rs_h5, h5py.File(ev_path, "r") as ev_h5:
        depth_ds = rs_h5["depth"]
        rgb_ds = rs_h5["rgb"]
        ev_frames_ds = ev_h5["events/frames"]

        N = min(depth_ds.shape[0], ev_frames_ds.shape[0])

        out_path = out_dir / "rgb_on_events.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (ev_w, ev_h), isColor=True,
        )

        for i in tqdm(range(N), desc="  rgb-on-events", unit="frame"):
            depth_u16 = depth_ds[i]                   # (dep_h, dep_w) uint16
            rgb_img = rgb_ds[i]                        # (dep_h, dep_w, 3) uint8 BGR
            ev_gray = ev_frames_ds[i]                  # (ev_h, ev_w) uint8

            depth_m = depth_u16.astype(np.float64).ravel() * depth_scale
            valid = depth_m > 0.0
            if not np.any(valid):
                writer.write(cv2.cvtColor(ev_gray, cv2.COLOR_GRAY2BGR))
                continue

            # 1  Unproject depth pixels to 3-D (depth frame)
            pts_depth = rays[valid] * depth_m[valid, None]       # (M, 3)

            # 2  Transform to colour frame → project → sample RGB
            pts_color = (R_cd @ pts_depth.T + t_cd).T            # (M, 3)
            uv_rgb, _ = cv2.projectPoints(
                pts_color, rvec_id, tvec_zero, K_rgb, dist_rgb,
            )
            uv_rgb = uv_rgb.reshape(-1, 2)
            cu = np.round(uv_rgb[:, 0]).astype(np.int32)
            cv_ = np.round(uv_rgb[:, 1]).astype(np.int32)
            rgb_h, rgb_w = rgb_img.shape[:2]
            rgb_ok = (cu >= 0) & (cu < rgb_w) & (cv_ >= 0) & (cv_ < rgb_h)

            # 3  Transform to event frame → project
            pts_event = (R_ed @ pts_depth.T + t_ed).T            # (M, 3)
            in_front = pts_event[:, 2] > 0.0
            uv_ev, _ = cv2.projectPoints(
                pts_event, rvec_id, tvec_zero, K_event, dist_event,
            )
            uv_ev = uv_ev.reshape(-1, 2)
            eu = np.round(uv_ev[:, 0]).astype(np.int32)
            ev_ = np.round(uv_ev[:, 1]).astype(np.int32)
            ev_ok = (eu >= 0) & (eu < ev_w) & (ev_ >= 0) & (ev_ < ev_h) & in_front

            # 4  Scatter RGB colours into event-sized image
            keep = rgb_ok & ev_ok
            rgb_proj = np.zeros((ev_h, ev_w, 3), dtype=np.uint8)
            rgb_proj[ev_[keep], eu[keep]] = rgb_img[cv_[keep], cu[keep]]

            # 5  Blend: event frame + projected RGB
            ev_bgr = cv2.cvtColor(ev_gray, cv2.COLOR_GRAY2BGR)
            mask = rgb_proj.any(axis=-1)
            frame = ev_bgr.copy()
            frame[mask] = cv2.addWeighted(ev_bgr, 0.35, rgb_proj, 0.65, 0)[mask]
            writer.write(rgb_proj)

        writer.release()
    print(f"  -> {out_path}  ({N} frames, {ev_w}x{ev_h})")


def make_rgb_on_events_naive_video(
    hdf5_dir: Path, out_dir: Path, fps: float,
) -> None:
    """Overlay RGB on event frames by simple resize (no calibration)."""
    rs_path = hdf5_dir / "realsense.h5"
    ev_path = hdf5_dir / "events_cam0.h5"
    if not rs_path.exists() or not ev_path.exists():
        print("[RGB-on-Events-naive] Need both realsense.h5 and events_cam0.h5, skipping")
        return

    print("\n[RGB-on-Events-naive] Resizing RGB directly onto event frames")

    with h5py.File(rs_path, "r") as rs_h5, h5py.File(ev_path, "r") as ev_h5:
        rgb_ds = rs_h5["rgb"]
        ev_frames_ds = ev_h5["events/frames"]

        N = min(rgb_ds.shape[0], ev_frames_ds.shape[0])
        ev_h, ev_w = ev_frames_ds.shape[1], ev_frames_ds.shape[2]

        out_path = out_dir / "rgb_on_events_naive.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (ev_w, ev_h), isColor=True,
        )

        for i in tqdm(range(N), desc="  rgb-on-events-naive", unit="frame"):
            rgb_img = rgb_ds[i]                        # (rgb_h, rgb_w, 3) uint8 BGR
            ev_gray = ev_frames_ds[i]                  # (ev_h, ev_w) uint8

            rgb_resized = cv2.resize(rgb_img, (ev_w, ev_h), interpolation=cv2.INTER_LINEAR)
            ev_bgr = cv2.cvtColor(ev_gray, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(ev_bgr, 0.35, rgb_resized, 0.65, 0)
            writer.write(frame)

        writer.release()
    print(f"  -> {out_path}  ({N} frames, {ev_w}x{ev_h})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check HDF5 recording output by exporting test videos"
    )
    parser.add_argument(
        "object_dir",
        type=Path,
        help="Path to the object recording directory (e.g. data/real/mug)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate for output videos (default: 30)",
    )
    parser.add_argument(
        "--event-cameras",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Number of event cameras to check (default: 2)",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=CALIB_DIR,
        help="Path to camera_data/ calibration directory",
    )
    args = parser.parse_args()

    object_dir: Path = args.object_dir
    if not object_dir.exists():
        print(f"ERROR: directory not found: {object_dir}")
        sys.exit(1)

    hdf5_dir = object_dir / "hdf5"
    if not hdf5_dir.exists():
        print(f"ERROR: hdf5 subdirectory not found: {hdf5_dir}")
        sys.exit(1)

    out_dir = object_dir / "check_videos"
    out_dir.mkdir(exist_ok=True)
    print(f"Output videos will be written to: {out_dir}")

    check_metadata(hdf5_dir)
    check_realsense(hdf5_dir, out_dir, args.fps)
    check_events(hdf5_dir, out_dir, args.fps, args.event_cameras)
    check_poses(hdf5_dir)
    make_rgb_on_events_video(hdf5_dir, out_dir, args.fps, args.calib_dir)
    make_rgb_on_events_naive_video(hdf5_dir, out_dir, args.fps)

    print(f"\nDone. Check {out_dir} for the output videos.")


if __name__ == "__main__":
    main()
