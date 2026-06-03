#!/usr/bin/env python3
"""
emvs_from_unet_data.py

Minimal EMVS-style depth reconstruction for the same data layout used by
train_unet_2.py.

Expected sequence layout:
    seq_dir/
        events/voxels_cam0.h5              dataset: "voxels"  (N,C,H,W)
        hdf5/poses.h5                      dataset: "ee_T"    (N,4,4)
    camera_data/
        event_intrinsics.npz               key: "camera_matrix"
        T_event_from_rgb.npz               key: "T"
        T_rgb_from_ee.npz                  key: "T"

This is not the original RPG EMVS implementation. It is an EMVS-like
depth-space-image plane sweep that uses your pre-binned event voxel frames:

    event voxels + known poses -> sweep depths -> DSI volume -> depth map -> point cloud

For each source voxel frame near a target frame, active event pixels are
backprojected at candidate depths, transformed into the target camera, and
accumulated in a target-view DSI. The winning depth per pixel is the depth
with the largest event support.

Example:
    python3 emvs_from_unet_data.py \
        --seq_dir data/lego/lego_1 \
        --camera_data camera_data \
        --target 100 \
        --window 15 \
        --num_depths 128 \
        --dmin 0.2 --dmax 4.0 \
        --out_dir emvs_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_event_K_scaled(
    camera_data: Path,
    h_crop: int,
    w_crop: int,
    h_full: int = 720,
    w_full: int = 1280,
) -> np.ndarray:
    """Load event-camera intrinsics and scale them to the voxel resolution."""
    K = np.load(camera_data / "event_intrinsics.npz")["camera_matrix"].astype(np.float64).copy()
    K[0, :] *= w_crop / w_full
    K[1, :] *= h_crop / h_full
    return K


def load_T_event_from_ee(camera_data: Path) -> np.ndarray:
    """
    Same convention as train_unet_2.py:
        T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee
    """
    T_event_from_rgb = np.load(camera_data / "T_event_from_rgb.npz")["T"]
    T_rgb_from_ee = np.load(camera_data / "T_rgb_from_ee.npz")["T"]
    return (T_event_from_rgb @ T_rgb_from_ee).astype(np.float64)


def load_world_from_event_poses(seq_dir: Path, camera_data: Path, n_frames: int) -> np.ndarray:
    """
    Convert end-effector poses to event-camera poses.

    poses.h5 stores ee_T, interpreted exactly like your UNet script:
        T_world_from_event = ee_T @ inv(T_event_from_ee)
    """
    T_event_from_ee = load_T_event_from_ee(camera_data)
    T_ee_from_event = np.linalg.inv(T_event_from_ee)

    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_T = f["ee_T"][:n_frames].astype(np.float64)

    return ee_T @ T_ee_from_event[None]


def voxel_to_event_image(voxel: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert a voxel grid (C,H,W) to one 2D event activity image (H,W).

    We use summed absolute event activity over bins. This discards exact
    timestamps, so this is an approximation of EMVS, not full asynchronous EMVS.
    """
    img = np.abs(voxel.astype(np.float32)).sum(axis=0)
    if threshold > 0.0:
        img = np.where(img >= threshold, img, 0.0)
    return img


def active_pixels(
    img: np.ndarray,
    max_events: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return active event pixel coordinates and weights:
        xs, ys, weights
    """
    ys, xs = np.nonzero(img > 0)
    weights = img[ys, xs].astype(np.float32)

    if max_events is not None and len(xs) > max_events:
        # Keep strongest event pixels for speed.
        keep = np.argpartition(weights, -max_events)[-max_events:]
        xs, ys, weights = xs[keep], ys[keep], weights[keep]

    return xs.astype(np.float64), ys.astype(np.float64), weights


def accumulate_source_into_dsi(
    dsi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    T_tgt_from_src: np.ndarray,
) -> None:
    """
    Add one source event image into target-view DSI.

    dsi shape: (D,H,W), where D=len(depths)
    xs, ys are active source-frame event pixels.
    T_tgt_from_src maps source camera coordinates into target camera coordinates.
    """
    D, H, W = dsi.shape
    if len(xs) == 0:
        return

    K_inv = np.linalg.inv(K)

    pix = np.stack([xs, ys, np.ones_like(xs)], axis=0)  # (3,M)
    rays = K_inv @ pix                                  # (3,M)

    R = T_tgt_from_src[:3, :3]
    t = T_tgt_from_src[:3, 3:4]

    for k, depth in enumerate(depths):
        P_src = rays * depth                            # (3,M)
        P_tgt = R @ P_src + t                           # (3,M)

        z = P_tgt[2]
        valid_z = z > 1e-6
        if not np.any(valid_z):
            continue

        x_proj = K[0, 0] * (P_tgt[0] / z) + K[0, 2]
        y_proj = K[1, 1] * (P_tgt[1] / z) + K[1, 2]

        xi = np.rint(x_proj).astype(np.int32)
        yi = np.rint(y_proj).astype(np.int32)

        valid = valid_z & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if not np.any(valid):
            continue

        np.add.at(dsi[k], (yi[valid], xi[valid]), weights[valid])


def estimate_depth_from_dsi(
    dsi: np.ndarray,
    depths: np.ndarray,
    min_support: float,
    smooth_support: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Winner-take-all depth from DSI.

    Returns:
        depth_map: (H,W), zero where invalid
        support:   (H,W), max accumulated event support
    """
    # Optional 3-point smoothing over the depth axis. This reduces isolated
    # depth-bin spikes without needing scipy.
    if smooth_support and dsi.shape[0] >= 3:
        dsi_score = dsi.copy()
        dsi_score[1:-1] = 0.25 * dsi[:-2] + 0.5 * dsi[1:-1] + 0.25 * dsi[2:]
    else:
        dsi_score = dsi

    best_idx = np.argmax(dsi_score, axis=0)
    support = np.take_along_axis(dsi_score, best_idx[None], axis=0)[0]

    depth = depths[best_idx].astype(np.float32)
    depth[support < min_support] = 0.0
    return depth, support.astype(np.float32)


def depth_to_points(depth: np.ndarray, K: np.ndarray, T_world_from_cam: np.ndarray) -> np.ndarray:
    """Backproject a target-frame depth map to a world-frame point cloud."""
    H, W = depth.shape
    ys, xs = np.nonzero(depth > 0)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs].astype(np.float64)
    pix = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones_like(z)], axis=0)
    rays = np.linalg.inv(K) @ pix
    P_cam = rays * z[None]

    P_h = np.concatenate([P_cam, np.ones((1, P_cam.shape[1]))], axis=0)
    P_w = T_world_from_cam @ P_h
    return P_w[:3].T.astype(np.float32)


def write_ply(path: Path, points: np.ndarray) -> None:
    """Write an ASCII PLY point cloud."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def run_emvs(args: argparse.Namespace) -> None:
    seq_dir = args.seq_dir
    voxels_path = seq_dir / "events" / "voxels_cam0.h5"
    poses_path = seq_dir / "hdf5" / "poses.h5"

    if not voxels_path.exists():
        raise FileNotFoundError(voxels_path)
    if not poses_path.exists():
        raise FileNotFoundError(poses_path)

    with h5py.File(voxels_path, "r") as f:
        voxels = f["voxels"]
        n_frames, _, H, W = voxels.shape

        target = args.target
        if target < 0:
            target = n_frames // 2
        if not (0 <= target < n_frames):
            raise ValueError(f"--target must be in [0, {n_frames - 1}]")

        K = load_event_K_scaled(args.camera_data, H, W)
        T_world_from_event = load_world_from_event_poses(seq_dir, args.camera_data, n_frames)

        depths = np.linspace(args.dmin, args.dmax, args.num_depths, dtype=np.float64)
        if args.inverse_depth:
            inv = np.linspace(1.0 / args.dmax, 1.0 / args.dmin, args.num_depths)
            depths = (1.0 / inv)[::-1].astype(np.float64)

        dsi = np.zeros((len(depths), H, W), dtype=np.float32)

        src_start = max(0, target - args.window)
        src_end = min(n_frames, target + args.window + 1)

        T_tgt_from_world = np.linalg.inv(T_world_from_event[target])

        used_sources = []
        for src in range(src_start, src_end, args.frame_step):
            if src == target:
                continue

            voxel = voxels[src]
            if voxel.dtype == np.float16:
                voxel = voxel.astype(np.float32)

            img = voxel_to_event_image(voxel, threshold=args.event_threshold)
            xs, ys, weights = active_pixels(img, max_events=args.max_events_per_frame)

            if len(xs) == 0:
                continue

            T_tgt_from_src = T_tgt_from_world @ T_world_from_event[src]
            accumulate_source_into_dsi(
                dsi=dsi,
                xs=xs,
                ys=ys,
                weights=weights,
                depths=depths,
                K=K,
                T_tgt_from_src=T_tgt_from_src,
            )
            used_sources.append(src)

    depth, support = estimate_depth_from_dsi(
        dsi,
        depths,
        min_support=args.min_support,
        smooth_support=args.smooth_depth_axis,
    )

    points = depth_to_points(depth, K, T_world_from_event[target])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"emvs_t{target:06d}"

    np.savez_compressed(
        args.out_dir / f"{stem}.npz",
        depth=depth,
        support=support,
        dsi=dsi if args.save_dsi else np.empty((0,), dtype=np.float32),
        depths=depths.astype(np.float32),
        K=K.astype(np.float32),
        T_world_from_event=T_world_from_event[target].astype(np.float32),
        used_sources=np.array(used_sources, dtype=np.int32),
    )
    write_ply(args.out_dir / f"{stem}.ply", points)

    valid = int((depth > 0).sum())
    print(f"Target frame: {target}")
    print(f"Used source frames: {len(used_sources)} from [{src_start}, {src_end})")
    print(f"Depth bins: {len(depths)} from {depths.min():.3f} m to {depths.max():.3f} m")
    print(f"Valid depth pixels: {valid} / {depth.size}")
    print(f"Point cloud points: {len(points)}")
    print(f"Saved: {args.out_dir / (stem + '.npz')}")
    print(f"Saved: {args.out_dir / (stem + '.ply')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=Path, required=True)
    parser.add_argument("--camera_data", type=Path, default=Path("camera_data"))
    parser.add_argument("--target", type=int, default=-1, help="Target frame index. Default: middle frame.")
    parser.add_argument("--window", type=int, default=15, help="Use source frames target±window.")
    parser.add_argument("--frame_step", type=int, default=1, help="Use every k-th source frame in the window.")

    parser.add_argument("--dmin", type=float, default=0.2)
    parser.add_argument("--dmax", type=float, default=4.0)
    parser.add_argument("--num_depths", type=int, default=128)
    parser.add_argument("--inverse_depth", action="store_true", help="Sample uniformly in inverse depth.")

    parser.add_argument("--event_threshold", type=float, default=0.0)
    parser.add_argument("--max_events_per_frame", type=int, default=30000)
    parser.add_argument("--min_support", type=float, default=3.0)
    parser.add_argument("--smooth_depth_axis", action="store_true")

    parser.add_argument("--save_dsi", action="store_true", help="Store full DSI in the .npz file. Can be large.")
    parser.add_argument("--out_dir", type=Path, default=Path("emvs_out"))

    args = parser.parse_args()
    run_emvs(args)


if __name__ == "__main__":
    main()
