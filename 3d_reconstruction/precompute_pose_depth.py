#!/usr/bin/env python3
"""
Augment precomputed voxels with a pose depth-to-table-plane channel.

Reads existing precomputed voxels from events/voxels_cam0/ (shape num_bins×H×W)
and robot end-effector poses from hdf5/poses.h5.  For each frame, a depth-to-
z=0-plane image is computed by ray-casting every pixel from the event camera
through the calibrated extrinsics into the robot base frame.  The resulting
metric depth is log-encoded (POSE_D_MAX / POSE_ALPHA) and appended as a final
channel, producing voxels of shape (num_bins+1)×H×W saved to
events/voxels_pose_cam0/.

Usage:
    python precompute_pose_depth.py --data_root data/real
    python precompute_pose_depth.py --data_dir data/real/bottle data/real/cube_medium
    python precompute_pose_depth.py --data_root data/real --calib_dir camera_data
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from reconstruction_config import POSE_D_MAX, POSE_ALPHA, CALIB_DIR as _CALIB_DIR

CALIB_DIR = Path(__file__).resolve().parent / _CALIB_DIR


def compute_pose_depth_to_plane(
    ee_T: np.ndarray,
    T_base_from_event_static: np.ndarray,
    rays_cam: np.ndarray,
    H: int,
    W: int,
    log_depth: bool = False,
) -> np.ndarray:
    """
    Compute normalised depth from the event camera to the z=0 (table) plane.

    For every pixel a ray is cast from the camera through the pixel into the
    robot base frame and intersected with the z=0 plane.  The resulting metric
    depth is normalised with POSE_D_MAX / POSE_ALPHA.

    Args:
        ee_T:                    (4, 4) T_base_from_ee for the current frame.
        T_base_from_event_static:(4, 4) inv(T_event_from_ee) — precomputed.
        rays_cam:                (H*W, 3) ray directions in event-camera frame.
        H, W:                    Event-camera image dimensions.
        log_depth:               If True, use log encoding; otherwise linear.

    Returns:
        (H, W) float32 array in [0, 1].
    """
    T_base_from_event = ee_T @ T_base_from_event_static
    R = T_base_from_event[:3, :3]
    cam_origin = T_base_from_event[:3, 3]

    rays_base = (R @ rays_cam.T).T  # (H*W, 3)

    ray_z = rays_base[:, 2]
    valid = np.abs(ray_z) > 1e-8
    t = np.full(H * W, POSE_D_MAX, dtype=np.float64)
    t[valid] = -cam_origin[2] / ray_z[valid]

    t[t <= 0] = POSE_D_MAX
    depth = np.clip(t, 1e-4, POSE_D_MAX).reshape(H, W)

    if log_depth:
        depth = (1.0 + (1.0 / POSE_ALPHA) * np.log(depth / POSE_D_MAX)).clip(0.0, 1.0)
    else:
        depth = (depth / POSE_D_MAX).clip(0.0, 1.0)
    return depth.astype(np.float32)


def process_sequence(
    sequence_dir: Path,
    calib_dir: Path,
    overwrite: bool = False,
    workers: int = 4,
    log_depth: bool = False,
) -> dict:
    """
    Augment precomputed voxels in a single sequence with pose depth.

    Reads:
        events/voxels_cam0/voxel_NNNNNN.npy   (num_bins, H, W)
        hdf5/poses.h5                          ee_T (N, 4, 4)
    Writes:
        events/voxels_pose_cam0/voxel_NNNNNN.npy  (num_bins+1, H, W)
    """
    sequence_dir = Path(sequence_dir)
    result = {
        "name": sequence_dir.name,
        "success": False,
        "n_frames": 0,
        "error": None,
    }

    # ---- Locate source voxels ----
    src_voxels_dir = None
    for _vd in [
        sequence_dir / "events" / "voxels_cam0",
        sequence_dir / "events" / "voxels",
    ]:
        if _vd.exists() and list(_vd.glob("voxel_*.npy")):
            src_voxels_dir = _vd
            break

    if src_voxels_dir is None:
        result["error"] = "No precomputed voxels found (run precompute_voxels.py first)"
        return result

    voxel_files = sorted(src_voxels_dir.glob("voxel_*.npy"))
    n_frames = len(voxel_files)

    # ---- Locate poses ----
    poses_path = sequence_dir / "hdf5" / "poses.h5"
    if not poses_path.exists():
        result["error"] = f"poses.h5 not found: {poses_path}"
        return result

    # ---- Output directory ----
    dst_voxels_dir = sequence_dir / "events" / "voxels_pose_cam0"

    # Fast-path: already done
    if not overwrite and dst_voxels_dir.exists():
        existing = len(list(dst_voxels_dir.glob("voxel_*.npy")))
        if existing >= n_frames:
            result["success"] = True
            result["n_frames"] = n_frames
            result["error"] = "Already processed (use --overwrite to reprocess)"
            return result

    # ---- Load poses ----
    with h5py.File(poses_path, "r") as pf:
        ee_T = pf["ee_T"][:]  # (N, 4, 4)

    n_poses = len(ee_T)
    if n_poses < n_frames:
        print(f"  [{sequence_dir.name}] WARNING: {n_poses} poses < {n_frames} voxels; truncating")
        n_frames = n_poses

    # ---- Load calibration and precompute static parts ----
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"]
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"]
    K_event = np.load(calib_dir / "event_intrinsics.npz")["camera_matrix"]

    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee
    T_base_from_event_static = np.linalg.inv(T_event_from_ee)
    K_event_inv = np.linalg.inv(K_event)

    # Get spatial dims from first voxel
    sample_voxel = np.load(voxel_files[0])
    H, W = sample_voxel.shape[1], sample_voxel.shape[2]

    # Precompute pixel rays  (H*W, 3)
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    rays_cam = (K_event_inv @ pixels.reshape(-1, 3).T).T.copy()

    # ---- Process frames ----
    dst_voxels_dir.mkdir(parents=True, exist_ok=True)

    def _process_frame(i: int) -> None:
        dst_path = dst_voxels_dir / f"voxel_{i:06d}.npy"
        if not overwrite and dst_path.exists():
            return
        voxel = np.load(voxel_files[i])  # (num_bins, H, W)
        pose_depth = compute_pose_depth_to_plane(
            ee_T[i], T_base_from_event_static, rays_cam, H, W,
            log_depth=log_depth,
        )
        augmented = np.concatenate([voxel, pose_depth[None]], axis=0)
        np.save(dst_path, augmented)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in tqdm(
                pool.map(_process_frame, range(n_frames)),
                total=n_frames, desc=f"  {sequence_dir.name}", leave=False,
            ):
                pass

        result["success"] = True
        result["n_frames"] = n_frames
    except Exception as e:
        result["error"] = str(e)

    return result


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """Find sequence directories that have precomputed voxels + poses."""
    sequence_dirs = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        has_voxels = (
            (d / "events" / "voxels_cam0").exists()
            or (d / "events" / "voxels").exists()
        )
        has_poses = (d / "hdf5" / "poses.h5").exists()
        if has_voxels and has_poses:
            sequence_dirs.append(d)
    return sequence_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Augment precomputed voxels with pose depth-to-table-plane channel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", nargs="+", type=str, default=None,
                        help="Sequence directory(ies) to process")
    parser.add_argument("--data_root", type=str, default="data/real",
                        help="Root data directory (will process all subdirs)")
    parser.add_argument("--calib_dir", type=str, default=str(CALIB_DIR),
                        help="Calibration data directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing pose-augmented voxels")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for frame processing")
    parser.add_argument("--log_depth", action="store_true",
                        help="Use log encoding for pose depth (default: linear normalization)")

    args = parser.parse_args()
    calib_dir = Path(args.calib_dir)

    # Validate calibration files
    for f in ["T_rgb_from_ee.npz", "T_event_from_rgb.npz", "event_intrinsics.npz"]:
        if not (calib_dir / f).exists():
            print(f"ERROR: calibration file not found: {calib_dir / f}")
            return

    # Find sequences
    if args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))

    if not sequence_dirs:
        print("No valid sequences found!")
        return

    print(f"Found {len(sequence_dirs)} sequences to process")
    print(f"Calibration: {calib_dir}")
    print(f"Pose encoding: D_MAX={POSE_D_MAX}, ALPHA={POSE_ALPHA}, mode={'log' if args.log_depth else 'linear'}")
    print(f"Overwrite: {args.overwrite}")
    print()

    results = []
    for seq_dir in sequence_dirs:
        result = process_sequence(seq_dir, calib_dir, args.overwrite, workers=args.workers,
                                  log_depth=args.log_depth)
        results.append(result)
        if result["success"]:
            print(f"  ✓ {result['name']}: {result['n_frames']} frames")
        else:
            print(f"  ✗ {result['name']}: {result['error']}")

    # Summary
    print()
    print("=" * 60)
    n_success = sum(1 for r in results if r["success"])
    n_total_frames = sum(r["n_frames"] for r in results if r["success"])
    print(f"Processed: {n_success}/{len(results)} sequences")
    print(f"Total frames: {n_total_frames}")
    print(f"Output: events/voxels_pose_cam0/")

    failures = [r for r in results if not r["success"] and "Already" not in str(r["error"])]
    if failures:
        print(f"\nFailures:")
        for r in failures:
            print(f"  - {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
