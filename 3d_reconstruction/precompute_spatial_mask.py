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
    hdf5/spatial_mask.h5  — dataset "mask" (N, H, W) uint8 {0, 1}

Usage:
    python3 precompute_spatial_mask.py --data_root data/real
    python3 precompute_spatial_mask.py --data_dir data/real/box --cube_side 0.4
    python3 precompute_spatial_mask.py --data_root data/real --overwrite
"""

import argparse
from pathlib import Path
from typing import List

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from reconstruction_config import (
    CALIB_DIR as _CALIB_DIR,
    DATA_ROOT as _DATA_ROOT,
    SPATIAL_CUBE_SIDE,
    SPATIAL_CUBE_CENTER_Z,
)

CALIB_DIR = Path(__file__).resolve().parent / _CALIB_DIR

# Default target point in robot base frame (where the object is placed)
# X/Y match franka_pipeline/config_defaults.py TARGET_X/Y; Z comes from reconstruction_config
DEFAULT_TARGET_X = 0.3
DEFAULT_TARGET_Y = 0.0
DEFAULT_TARGET_Z = SPATIAL_CUBE_CENTER_Z


def load_calibration(calib_dir: Path) -> dict:
    """Load calibration needed for event-pixel → base-frame unprojection."""
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"]       # (4,4)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"] # (4,4)
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K_event = ev["camera_matrix"]      # (3,3)
    dist_event = ev["dist_coeffs"].ravel()  # (5,)
    ev_size = ev["image_size"]         # [W, H]

    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee
    # inv(T_event_from_ee) brings event-cam → ee → base when composed with ee_T
    T_ee_from_event = np.linalg.inv(T_event_from_ee)

    return {
        "T_ee_from_event": T_ee_from_event,
        "K_event": K_event,
        "dist_event": dist_event,
        "ev_w": int(ev_size[0]),
        "ev_h": int(ev_size[1]),
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


def process_sequence(
    seq_dir: Path,
    calib: dict,
    cube_side: float,
    target_point: np.ndarray,
    overwrite: bool = False,
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

    # Build rays for event camera
    rays = build_event_rays(calib["K_event"], H, W)
    half_side = cube_side / 2.0

    # Process frames
    with h5py.File(depth_h5_path, "r") as df, \
         h5py.File(out_path, "w") as of:
        mask_ds = of.create_dataset(
            "mask", shape=(n_frames, H, W), dtype=np.uint8,
            chunks=(1, H, W), compression="gzip", compression_opts=4,
        )
        of.attrs["cube_side_m"] = cube_side
        of.attrs["target_point"] = target_point
        of.attrs["description"] = (
            f"Spatial mask: 1 inside {cube_side:.2f}m cube around "
            f"target ({target_point[0]:.3f}, {target_point[1]:.3f}, {target_point[2]:.3f}), 0 outside"
        )

        for i in tqdm(range(n_frames), desc=f"  {seq_dir.name}", leave=False):
            depth = df["depth"][i].astype(np.float32)
            m = compute_spatial_mask(
                depth, ee_T_all[i], calib["T_ee_from_event"], rays, half_side,
                target_point,
            )
            mask_ds[i] = m

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
    parser.add_argument("--data_root", type=str, default=str(_DATA_ROOT),
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
        result = process_sequence(d, calib, args.cube_side, target_point, overwrite=args.overwrite)
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
