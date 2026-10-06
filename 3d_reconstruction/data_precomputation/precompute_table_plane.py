#!/usr/bin/env python3
"""
Precompute the table-plane depth prior for each recording.

For each depth frame the script intersects the per-pixel camera rays with the
horizontal plane  z = table_z  (in the robot base frame) and stores the
resulting camera-space depth, normalised to [0, 1] using the training range
[DEPTH_MIN, D_MAX].  The result can be concatenated with the event voxels as
an extra input channel during training (train_mvs.py).

Output (per recording):
    hdf5/table_plane.h5 — dataset "table_plane" (N, H, W) float32 in [0, 1]
                         using intrinsics transformed through resize + centred crop
                         — attrs: table_z_m, depth_min, depth_max, description

Optional debug image:
    debug/table_plane_debug.png  — GT depth | table-plane depth side-by-side
                                   for 8 evenly-spaced frames

Usage:
    python3 data_precomputation/precompute_table_plane.py
    python3 data_precomputation/precompute_table_plane.py --data_dir data/new/train/1
    python3 data_precomputation/precompute_table_plane.py --debug --overwrite
    python3 data_precomputation/precompute_table_plane.py --table_z 0.02
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # 3d_reconstruction/

from config import (
    CALIB_DIR as _CALIB_DIR,
    DATA_ROOT  as _DATA_ROOT,
    DEPTH_MIN,
    D_MAX,
    DEPTH_VIZ_MIN,
    DEPTH_VIZ_MAX,
    PREPROCESS_CROP_HW,
    PREPROCESS_RESIZE_HW,
    SPATIAL_TARGET_Z,
    SPATIAL_CUBE_SIDE,
    TABLE_Z_OFFSET,
)
from helpers import INTRINSICS_TRANSFORM, resolve_path, transform_intrinsics

CALIB_DIR = _HERE.parent / _CALIB_DIR
DATA_ROOT = _HERE.parent / _DATA_ROOT


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _load_calibration() -> dict:
    """Load event-camera intrinsics and the T_ee_from_event transform."""
    ev = np.load(CALIB_DIR / "event_intrinsics.npz")
    K_native = ev["camera_matrix"].astype(np.float64)
    native_W = int(ev["image_size"][0])
    native_H = int(ev["image_size"][1])

    T_er = np.load(CALIB_DIR / "T_event_from_rgb.npz")["T"].astype(np.float64)
    T_re = np.load(CALIB_DIR / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_event_from_ee = T_er @ T_re
    T_ee_from_event = np.linalg.inv(T_event_from_ee)

    return {
        "K_native":       K_native,
        "native_H":       native_H,
        "native_W":       native_W,
        "T_ee_from_event": T_ee_from_event,
    }


# ---------------------------------------------------------------------------
# Per-frame computation
# ---------------------------------------------------------------------------

def _compute_channel(
    T_base_from_event: np.ndarray,   # (4, 4)
    K_native:          np.ndarray,   # (3, 3) at native camera resolution
    native_H:          int,
    native_W:          int,
    out_H:             int,
    out_W:             int,
    table_z:           float,
    depth_min:         float = DEPTH_MIN,
    depth_max:         float = D_MAX,
) -> np.ndarray:
    """
    Return (out_H, out_W) float32 channel: per-pixel depth [m] to the table
    plane  z = table_z  in the robot base frame, normalised to [0, 1].

    Pixels whose rays are parallel to the plane or face away from it are set
    to 0 (they would need depth_max or more, so they're outside the range).
    """
    K = K_native.copy()
    K[0, :] *= out_W / native_W
    K[1, :] *= out_H / native_H
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(out_W, dtype=np.float64)
    v = np.arange(out_H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    dz = np.ones_like(dx)

    R         = T_base_from_event[:3, :3].astype(np.float64)
    rays_flat = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # (HW, 3)
    d_base    = (R @ rays_flat.T).T                                       # (HW, 3)
    t_cam     = T_base_from_event[:3, 3].astype(np.float64)

    # Solve: (t_cam + depth * d_base)[2] = table_z
    denom = d_base[:, 2]
    numer = float(table_z) - float(t_cam[2])
    depth = np.where(np.abs(denom) > 1e-6, numer / denom, depth_max)
    depth = np.where(depth > 0.0, depth, depth_max)

    channel = ((depth - depth_min) / max(float(depth_max - depth_min), 1e-6)).clip(0.0, 1.0)
    return channel.reshape(out_H, out_W).astype(np.float32)


# ---------------------------------------------------------------------------
# Debug image
# ---------------------------------------------------------------------------

def _save_debug_png(
    seq_dir:   Path,
    depths:    list,   # list of (H, W) float32 metres
    tables:    list,   # list of (H, W) float32 normalised [0, 1]
    depth_min: float,
    depth_max: float,
    n_show:    int = 8,
) -> None:
    """Save GT depth | table-plane depth side-by-side for n_show evenly-spaced frames."""
    n       = len(depths)
    indices = np.round(np.linspace(0, n - 1, min(n_show, n))).astype(int)
    rows    = []

    def _to_bgr(arr_m: np.ndarray) -> np.ndarray:
        norm = np.clip(
            (arr_m - DEPTH_VIZ_MIN) / max(DEPTH_VIZ_MAX - DEPTH_VIZ_MIN, 1e-6),
            0.0, 1.0,
        )
        u8  = (norm * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        bgr[arr_m <= 0] = 0
        return bgr

    for i in indices:
        gt    = depths[i]                                     # metres
        tbl_m = tables[i] * (depth_max - depth_min) + depth_min  # metres

        left  = _to_bgr(gt)
        right = _to_bgr(tbl_m)

        for panel, label in (
            (left,  f"GT depth      frame {i}"),
            (right, f"Table plane   frame {i}"),
        ):
            cv2.putText(panel, label, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        rows.append(np.hstack([left, right]))

    sheet   = np.vstack(rows)
    out_dir = seq_dir / "debug"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "table_plane_debug.png"
    cv2.imwrite(str(out_path), sheet)
    print(f"  [{seq_dir.name}] debug PNG → {out_path}")


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq_dir:   Path,
    calib:     dict,
    table_z:   float,
    overwrite: bool = False,
    debug:     bool = False,
) -> dict:
    """Compute and save the canonical corrected table_plane.h5 prior."""
    result = {"name": seq_dir.name, "success": False, "n_frames": 0, "error": None}

    depth_h5_path = seq_dir / "hdf5" / "depth_in_event_frame.h5"
    poses_path    = seq_dir / "hdf5" / "poses.h5"

    if not depth_h5_path.exists():
        result["error"] = "depth_in_event_frame.h5 not found (run project_realsense_to_event.py first)"
        return result
    if not poses_path.exists():
        result["error"] = "poses.h5 not found"
        return result

    out_path = seq_dir / "hdf5" / "table_plane.h5"

    with h5py.File(depth_h5_path, "r") as df:
        n_depth = df["depth"].shape[0]
        H       = df["depth"].shape[1]
        W       = df["depth"].shape[2]
        native_H = int(df.attrs.get("native_ev_h", calib["native_H"]))
        native_W = int(df.attrs.get("native_ev_w", calib["native_W"]))
        resize_H = int(df.attrs.get("resize_h", native_H))
        resize_W = int(df.attrs.get("resize_w", native_W))
        crop_H = int(df.attrs.get("crop_h", H))
        crop_W = int(df.attrs.get("crop_w", W))
        stored_transform = df.attrs.get("intrinsics_transform", "")
        if isinstance(stored_transform, bytes):
            stored_transform = stored_transform.decode("utf-8", errors="replace")

    expected_transform = INTRINSICS_TRANSFORM
    if stored_transform != expected_transform:
        result["error"] = (
            f"depth uses {stored_transform!r}, requested {expected_transform!r}; "
            "regenerate projected depth with the matching mode first"
        )
        return result

    with h5py.File(poses_path, "r") as pf:
        ee_T_all = pf["ee_T"][:]

    n_frames = min(n_depth, len(ee_T_all))
    if n_depth != len(ee_T_all):
        print(f"  [{seq_dir.name}] WARNING: {n_depth} depth frames != {len(ee_T_all)} poses; "
              f"using {n_frames}")

    if not overwrite and out_path.exists():
        with h5py.File(out_path, "r") as ef:
            if ef["table_plane"].shape[0] >= n_frames:
                transform = ef.attrs.get("intrinsics_transform", "")
                if isinstance(transform, bytes):
                    transform = transform.decode("utf-8", errors="replace")
                corrected_transform = transform == expected_transform
                if corrected_transform:
                    result.update(success=True, n_frames=n_frames,
                                  error="Already computed (use --overwrite)")
                    return result
                print(
                    f"  [{seq_dir.name}] Existing table_plane.h5 uses the obsolete "
                    "direct-scaling geometry; recomputing with centred crop + resize."
                )

    # Build T_base_from_event for every frame
    T_ee_from_event        = calib["T_ee_from_event"]
    T_base_from_event_all  = (ee_T_all[:n_frames] @ T_ee_from_event[None]).astype(np.float64)

    depths: list = [] if debug else None   # type: ignore
    tables: list = [] if debug else None   # type: ignore

    with h5py.File(depth_h5_path, "r") as df, \
         h5py.File(out_path, "w") as of:

        ds = of.create_dataset(
            "table_plane",
            shape=(n_frames, H, W),
            dtype=np.float32,
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=4,
        )
        of.attrs["table_z_m"]  = float(table_z)
        of.attrs["depth_min"]  = float(DEPTH_MIN)
        of.attrs["depth_max"]  = float(D_MAX)
        of.attrs["intrinsics_transform"] = expected_transform
        of.attrs["description"] = (
            f"Per-pixel normalised depth [0,1] to table plane z={table_z:.4f} m "
            f"(robot base frame). Training depth range: [{DEPTH_MIN}, {D_MAX}] m."
        )

        K_for_table = transform_intrinsics(
            calib["K_native"].astype(np.float64),
            (native_H, native_W),
            (resize_H, resize_W),
            (crop_H, crop_W),
        )
        K_native_H, K_native_W = H, W

        for fi in tqdm(range(n_frames), desc=seq_dir.name, leave=False):
            channel = _compute_channel(
                T_base_from_event_all[fi],
                K_for_table, K_native_H, K_native_W,
                H, W, table_z,
            )
            ds[fi] = channel

            if debug:
                depths.append(df["depth"][fi].astype(np.float32))
                tables.append(channel)

    if debug:
        _save_debug_png(seq_dir, depths, tables, DEPTH_MIN, D_MAX)

    result.update(success=True, n_frames=n_frames)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute the corrected table-plane depth prior "
            "(hdf5/table_plane.h5)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", nargs="+", type=str, default=None,
        help="Sequence directory(ies) to process. A parent directory containing "
             "multiple sequences is also accepted.",
    )
    parser.add_argument(
        "--data_root", type=str, default=str(DATA_ROOT),
        help="Root data directory to scan when --data_dir is not provided.",
    )
    parser.add_argument(
        "--table_z", type=float, default=None,
        help="Table plane Z in robot base frame [m]. "
             "Defaults to SPATIAL_TARGET_Z - SPATIAL_CUBE_SIDE/2 + TABLE_Z_OFFSET.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute even if the selected table-plane file already exists.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save debug/table_plane_debug.png (GT depth | table-plane depth) "
             "for each sequence.",
    )
    args = parser.parse_args()

    if args.table_z is None:
        args.table_z = SPATIAL_TARGET_Z - SPATIAL_CUBE_SIDE / 2.0 + TABLE_Z_OFFSET

    print(f"Table plane Z : {args.table_z:.4f} m")
    crop_h, crop_w = PREPROCESS_CROP_HW
    resize_h, resize_w = PREPROCESS_RESIZE_HW
    print(
        f"Preprocessing : crop -> {crop_w}x{crop_h} -> "
        f"resize -> {resize_w}x{resize_h}"
    )
    data_root = resolve_path(args.data_root, _HERE.parent)
    print(f"Data root     : {data_root}")

    calib = _load_calibration()

    def _is_sequence(p: Path) -> bool:
        return (
            (p / "hdf5" / "depth_in_event_frame.h5").exists()
            and (p / "hdf5" / "poses.h5").exists()
        )

    def _collect_sequences(path: Path) -> list[Path]:
        if _is_sequence(path):
            return [path]
        if not path.is_dir():
            return []
        return sorted(
            d for d in path.rglob("*")
            if d.is_dir() and _is_sequence(d)
        )

    searched_roots: list[Path]
    if args.data_dir:
        seq_dirs = []
        searched_roots = []
        for data_dir in args.data_dir:
            path = resolve_path(data_dir, _HERE.parent)
            searched_roots.append(path)
            seq_dirs.extend(_collect_sequences(path))
        seq_dirs = sorted(set(seq_dirs))
    else:
        searched_roots = [data_root]
        seq_dirs = _collect_sequences(data_root)

    if not seq_dirs:
        searched = ", ".join(str(p) for p in searched_roots)
        sys.exit(
            f"[ERROR] No valid sequences found at or under {searched}\n"
            "        Make sure depth_in_event_frame.h5 and poses.h5 exist."
        )

    print(f"Found {len(seq_dirs)} sequence(s)\n")

    results = []
    for seq_dir in seq_dirs:
        r = process_sequence(
            seq_dir, calib, args.table_z,
            overwrite=args.overwrite,
            debug=args.debug,
        )
        results.append(r)
        status = "OK"   if r["success"] else "FAIL"
        detail = f"  ({r['error']})" if r["error"] else f"  ({r['n_frames']} frames)"
        print(f"  [{status}] {r['name']}{detail}")

    n_ok   = sum(1 for r in results if r["success"] and not r["error"])
    n_skip = sum(1 for r in results if r["success"] and     r["error"])
    n_fail = sum(1 for r in results if not r["success"])
    print(f"\nDone: {n_ok} computed, {n_skip} skipped, {n_fail} failed.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
