#!/usr/bin/env python3
"""
Visualize spatial masks for 10 random frames across a range of Z offsets.

For each Z offset (10 values ±9 mm around 0, step 2 mm) the mask is
recomputed on-the-fly for 10 randomly selected frames.  All results are
saved as one large PNG:  rows = frames,  columns = Z offsets.

Usage:
    python3 visualize_spatial_mask_offsets.py --data_dir data/real/box
    python3 visualize_spatial_mask_offsets.py --data_dir data/real/box --seed 42
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_precomputation"))

from config import (
    CALIB_DIR as _CALIB_DIR,
    SPATIAL_CUBE_SIDE,
    SPATIAL_CUBE_X_OFFSET,
    SPATIAL_TARGET_X,
    SPATIAL_TARGET_Y,
)
from precompute_spatial_mask import (
    build_event_rays,
    compute_spatial_mask,
    load_calibration,
)

CALIB_DIR = Path(__file__).resolve().parent.parent / _CALIB_DIR

N_FRAMES  = 10      # random frames to sample
N_OFFSETS = 10      # Z offsets to test
STEP_M    = 0.002   # 2 mm step between offsets


def make_overlay(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image: depth in grey + green tint scaled by mask value.

    Works for both binary uint8 masks ({0, 1}) and soft float32 masks ([0, 1]).
    """
    alpha = mask.astype(np.float32)  # (H, W) in [0, 1]
    valid = depth > 0
    d_vis = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        lo, hi = depth[valid].min(), depth[valid].max()
        span = hi - lo if hi > lo else 1.0
        d_vis[valid] = (depth[valid] - lo) / span
    rgb = np.stack([d_vis, d_vis, d_vis], axis=-1)  # grey
    # tint pixels green proportionally to mask value
    green_boost = 0.45
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * (1.0 - 0.5 * alpha), 0, 1)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] * (1.0 - 0.5 * alpha) + green_boost * alpha, 0, 1)
    rgb[:, :, 2] = np.clip(rgb[:, :, 2] * (1.0 - 0.5 * alpha), 0, 1)
    return (rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize spatial masks over a range of Z offsets for random frames."
    )
    parser.add_argument("--data_dir", required=True, help="Path to recording directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for frame selection")
    parser.add_argument("--out", type=str, default="", help="Output PNG path (default: <data_dir>/spatial_mask_offsets.png)")
    parser.add_argument("--soft_mask", action="store_true",
                        help="Use soft float32 mask: linear ramp 0→1 over the bottom 1 cm of the cube in Z")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else seq_dir / "spatial_mask_offsets.png"

    depth_h5_path = seq_dir / "hdf5" / "depth_in_event_frame.h5"
    poses_path    = seq_dir / "hdf5" / "poses.h5"
    for p in (depth_h5_path, poses_path):
        if not p.exists():
            sys.exit(f"Required file not found: {p}")

    # Load calibration
    calib = load_calibration(CALIB_DIR)
    rays = None  # built lazily after we know H, W

    # Load poses
    with h5py.File(poses_path, "r") as pf:
        ee_T_all = pf["ee_T"][:]  # (N, 4, 4)

    with h5py.File(depth_h5_path, "r") as df:
        n_total = df["depth"].shape[0]
        H, W = df["depth"].shape[1], df["depth"].shape[2]
        # Read stored resolution attrs so K_event can be adjusted when the depth
        # was stored at a downscaled/cropped resolution by project_realsense_to_event.py.
        native_ev_h = int(df.attrs.get("native_ev_h", calib["ev_h"]))
        native_ev_w = int(df.attrs.get("native_ev_w", calib["ev_w"]))
        _resize_h   = int(df.attrs.get("resize_h",   native_ev_h))
        _resize_w   = int(df.attrs.get("resize_w",   native_ev_w))
        _crop_h     = int(df.attrs.get("crop_h",     H))
        _crop_w     = int(df.attrs.get("crop_w",     W))

    # Scale K_event from native event-camera resolution to the stored resolution.
    K_event = calib["K_event"].copy().astype(np.float64)
    if H != native_ev_h or W != native_ev_w:
        K_event[0] *= _resize_w / native_ev_w   # fx, cx scale by resize
        K_event[1] *= _resize_h / native_ev_h   # fy, cy scale by resize
        K_event[0, 2] -= (_resize_w - _crop_w) / 2   # cx shift by crop
        K_event[1, 2] -= (_resize_h - _crop_h) / 2   # cy shift by crop

    n_frames = min(n_total, len(ee_T_all))
    rays = build_event_rays(K_event, H, W)
    half_side = SPATIAL_CUBE_SIDE / 2.0

    # Pick random frame indices
    rng = np.random.default_rng(args.seed)
    frame_indices = sorted(rng.choice(n_frames, size=min(N_FRAMES, n_frames), replace=False).tolist())

    # Z offsets: 10 values from -(N_OFFSETS-1)/2 to +(N_OFFSETS-1)/2 * step
    # e.g. for N=10, step=2mm: -9,-7,-5,-3,-1,+1,+3,+5,+7,+9 mm
    half = (N_OFFSETS - 1) / 2.0
    z_offsets_m = np.array([(i - half) * STEP_M for i in range(N_OFFSETS)])  # metres

    print(f"Recording:   {seq_dir}")
    print(f"Frames ({len(frame_indices)}): {frame_indices}")
    print(f"Z offsets:   {[f'{v*1000:+.0f}mm' for v in z_offsets_m]}")

    # Preload the needed depth frames
    with h5py.File(depth_h5_path, "r") as df:
        depths = {fi: df["depth"][fi].astype(np.float32) for fi in frame_indices}

    # Build figure: rows = frames, columns = offsets
    nrows = len(frame_indices)
    ncols = N_OFFSETS
    cell_h, cell_w = H, W  # pixels per cell in display
    dpi = 100
    thumb_scale = 1.5          # upscale factor for readability — increase if needed
    fig_w = ncols * cell_w * thumb_scale / dpi
    fig_h = nrows * cell_h * thumb_scale / dpi + 1.5  # +1.5 inch for column headers
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Spatial mask — {seq_dir.name}\n"
        f"rows: frame index  |  cols: Z offset (cube_side={SPATIAL_CUBE_SIDE*100:.0f} cm)",
        fontsize=10, fontweight="bold",
    )

    for ci, z_off in enumerate(z_offsets_m):
        # Center Z = offset + half_side (as specified)
        center_z = z_off + SPATIAL_CUBE_SIDE / 2.0
        target_point = np.array([SPATIAL_TARGET_X, SPATIAL_TARGET_Y, center_z])

        for ri, fi in enumerate(frame_indices):
            ax = axes[ri, ci]
            depth = depths[fi]
            mask = compute_spatial_mask(
                depth, ee_T_all[fi], calib["T_ee_from_event"], rays,
                half_side, target_point, soft=args.soft_mask,
            )
            overlay = make_overlay(depth, mask)
            ax.imshow(overlay, interpolation="nearest")

            # Labels
            if ri == 0:
                ax.set_title(f"{z_off*1000:+.0f} mm\nctr_z={center_z*100:.1f} cm",
                             fontsize=6, pad=2)
            if ci == 0:
                ax.set_ylabel(f"fr {fi}", fontsize=6, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])

            n_valid = int((depth > 0).sum())
            if args.soft_mask:
                wgt = float(mask.sum())
                frac = wgt / n_valid * 100 if n_valid > 0 else 0.0
                ax.set_xlabel(f"wgt {wgt:.0f} ({frac:.0f}%)", fontsize=5, labelpad=1)
            else:
                n_masked = int(mask.sum())
                frac = n_masked / n_valid * 100 if n_valid > 0 else 0.0
                ax.set_xlabel(f"{n_masked} px ({frac:.0f}%)", fontsize=5, labelpad=1)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
