#!/usr/bin/env python3
"""
Visualize depth mask, white mask, and combined mask for every object in data/real/.

For each object, picks 3 frames (start, middle, end) and shows them after the
same resize + center-crop pipeline used by real_train.py, so the visualization
exactly matches what the model sees during training.

Produces a single image with one row per (object, frame) showing:
    RGB | depth GT | depth mask | white mask | combined mask

All data is shown projected into the event camera frame when available
(depth_in_event_frame.h5 / rgb_in_event_frame.h5), with per-frame
fallback to the raw realsense.h5 for frames where the projection is empty.

Usage:
    python3 viz_and_tests/visualize_masks.py
    python3 viz_and_tests/visualize_masks.py --data_dir data/real/box
    python3 viz_and_tests/visualize_masks.py --data_root data/real --out masks_overview.png
    python3 viz_and_tests/visualize_masks.py --resize_h 288 --resize_w 384 --crop_h 240 --crop_w 320
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconstruction_config import (
    D_MAX, ALPHA, DEPTH_MIN, WHITE_THRESH, DATA_ROOT, SPATIAL_CUBE_SIDE,
)


def find_objects(data_root: Path) -> list[Path]:
    """Find all recording directories that contain depth data."""
    objects = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "hdf5" / "realsense.h5").exists():
            objects.append(p)
    return objects


def load_depth_frame(seq_dir: Path, idx: int):
    """Load a single depth frame in metres (float32).

    Prefers projected event-frame data; falls back to raw realsense
    when the projected frame is all zeros (incomplete projection).
    """
    proj = seq_dir / "hdf5" / "depth_in_event_frame.h5"
    if proj.exists():
        with h5py.File(proj, "r") as f:
            frame = f["depth"][idx].astype(np.float32)
            if frame.max() > 0:
                return frame, True  # already metres
    raw = seq_dir / "hdf5" / "realsense.h5"
    with h5py.File(raw, "r") as f:
        return f["depth"][idx].astype(np.float32) / 1000.0, False


def load_rgb_frame(seq_dir: Path, idx: int):
    """Load a single RGB frame (uint8, H×W×3).

    Prefers projected event-frame data; falls back to raw realsense
    when the projected frame is all zeros (incomplete projection).
    """
    proj = seq_dir / "hdf5" / "rgb_in_event_frame.h5"
    if proj.exists():
        with h5py.File(proj, "r") as f:
            frame = f["rgb"][idx]
            if frame.max() > 0:
                return frame, True
    raw = seq_dir / "hdf5" / "realsense.h5"
    with h5py.File(raw, "r") as f:
        return f["rgb"][idx], False


def get_n_frames(seq_dir: Path) -> int:
    """Return number of frames available (from realsense.h5)."""
    raw = seq_dir / "hdf5" / "realsense.h5"
    with h5py.File(raw, "r") as f:
        return f["depth"].shape[0]


def resize_and_crop(img: np.ndarray, resize_hw, crop_hw) -> np.ndarray:
    """Apply bilinear resize then center crop, matching real_train.py pipeline.

    Works on 2-D (H, W) and 3-D (H, W, C) arrays.
    Skips the transform when the image is already at the final target size
    (i.e. precomputed files were stored pre-downscaled/cropped).
    """
    # If data is already at the target resolution, nothing to do.
    target_hw = crop_hw if crop_hw is not None else resize_hw
    if target_hw is not None and img.shape[:2] == tuple(target_hw):
        return img

    is_mask = img.dtype == np.float32 and img.ndim == 2

    if resize_hw is not None:
        rh, rw = resize_hw
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        img = cv2.resize(img, (rw, rh), interpolation=interp)

    if crop_hw is not None:
        ch, cw = crop_hw
        cur_h, cur_w = img.shape[:2]
        y0 = (cur_h - ch) // 2
        x0 = (cur_w - cw) // 2
        img = img[y0:y0 + ch, x0:x0 + cw]

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize depth, white, and combined masks for all objects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Single recording directory (mutually exclusive with --data_root)")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                        help="Root directory containing recording subdirs")
    parser.add_argument("--out", type=str, default="masks_overview.png",
                        help="Output image path")
    parser.add_argument("--resize_h", type=int, default=288)
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--crop_h", type=int, default=240)
    parser.add_argument("--crop_w", type=int, default=320)
    parser.add_argument("--depth_min", type=float, default=DEPTH_MIN)
    parser.add_argument("--depth_max", type=float, default=D_MAX)
    parser.add_argument("--white_thresh", type=int, default=WHITE_THRESH)
    parser.add_argument("--no_spatial_mask", action="store_true",
                        help="Hide the spatial mask column (shown by default)")
    parser.add_argument("--depth_mask", action="store_true",
                        help="Show depth-range mask column (hidden by default)")
    parser.add_argument("--white_mask", action="store_true",
                        help="Show white-pixel mask column (hidden by default)")
    parser.add_argument("--log_depth", action="store_true",
                        help="Display depth with log encoding (default: linear normalization)")
    args = parser.parse_args()

    if args.data_dir:
        obj_path = Path(args.data_dir)
        if not (obj_path / "hdf5" / "realsense.h5").exists():
            sys.exit(f"No realsense.h5 found in {obj_path}")
        objects = [obj_path]
    else:
        data_root = Path(args.data_root)
        objects = find_objects(data_root)
        if not objects:
            sys.exit(f"No recordings found under {data_root}")

    resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None
    crop_hw = (args.crop_h, args.crop_w) if args.crop_h > 0 and args.crop_w > 0 else None

    has_spatial    = not args.no_spatial_mask
    show_depth_mask = args.depth_mask
    show_white_mask = args.white_mask

    # Collect rows: each row is (object_name, frame_idx, depth, depth_mask, white_mask, combined)
    rows = []
    for obj_dir in objects:
        n_frames = get_n_frames(obj_dir)
        if n_frames < 3:
            print(f"[skip] {obj_dir.name}: only {n_frames} frames")
            continue

        # Pick start, middle, end
        indices = [0, n_frames // 2, n_frames - 1]

        for idx in indices:
            depth, d_proj = load_depth_frame(obj_dir, idx)
            rgb, r_proj = load_rgb_frame(obj_dir, idx)
            src = "proj" if (d_proj and r_proj) else "raw"

            # Depth mask: valid depth range
            depth_mask = ((depth > args.depth_min) & (depth < args.depth_max)).astype(np.float32)

            # White mask from RGB: flag pixels that are white (background table)
            # OR have no projected RGB value (event-frame pixels with no coverage).
            no_rgb = np.all(rgb == 0, axis=-1)
            white_mask = (np.all(rgb > args.white_thresh, axis=-1) | no_rgb).astype(np.float32)

            # Spatial mask (precomputed, shown by default)
            spatial_raw = None
            if has_spatial:
                sp_path = obj_dir / "hdf5" / "spatial_mask.h5"
                if sp_path.exists():
                    with h5py.File(sp_path, "r") as f:
                        spatial_raw = f["mask"][idx].astype(np.float32)
                else:
                    # zeros = everything masked out → still shows the column
                    spatial_raw = np.zeros_like(depth_mask)

            # Combined mask (depth valid AND not white AND spatial if enabled)
            combined_mask = depth_mask.copy()
            combined_mask[white_mask > 0.5] = 0.0
            if spatial_raw is not None:
                combined_mask[spatial_raw == 0] = 0.0

            # Normalize depth for display
            depth_clipped = np.clip(depth, args.depth_min, args.depth_max)
            if args.log_depth:
                depth_norm = 1.0 + (1.0 / ALPHA) * np.log(depth_clipped / D_MAX)
            else:
                depth_norm = (depth_clipped - args.depth_min) / (args.depth_max - args.depth_min)
            depth_norm = np.clip(depth_norm, 0, 1)

            # Apply resize + center crop (same as training)
            rgb_vis = resize_and_crop(rgb, resize_hw, crop_hw)
            depth_norm = resize_and_crop(depth_norm, resize_hw, crop_hw)
            depth_mask = resize_and_crop(depth_mask, resize_hw, crop_hw)
            white_mask = resize_and_crop(white_mask, resize_hw, crop_hw)
            combined_mask = resize_and_crop(combined_mask, resize_hw, crop_hw)
            if spatial_raw is not None:
                spatial_raw = resize_and_crop(spatial_raw, resize_hw, crop_hw)
                spatial_raw = (spatial_raw > 0.5).astype(np.float32)

            # Re-binarize masks after resize (nearest interp keeps them sharp,
            # but just in case)
            depth_mask = (depth_mask > 0.5).astype(np.float32)
            white_mask = (white_mask > 0.5).astype(np.float32)
            combined_mask = (combined_mask > 0.5).astype(np.float32)

            rows.append({
                "name": obj_dir.name,
                "idx": idx,
                "src": src,
                "rgb": rgb_vis,
                "depth_norm": depth_norm,
                "depth_mask": depth_mask,
                "white_mask": white_mask,
                "spatial_mask": spatial_raw,
                "combined_mask": combined_mask,
            })

    if not rows:
        sys.exit("No valid data found")

    # Plot
    n_rows = len(rows)
    depth_label = "Depth (log)" if args.log_depth else "Depth (linear)"
    col_titles = ["RGB", depth_label]
    if show_depth_mask:
        col_titles.append("Depth mask")
    if show_white_mask:
        col_titles.append("White mask")
    if has_spatial:
        col_titles.append(f"Spatial mask ({SPATIAL_CUBE_SIDE*100:.0f}cm cube)")
    col_titles.append("Combined mask")
    n_cols = len(col_titles)
    cell_h, cell_w = 2.0, 3.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 1.5, cell_h * n_rows + 1.0),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.08, wspace=0.05)

    for ci, title in enumerate(col_titles):
        axes[0][ci].set_title(title, fontsize=9, fontweight="bold")

    for ri, row in enumerate(rows):
        ci = 0

        # RGB
        ax = axes[ri][ci]; ci += 1
        ax.imshow(row["rgb"])
        ax.set_ylabel(f"{row['name']}\n#{row['idx']} ({row['src']})", fontsize=7,
                      rotation=0, labelpad=65, va="center")
        ax.set_xticks([]); ax.set_yticks([])

        # Depth normalized
        ax = axes[ri][ci]; ci += 1
        ax.imshow(row["depth_norm"], cmap="plasma", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])

        # Depth mask (optional)
        if show_depth_mask:
            ax = axes[ri][ci]; ci += 1
            ax.imshow(row["depth_mask"], cmap="gray", vmin=0, vmax=1)
            pct = row["depth_mask"].mean() * 100
            ax.text(0.98, 0.02, f"{pct:.0f}%", transform=ax.transAxes,
                    fontsize=6, color="lime", ha="right", va="bottom",
                    fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

        # White mask (optional)
        if show_white_mask:
            ax = axes[ri][ci]; ci += 1
            ax.imshow(row["white_mask"], cmap="Reds", vmin=0, vmax=1)
            pct = row["white_mask"].mean() * 100
            ax.text(0.98, 0.02, f"{pct:.0f}%", transform=ax.transAxes,
                    fontsize=6, color="red", ha="right", va="bottom",
                    fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

        # Spatial mask
        if has_spatial:
            ax = axes[ri][ci]; ci += 1
            sm = row["spatial_mask"]
            if sm is not None:
                ax.imshow(sm, cmap="gray", vmin=0, vmax=1)
                pct = sm.mean() * 100
                ax.text(0.98, 0.02, f"{pct:.0f}%", transform=ax.transAxes,
                        fontsize=6, color="cyan", ha="right", va="bottom",
                        fontweight="bold")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_xticks([]); ax.set_yticks([])

        # Combined mask
        ax = axes[ri][ci]
        ax.imshow(row["combined_mask"], cmap="gray", vmin=0, vmax=1)
        pct = row["combined_mask"].mean() * 100
        ax.text(0.98, 0.02, f"{pct:.0f}%", transform=ax.transAxes,
                fontsize=6, color="lime", ha="right", va="bottom",
                fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])


    preproc = ""
    if resize_hw:
        preproc += f"resize {resize_hw[1]}x{resize_hw[0]}"
    if crop_hw:
        preproc += f" → crop {crop_hw[1]}x{crop_hw[0]}"
    fig.suptitle(
        f"Mask overview  |  depth=[{args.depth_min}, {args.depth_max}]m  "
        f"white>{args.white_thresh}  |  {preproc}",
        fontsize=10, fontweight="bold", y=0.995,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  ({n_rows} rows, {len(objects)} objects)")


if __name__ == "__main__":
    main()
