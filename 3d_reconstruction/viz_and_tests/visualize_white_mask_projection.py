#!/usr/bin/env python3
"""
Visualize the white-pixel mask (as used in real_train --rgb_mask) and its
projection onto the event camera plane.

For each of 5 randomly chosen (or specified) frame indices the script draws
one row of 4 panels:

    [original RGB] | [white-pixel mask on RGB] | [event frame] | [white-pixel mask on event frame]

The white-pixel mask is defined the same way as in RealDataset.__getitem__:
    white = np.all(rgb > 200, axis=-1)

For the "mask on event frame" panel the projected RGB (rgb_in_event_frame.h5,
produced by project_realsense_to_event.py) is used.

Usage:
    python visualize_white_mask_projection.py --data_dir data/real/bottle
    python visualize_white_mask_projection.py --data_dir data/real/bottle --indices 0 42 99 7 200
    python visualize_white_mask_projection.py --data_dir data/real/bottle --n 3
    python visualize_white_mask_projection.py --data_dir data/real/bottle --save mask_proj.png
    python visualize_white_mask_projection.py --data_dir data/real/bottle --white_thresh 220
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ------------------------------------------------------------------ helpers

def _load_rgb_frame(realsense_h5: Path, idx: int) -> np.ndarray | None:
    """Load one RGB frame from realsense.h5."""
    if not realsense_h5.exists():
        return None
    with h5py.File(realsense_h5, "r") as f:
        if "rgb" not in f:
            return None
        n = f["rgb"].shape[0]
        if idx >= n:
            return None
        return f["rgb"][idx]  # (H, W, 3) uint8


def _load_proj_rgb_frame(proj_rgb_h5: Path, idx: int) -> np.ndarray | None:
    """Load one projected-RGB frame from rgb_in_event_frame.h5."""
    if not proj_rgb_h5.exists():
        return None
    with h5py.File(proj_rgb_h5, "r") as f:
        if "rgb" not in f:
            return None
        n = f["rgb"].shape[0]
        if idx >= n:
            return None
        return f["rgb"][idx]  # (H, W, 3) uint8


def _load_voxel_sum(voxels_dir: Path, idx: int) -> np.ndarray | None:
    """Load voxel_XXXXXX.npy and return sum across bins as a 2-D array."""
    path = voxels_dir / f"voxel_{idx:06d}.npy"
    if not path.exists():
        return None
    voxel = np.load(path)  # (B, H, W)
    return voxel.sum(axis=0)  # (H, W)


def _count_voxel_frames(voxels_dir: Path) -> int:
    return len(list(voxels_dir.glob("voxel_*.npy")))


def _white_mask(rgb: np.ndarray, thresh: int) -> np.ndarray:
    """Return boolean mask (H, W) where all channels exceed thresh."""
    return np.all(rgb > thresh, axis=-1)


def _render_mask_overlay(bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay a boolean mask on top of a uint8 RGB image.
    Masked pixels are rendered in red; non-masked pixels are dimmed.
    """
    out = bg.copy().astype(np.float32) / 255.0
    # Dim non-masked area slightly
    out[~mask] *= 0.5
    # Paint masked pixels red
    out[mask] = [1.0, 0.0, 0.0]
    return (out * 255).astype(np.uint8)


def _event_to_rgb(voxel_sum: np.ndarray) -> np.ndarray:
    """Convert a raw voxel-sum map to a displayable uint8 RGB image."""
    v = voxel_sum.astype(np.float32)
    # Normalise using absolute max so polarity sign is preserved
    vmax = np.abs(v).max()
    if vmax > 0:
        v = v / vmax  # in [-1, 1]
    # Map to [0, 255] (grey = 0 events, brighter/darker = +/- events)
    img = ((v + 1.0) * 0.5 * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)  # (H, W, 3)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(
        description="Visualize white-pixel mask and its projection onto the event plane",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Sequence directory (must contain hdf5/ and events/voxels_cam0/)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Frame indices to visualize (default: 5 random)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of random frames to pick when --indices is not given")
    parser.add_argument("--cam", type=int, default=0,
                        help="Event camera index")
    parser.add_argument("--white_thresh", type=int, default=200,
                        help="All-channel threshold for white-pixel detection (same as real_train)")
    parser.add_argument("--save", type=str, default=None,
                        help="Output PNG path (default: white_mask_proj_<seq>.png next to this script)")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)
    if not seq_dir.exists():
        sys.exit(f"Directory not found: {seq_dir}")

    # ---- paths ----
    realsense_h5  = seq_dir / "hdf5" / "realsense.h5"
    proj_rgb_h5   = seq_dir / "hdf5" / "rgb_in_event_frame.h5"
    voxels_dir    = seq_dir / "events" / f"voxels_cam{args.cam}"
    if not voxels_dir.exists():
        # fall back to legacy path
        voxels_dir = seq_dir / "events" / "voxels"

    if not realsense_h5.exists():
        sys.exit(f"realsense.h5 not found: {realsense_h5}")
    if not proj_rgb_h5.exists():
        sys.exit(
            f"rgb_in_event_frame.h5 not found: {proj_rgb_h5}\n"
            "Run project_realsense_to_event.py first."
        )
    if not voxels_dir.exists():
        sys.exit(
            f"Voxels directory not found: {voxels_dir}\n"
            "Run precompute_voxels.py first."
        )

    n_frames = _count_voxel_frames(voxels_dir)
    if n_frames == 0:
        sys.exit(f"No voxel files found in {voxels_dir}")

    print(f"Sequence : {seq_dir.name}")
    print(f"Frames   : {n_frames}  (cam{args.cam})")
    print(f"Thresh   : {args.white_thresh}")

    # ---- index selection ----
    if args.indices is not None:
        indices = [i % n_frames for i in args.indices]
    else:
        rng = np.random.default_rng()
        indices = sorted(
            rng.choice(n_frames, size=min(args.n, n_frames), replace=False).tolist()
        )
    print(f"Indices  : {indices}")

    # ---- figure layout ----
    n_rows = len(indices)
    n_cols = 4  # RGB | mask-on-RGB | event | mask-on-event
    col_w  = 3.5
    row_h  = 3.0

    fig = plt.figure(figsize=(n_cols * col_w, n_rows * row_h + 0.6))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.05, wspace=0.05,
                            left=0.02, right=0.98,
                            top=0.94, bottom=0.02)

    col_titles = [
        "Original RGB",
        f"White mask on RGB (>{args.white_thresh})",
        "Event frame (voxel sum)",
        f"White mask on event frame (>{args.white_thresh})",
    ]

    for col_idx, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_title(title, fontsize=8, pad=3)

    for row, idx in enumerate(indices):
        # --- load data ---
        rgb       = _load_rgb_frame(realsense_h5, idx)
        proj_rgb  = _load_proj_rgb_frame(proj_rgb_h5, idx)
        vox_sum   = _load_voxel_sum(voxels_dir, idx)

        if rgb is None:
            print(f"  [idx={idx}] Warning: no RGB frame, skipping")
            continue
        if proj_rgb is None:
            print(f"  [idx={idx}] Warning: no projected RGB frame, skipping")
            continue
        if vox_sum is None:
            print(f"  [idx={idx}] Warning: no voxel file, skipping")
            continue

        # --- compute masks ---
        white_rgb   = _white_mask(rgb,       args.white_thresh)   # in RGB frame
        white_event = _white_mask(proj_rgb,  args.white_thresh)   # in event frame

        mask_pct_rgb   = 100.0 * white_rgb.mean()
        mask_pct_event = 100.0 * white_event.mean()

        # --- build display images ---
        overlay_rgb   = _render_mask_overlay(rgb,      white_rgb)
        event_img     = _event_to_rgb(vox_sum)

        # Resize event-frame images to event resolution (already there), then
        # overlay mask on a copy tinted with the event image as background.
        # For the event background we produce a colour event image first.
        overlay_event = _render_mask_overlay(event_img, white_event)

        panels = [rgb, overlay_rgb, event_img, overlay_event]

        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row, col_idx])
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])

            # Row label on left-most panel
            if col_idx == 0:
                ax.set_ylabel(f"idx={idx}", fontsize=7)

            # Mask percentage annotation
            if col_idx == 1:
                ax.set_xlabel(f"masked: {mask_pct_rgb:.1f}%", fontsize=6)
            elif col_idx == 3:
                ax.set_xlabel(f"masked: {mask_pct_event:.1f}%", fontsize=6)

    fig.suptitle(
        f"White-pixel mask projection  |  {seq_dir.name}",
        fontsize=10, y=0.98,
    )

    # ---- save ----
    if args.save:
        out_path = Path(args.save)
    else:
        script_dir = Path(__file__).parent
        out_path = script_dir / f"white_mask_proj_{seq_dir.name}.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
