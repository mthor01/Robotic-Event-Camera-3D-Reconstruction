#!/usr/bin/env python3
"""
Visualize precomputed voxel grids alongside the corresponding event-camera
and RGB frames stored in HDF5.

For each of 3 randomly chosen (or specified) frame indices the script draws
one row of panels:

    [event frame (HDF5)] | [voxel bin-0] ... [voxel bin-N] | [RGB frame]

Usage:
    python visualize_voxels.py --data_dir data/real/bottle
    python visualize_voxels.py --data_dir data/real/bottle --indices 0 42 99
    python visualize_voxels.py --data_dir data/real/bottle --cam 1
    python visualize_voxels.py --data_dir data/real/bottle --save vis_voxels.png
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_voxel(voxels_dir: Path, idx: int) -> np.ndarray:
    """Load voxel_XXXXXX.npy for the given frame index."""
    path = voxels_dir / f"voxel_{idx:06d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Voxel file not found: {path}")
    return np.load(path)  # (num_bins, H, W)


def load_event_frame(events_h5: Path, idx: int) -> np.ndarray | None:
    """Load one grayscale event-accumulation frame from the HDF5 file."""
    if not events_h5.exists():
        return None
    with h5py.File(events_h5, "r") as f:
        if "events/frames" not in f:
            return None
        n = f["events/frames"].shape[0]
        if idx >= n:
            return None
        return f["events/frames"][idx]  # (H, W) uint8


def load_rgb_frame(realsense_h5: Path, idx: int) -> np.ndarray | None:
    """Load one RGB frame from the RealSense HDF5 file."""
    if not realsense_h5.exists():
        return None
    with h5py.File(realsense_h5, "r") as f:
        if "rgb" not in f:
            return None
        n = f["rgb"].shape[0]
        if idx >= n:
            return None
        return f["rgb"][idx]  # (H, W, 3) uint8


def count_voxel_frames(voxels_dir: Path) -> int:
    return len(list(voxels_dir.glob("voxel_*.npy")))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize precomputed voxel grids + HDF5 event/RGB frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Sequence directory (must contain events/voxels_cam* and hdf5/)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Frame indices to visualize (default: 3 random)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of random frames to pick when --indices is not set")
    parser.add_argument("--cam", type=int, default=0,
                        help="Event camera index to use")
    parser.add_argument("--save", type=str, default=None,
                        help="Output image path (default: vis_voxels_<seq>.png)")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)
    if not seq_dir.exists():
        sys.exit(f"Directory not found: {seq_dir}")

    voxels_dir    = seq_dir / "events" / f"voxels_cam{args.cam}"
    events_h5     = seq_dir / "hdf5"   / f"events_cam{args.cam}.h5"
    realsense_h5  = seq_dir / "hdf5"   / "realsense.h5"

    if not voxels_dir.exists():
        sys.exit(f"Voxels directory not found: {voxels_dir}\n"
                 "Run precompute_voxels.py first.")

    n_frames = count_voxel_frames(voxels_dir)
    if n_frames == 0:
        sys.exit(f"No voxel files found in {voxels_dir}")

    print(f"Sequence : {seq_dir.name}")
    print(f"Frames   : {n_frames}  (cam{args.cam})")

    # ------------------------------------------------------------------ indices
    if args.indices is not None:
        indices = [i % n_frames for i in args.indices]
    else:
        rng = np.random.default_rng()
        indices = sorted(rng.choice(n_frames, size=min(args.n, n_frames), replace=False).tolist())

    print(f"Indices  : {indices}")

    # ----------------------------------------------------- figure layout
    # Load one voxel to find num_bins
    sample_voxel = load_voxel(voxels_dir, indices[0])
    num_bins = sample_voxel.shape[0]

    # Columns per row:  event_frame | bin_0 … bin_{B-1} | voxel_sum | rgb_frame
    n_rows  = len(indices)
    n_cols  = 1 + num_bins + 1 + 1   # event | bins | sum | rgb
    col_w   = 2.2

    fig = plt.figure(figsize=(col_w * n_cols + 0.5, col_w * n_rows + 1.0))
    fig.suptitle(f"{seq_dir.name}  –  cam{args.cam}  –  indices {indices}", fontsize=11)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.08)

    for row, idx in enumerate(indices):
        voxel      = load_voxel(voxels_dir, idx)          # (B, H, W)
        ev_frame   = load_event_frame(events_h5, idx)     # (H, W) uint8 or None
        rgb_frame  = load_rgb_frame(realsense_h5, idx)    # (H, W, 3) uint8 or None
        voxel_sum  = voxel.sum(axis=0)                    # (H, W)

        col = 0

        # -- event HDF5 frame
        ax = fig.add_subplot(gs[row, col]); col += 1
        if ev_frame is not None:
            ax.imshow(ev_frame, cmap="gray", vmin=0, vmax=255)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        if row == 0:
            ax.set_title("event\nframe", fontsize=8)
        ax.set_ylabel(f"idx {idx}", fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # -- individual voxel bins
        # Render like the HDF5 event frame: gray background (128),
        # positive events → white, negative → black.
        # Scale by ±3σ of non-zero values for good contrast.
        nonzero = voxel[voxel != 0]
        scale = (3.0 * nonzero.std()) if len(nonzero) > 0 and nonzero.std() > 0 else 1.0
        for b in range(num_bins):
            ax = fig.add_subplot(gs[row, col]); col += 1
            gray = np.clip(voxel[b] / scale * 127 + 128, 0, 255).astype(np.uint8)
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            if row == 0:
                ax.set_title(f"bin {b}", fontsize=8)
            ax.set_yticks([]); ax.set_xticks([])

        # -- voxel sum
        ax = fig.add_subplot(gs[row, col]); col += 1
        nonzero_sum = voxel_sum[voxel_sum != 0]
        scale_sum = (3.0 * nonzero_sum.std()) if len(nonzero_sum) > 0 and nonzero_sum.std() > 0 else 1.0
        gray_sum = np.clip(voxel_sum / scale_sum * 127 + 128, 0, 255).astype(np.uint8)
        ax.imshow(gray_sum, cmap="gray", vmin=0, vmax=255)
        if row == 0:
            ax.set_title("voxel\nsum", fontsize=8)
        ax.set_yticks([]); ax.set_xticks([])

        # -- RGB frame
        ax = fig.add_subplot(gs[row, col]); col += 1
        if rgb_frame is not None:
            ax.imshow(rgb_frame)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        if row == 0:
            ax.set_title("RGB", fontsize=8)
        ax.set_yticks([]); ax.set_xticks([])

    plt.tight_layout()

    out_path = Path(args.save) if args.save else Path(__file__).parent / f"vis_voxels_{seq_dir.name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
