"""
Quick visualisation of a random input/depth-GT pair from the real dataset.

Usage:
    python show_sample.py                          # uses default data/real
    python show_sample.py --data_dir data/real/seq1
    python show_sample.py --idx 42                 # specific sample index
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Re-use dataset + constants from real_train
sys.path.insert(0, str(Path(__file__).parent))
from real_train import RealDataset, DataConfig, D_MAX, ALPHA

DATA_ROOT = Path("data/real")


def decode_log_depth(d_norm: np.ndarray) -> np.ndarray:
    """Invert the log-normalisation applied in __getitem__."""
    # d_norm = 1 + (1/ALPHA) * log(d / D_MAX)
    # => d = D_MAX * exp(ALPHA * (d_norm - 1))
    return D_MAX * np.exp(ALPHA * (d_norm - 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Sequence directory (default: first subdir of data/real)")
    parser.add_argument("--idx", type=int, default=None,
                        help="Sample index (default: random)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    args = parser.parse_args()

    # Resolve sequence directory
    if args.data_dir:
        seq_dir = Path(args.data_dir)
    else:
        candidates = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
        if not candidates:
            sys.exit(f"No subdirectories found in {DATA_ROOT}")
        seq_dir = candidates[0]
        print(f"Using sequence: {seq_dir}")

    cfg = DataConfig(seq_len=1, augment=False)
    ds = RealDataset(str(seq_dir), cfg, split=args.split)

    idx = args.idx if args.idx is not None else np.random.randint(len(ds))
    idx = idx % len(ds)
    print(f"Showing sample index {idx} / {len(ds) - 1}")

    events, depths, masks = ds[idx]   # (T, C, H, W) each; T=1 here

    # Take the single frame (T=0)
    voxel = events[0]   # (num_bins, H, W)
    depth_norm = depths[0, 0]  # (H, W) – log-normalised [0, 1]
    mask = masks[0, 0]         # (H, W)
    depth_m = decode_log_depth(depth_norm)

    # Collapse voxel bins: net polarity map
    event_display = voxel.sum(axis=0)  # (H, W)

    # Load the actual event camera frame from HDF5
    frame_idx = int(ds.indices[idx])
    event_frame = None
    events_h5_path = seq_dir / "hdf5" / "events_cam0.h5"
    if events_h5_path.exists():
        with h5py.File(events_h5_path, "r") as f:
            if "events/frames" in f and frame_idx < f["events/frames"].shape[0]:
                event_frame = f["events/frames"][frame_idx]  # (H, W) uint8

    # ------------------------------------------------------------------ plot
    n_cols = 4 if event_frame is not None else 3
    fig = plt.figure(figsize=(4 * n_cols + 2, 5))
    fig.suptitle(f"{seq_dir.name}  –  sample {idx}", fontsize=12)
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.35)

    col = 0

    # 1) Event camera frame from HDF5 (greyscale)
    if event_frame is not None:
        ax0 = fig.add_subplot(gs[col]); col += 1
        ax0.imshow(event_frame, cmap="gray", vmin=0, vmax=255)
        ax0.set_title("Event cam frame (HDF5)")
        ax0.axis("off")

    # 2) Event voxel (sum across bins)
    ax = fig.add_subplot(gs[col]); col += 1
    vmax = np.abs(event_display).max() + 1e-6
    ax.imshow(event_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Events (voxel sum, R=pos / B=neg)")
    ax.axis("off")

    # 3) Depth GT (normalised)
    ax3 = fig.add_subplot(gs[col]); col += 1
    d_masked = np.where(mask > 0, depth_norm, np.nan)
    im3 = ax3.imshow(d_masked, cmap="plasma", vmin=0, vmax=1)
    ax3.set_title("GT depth (log-norm)")
    fig.colorbar(im3, ax=ax3, fraction=0.046)
    ax3.axis("off")

    # 4) Depth GT in metres
    ax4 = fig.add_subplot(gs[col]); col += 1
    d_m_masked = np.where(mask > 0, depth_m, np.nan)
    im4 = ax4.imshow(d_m_masked, cmap="plasma")
    ax4.set_title("GT depth (metres)")
    fig.colorbar(im4, ax=ax4, fraction=0.046, label="m")
    ax4.axis("off")

    plt.tight_layout()
    out_path = Path(__file__).parent / f"sample_{seq_dir.name}_{idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
