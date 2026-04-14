#!/usr/bin/env python3
"""
Overlay net event polarity, EE translation speed, and EE angular speed
on a single plot (z-score normalised so they share the same y-axis scale).

Usage:
    python3 plot_polarity_vs_speed.py --data_dir data/real/1
    python3 plot_polarity_vs_speed.py --data_dir data/real/1 --smooth 5 --out overlay.png
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SMOOTH = 5
DEFAULT_OUT    = str(Path(__file__).parent / "polarity_vs_speed.png")


def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.copy()
    half = k // 2
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()


def rotation_angle_per_frame(rotations: np.ndarray) -> np.ndarray:
    """Angular step size (rad) between consecutive rotation matrices → (N,) padded."""
    n = len(rotations)
    ang = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        R_rel = rotations[i - 1].T @ rotations[i]
        cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        ang[i] = float(np.arccos(cos_a))
    return ang


def load_polarity(seq_dir: Path) -> np.ndarray:
    for vdir in [seq_dir / "events" / "voxels_cam0",
                 seq_dir / "events" / "voxels"]:
        files = sorted(vdir.glob("voxel_*.npy")) if vdir.exists() else []
        if files:
            return np.array([np.load(f).mean() for f in files], dtype=np.float32)
    ev_h5 = seq_dir / "hdf5" / "events_cam0.h5"
    if ev_h5.exists():
        with h5py.File(ev_h5, "r") as f:
            if "events/frames" in f:
                frames = f["events/frames"][:]
                return frames.astype(np.float32).mean(axis=(1, 2)) - 128.0
    raise FileNotFoundError(f"No event data found in {seq_dir}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--smooth",   type=int, default=DEFAULT_SMOOTH,
                        help="Box-filter width applied to all three signals")
    parser.add_argument("--out",      type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)

    # ── load signals ───────────────────────────────────────────────────────────
    polarity = load_polarity(seq_dir)
    n = len(polarity)

    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]           # (N, 4, 4)

    positions  = ee_Ts[:, :3, 3]
    rotations  = ee_Ts[:, :3, :3]

    # translation speed: |Δpos| per frame, padded to length N
    trans_speed = np.zeros(n, dtype=np.float32)
    trans_speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    ang_speed = rotation_angle_per_frame(rotations)   # (N,) radians/frame

    # ── smooth + normalise ─────────────────────────────────────────────────────
    pol_s   = zscore(box_smooth(polarity,    args.smooth))
    trans_s = zscore(box_smooth(trans_speed, args.smooth))
    ang_s   = zscore(box_smooth(np.degrees(ang_speed), args.smooth))

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))

    x = np.arange(n)
    ax.plot(x, pol_s,   color="steelblue", lw=1.2, alpha=0.9, label="net polarity")
    ax.plot(x, trans_s, color="forestgreen", lw=1.2, alpha=0.85, label="translation speed")
    ax.plot(x, ang_s,   color="tomato",  lw=1.2, alpha=0.85, label="angular speed")

    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Z-score (zero-mean, unit-std)")
    ax.set_title(
        f"Net polarity · translation speed · angular speed  |  {seq_dir.name}"
        f"  (smoothed k={args.smooth}, z-score normalised)",
        fontsize=10,
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
