#!/usr/bin/env python3
"""
Visualize all camera poses for an object sequence.

Reads:
    <data_dir>/hdf5/poses.h5
    camera_data/T_rgb_from_ee.npz
    camera_data/T_event_from_rgb.npz

Output:
    viz_and_tests/plots/all_poses_<object_name>.png by default

Usage:
    python3 viz_and_tests/visualize_all_poses.py --data_dir data/new/train/1
    python3 viz_and_tests/visualize_all_poses.py --data_dir data/new/train/1 --stride 5
    python3 viz_and_tests/visualize_all_poses.py --data_dir data/new/train/1 --out_png /tmp/poses.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
from helpers import set_3d_axes_equal


def _load_numpy():
    import numpy as np
    return np


def load_T_event_from_ee(calib_dir: Path) -> np.ndarray:
    np = _load_numpy()
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float64)
    return T_event_from_rgb @ T_rgb_from_ee


def load_camera_poses(data_dir: Path, calib_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    np = _load_numpy()
    import h5py

    poses_path = data_dir / "hdf5" / "poses.h5"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")

    with h5py.File(poses_path, "r") as f:
        if "ee_T" not in f:
            raise KeyError(f"{poses_path} does not contain dataset 'ee_T'")
        ee_T = f["ee_T"][:].astype(np.float64)

    T_event_from_ee = load_T_event_from_ee(calib_dir)
    T_base_from_event = ee_T @ np.linalg.inv(T_event_from_ee)
    centers = T_base_from_event[:, :3, 3]
    return T_base_from_event, centers


def plot_poses(
    data_dir: Path,
    out_png: Path,
    stride: int = 1,
    max_arrows: int = 120,
) -> Path:
    np = _load_numpy()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calib_dir = _ROOT / "camera_data"
    poses, centers = load_camera_poses(data_dir, calib_dir)

    stride = max(1, int(stride))
    draw_idx = np.arange(0, len(poses), stride, dtype=np.int64)
    if len(draw_idx) == 0:
        raise RuntimeError(f"No poses to draw in {data_dir}")

    arrow_idx = draw_idx
    if len(arrow_idx) > max_arrows:
        arrow_idx = np.linspace(0, len(poses) - 1, max_arrows, dtype=np.int64)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_idx = np.arange(len(centers))
    sc = ax.scatter(
        centers[draw_idx, 0],
        centers[draw_idx, 1],
        centers[draw_idx, 2],
        c=draw_idx,
        cmap="viridis",
        s=12,
        alpha=0.9,
        label="camera centers",
    )
    ax.plot(
        centers[draw_idx, 0],
        centers[draw_idx, 1],
        centers[draw_idx, 2],
        color="0.45",
        linewidth=0.8,
        alpha=0.75,
    )

    axis_len = max(float(np.ptp(centers[draw_idx], axis=0).max()) * 0.04, 0.015)
    axis_colors = ("tab:red", "tab:green", "tab:blue")
    for idx in arrow_idx:
        T = poses[idx]
        c = T[:3, 3]
        R = T[:3, :3]
        for axis_i, color in enumerate(axis_colors):
            direction = R[:, axis_i] * axis_len
            ax.quiver(
                c[0], c[1], c[2],
                direction[0], direction[1], direction[2],
                color=color,
                linewidth=0.6,
                arrow_length_ratio=0.25,
                alpha=0.75,
            )

    start = centers[0]
    end = centers[-1]
    ax.scatter(start[0], start[1], start[2], color="black", marker="o", s=55, label="start")
    ax.scatter(end[0], end[1], end[2], color="tab:orange", marker="*", s=90, label="end")

    fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08, label="frame index")
    ax.set_title(f"All event-camera poses: {data_dir.name} ({len(poses)} total, stride={stride})")
    ax.set_xlabel("base/world x [m]")
    ax.set_ylabel("base/world y [m]")
    ax.set_zlabel("base/world z [m]")
    ax.legend(loc="upper right")
    set_3d_axes_equal(ax)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[visualize_all_poses] poses: {len(poses)}")
    print(f"[visualize_all_poses] plotted centers: {len(draw_idx)}")
    print(f"[visualize_all_poses] saved -> {out_png}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize all event-camera poses for an object sequence."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Object sequence directory containing hdf5/poses.h5.",
    )
    parser.add_argument(
        "--out_png",
        type=Path,
        default=None,
        help="Output PNG path. Default: viz_and_tests/plots/all_poses_<object>.png",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Plot every Nth camera center. Arrows are additionally capped by --max_arrows.",
    )
    parser.add_argument(
        "--max_arrows",
        type=int,
        default=120,
        help="Maximum number of pose axis triads to draw.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_png = (
        args.out_png.resolve()
        if args.out_png is not None
        else _HERE / "plots" / f"all_poses_{data_dir.name}.png"
    )
    plot_poses(data_dir, out_png, stride=args.stride, max_arrows=args.max_arrows)


if __name__ == "__main__":
    main()
