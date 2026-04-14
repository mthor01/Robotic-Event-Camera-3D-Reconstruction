#!/usr/bin/env python3
"""
Visualize camera poses alongside their RGB images.

For a given recording, samples N arm (EE) poses at random, transforms them
into RGB-camera world poses using T_rgb_from_ee, and produces a single PNG
with:
  - Left:  3D scatter plot of the sampled camera positions with EE-frame axes,
           viewing-direction arrows toward the mean scene point, and pose IDs.
  - Right: Grid of the corresponding RGB frames, each labelled with its pose ID
           and HDF5 frame index.

Usage:
    python3 visualize_poses.py --data_dir data/real/1
    python3 visualize_poses.py --data_dir data/real/1 --n 8 --out poses_viz.png --seed 0
    python3 visualize_poses.py --data_dir data/real/1 --calib_dir camera_data
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── defaults ──────────────────────────────────────────────────────────────────
CALIB_DIR   = Path("camera_data")
DEFAULT_N   = 6
DEFAULT_OUT = str(Path(__file__).parent / "poses_viz.png")
AXIS_LEN    = 0.03   # metres – length of the drawn coordinate axes
ARROW_LEN   = 0.06   # metres – length of the viewing-direction arrow


# ── helpers ───────────────────────────────────────────────────────────────────

def load_T_rgb_from_ee(calib_dir: Path) -> np.ndarray:
    """Return (4,4) float64 transform: T_rgb_from_ee."""
    return np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)


def ee_to_rgb_pose(ee_T: np.ndarray, T_rgb_from_ee: np.ndarray) -> np.ndarray:
    """
    Given EE pose in world frame (4×4) and the hand-eye calibration,
    return the RGB-camera pose in world frame (4×4).

    T_world_rgb = T_world_ee  @  T_ee_rgb
    where T_ee_rgb = inv(T_rgb_from_ee).
    """
    T_ee_rgb = np.linalg.inv(T_rgb_from_ee)
    return ee_T @ T_ee_rgb


def draw_camera_axes(ax, R: np.ndarray, origin: np.ndarray, length: float):
    """Draw RGB (red), YG (green), ZB (blue) axes for a camera pose."""
    colors = ["red", "green", "blue"]
    for i, col in enumerate(colors):
        direction = R[:, i] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=col, linewidth=1.2, arrow_length_ratio=0.3,
        )


def draw_viewing_direction(ax, R: np.ndarray, origin: np.ndarray,
                           length: float, color: str = "black"):
    """Draw the camera's +Z viewing direction."""
    z_dir = R[:, 2] * length
    ax.quiver(
        origin[0], origin[1], origin[2],
        z_dir[0], z_dir[1], z_dir[2],
        color=color, linewidth=1.5, arrow_length_ratio=0.25, alpha=0.7,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize EE→RGB camera poses + corresponding RGB images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Recording directory (must contain hdf5/poses.h5 and hdf5/realsense.h5)")
    parser.add_argument("--calib_dir",  type=str, default=str(CALIB_DIR),
                        help="Directory with T_rgb_from_ee.npz")
    parser.add_argument("--n",          type=int, default=DEFAULT_N,
                        help="Number of poses to select via farthest-point sampling")
    parser.add_argument("--out",        type=str, default=DEFAULT_OUT,
                        help="Output PNG path")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Random seed for the FPS starting point")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    calib_dir = Path(args.calib_dir)

    # ── load calibration ─────────────────────────────────────────────────────
    T_rgb_from_ee = load_T_rgb_from_ee(calib_dir)

    # ── load poses and sample ─────────────────────────────────────────────────
    poses_h5_path = data_dir / "hdf5" / "poses.h5"
    rs_h5_path    = data_dir / "hdf5" / "realsense.h5"

    with h5py.File(poses_h5_path, "r") as f:
        ee_Ts = f["ee_T"][:]          # (N_frames, 4, 4)

    n_frames = ee_Ts.shape[0]
    n_samples = min(args.n, n_frames)

    # Farthest-point sampling on all EE positions so selected poses are
    # maximally spread across the trajectory.
    all_positions = ee_Ts[:, :3, 3]      # (N_frames, 3)

    rng = np.random.default_rng(args.seed)
    first = int(rng.integers(0, n_frames))

    selected = [first]
    # Maintain distance from each candidate to the nearest already-selected point
    min_dists = np.full(n_frames, np.inf)
    diff = all_positions - all_positions[first]
    min_dists = np.minimum(min_dists, (diff * diff).sum(axis=1))

    for _ in range(n_samples - 1):
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        diff = all_positions - all_positions[farthest]
        min_dists = np.minimum(min_dists, (diff * diff).sum(axis=1))

    sampled_indices = sorted(selected)

    # ── convert EE poses → RGB camera poses ───────────────────────────────────
    rgb_Ts = np.stack(
        [ee_to_rgb_pose(ee_Ts[i], T_rgb_from_ee) for i in sampled_indices],
        axis=0,
    )  # (n_samples, 4, 4)

    positions = rgb_Ts[:, :3, 3]          # (n_samples, 3)
    rotations = rgb_Ts[:, :3, :3]         # (n_samples, 3, 3)

    # ── load RGB images ───────────────────────────────────────────────────────
    with h5py.File(rs_h5_path, "r") as f:
        rgb_frames = [f["rgb"][i] for i in sampled_indices]

    # ── layout ────────────────────────────────────────────────────────────────
    # Left column: 3-D trajectory plot
    # Right columns: grid of RGB images (ceil(n/2) rows × 2 cols)
    img_cols = 2
    img_rows = int(np.ceil(n_samples / img_cols))

    fig_w = 7 + img_cols * 3.2
    fig_h = max(6, img_rows * 3.2)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: 1 col for 3D plot + img_cols cols for images
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        img_rows, 1 + img_cols,
        figure=fig,
        width_ratios=[2.5] + [1] * img_cols,
        wspace=0.05, hspace=0.35,
    )

    # ── 3D plot ────────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")

    # Scatter camera positions
    ax3d.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        s=60, c="steelblue", zorder=5,
    )

    # Per-pose axes + viewing direction + label
    for k, (pos, rot, frame_idx) in enumerate(zip(positions, rotations, sampled_indices)):
        draw_camera_axes(ax3d, rot, pos, length=AXIS_LEN)
        draw_viewing_direction(ax3d, rot, pos, length=ARROW_LEN)
        ax3d.text(
            pos[0], pos[1], pos[2],
            f"  #{k}",
            fontsize=8, color="steelblue", zorder=10,
        )

    # Connect poses in order
    ax3d.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="steelblue", linewidth=0.8, alpha=0.5,
    )

    # Mark the approximate scene centre (mean camera position projected forward)
    scene_centre = positions.mean(axis=0)
    ax3d.scatter(*scene_centre, s=120, marker="*", c="red", label="mean pos", zorder=10)

    ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
    ax3d.set_title(
        f"RGB camera poses  ({data_dir.name})\n"
        f"axes: X=red Y=green Z=blue  |  arrow=view dir",
        fontsize=9,
    )

    # Equal aspect ratio
    all_pts = positions
    ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_range = ranges.max() / 2 if ranges.max() > 0 else 0.1
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    ax3d.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax3d.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax3d.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # ── RGB image grid ─────────────────────────────────────────────────────────
    for k, (rgb, frame_idx) in enumerate(zip(rgb_frames, sampled_indices)):
        row = k // img_cols
        col = k %  img_cols
        ax_img = fig.add_subplot(gs[row, 1 + col])
        ax_img.imshow(rgb)
        ax_img.set_title(f"#{k}  [HDF idx {frame_idx}]", fontsize=8, pad=3)
        ax_img.axis("off")

    # Blank out any unused image cells
    for k in range(n_samples, img_rows * img_cols):
        row = k // img_cols
        col = k %  img_cols
        fig.add_subplot(gs[row, 1 + col]).axis("off")

    fig.suptitle(
        f"Camera pose visualisation — {data_dir.name}  "
        f"({n_samples} poses via farthest-point sampling, seed={args.seed})",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
