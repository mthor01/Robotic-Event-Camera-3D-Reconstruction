#!/usr/bin/env python3
"""
Visualise all raw poses from raw_poses.h5.

Plots EE position (X/Y/Z), EE orientation (roll/pitch/yaw extracted from
the rotation matrix), joint positions, and gripper opening over time.

Usage:
    python3 visualize_raw_poses.py --data_dir data/real/my_recording
    python3 visualize_raw_poses.py --data_dir data/real/my_recording --out poses.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZYX Euler angles (yaw, pitch, roll) in degrees from (N,3,3) rotation matrices."""
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6
    roll  = np.where(singular, np.arctan2(-R[:, 1, 2], R[:, 1, 1]),
                               np.arctan2( R[:, 2, 1], R[:, 2, 2]))
    pitch = np.where(singular, np.arctan2(-R[:, 2, 0], sy),
                               np.arctan2(-R[:, 2, 0], sy))
    yaw   = np.where(singular, np.zeros(len(R)),
                               np.arctan2( R[:, 1, 0], R[:, 0, 0]))
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)


def visualize_raw_poses(seq_dir: Path, out_path: Path | None = None) -> None:
    if out_path is None:
        out_path = seq_dir / "raw_poses.png"

    raw_poses_path = seq_dir / "hdf5" / "raw_poses.h5"
    if not raw_poses_path.exists():
        raise FileNotFoundError(f"No raw_poses.h5 in {seq_dir / 'hdf5'}")

    with h5py.File(raw_poses_path, "r") as f:
        t_ns          = f["t_ns"][:]           # (P,) int64
        ee_T          = f["ee_T"][:]           # (P, 4, 4)
        joint_pos     = f["joint_positions"][:] # (P, 7)
        gripper_q     = f["gripper_q"][:]      # (P,)

    P = len(t_ns)
    t_sec = (t_ns - t_ns[0]) / 1e9  # seconds from start

    # EE position
    pos = ee_T[:, :3, 3]   # (P, 3)  X, Y, Z

    # EE orientation as Euler angles
    R = ee_T[:, :3, :3]
    yaw, pitch, roll = rotation_matrix_to_euler_zyx(R)
    z_angle = np.arctan2(R[:, 1, 0], R[:, 0, 0])  # Z-rotation (matches temporal analysis)

    # Pose rate
    dt = np.diff(t_sec)
    dt[dt == 0] = 1e-6
    pose_rate = 1.0 / dt   # Hz

    # EE rotation speed (Z-angle finite difference, rad/s)
    z_angle = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    rot_vel = np.zeros(P, dtype=np.float64)
    rot_vel[1:] = np.diff(z_angle) / dt
    rot_vel[np.abs(rot_vel) > 20.0] = 0.0  # clamp wrap-around artefacts

    print(f"Raw poses:   {P}")
    print(f"Duration:    {t_sec[-1]:.2f} s")
    print(f"Median rate: {np.median(pose_rate):.1f} Hz")
    print(f"EE X range:  [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}] m")
    print(f"EE Y range:  [{pos[:, 1].min():.3f}, {pos[:, 1].max():.3f}] m")
    print(f"EE Z range:  [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}] m")

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 19))
    gs = GridSpec(6, 1, figure=fig, hspace=0.45)

    idx = np.arange(P)

    # --- Panel 1: EE position ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_sec, pos[:, 0], lw=1.0, label="X")
    ax1.plot(t_sec, pos[:, 1], lw=1.0, label="Y")
    ax1.plot(t_sec, pos[:, 2], lw=1.0, label="Z")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("EE Position (XYZ)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, lw=0.4, alpha=0.5)

    # --- Panel 2: EE orientation ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_sec, yaw,   lw=1.0, label="Yaw (Z)")
    ax2.plot(t_sec, pitch, lw=1.0, label="Pitch (Y)")
    ax2.plot(t_sec, roll,  lw=1.0, label="Roll (X)")
    ax2.set_ylabel("Angle (°)")
    ax2.set_title("EE Orientation — ZYX Euler angles", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, lw=0.4, alpha=0.5)

    # --- Panel 3: joint positions ---
    ax3 = fig.add_subplot(gs[2])
    n_joints = joint_pos.shape[1]
    for j in range(n_joints):
        ax3.plot(t_sec, np.rad2deg(joint_pos[:, j]), lw=0.8, label=f"J{j+1}")
    ax3.set_ylabel("Angle (°)")
    ax3.set_title("Joint Positions", fontsize=11)
    ax3.legend(fontsize=7, loc="upper right", ncol=4)
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, lw=0.4, alpha=0.5)

    # --- Panel 4: rotation speed ---
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t_sec, np.rad2deg(rot_vel), lw=0.8, color="#e08020", label="Z-rot speed")
    ax4.axhline(0, color="gray", lw=0.5, ls="--")
    ax4.set_ylabel("Speed (°/s)")
    ax4.set_title("EE Rotation Speed (Z-axis)", fontsize=11)
    ax4.set_xlabel("Time (s)")
    ax4.grid(True, lw=0.4, alpha=0.5)

    # --- Panel 5: gripper ---
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(t_sec, gripper_q, lw=1.0, color="#555555")
    ax5.set_ylabel("Gripper opening")
    ax5.set_title("Gripper", fontsize=11)
    ax5.set_xlabel("Time (s)")
    ax5.grid(True, lw=0.4, alpha=0.5)

    # --- Panel 6: pose publish rate ---
    ax5 = fig.add_subplot(gs[5])
    ax5.plot(t_sec[1:], pose_rate, lw=0.7, color="#888888")
    ax5.axhline(np.median(pose_rate), color="red", lw=1.0, ls="--",
                label=f"median = {np.median(pose_rate):.1f} Hz")
    ax5.set_ylabel("Rate (Hz)")
    ax5.set_title("Pose publish rate", fontsize=11)
    ax5.legend(fontsize=8)
    ax5.set_xlabel("Time (s)")
    ax5.grid(True, lw=0.4, alpha=0.5)

    fig.suptitle(
        f"Raw Poses — {seq_dir.name}  ({P} poses, {t_sec[-1]:.1f} s)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualise raw poses from raw_poses.h5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Recording directory containing hdf5/raw_poses.h5")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path (default: <data_dir>/raw_poses.png)")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else None
    visualize_raw_poses(seq_dir, out_path=out_path)


if __name__ == "__main__":
    main()
