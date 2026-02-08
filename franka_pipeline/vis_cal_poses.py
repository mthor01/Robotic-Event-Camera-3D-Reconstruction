#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def _set_axes_equal_3d(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max()
    half = 0.5 * span if span > 0 else 1.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def plot_calibration_poses(
    poses: np.ndarray,              # (N,7) [x,y,z,qx,qy,qz,qw]
    base_point: np.ndarray | None = None,
    axis_len: float = 0.05,
    show_indices: bool = True,
):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    assert poses.shape[1] == 7, f"Expected (N,7), got {poses.shape}"

    pts = poses[:, :3]
    quats = poses[:, 3:7]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=35, label="calib poses")

    if base_point is not None:
        base_point = np.asarray(base_point, dtype=float).reshape(3,)
        ax.scatter([base_point[0]], [base_point[1]], [base_point[2]],
                   s=80, marker="x", label="base")

    # Draw local axes from quaternion (RGB)
    for i in range(len(poses)):
        p = pts[i]
        q = quats[i]

        # SciPy expects [x,y,z,w] which matches your codebase
        Rm = Rotation.from_quat(q).as_matrix()
        x_axis = Rm[:, 0]
        y_axis = Rm[:, 1]
        z_axis = Rm[:, 2]

        ax.quiver(p[0], p[1], p[2], x_axis[0], x_axis[1], x_axis[2],
                  length=axis_len, normalize=True, color="r", linewidth=1.0)
        ax.quiver(p[0], p[1], p[2], y_axis[0], y_axis[1], y_axis[2],
                  length=axis_len, normalize=True, color="g", linewidth=1.0)
        ax.quiver(p[0], p[1], p[2], z_axis[0], z_axis[1], z_axis[2],
                  length=axis_len, normalize=True, color="b", linewidth=1.0)

        if show_indices:
            ax.text(p[0], p[1], p[2], f"{i}", fontsize=9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal_3d(ax, pts)
    ax.legend()
    plt.tight_layout()
    plt.show()


def load_poses_npy(path: str) -> np.ndarray:
    poses = np.load(path)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    return poses


if __name__ == "__main__":
    # Option A: visualize poses saved by your calibration agent
    # (this file is created by save_poses_to_file: calibration_poses.npy)
    poses = load_poses_npy("calibration_data/calibration_poses.npy")
    plot_calibration_poses(poses, base_point=None, axis_len=0.05, show_indices=True)

    # Option B: if you already have them in memory (agent.calibration_poses):
    # plot_calibration_poses(np.array(agent.calibration_poses), axis_len=0.05)
