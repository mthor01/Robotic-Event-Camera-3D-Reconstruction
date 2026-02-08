#!/usr/bin/env python3
"""
Visualize poses on a half-sphere using the *exact same sampling + filtering*
as HalfSphereRecordingAgent, and visualize the viewing direction (point -> base).

Changes vs your original:
- Sampling matches agent: theta loop uses range(num_theta-1), not range(num_theta)
- base_pose defaults match agent example (0.5, 0.0, -0.1, ...)
- radius/num_theta/num_phi defaults match agent (0.2, 4, 7)
- pose filtering matches agent: only keep poses where p > 1
- Quaternion matches agent: quat_z_points_only(direction) (agent currently uses that)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def quat_z_points_keep_y_horizontal(
    direction: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    d = direction.astype(float)
    d /= np.linalg.norm(d)

    z = d
    y = np.cross(z, world_up)
    if np.linalg.norm(y) < 1e-8:
        y = np.array([0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)

    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    Rm = np.column_stack([x, y, z])

    # match your agent's extra flips
    Rm[:, 0] *= -1
    Rm[:, 1] *= -1

    q = Rotation.from_matrix(Rm).as_quat()  # [x,y,z,w]
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def quat_z_points_only(
    direction: np.ndarray,
    reference_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """
    Construct a quaternion whose z-axis points along `direction`.
    Rotation around z is chosen to be as close as possible to `reference_up`.
    Matches HalfSphereRecordingAgent.quat_z_points_only().
    """
    z = direction.astype(float)
    z /= np.linalg.norm(z)

    up = reference_up.astype(float)
    up -= np.dot(up, z) * z

    if np.linalg.norm(up) < 1e-8:
        up = np.array([1.0, 0.0, 0.0])
        up -= np.dot(up, z) * z

    x = up / np.linalg.norm(up)
    y = np.cross(z, x)

    Rm = np.column_stack([x, y, z])
    q = Rotation.from_matrix(Rm).as_quat()  # [x,y,z,w]
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def generate_half_sphere_poses_like_agent(
    base_pose: np.ndarray,
    radius: float = 0.2,
    num_theta: int = 4,
    num_phi: int = 7,
    use_quat: str = "z_only",   # "z_only" (agent) or "keep_y_horizontal"
    filter_p_gt: int = 1,       # agent: if p > 1 (keeps p=2..)
):
    """
    Matches HalfSphereRecordingAgent pose generation.

    Returns:
      poses: (N,7) [x,y,z,qx,qy,qz,qw]
      dirs:  (N,3) direction vectors (point -> base)
    """
    base_point = base_pose[:3].copy()

    poses = []
    dirs = []

    # agent uses: for t in range(num_theta-1)
    for t in range(num_theta - 1):
        theta = (np.pi / 2) * t / (num_theta - 1)  # 0..pi/2
        for p in range(num_phi):
            phi = 2 * np.pi * p / num_phi  # 0..2pi (exclusive endpoint)

            dx = radius * np.sin(theta) * np.cos(phi)
            dy = radius * np.sin(theta) * np.sin(phi)
            dz = radius * np.cos(theta)

            pose = base_pose.copy()
            pose[:3] += [dx, dy, dz]

            direction = base_point - pose[:3]  # point back toward base

            if use_quat == "keep_y_horizontal":
                q = quat_z_points_keep_y_horizontal(direction)
            else:
                q = quat_z_points_only(direction)

            pose[3:7] = q

            # agent filter: if p > 1
            if p > filter_p_gt:
                poses.append(pose)
                dirs.append(direction)

    return np.array(poses, dtype=float), np.array(dirs, dtype=float)


def plot_poses(poses: np.ndarray, dirs: np.ndarray, base_point: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    pts = poses[:, :3]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=40, label="poses")
    ax.scatter([base_point[0]], [base_point[1]], [base_point[2]], s=80, marker="x", label="base")

    arrow_len = 0.06  # scale up a bit for radius=0.2
    dnorm = np.linalg.norm(dirs, axis=1, keepdims=True)
    dunit = dirs / np.maximum(dnorm, 1e-12)

    ax.quiver(
        pts[:, 0], pts[:, 1], pts[:, 2],
        dunit[:, 0], dunit[:, 1], dunit[:, 2],
        length=arrow_len,
        normalize=True,
        linewidth=1.5,
        color="k",
        label="view dir (to base)",
    )

    draw_axes = True
    if draw_axes:
        axis_len = 0.04
        for i in range(len(poses)):
            q = poses[i, 3:7]
            Rm = Rotation.from_quat(q).as_matrix()
            x_axis = Rm[:, 0]
            y_axis = Rm[:, 1]
            z_axis = Rm[:, 2]
            p = pts[i]

            ax.quiver(p[0], p[1], p[2], x_axis[0], x_axis[1], x_axis[2],
                      length=axis_len, normalize=True, color="r", linewidth=1.0)
            ax.quiver(p[0], p[1], p[2], y_axis[0], y_axis[1], y_axis[2],
                      length=axis_len, normalize=True, color="g", linewidth=1.0)
            ax.quiver(p[0], p[1], p[2], z_axis[0], z_axis[1], z_axis[2],
                      length=axis_len, normalize=True, color="b", linewidth=1.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins).max()
    half = 0.5 * span if span > 0 else 1.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    # match your agent defaults
    base_pose = np.array([0.5, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=float)

    poses, dirs = generate_half_sphere_poses_like_agent(
        base_pose=base_pose,
        radius=0.2,
        num_theta=4,
        num_phi=7,
        use_quat="keep_y_horizontal",   # agent currently uses quat_z_points_only
        filter_p_gt=-1,       # agent: if p > 1
    )

    print(f"Generated {len(poses)} poses [x,y,z,qx,qy,qz,qw]:")
    np.set_printoptions(precision=6, suppress=True)
    print(poses)

    plot_poses(poses, dirs, base_pose[:3])


if __name__ == "__main__":
    main()
