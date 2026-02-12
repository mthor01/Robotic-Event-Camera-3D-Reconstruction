# TODO: This is heavily AI-coded, future review is needed

import time
from typing import Any, Dict, Tuple

import numpy as np

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

from scipy.spatial.transform import Rotation

logger = get_logger(__name__)


class HalfSphereRecordingAgent(Agent):
    """Agent that moves between sampled hemisphere poses in a loop to test OscPoseTargetController."""

    def __init__(
        self,
        wait_time: float = 0.0,
    ) -> None:
        super().__init__(action_type="OSC_POSE")
        self.wait_time = wait_time

        radius = 0.15
        base_pose = np.array([0.35, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=float)

        num_theta = 5
        num_phi = 7

        # Replace quarter-sphere with hemisphere sampling logic (with x-cut via dx_min)
        self.poses = self.generate_hemisphere_poses(
            base_pose=base_pose,
            radius=radius,
            num_theta=num_theta,
            num_phi=num_phi,
            use_quat="keep_y_horizontal",  # or "z_only"
            filter_p_gt=-1
        )

        logger.info(f"Generated {len(self.poses)} poses for HalfSphereRecordingAgent.")

        self.current_pose_idx = 0
        self.state = "MOVING"  # States: MOVING, WAITING
        self.wait_start_time: float | None = None
        self.osc_controller: OscPoseTargetController | None = None

    def quat_z_points_keep_y_horizontal(
        self,
        direction: np.ndarray,
        world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> np.ndarray:
        d = direction.astype(float)
        d /= np.linalg.norm(d)

        z = d

        # Make EE y-axis horizontal and perpendicular to z
        y = np.cross(z, world_up)
        if np.linalg.norm(y) < 1e-8:
            y = np.array([0.0, 1.0, 0.0])
        y /= np.linalg.norm(y)

        # Right-hand rule to get x
        x = np.cross(y, z)
        x /= np.linalg.norm(x)

        R = np.column_stack([x, y, z])

        # match script's extra flips
        R[:, 0] *= -1
        R[:, 1] *= -1

        q = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
        q /= np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        return q

    def quat_z_points_only(
        self,
        direction: np.ndarray,
        reference_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> np.ndarray:
        z = direction.astype(float)
        z /= np.linalg.norm(z)

        up = reference_up.astype(float)
        up -= np.dot(up, z) * z

        if np.linalg.norm(up) < 1e-8:
            up = np.array([1.0, 0.0, 0.0])
            up -= np.dot(up, z) * z

        x = up / np.linalg.norm(up)
        y = np.cross(z, x)

        R = np.column_stack([x, y, z])

        q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
        q /= np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        return q

    def _init_osc_controller(self) -> None:
        target_pose = self.poses[self.current_pose_idx]
        # command is [x, y, z, qx, qy, qz, qw, gripper]
        position_tolerance: float = 0.1
        rotation_tolerance: float = 0.1

        command = np.concatenate([target_pose, [-1.0]])
        self.osc_controller = OscPoseTargetController(
            target_pose=command
            #threshold_reach=position_tolerance,
            #threshold_rotation=rotation_tolerance,
        )

    def generate_hemisphere_poses(
        self,
        base_pose: np.ndarray,
        radius: float = 0.2,
        num_theta: int = 4,
        num_phi: int = 7,
        use_quat: str = "z_only",       # "z_only" or "keep_y_horizontal"
        filter_p_gt: int = -1           # keep pose only if p > filter_p_gt
    ) -> list[np.ndarray]:
        """
        Port of the sampling logic you pasted:
        - theta in linspace(0..pi/2)
        - phi uses: 2*pi*(p/num_phi + 0.5)
        - alternates sweep direction per theta-level (start_from_right)
        - breaks after first phi sample at the pole (theta == theta_vals[0])
        - also applies x-cut: dx > dx_min
        """
        base_point = base_pose[:3].copy()
        poses: list[np.ndarray] = []


        theta_vals = (
            np.array([0.0]) if num_theta <= 1 else np.linspace(0.0, np.pi / 2.0, num_theta)
        )

        start_from_right = True
        eps = 1e-12

        for theta in theta_vals[:-1]:
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)

            for p in range(1, num_phi):
                if p <= filter_p_gt:
                    continue

                phi = 2.0 * np.pi * (p / num_phi + 0.5)
                if start_from_right:
                    phi = -phi

                dx = radius * sin_t * np.cos(phi)
                dy = radius * sin_t * np.sin(phi)
                dz = radius * cos_t

                pose = base_pose.copy()
                pose[:3] += np.array([dx, dy, dz], dtype=float)

                direction = base_point - pose[:3]

                if use_quat == "keep_y_horizontal":
                    q = self.quat_z_points_keep_y_horizontal(direction)
                else:
                    q = self.quat_z_points_only(direction)

                pose[3:7] = q

                poses.append(pose)

                if theta == theta_vals[0]:
                    break

            start_from_right = not start_from_right

        return poses

    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pos = robot_state["osc_position"]
        rot = robot_state["osc_rotation_quaternion"]
        current_pose = np.concatenate([pos.flatten(), rot.flatten()]).astype(float)

        if self.osc_controller is None:
            self._init_osc_controller()

        action, is_finished = self.osc_controller.calculate_action(current_pose)

        if self.state == "MOVING":
            if is_finished:
                logger.info(
                    f"Reached pose {self.current_pose_idx}. Waiting for {self.wait_time}s."
                )
                self.state = "WAITING"
                self.wait_start_time = time.time()
            else:
                logger.debug(f"Moving to pose {self.current_pose_idx}.")

        elif self.state == "WAITING":
            if self.wait_start_time is not None and (
                time.time() - self.wait_start_time > self.wait_time
            ):
                self.current_pose_idx = (self.current_pose_idx + 1) % len(self.poses)
                logger.info(f"Wait finished. Moving to pose {self.current_pose_idx}.")
                self.state = "MOVING"
                self._init_osc_controller()
                action, is_finished = self.osc_controller.calculate_action(current_pose)

        logger.debug(f"Action: {action}")

        metadata = {
            "action_type": self.action_type,
            "state": self.state,
            "target_pose_idx": self.current_pose_idx,
            "is_finished": is_finished,
        }

        return action, metadata

    def reset(self) -> None:
        self.current_pose_idx = 0
        self.state = "MOVING"
        self.wait_start_time = None
        self.osc_controller = None
