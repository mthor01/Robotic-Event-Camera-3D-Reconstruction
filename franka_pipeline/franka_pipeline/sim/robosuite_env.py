"""Robosuite simulation environment wrapper.

This module provides a wrapper around the robosuite library for
running Franka Panda robot simulations with MuJoCo.
"""

import time
from typing import Any

import numpy as np
import robosuite
import robosuite.macros as macros
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# Set the image convention to opencv to avoid flipped images
macros.IMAGE_CONVENTION = "opencv"

from robosuite.utils.transform_utils import quat2mat
from scipy.spatial.transform import Rotation

from .sim_env import SimEnv


class RobosuiteSimEnv(SimEnv):
    """Robosuite-based simulation environment for Franka Panda.

    This class wraps a robosuite Lift environment with configurable
    controller type and rendering.

    Attributes:
        controller_type: Type of controller (e.g., 'OSC_POSE').
        control_freq: Control frequency in Hz.
        env: The underlying robosuite environment.
    """

    def __init__(
        self,
        controller_type: str = "OSC_POSE",
        control_freq: int = 20,
        camera_width: int = 640,
        camera_height: int = 480,
        has_renderer: bool = True,
    ) -> None:
        """Initialize the robosuite simulation environment.

        Args:
            controller_type: Controller type for the robot arm.
            control_freq: Control loop frequency in Hz.
            camera_width: Width of the camera images.
            camera_height: Height of the camera images.
            has_renderer: Whether to show the rendering window (False for headless mode).
        """
        self.controller_type = controller_type
        self.control_freq = control_freq
        self.camera_width = camera_width
        self.camera_height = camera_height

        arm_controller_config = robosuite.load_part_controller_config(
            default_controller=controller_type
        )

        controller_configs = refactor_composite_controller_config(
            arm_controller_config, "Panda", ["right"]
        )

        self.env = robosuite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=has_renderer,
            ignore_done=True,
            control_freq=self.control_freq,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["frontview", "robot0_eye_in_hand"],
            camera_depths=True,
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            controller_configs=controller_configs,
            # renderer="mujoco",
            renderer="mjviewer",
            hard_reset=False,
        )

        # Correction for the 90-degree rotation discrepancy between robosuite and real robot (deoxys)
        # Robosuite's Panda EE site is rotated by -90 degrees around Z relative to the standard Franka hand frame.
        self._ee_correction_matrix = Rotation.from_euler(
            "z", 90, degrees=True
        ).as_matrix()

        self._last_obs = self.env.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the simulation environment.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial observation dictionary.
        """
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        self._last_obs = self.env.reset()
        return self._last_obs

    def step(self, action: Any) -> dict[str, Any]:
        """Execute one simulation step.

        Args:
            action: Robot action to execute.

        Returns:
            Observation dictionary after the step.
        """
        start = time.perf_counter()

        # TODO The following code is not checked, and might not be correct >>>>
        # If using OSC_POSE and the controller is configured for EE-frame input,
        # we need to transform the action to account for our EE correction.
        # However, in this pipeline, OSC_POSE is typically used with base-frame input.
        # We check the controller config to be sure.
        robot = self.env.robots[0]
        controllers_to_check = []
        if hasattr(robot, "composite_controller"):
            controllers_to_check.extend(
                robot.composite_controller.part_controllers.values()
            )
        if hasattr(robot, "controller"):
            controllers_to_check.append(robot.controller)

        for controller in controllers_to_check:
            if controller.name == "OSC_POSE":
                if getattr(controller, "input_ref_frame", "base") == "ee":
                    # Transform delta position and rotation from corrected EE frame to robosuite EE frame
                    # action_robosuite = R_corr @ action_agent
                    action = action.copy()
                    action[:3] = self._ee_correction_matrix @ action[:3]
                    action[3:6] = self._ee_correction_matrix @ action[3:6]
                break
        # <<<<< End of unchecked code

        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs

        if self.env.viewer is not None:
            self.env.render()

        # TODO: Hacky enforce control frequency pacing inside the environment wrapper by sleeping.
        time_per_step = 1.0 / self.control_freq
        elapsed = time.perf_counter() - start
        if elapsed < time_per_step:
            time.sleep(time_per_step - elapsed)
        return obs

    def get_state(self) -> dict[str, Any]:
        # TODO NOTE: This get_state is mostly AI generated, and not tested at all, might have some bugs
        """Get the current robot state from simulation.

        Returns:
            Dictionary containing pose, joint positions, velocities,
            and gripper state.
        """
        robot = self.env.robots[0]
        sim = self.env.sim

        arm_name = list(robot.eef_site_id.keys())[0]
        eef_site_id = robot.eef_site_id[arm_name]

        # Get world poses directly from MuJoCo sites to be consistent with extrinsics
        osc_position_world = sim.data.site_xpos[eef_site_id]
        osc_rotation_matrix_world = sim.data.site_xmat[eef_site_id].reshape(3, 3)

        T_world_ee = np.eye(4)
        T_world_ee[:3, :3] = osc_rotation_matrix_world
        T_world_ee[:3, 3] = osc_position_world

        # Get base pose in world
        # In robosuite, robot.base_pos and robot.base_ori are available
        T_world_base = np.eye(4)
        if robot.base_ori.ndim == 2:
            T_world_base[:3, :3] = robot.base_ori
        else:
            # Use scipy for rotation conversion to avoid robosuite/numba quat2mat issues
            T_world_base[:3, :3] = Rotation.from_quat(robot.base_ori).as_matrix()
        T_world_base[:3, 3] = robot.base_pos

        # Compute base-relative pose
        T_base_ee = np.linalg.inv(T_world_base) @ T_world_ee

        # Apply EE correction to match real robot convention
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ self._ee_correction_matrix

        osc_pose = T_base_ee
        osc_position = T_base_ee[:3, 3]
        osc_rotation_matrix = T_base_ee[:3, :3]

        osc_rotation_quaternion = Rotation.from_matrix(
            osc_rotation_matrix
        ).as_quat()  # [x, y, z, w]

        joint_positions_raw = robot._joint_positions
        joint_velocities_raw = robot._joint_velocities
        if isinstance(joint_positions_raw, dict):
            joint_position = np.array(joint_positions_raw[arm_name]).flatten()
            joint_velocity = np.array(joint_velocities_raw[arm_name]).flatten()
        else:
            joint_position = np.array(joint_positions_raw).flatten()
            joint_velocity = np.array(joint_velocities_raw).flatten()

        gripper_joints = robot.gripper_joints
        if gripper_joints is not None and hasattr(robot, "sim"):
            if isinstance(gripper_joints, dict):
                gripper_joint_names = gripper_joints[arm_name]
            else:
                gripper_joint_names = gripper_joints

            gripper_joint_positions = np.array(
                [
                    robot.sim.data.qpos[robot.sim.model.joint_name2id(joint)]
                    for joint in gripper_joint_names
                ]
            )
            gripper_q = np.array(np.sum(np.abs(gripper_joint_positions)))
        else:
            gripper_q = np.array(0.0)

        return {
            "osc_pose": osc_pose,
            "osc_position": osc_position,
            "osc_rotation_matrix": osc_rotation_matrix,
            "osc_rotation_quaternion": osc_rotation_quaternion,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_q": gripper_q,
        }

    def get_camera_intrinsics(self, camera_id: str) -> dict[str, float] | None:
        """Get camera intrinsics from simulation.

        Args:
            camera_id: Camera identifier.

        Returns:
            Dictionary with fx, fy, cx, cy.
        """
        if not hasattr(self.env, "sim"):
            return None

        try:
            sim = self.env.sim
            cam_id = sim.model.camera_name2id(camera_id)
            fovy = sim.model.cam_fovy[cam_id]

            # Use configured resolution
            height = self.camera_height
            width = self.camera_width

            # fovy is in degrees
            fovy_rad = np.deg2rad(fovy)
            f = (height / 2) / np.tan(fovy_rad / 2)

            return {"fx": f, "fy": f, "cx": width / 2, "cy": height / 2}
        except Exception:
            return None

    def get_camera_extrinsics(self, camera_id: str) -> np.ndarray | None:
        # TODO this is AI generated and not tested yet
        """Get camera extrinsics (T_ee2cam) from simulation.

        Args:
            camera_id: Camera identifier.

        Returns:
            4x4 transformation matrix T_ee2cam.
        """
        if not hasattr(self.env, "sim"):
            return None

        try:
            sim = self.env.sim
            robot = self.env.robots[0]

            # Get end-effector pose in world frame
            arm_name = list(robot.eef_site_id.keys())[0]
            eef_site_id = robot.eef_site_id[arm_name]

            eef_pos = sim.data.site_xpos[eef_site_id]
            eef_mat = sim.data.site_xmat[eef_site_id].reshape(3, 3)

            T_world_ee = np.eye(4)
            T_world_ee[:3, :3] = eef_mat
            T_world_ee[:3, 3] = eef_pos

            # Get camera pose in world frame
            cam_id = sim.model.camera_name2id(camera_id)
            cam_pos = sim.data.cam_xpos[cam_id]
            cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = cam_mat
            T_world_cam[:3, 3] = cam_pos

            # MuJoCo camera has Z-axis pointing away from the scene (backward).
            # OpenCV camera (used by point cloud generation) has Z-axis pointing into the scene (forward),
            # and Y-axis pointing down.
            # We rotate the MuJoCo camera pose by 180 degrees around the X-axis to match OpenCV convention.
            R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            T_flip = np.eye(4)
            T_flip[:3, :3] = R_flip
            T_world_cam = T_world_cam @ T_flip

            # T_ee_cam = T_ee_world * T_world_cam = inv(T_world_ee) * T_world_cam
            T_ee_cam = np.linalg.inv(T_world_ee) @ T_world_cam

            # Apply EE correction to match real robot convention
            T_ee_cam[:3, :3] = self._ee_correction_matrix.T @ T_ee_cam[:3, :3]
            T_ee_cam[:3, 3] = self._ee_correction_matrix.T @ T_ee_cam[:3, 3]

            return T_ee_cam
        except Exception:
            return None

    def get_image(
        self,
        camera_id: str,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> Any:
        """Get an image from a simulated camera.

        Args:
            camera_id: Camera identifier (e.g., 'frontview').
            rgb: Whether to get RGB image (default True).
            depth: Whether to get depth image.
            segmentation: Whether to get segmentation (not implemented).

        Returns:
            RGB image as numpy array, or None if unavailable.
        """

        if segmentation:
            # TODO: implement segmentation retrieval
            raise NotImplementedError(
                "RobosuiteSimEnv.get_image: segmentation not implemented."
            )

        suffix = "_image"
        if depth:
            suffix = "_depth"

        # Robosuite naming convention
        obs_key = camera_id + suffix

        if obs_key and self._last_obs is not None and obs_key in self._last_obs:
            img = self._last_obs[obs_key]
            if depth and img is not None:
                # MuJoco uses normalized depth, that ha sto be converted back to meters
                # Convert normalized depth to meters
                # z = znear / (1 - d * (1 - znear / zfar))
                sim = self.env.sim
                extent = sim.model.stat.extent
                znear = sim.model.vis.map.znear * extent
                zfar = sim.model.vis.map.zfar * extent

                # Avoid division by zero if d is exactly 1/(1-znear/zfar) which shouldn't happen for valid depth
                # But d is in [0, 1]

                # img is (H, W, 1)
                d = img

                # Optimization: precompute constant
                c = 1.0 - (znear / zfar)
                z_meters = znear / (1.0 - d * c)
                return z_meters

            return img
        return None

    def close(self) -> None:
        """Close the simulation environment and release resources."""
        try:
            self.env.close()
        except Exception:
            pass
