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
    """Agent that moves between 4 predefined poses in a loop to test OscPoseTargetController.

    This agent implements a state machine to:
    1. Move to a target pose until it is reached.
    2. Wait for a specified duration (2 seconds).
    3. Cycle to the next pose in a loop of 4 predefined poses.
    """

    def __init__(
        self,
        wait_time: float = 0.0,
    ) -> None:
        """Initialize the OscPoseTargetDemoAgent.

        Args:
            wait_time: Time to wait (seconds) at each pose.
        """
        super().__init__(action_type="OSC_POSE")
        self.wait_time = wait_time


        # Define poses as a grid of evenly spaced points on a half sphere above the base point
        radius = 0.2  # Sphere radius (meters)
        
        base_pose = np.array([0.5, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0])


        num_theta = 4  # Number of steps for polar angle (from 0 to pi/2)
        num_phi = 7    # Number of steps for azimuthal angle (from 0 to 2*pi)
        self.poses = []
        for t in range(num_theta-1):
            theta = (np.pi / 2) * t / (num_theta - 1)  # 0 to pi/2
            for p in range(num_phi):
                phi = 2 * np.pi * p / num_phi  # 0 to 2*pi
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                pose = base_pose.copy()
                pose[0] += x  # x
                pose[1] += y  # y
                pose[2] += z  # z

                base_point = base_pose[:3].copy()

                pose = base_pose.copy()
                pose[:3] += [x, y, z]

                direction = base_point - pose[:3]   # point back toward base
                pose[3:7] = self.quat_z_points_keep_y_horizontal(direction)
                #pose[3:7] = self.quat_z_points_only(direction)



                self.poses.append(pose)

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
        y = np.cross(z, world_up)   # this is guaranteed orthogonal to world_up => horizontal
        if np.linalg.norm(y) < 1e-8:
            # z parallel to world_up (pointing straight up/down): y is undefined -> pick a fixed horizontal y
            y = np.array([0.0, 1.0, 0.0])
        y /= np.linalg.norm(y)

        # Right-hand rule to get x
        x = np.cross(y, z)
        x /= np.linalg.norm(x)

        R = np.column_stack([x, y, z])  # columns are EE axes in world frame

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
        """
        Construct a quaternion whose z-axis points along `direction`.
        Rotation around z is chosen to be as close as possible to `reference_up`.
        """

        z = direction.astype(float)
        z /= np.linalg.norm(z)

        # Make reference_up orthogonal to z
        up = reference_up.astype(float)
        up -= np.dot(up, z) * z

        if np.linalg.norm(up) < 1e-8:
            # direction parallel to reference_up -> choose arbitrary orthogonal x
            up = np.array([1.0, 0.0, 0.0])
            up -= np.dot(up, z) * z

        x = up / np.linalg.norm(up)
        y = np.cross(z, x)

        R = np.column_stack([x, y, z])  # EE axes in world frame

        q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
        q /= np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        return q


    def _init_osc_controller(self) -> None:
        target_pose = self.poses[self.current_pose_idx]
        # command is [x, y, z, qx, qy, qz, qw, gripper]
        # We'll keep gripper open (-1.0)
        position_tolerance: float = 0.1   # 3cm (tune)
        rotation_tolerance: float = 0.1 # 

        command = np.concatenate([target_pose, [-1.0]])
        self.osc_controller = OscPoseTargetController(
            target_pose=command,
            threshold_reach=position_tolerance,
            threshold_rotation=rotation_tolerance,
            # optional: stop fighting orientation so hard
            #kp_rot=1.5,
            #ki_rot=2.0,
            # optional: reduce integral action kicking in far away
            #i_zone=min(0.02, position_tolerance),
        )



    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate robot actions based on the current state machine."""
        # Get current pose for OSC control
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
                # Re-initialize controller for new target
                self._init_osc_controller()
                # Recalculate action for new target
                action, is_finished = self.osc_controller.calculate_action(current_pose)
        
        #logger.info(f"Target Pose: {self.poses[self.current_pose_idx]}, Current Pose: {current_pose}, Difference: {current_pose - self.poses[self.current_pose_idx]}")

        logger.debug(f"Action: {action}")

        metadata = {
            "action_type": self.action_type,
            "state": self.state,
            "target_pose_idx": self.current_pose_idx,
            "is_finished": is_finished,
        }

        return action, metadata

    def reset(self) -> None:
        """Reset the agent state."""
        self.current_pose_idx = 0
        self.state = "MOVING"
        self.wait_start_time = None
        self.osc_controller = None
