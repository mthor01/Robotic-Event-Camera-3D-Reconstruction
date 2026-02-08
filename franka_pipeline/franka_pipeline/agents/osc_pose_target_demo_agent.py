# TODO: This is heavily AI-coded, future review is needed

import time
from typing import Any, Dict, Tuple

import numpy as np

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

logger = get_logger(__name__)


class OscPoseTargetDemoAgent(Agent):
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

        # Define 4 poses [x, y, z, qx, qy, qz, qw]
        # Using a safe base pose and creating a small square movement
        base_pose = np.array([0.45, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])
        self.poses = [
            base_pose.copy() + np.array([-0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            base_pose + np.array([0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            base_pose + np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            base_pose + np.array([-0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]

        self.current_pose_idx = 0
        self.state = "MOVING"  # States: MOVING, WAITING
        self.wait_start_time: float | None = None
        self.osc_controller: OscPoseTargetController | None = None

    def _init_osc_controller(self) -> None:
        target_pose = self.poses[self.current_pose_idx]
        # command is [x, y, z, qx, qy, qz, qw, gripper]
        # We'll keep gripper open (-1.0)
        command = np.concatenate([target_pose, [-1.0]])
        self.osc_controller = OscPoseTargetController(target_pose=command)

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
