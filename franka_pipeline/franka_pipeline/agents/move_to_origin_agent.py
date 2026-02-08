import time
from typing import Any, Dict, Tuple

import numpy as np

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

logger = get_logger(__name__)

class MoveToOriginAgent(Agent):
    """Agent that moves the robot arm to the origin and stops."""

    def __init__(
        self,
        origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
        gripper: float = -1.0,
    ) -> None:
        """Initialize the agent.

        Args:
            origin: Target position (default: [0,0,0]).
            orientation: Target orientation quaternion (default: pointing down).
            gripper: Gripper command (default open).
        """
        super().__init__(action_type="OSC_POSE")
        self.target_pose = np.concatenate([origin, orientation])
        self.gripper = gripper
        self.osc_controller = None
        self.state = "MOVING"  # States: MOVING, DONE

    def _init_osc_controller(self) -> None:
        command = np.concatenate([self.target_pose, [self.gripper]])
        self.osc_controller = OscPoseTargetController(target_pose=command)

    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pos = robot_state["osc_position"]
        rot = robot_state["osc_rotation_quaternion"]
        current_pose = np.concatenate([pos.flatten(), rot.flatten()]).astype(float)

        if self.state == "DONE":
            return np.zeros_like(current_pose), {"quit": True, "state": "DONE"}

        if self.osc_controller is None:
            self._init_osc_controller()

        action, is_finished = self.osc_controller.calculate_action(current_pose)

        if self.state == "MOVING":
            if is_finished:
                logger.info("Reached origin. Stopping.")
                self.state = "DONE"
            else:
                logger.debug("Moving to origin.")

        metadata = {
            "action_type": self.action_type,
            "state": self.state,
            "is_finished": is_finished,
        }

        return action, metadata

    def reset(self) -> None:
        self.osc_controller = None
        self.state = "MOVING"
