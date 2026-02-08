"""Base classes for all robot control agents."""

from abc import ABC, abstractmethod
from time import time
from typing import Any, Optional

import numpy as np

from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

logger = get_logger(__name__)


class Agent(ABC):
    """Abstract base class for all robot control agents.

    Agents receive robot state, observations, and instructions, and produce
    actions for the robot to execute.
    """

    # TODO do i want to have this abstraction for the action type at this place? -> probably yes
    def __init__(self, action_type: str | None = None) -> None:
        """Initialize the agent.

        Args:
            action_type: Type of action space. Supported values include:
                "OSC_POSE", "OSC_POSITION", "OSC_YAW", "OSC_POSE_TARGET",
                "JOINT_IMPEDANCE", "JOINT_POSITION", "CARTESIAN_VELOCITY".
        """
        self.action_type = action_type

    def __call__(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str = "",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Convenience method to call act."""
        return self.act(robot_state, observation, instruction)

    @abstractmethod
    def act(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate an action based on current state and observations.

        Args:
            robot_state: Current robot state containing joint positions,
                velocities, end-effector pose, etc.
            observation: Sensor observations (e.g., camera images by name).
            instruction: Natural language instruction for the task.

        Returns:
            Tuple of (action, metadata) where action is the combined
            robot arm and gripper command, and metadata contains
            additional information like action_type or confidence.
        """
        # return action, metadata

    def reset(self) -> None:
        """Reset the agent state. Called between episodes."""


class ReplayAgent(Agent):
    """Agent that replays recorded trajectories."""

    # TODO: Implement trajectory replay functionality


class DummyAgentOscPose(Agent):
    """Dummy agent implementing sinusoidal left-right movement.

    Useful for testing the OSC_POSE controller interface.
    """

    def __init__(self, amplitude: float = 0.4, duration: float = 5.0) -> None:
        """Initialize the dummy agent.

        Args:
            amplitude: Maximum displacement amplitude.
            duration: Period of the sinusoidal movement in seconds.
        """
        super().__init__(action_type="OSC_POSE")
        self.amplitude = amplitude
        self.duration = duration
        self.start_time: float | None = None

    def act(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.start_time is None:
            self.start_time = time()
        elapsed = time() - self.start_time

        # Simple left-right sinusoidal movement in the y direction
        y_offset = self.amplitude * np.sin(2 * np.pi / self.duration * elapsed)

        robot_action = np.zeros(6)
        robot_action[1] = y_offset

        # Alternate gripper open/close each half period
        gripper_action = -0.5 if elapsed % self.duration < self.duration / 2 else 0.5

        action = np.concatenate([robot_action, [gripper_action]])
        metadata = {"action_type": self.action_type}
        return action, metadata

    def reset(self) -> None:
        """Reset the agent state for a new episode."""
        self.start_time = None


class DummyAgentDoNothing(Agent):
    """Dummy agent that outputs zero actions."""

    def __init__(self) -> None:
        """Initialize the dummy agent.

        Args:
            action_type: Type of action space to output.
        """
        super().__init__(action_type="OSC_POSE")

    def act(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        action = np.zeros(7, dtype=np.float32)

        metadata = {"action_type": self.action_type}
        return action, metadata
