"""Teleoperation agent for controlling the robot via human input devices."""

from typing import Any

import numpy as np

from franka_pipeline.agents.agent import Agent
from franka_pipeline.input_controllers.input_controller import InputController


# TODO: currently we are primarily interested in teleoperation for the EE-pose. One could think about a teleoperation agent for joint actions
class TeleoperationAgent(Agent):
    """Agent for teleoperation control of the robot.

    Uses human input devices (e.g., SpaceMouse, PS4 controller) to control
    the robot's end-effector pose via the OSC_POSE interface.
    """

    def __init__(self, input_controller: InputController) -> None:
        """Initialize the teleoperation agent.

        Args:
            input_controller: The input device controller to use for
                receiving human teleoperation commands.
        """
        super().__init__(action_type="OSC_POSE")
        self.input_controller = input_controller
        self.input_controller.connect()

    def act(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate action from teleoperation input.

        Args:
            robot_state: Current robot state (unused for teleoperation).
            observation: Sensor observations (unused for teleoperation).
            instruction: Task instruction (unused for teleoperation).

        Returns:
            Tuple of (action, metadata) where action combines translation,
            rotation, and gripper commands from the input controller.
        """
        control = self.input_controller.get_control()

        action = np.concatenate(
            [control["translation"], control["rotation"], [control["gripper"]]]
        )

        metadata: dict[str, Any] = {"action_type": self.action_type}
        metadata.update(control.get("other_controls", {}))

        return action, metadata

    def reset(self) -> None:
        """Reset the input controller state."""
        self.input_controller.reset()
