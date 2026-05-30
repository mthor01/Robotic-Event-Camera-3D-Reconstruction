"""Agent wrapper for adding keyboard-based episode controls.

This wrapper adds keyboard shortcuts for episode control ("e": end, "r": reset, "q": quit)
to any wrapped agent.
"""

from typing import Any

import numpy as np
from pynput import keyboard

from franka_pipeline.agents.agent import Agent


class EpisodeControlWrapperAgent(Agent):
    """Wrapper that adds keyboard controls to any agent.

    Provides standard keyboard shortcuts:
        - 'e': End the current episode
        - 'r': Reset the current episode
        - 'q': Quit the application

    These controls are merged into the agent's metadata output.

    Example:
        >>> agent = AgentEpisodeControlWrapper(TeleoperationAgent(SpaceMouseController()))
        >>> action, metadata = agent.act(robot_state, observation, instruction)
        >>> if metadata.get("quit"):
        ...     break
    """

    def __init__(self, wrapped_agent: Agent) -> None:
        """Initialize the agent episode control wrapper.

        Args:
            wrapped_agent: The underlying agent to wrap.
        """
        super().__init__(action_type=wrapped_agent.action_type)
        self.wrapped_agent = wrapped_agent

        self._keyboard_control: dict[str, bool] = {}
        self._reset_keyboard_state()

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def act(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate action from wrapped agent with keyboard control overrides.

        Args:
            robot_state: Current robot state containing joint positions,
                velocities, end-effector pose, etc.
            observation: Sensor observations (e.g., camera images by name).
            instruction: Natural language instruction for the task.

        Returns:
            Tuple of (action, metadata) where metadata includes keyboard
            controls merged in. Keyboard controls take precedence (OR logic).
        """
        action, metadata = self.wrapped_agent.act(robot_state, observation, instruction)

        # Merge keyboard controls into metadata
        for key, value in self._keyboard_control.items():
            if value:
                metadata[key] = True

        return action, metadata

    def reset(self) -> None:
        """Reset both the wrapped agent and keyboard state."""
        self.wrapped_agent.reset()
        self._reset_keyboard_state()

    def stop(self) -> None:
        """Stop the wrapped agent and cleanup resources."""
        if hasattr(self.wrapped_agent, "stop"):
            self.wrapped_agent.stop()
        if self._listener is not None:
            self._listener.stop()

    def _reset_keyboard_state(self) -> None:
        """Reset the keyboard control state."""
        self._keyboard_control = {
            "reset_episode": False,
            "end_episode": False,
            "quit": False,
            "trigger_grasp": False,
        }

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle keyboard key press events.

        Controls:
            Ctrl+E  – end episode
            Ctrl+R  – reset episode
            Ctrl+Q  – quit
            g       – trigger grasp
            h       – trigger execution
        """
        # Disable episode controls if the wrapped agent is currently taking terminal input # TODO examine this interface again, maybe make it more general
        if getattr(self.wrapped_agent, "is_input_active", False):
            return

        try:
            # Ctrl+key combinations produce key.char values in the range \x01-\x1a
            if hasattr(key, "char") and key.char is not None:
                if key.char == "\x11":      # Ctrl+Q
                    self._keyboard_control["quit"] = True
                elif key.char == "\x12":    # Ctrl+R
                    self._keyboard_control["reset_episode"] = True
                elif key.char == "\x05":    # Ctrl+E
                    self._keyboard_control["end_episode"] = True
                elif key.char == "g":
                    self._keyboard_control["trigger_grasp"] = True
                    if hasattr(self.wrapped_agent, "_trigger_grasp"):
                        self.wrapped_agent._trigger_grasp = True
                elif key.char == "h":
                    if hasattr(self.wrapped_agent, "_trigger_execution"):
                        self.wrapped_agent._trigger_execution = True
        except AttributeError:
            pass

    def stop(self) -> None:
        """Stop the keyboard listener.

        Call this method when the agent is no longer needed to clean up
        the keyboard listener thread.
        """
        self._listener.stop()
