"""Base interface for input controllers."""

from abc import ABC, abstractmethod
from typing import Any


class InputController(ABC):
    """Abstract base class for human input devices.

    Input controllers translate human input (e.g., from a SpaceMouse,
    PS4 controller, or gesture recognition) into robot control commands.

    Attributes:
        name: Identifier for this input controller.
        is_connected: Whether the device is currently connected.
    """

    def __init__(self, name: str) -> None:
        """Initialize the input controller.

        Args:
            name: Name of the input controller.
        """
        self.name = name
        self.is_connected = False

    @abstractmethod
    def connect(self) -> None:
        """Connect to the input device."""
        self.is_connected = True

    @abstractmethod
    def get_control(self) -> dict[str, Any]:
        """Get the current state of the input device.

        Returns:
            Dictionary containing input state with keys:
                - 'translation': 3D translation vector [x, y, z]
                - 'rotation': 3D rotation vector [rx, ry, rz]
                - 'gripper': Gripper command (scalar, negative=open, positive=close)
                - 'other_controls': Dict with additional controls
                    (e.g., end_episode, reset_episode, quit)

        Raises:
            RuntimeError: If called before connect().
        """
        if not self.is_connected:
            raise RuntimeError(f"Input device {self.name} not connected.")

    def reset(self) -> None:
        """Reset the input device state if applicable."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the input device."""
