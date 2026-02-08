"""Abstract simulation environment interface.

This module defines the abstract base class for simulation environments,
providing a backend-agnostic API for control, observation, and images.
"""

# TODO: I don't like, that this is a duplication of the simulation interface already contained in the simulation software (e.g. robosuite, Isaac Sim).

from abc import ABC, abstractmethod
from typing import Any


class SimEnv(ABC):
    """Abstract interface for a simulator environment backend.

    Concrete implementations wrap a specific simulator (e.g., robosuite, Isaac Sim)
    and expose a minimal, backend-agnostic API for control, observation, and images.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial observation dictionary.
        """

    @abstractmethod
    def step(self, action: Any) -> dict[str, Any]:
        """Execute one environment step.

        Args:
            action: Control action to apply.

        Returns:
            Observation dictionary after step.
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get the current robot state.

        Returns:
            Dictionary containing robot state information.
        """

    @abstractmethod
    def get_image(
        self,
        camera_id: str,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> Any:
        """Get camera image from the simulation.

        Args:
            camera_id: Identifier for the camera view.
            rgb: Whether to return RGB image.
            depth: Whether to return depth image.
            segmentation: Whether to return segmentation mask.

        Returns:
            Requested image data.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the simulation environment and release resources."""
