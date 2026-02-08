"""Base camera interface for vision sensors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseCamera(ABC):
    """Abstract base class for camera sensors.

    All camera implementations should inherit from this class and implement
    the required methods for capturing and managing camera streams.

    Attributes:
        name: Unique identifier for this camera instance.
        is_started: Whether the camera stream is currently active.
    """

    _instance_count = 0

    def __init__(self, name: str | None = None) -> None:
        """Initialize the camera.

        Args:
            name: Optional custom name for the camera. If not provided,
                a default name will be generated as 'camera_N'.
        """
        BaseCamera._instance_count += 1
        self.name = name if name is not None else f"camera_{BaseCamera._instance_count}"
        self.is_started = False

    @abstractmethod
    def start(self) -> None:
        """Start the camera device and begin streaming."""
        self.is_started = True

    @abstractmethod
    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        # TODO maybe make this interface different, allowing for frame, camera or segmentation
        """Capture a frame from the camera.

        Returns:
            Tuple of (color_image, depth_image) where depth may be None.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera stream."""
        self.is_started = False

    def release(self) -> None:
        """Release camera resources and clean up."""
        self.stop()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.release()
