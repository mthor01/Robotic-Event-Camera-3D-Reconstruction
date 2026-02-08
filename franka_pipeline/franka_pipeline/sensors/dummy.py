"""Dummy camera interface for testing."""

import numpy as np

from .base_camera import BaseCamera


class DummyCamera(BaseCamera):
    """A dummy camera that returns black images.

    Useful for debugging or testing when a real camera is not available.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        name: str | None = None,
        enable_depth: bool = False,
    ) -> None:
        """Initialize the dummy camera.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            name: Optional custom name for the camera.
            enable_depth: Whether to generate a dummy depth image.
        """
        super().__init__(name)
        self.width = width
        self.height = height
        self.enable_depth = enable_depth

    def start(self) -> None:
        """Start the dummy camera."""
        super().start()

    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Generate black frames.

        Returns:
            Tuple of (color_image, depth_image).
        """
        if not self.is_started:
            return None, None

        color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_image = None
        if self.enable_depth:
            depth_image = np.zeros((self.height, self.width), dtype=np.uint16)

        return color_image, depth_image

    def stop(self) -> None:
        """Stop the dummy camera."""
        super().stop()
