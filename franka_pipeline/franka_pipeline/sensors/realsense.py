"""Intel RealSense camera interface."""

import numpy as np
import pyrealsense2 as rs

from franka_pipeline.logging import get_logger

from .base_camera import BaseCamera

logger = get_logger(__name__)


class RealsenseCamera(BaseCamera):
    """Intel RealSense camera (D435, D455, etc.).

    Provides RGB and optional depth streams aligned to the color frame.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fps: Frame rate.
        enable_depth: Whether depth stream is enabled.
    """

    def __init__(
        self,
        serial_number: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        name: str = "realsense",
        enable_depth: bool = True,
    ) -> None:
        """Initialize the RealSense camera.

        Args:
            serial_number: Camera serial number (for multi-camera setups).
            width: Image width in pixels.
            height: Image height in pixels.
            fps: Frame rate.
            name: Camera name.
            enable_depth: Whether to enable depth stream.
        """
        super().__init__(name)

        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_number is not None:
            self.config.enable_device(serial_number)

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            self.config.enable_stream(
                rs.stream.depth, width, height, rs.format.z16, fps
            )
            self.align = rs.align(rs.stream.color)

    def start(self) -> None:
        """Start the RealSense pipeline."""
        if not self.is_started:
            try:
                self.pipeline.start(self.config)
            except RuntimeError as e:
                logger.error(f"Failed to start RealSense camera '{self.name}': {e}")
                logger.error(
                    "This often happens if the requested resolution/FPS is not supported "
                    "Or if the camera is not properly connected via USB3 (read the appropriate sections in the README). "
                    "Note, that the highest possible resolution for depth and color is 1280x720 at 30fps. "
                    "You can enumerate the available settings using `rs-enumerate-devices`"
                )
                raise
            self.is_started = True
            logger.info(
                f"RealSense camera '{self.name}' started at {self.width}x{self.height}@{self.fps}fps"
            )

    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture RGB and depth frames.

        Returns:
            Tuple of (color_image, depth_image). Depth is None if not enabled.

        Raises:
            RuntimeError: If pipeline not started.
        """
        if not self.is_started:
            raise RuntimeError("Pipeline not started. Call start() first.")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to capture frames from RealSense camera. "
                "Are you running the realsense with correct USB3.2 connection and USB3.2 cable? "
                "Maybe just restart the script again. "
            ) from e

        if self.enable_depth:
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
        else:
            color_frame = frames.get_color_frame()
            depth_frame = None

        color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

        return color_image, depth_image

    def get_intrinsics(self) -> dict[str, float]:
        """Get camera intrinsics.

        Returns:
            Dictionary with keys 'fx', 'fy', 'cx', 'cy'.
        """
        if not self.is_started:
            raise RuntimeError("Pipeline not started. Call start() first.")

        profile = self.pipeline.get_active_profile()
        # We align depth to color, so we use color intrinsics
        stream = profile.get_stream(rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()

        return {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
        }

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self.is_started:
            self.pipeline.stop()
            self.is_started = False
            logger.info(f"RealSense camera '{self.name}' stopped.")
