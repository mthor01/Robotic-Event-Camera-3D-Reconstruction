"""Robosuite simulation camera interface."""

from typing import Any

import cv2
import numpy as np

from .base_camera import BaseCamera

from franka_pipeline.sim.robosuite_env import RobosuiteSimEnv


class RobosuiteCamera(BaseCamera):
    """Camera wrapper for robosuite simulation environment.

    Retrieves rendered images from a robosuite environment's camera views.

    Attributes:
        sim_env: The robosuite simulation environment.
        camera_id: Identifier for the camera view to capture.
    """

    def __init__(
        self, sim_env: RobosuiteSimEnv, camera_id: str, name: str | None = None
    ) -> None:
        """Initialize the robosuite camera.

        Args:
            sim_env: Robosuite simulation environment instance.
            camera_id: Camera view ID (e.g., 'frontview', 'robot0_eye_in_hand').
            name: Optional camera name. Defaults to camera_id.
        """
        super().__init__(name=name or camera_id)
        self.sim_env = sim_env
        self.camera_id = camera_id

    def start(self) -> None:
        """Mark camera as started (no-op for simulation)."""
        self.is_started = True

    def get_frames(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Get current camera frame from simulation.

        Returns:
            Tuple of (color_image, depth_image). Color image is in BGR format.

        Raises:
            RuntimeError: If image retrieval fails.
        """
        img = self.sim_env.get_image(self.camera_id, rgb=True)
        if img is None:
            raise RuntimeError(
                f"RobosuiteCamera: failed to get image for {self.camera_id}"
            )
        # Convert RGB to BGR for consistency with real cameras
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        depth = self.sim_env.get_image(self.camera_id, depth=True)

        if depth is not None:
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth.squeeze(2)
            # Convert meters to mm
            depth = depth * 1000.0

        return img, depth

    def get_intrinsics(self) -> dict[str, float] | None:
        """Get camera intrinsics.

        Returns:
            Dictionary with fx, fy, cx, cy.
        """
        return self.sim_env.get_camera_intrinsics(self.camera_id)

    def get_extrinsics(self) -> np.ndarray | None:
        """Get camera extrinsics (T_ee2cam).

        Returns:
            4x4 transformation matrix T_ee2cam.
        """
        return self.sim_env.get_camera_extrinsics(self.camera_id)

    def stop(self) -> None:
        """Mark camera as stopped (no-op for simulation)."""
        self.is_started = False
