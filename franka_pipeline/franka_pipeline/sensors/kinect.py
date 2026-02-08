# TODO copied from Marius, unmodified
# TODO make depth image optional
# TODO check if everything is correct (kinect Azure was never actually tried)
"""Kinect camera interface."""
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import numpy as np
import cv2
import os
from .base_camera import BaseCamera
from ..utils import crop_and_resize
from franka_pipeline.logging import get_logger

logger = get_logger(__name__)


class KinectAzureCamera(BaseCamera):
    def __init__(
        self,
        color_resolution=ColorResolution.RES_720P,
        depth_mode=None,  # DepthMode.WFOV_2X2BINNED,
        synchronized=False,
        name="external",
    ):
        """
        Mirrors the RealSense interface using pyk4a.
        """
        self.color_resolution = color_resolution
        self.depth_mode = depth_mode
        self.synchronized = synchronized
        self.k4a = None
        self.is_started = False
        self.name = name

    def start(self):
        """Starts the Azure Kinect device."""
        if not self.is_started:
            config = Config(
                color_resolution=self.color_resolution,
                depth_mode=self.depth_mode,
                synchronized_images_only=self.synchronized,
            )
            self.k4a = PyK4A(config)
            self.k4a.start()
            self.is_started = True
            logger.info("Kinect Azure device started.")

    def get_frames(self):
        """
        Captures a color and depth frame.
        :return: (color_image, depth_image)
        """
        if not self.is_started:
            raise RuntimeError("Device not started. Call start() first.")

        capture = self.k4a.get_capture()
        # Wait for valid frames
        while capture.color is None:
            capture = self.k4a.get_capture()

        # Color comes as BGRA; take first three channels
        color_image = capture.color[:, :, :3]
        # Ensure C-contiguous for OpenCV
        color_image = np.require(color_image, requirements=["C_CONTIGUOUS"])

        # Use the function in get_frames
        color_image = crop_and_resize(color_image, 224, 224)

        # Transformed depth is already aligned
        depth_image = capture.transformed_depth

        return color_image, depth_image

    def stop(self):
        """Stops the Azure Kinect device."""
        if self.is_started and self.k4a is not None:
            self.k4a.stop()
            self.is_started = False
            logger.info("Kinect Azure device stopped.")


# Quick demo
if __name__ == "__main__":
    cam = KinectAzureCamera()
    cam.start()
    try:
        while True:
            color_frame, depth_frame = cam.get_frames()
            cv2.imshow("Color Frame", color_frame)
            cv2.imshow("Depth Frame", depth_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
