"""SpaceMouse input controller for robot teleoperation.

This module provides a driver for 3Dconnexion SpaceMouse devices,
translating 6-DOF mouse inputs into robot control commands.

Based on the deoxys SpaceMouse driver implementation. (https://github.com/UT-Austin-RPL/deoxys_control/blob/main/deoxys/deoxys/utils/io_devices/spacemouse.py)
"""

import threading
import time
from collections import namedtuple
from typing import Any

import hid
import numpy as np

from deoxys.utils.transform_utils import rotation_matrix
from franka_pipeline.input_controllers import InputController
from franka_pipeline.logging import get_logger

logger = get_logger(__name__)

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

# SpaceMouse HID protocol axis specifications
SPACE_MOUSE_SPEC: dict[str, AxisSpec] = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}


def to_int16(y1: int, y2: int) -> int:
    """Convert two 8-bit bytes to a signed 16-bit integer.

    Args:
        y1: Low byte.
        y2: High byte.

    Returns:
        Signed 16-bit integer.
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(
    x: int, axis_scale: float = 350.0, min_v: float = -1.0, max_v: float = 1.0
) -> float:
    """Normalize raw HID readings to target range.

    Args:
        x: Raw reading from HID.
        axis_scale: Scaling factor for mapping raw input value.
        min_v: Minimum limit after scaling.
        max_v: Maximum limit after scaling.

    Returns:
        Clipped, scaled input from HID.
    """
    return min(max(x / axis_scale, min_v), max_v)


def convert(b1: int, b2: int) -> float:
    """Convert SpaceMouse message bytes to scaled control value.

    Args:
        b1: First byte.
        b2: Second byte.

    Returns:
        Scaled value from SpaceMouse message.
    """
    return scale_to_control(to_int16(b1, b2))


class SpaceMouse:
    """Low-level driver class for SpaceMouse with HID library.

    Provides 6-DOF input from 3Dconnexion SpaceMouse devices.
    Use hid.enumerate() to find vendor/product IDs for your device.

    Attributes:
        pos_sensitivity: Scaling factor for position commands.
        rot_sensitivity: Scaling factor for rotation commands.
    """

    def __init__(
        self,
        vendor_id: int = 9583,
        product_id: int = 50735,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
    ) -> None:
        """Initialize the SpaceMouse driver.

        Args:
            vendor_id: HID device vendor ID.
            product_id: HID device product ID.
            pos_sensitivity: Position command scaling factor.
            rot_sensitivity: Rotation command scaling factor.
        """
        logger.info("Opening SpaceMouse device")
        self.device = hid.device()
        self.device.open(vendor_id, product_id)

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        logger.info(f"Manufacturer: {self.device.get_manufacturer_string()}")
        logger.info(f"Product: {self.device.get_product_string()}")

        # 6-DOF state variables
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0

        self._display_controls()

        self.gripper_state = False
        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _display_controls() -> None:
        """Log SpaceMouse control instructions."""
        controls = [
            ("Control", "Command"),
            ("Right button", "reset simulation"),
            ("Left button (hold)", "close gripper"),
            ("Move mouse laterally", "move arm horizontally in x-y plane"),
            ("Move mouse vertically", "move arm vertically"),
            ("Twist mouse about an axis", "rotate arm about corresponding axis"),
            ("ESC", "quit"),
        ]
        logger.info("")
        for cmd, info in controls:
            logger.info(f"{cmd:<30}\t{info}")
        logger.info("")

    def _reset_internal_state(self) -> None:
        """Reset internal state of controller, except for the reset signal."""
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
        self._control = np.zeros(6).tolist()
        self.gripper_state = False

    def start_control(self) -> None:
        """Enable the controller for receiving commands."""
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self) -> dict[str, Any]:
        """Get the current state of the 3D mouse.

        Returns:
            Dictionary containing:
                - dpos: Position delta [x, y, z]
                - rotation: Absolute orientation matrix
                - raw_drotation: Raw rotation delta [roll, pitch, yaw]
                - grasp: Gripper state (True = closed)
                - reset: Reset button state
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def _run(self) -> None:
        """Background listener thread that continuously reads SpaceMouse data."""
        while True:
            d = self.device.read(13)
            if d is not None and self._enabled:
                if d[0] == 1:  # 6-DoF sensor readings
                    self.y = convert(d[1], d[2])
                    self.x = convert(d[3], d[4])
                    self.z = convert(d[5], d[6]) * -1.0

                    self.roll = convert(d[7], d[8])
                    self.pitch = convert(d[9], d[10])
                    self.yaw = convert(d[11], d[12])

                    self._control = [
                        self.x,
                        self.y,
                        self.z,
                        self.roll,
                        self.pitch,
                        self.yaw,
                    ]

                elif d[0] == 3:  # Side button readings
                    if d[1] == 1:  # Left button pressed
                        self.gripper_state = True
                    elif d[1] == 0:  # Left button released
                        self.gripper_state = False
                    elif d[1] == 2:  # Right button (reset)
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()

    @property
    def control(self) -> np.ndarray:
        """Current 6-DoF control values [x, y, z, roll, pitch, yaw]."""
        return np.array(self._control)

    @property
    def control_gripper(self) -> bool:
        """Current gripper state (True = closed)."""
        return self.gripper_state


class SpaceMouseController(InputController):
    """Input controller for 3Dconnexion SpaceMouse devices.

    Translates SpaceMouse 6-DOF inputs into robot control commands
    compatible with the teleoperation agent.

    Attributes:
        device: The underlying SpaceMouse driver instance.
    """

    # Other product_id for different SpaceMouse model: 50734
    def __init__(self, vendor_id: int = 9583, product_id: int = 50746) -> None:
        """Initialize the SpaceMouse controller.

        Args:
            vendor_id: HID device vendor ID (default: 3Dconnexion).
            product_id: HID device product ID (varies by model).
        """
        super().__init__(name="SpaceMouse")
        self.device = SpaceMouse(vendor_id=vendor_id, product_id=product_id)
        self.device.start_control()

    def connect(self) -> None:
        """Mark the controller as connected."""
        super().connect()

    def get_control(self) -> dict[str, Any]:
        """Get current control state from the SpaceMouse.

        Returns:
            Dictionary with keys:
                - translation: Scaled position delta [x, y, z]
                - rotation: Reoriented rotation delta [roll, pitch, yaw]
                - gripper: 1 for close, -1 for open
                - other_controls: {end_episode: bool}
        """
        state = self.device.get_controller_state()

        dpos = state["dpos"]
        raw_drotation = state["raw_drotation"]
        grasp = state["grasp"]
        reset = state["reset"]

        # Reorientation and scaling copied from deoxys.utils.input_utils.input2action
        drotation = raw_drotation[[1, 0, 2]]
        drotation[2] = -drotation[2]
        drotation *= 75
        dpos = dpos * 200
        # TODO We could also implement continuous grasping on button press, by changing the SpaceMouse class, but after some preliminary testing this does not feel good.
        gripper = 1 if grasp else -1

        return {
            "translation": dpos,
            "rotation": drotation,
            "gripper": gripper,
            "other_controls": {"end_episode": bool(reset)},
        }

    def reset(self) -> None:
        """Reset the SpaceMouse controller state."""
        self.device.start_control()

    def disconnect(self) -> None:
        """Disconnect from the SpaceMouse device."""


if __name__ == "__main__":
    space_mouse = SpaceMouse(product_id=50770)
    for _ in range(100):
        logger.info(f"{space_mouse.control}, {space_mouse.control_gripper}")
        time.sleep(0.02)
