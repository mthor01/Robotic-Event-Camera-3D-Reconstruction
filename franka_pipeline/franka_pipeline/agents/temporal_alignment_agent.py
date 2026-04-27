"""Agent that rotates the end-effector about the Z axis for temporal alignment checks.

The robot holds a fixed position pointing straight down and alternates
between +20° and −20° Z-rotation.  Each reversal produces a sharp
change in event-camera activity that can be compared against the pose
timestamps to verify temporal alignment."""

import time
from typing import Any, Dict, Tuple

import numpy as np

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

logger = get_logger(__name__)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-last quaternions [qx, qy, qz, qw]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Quaternion (scalar-last) from axis + angle (radians)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle / 2)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle / 2)])


class TemporalAlignmentAgent(Agent):
    """Rotate end-effector about Z while holding a fixed position.

    Parameters
    ----------
    center_x, center_y, center_z : float
        Fixed EE position (robot base frame, metres).
    rotation_deg : float
        Half-range of Z rotation in degrees.  The EE swings between
        ``-rotation_deg`` and ``+rotation_deg``.
    num_sweeps : int
        Number of **full** back-and-forth rotation sweeps.
    wait_at_end : float
        Seconds to dwell at each turning point (default 0).
    """

    def __init__(
        self,
        center_x: float = 0.45,
        center_y: float = 0.0,
        center_z: float = 0.25,
        rotation_deg: float = 20.0,
        num_sweeps: int = 40,
        wait_at_end: float = 0.0,
    ) -> None:
        super().__init__(action_type="OSC_POSE")

        self.center = np.array([center_x, center_y, center_z])
        self.rotation_rad = np.deg2rad(rotation_deg)
        self.num_sweeps = num_sweeps
        self.wait_at_end = wait_at_end

        # Base orientation: pointing straight down (180° around X, scalar-last)
        self.down_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self._waypoints = self._build_waypoints()
        self._wp_idx = 0

        self.state = "MOVING"
        self._wait_start: float | None = None
        self._osc: OscPoseTargetController | None = None
        self._complete = False

        logger.info(
            f"TemporalAlignmentAgent: center={self.center}, "
            f"rotation=±{rotation_deg}°, sweeps={self.num_sweeps}, "
            f"waypoints={len(self._waypoints)}"
        )

    # ------------------------------------------------------------------
    def _build_waypoints(self) -> list[np.ndarray]:
        """Create alternating +θ / −θ Z-rotation target poses."""
        wps: list[np.ndarray] = []
        for i in range(self.num_sweeps * 2):
            sign = 1.0 if (i % 2 == 0) else -1.0
            angle = sign * self.rotation_rad
            z_rot = _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)
            quat = _quat_multiply(z_rot, self.down_quat)
            pose = np.concatenate([self.center, quat])  # 7-vec
            wps.append(pose)
        return wps

    def _set_target(self) -> None:
        target = self._waypoints[self._wp_idx]
        cmd = np.concatenate([target, [-1.0]])  # gripper open
        self._osc = OscPoseTargetController(
            target_pose=cmd,
            threshold_reach=0.02,      # 20 mm (default 5 mm)
            threshold_rotation=0.15,   # ~8.6° (default ~2.9°)
        )

    # ------------------------------------------------------------------
    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pos = robot_state["osc_position"]
        rot = robot_state["osc_rotation_quaternion"]
        current = np.concatenate([pos.flatten(), rot.flatten()]).astype(float)

        if self._osc is None:
            self._set_target()

        action, reached = self._osc.calculate_action(current)

        meta: Dict[str, Any] = {
            "state": self.state,
            "waypoint_idx": self._wp_idx,
            "target_pose_idx": self._wp_idx,
            "temporal_alignment_complete": self._complete,
        }

        if self.state == "MOVING":
            if reached:
                logger.info(
                    f"Reached waypoint {self._wp_idx}/{len(self._waypoints)-1}"
                )
                meta["state"] = "WAITING"
                meta["target_pose_idx"] = self._wp_idx
                if self.wait_at_end > 0:
                    self.state = "WAITING"
                    self._wait_start = time.time()
                else:
                    self._advance()

        elif self.state == "WAITING":
            if time.time() - self._wait_start >= self.wait_at_end:
                self._advance()
                meta["state"] = self.state

        elif self.state == "COMPLETE":
            pass

        meta["temporal_alignment_complete"] = self._complete
        return action, meta

    # ------------------------------------------------------------------
    def _advance(self) -> None:
        self._wp_idx += 1
        if self._wp_idx >= len(self._waypoints):
            self.state = "COMPLETE"
            self._complete = True
            logger.info("Temporal alignment motion complete.")
        else:
            self.state = "MOVING"
            self._set_target()
