"""Agent that holds the arm at its current position for a fixed duration.

Intended for the light-check recording workflow: the arm stays completely
still while the cameras record a screen that periodically switches between
black and white, allowing temporal alignment between RealSense and event
cameras to be verified.
"""

import time
from typing import Any, Dict, Tuple

import numpy as np

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

logger = get_logger(__name__)


class LightCheckAgent(Agent):
    """Hold arm at the pose it has when first called, for ``duration_sec`` seconds.

    On the first ``act()`` call the current end-effector pose is latched as the
    target and an ``OscPoseTargetController`` is created to actively servo
    back to that position each step, countering gravity-induced drift.

    ``state="WAITING"`` and ``target_pose_idx=0`` are returned from the very
    first call, immediately satisfying the condition in my_main.py that
    triggers ``sync_server.send_start()``.  After the configured duration
    elapses, ``light_check_complete=True`` is added to metadata.

    Parameters
    ----------
    duration_sec : float
        How long to hold position before signalling completion (default 40 s).
    """

    def __init__(self, duration_sec: float = 40.0) -> None:
        super().__init__(action_type="OSC_POSE")
        self.duration_sec = duration_sec
        self._start_time: float | None = None
        self._complete = False
        self._osc: OscPoseTargetController | None = None
        logger.info(f"LightCheckAgent: will hold position for {duration_sec:.1f} s")

    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        now = time.time()

        # Latch current pose as the hold target on the first step
        if self._osc is None:
            pos = np.asarray(robot_state["osc_position"]).flatten()
            rot = np.asarray(robot_state["osc_rotation_quaternion"]).flatten()
            # OscPoseTargetController expects [x, y, z, qx, qy, qz, qw, gripper]
            target = np.concatenate([pos, rot, [-1.0]])
            self._osc = OscPoseTargetController(
                target_pose=target,
                threshold_reach=1e-3,
                threshold_rotation=1e-3,
            )
            self._start_time = now
            logger.info(f"LightCheckAgent: latched hold position {pos}, timer started")

        if not self._complete and (now - self._start_time) >= self.duration_sec:
            self._complete = True
            logger.info(
                f"LightCheckAgent: {self.duration_sec:.1f} s elapsed — signalling complete"
            )

        # Actively servo back to the latched position
        pos = np.asarray(robot_state["osc_position"]).flatten()
        rot = np.asarray(robot_state["osc_rotation_quaternion"]).flatten()
        current = np.concatenate([pos, rot])
        action, _ = self._osc.calculate_action(current)

        meta: Dict[str, Any] = {
            "state": "WAITING",
            "target_pose_idx": 0,
            "light_check_complete": self._complete,
        }
        return action, meta

    def reset(self) -> None:
        self._start_time = None
        self._complete = False
        self._osc = None
