"""Multi-camera calibration agent (ZMQ version).

Runs in Docker A (robot docker).  Controls robot movement through calibration
poses and communicates with a recording / calibration script in Docker B
(metavision docker) via ZeroMQ REQ/REP.

Docker B manages *both* cameras (RealSense RGB/depth + event camera) and runs
all calibration math.  This agent only handles:
  - Pose management (load / save / manual selection)
  - Robot movement to calibration poses
  - ZMQ signaling so Docker B knows when to capture

ZMQ protocol (this agent = REQ, recording script = REP):
  1.  Agent  ->  {"cmd": "INIT", "num_poses": N}
      Script <-  {"status": "READY"}

  2.  Agent  ->  {"cmd": "POSE_REACHED", "index": i, "ee_pose": <16 floats>}
      Script <-  {"status": "RGB_CAPTURED"}

  3.  Agent  ->  {"cmd": "WIGGLE_DONE", "index": i}
      Script <-  {"status": "EVENT_CAPTURED"}

  4.  Agent  ->  {"cmd": "ALL_DONE"}
      Script <-  {"status": "CALIBRATION_COMPLETE", "success": bool}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zmq
import msgpack
from pynput import keyboard

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)
from franka_pipeline.utils import pos_rot_to_transformation_matrix

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MultiCameraCalibrationConfig:
    output_dir: Path = Path("camera_data")
    position_tolerance: float = 0.005
    rotation_tolerance: float = 0.02
    settling_time: float = 0.5
    static_event_duration: float = 0.5  # seconds to record events before signalling done
    zmq_endpoint: str = "tcp://localhost:6002"  # REQ socket connects here
    zmq_timeout_s: float = 60.0         # per-message timeout


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class MultiCameraCalibrationAgent(Agent):
    """Agent that moves the robot through calibration poses and signals Docker B.

    Docker B (recording script) handles all camera capture and calibration.

    Constructor args match the original CameraEyeInHandCalibrationAgent:
        config:              Calibration settings.
        calibration_poses:   Pre-defined poses [x, y, z, qx, qy, qz, qw].
        poses_file:          Path to .npy file with poses.
        generate_poses_new:  If True, ignore poses_file and enter manual mode.
        input_controller:    SpaceMouse or other controller for manual mode.
    """

    class State:
        IDLE = "idle"
        MANUAL_POSE_SELECTION = "manual_pose_selection"
        MOVING_TO_POSE = "moving_to_pose"
        SETTLING = "settling"
        SIGNALLING_POSE_REACHED = "signalling_pose_reached"
        STATIC_RECORDING = "static_recording"
        SIGNALLING_WIGGLE_DONE = "signalling_wiggle_done"
        WAITING_CALIBRATION = "waiting_calibration"
        COMPLETE = "complete"

    def __init__(
        self,
        config: MultiCameraCalibrationConfig | None = None,
        calibration_poses: Sequence[np.ndarray] | None = None,
        poses_file: str | Path | None = None,
        generate_poses_new: bool = False,
        input_controller: Any | None = None,
    ) -> None:
        super().__init__(action_type="OSC_POSE")
        self.config = config or MultiCameraCalibrationConfig()
        self.calibration_poses: List[np.ndarray] = (
            list(calibration_poses) if calibration_poses else []
        )

        # State
        self.current_pose_index = 0
        self.state = self.State.IDLE
        self.wait_start_time = 0.0
        self.gripper_action = 0.5

        # OSC controller for moving to targets
        self.osc_controller: OscPoseTargetController | None = None

        # ZMQ REQ socket (lazy init in start())
        self._zmq_ctx: zmq.Context | None = None
        self._zmq_req: zmq.Socket | None = None

        # Keyboard / teleop for manual mode
        self._keyboard_control: Dict[str, bool] = {
            "capture_pose": False,
            "end_selection": False,
            "quit": False,
        }
        self._keyboard_listener: keyboard.Listener | None = None
        self._input_controller = input_controller
        self._teleop_agent: TeleoperationAgent | None = None

        # Pose file handling
        self._poses_file = (
            Path(poses_file) if poses_file and not generate_poses_new else None
        )
        self._use_manual_selection = True
        print(self._poses_file.exists())
        if self._poses_file and self._poses_file.exists():
            self.load_poses_from_file(self._poses_file)
            self._use_manual_selection = False
            logger.info(f"Loaded {len(self.calibration_poses)} poses from {self._poses_file}")
        elif self.calibration_poses:
            self._use_manual_selection = False
            logger.info(f"Using {len(self.calibration_poses)} provided calibration poses")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ZMQ helpers
    # ------------------------------------------------------------------
    def _zmq_init(self) -> None:
        if self._zmq_req is not None:
            return
        self._zmq_ctx = zmq.Context.instance()
        self._zmq_req = self._zmq_ctx.socket(zmq.REQ)
        self._zmq_req.setsockopt(zmq.RCVTIMEO, int(self.config.zmq_timeout_s * 1000))
        self._zmq_req.setsockopt(zmq.SNDTIMEO, int(self.config.zmq_timeout_s * 1000))
        self._zmq_req.connect(self.config.zmq_endpoint)
        logger.info(f"ZMQ REQ connected to {self.config.zmq_endpoint}")

    def _zmq_send(self, msg: dict) -> dict:
        """Send a message and wait for a reply.  Returns the reply dict."""
        payload = msgpack.packb(msg, use_bin_type=True)
        self._zmq_req.send(payload)
        reply_bytes = self._zmq_req.recv()
        return msgpack.unpackb(reply_bytes, raw=False)

    def _zmq_close(self) -> None:
        if self._zmq_req is not None:
            self._zmq_req.close()
            self._zmq_req = None

    # ------------------------------------------------------------------
    # Keyboard / teleoperation
    # ------------------------------------------------------------------
    def _print_controls(self) -> None:
        print("\n" + "=" * 60)
        print("Multi-Camera Calibration - Manual Pose Selection")
        print("=" * 60)
        print("Use the SpaceMouse to move the robot.")
        print("Make sure the ChArUco board is visible in the cameras.")
        print()
        print("  'c' - capture current pose")
        print("  's' - start calibration (end pose selection)")
        print("  'q' - quit")
        print("=" * 60 + "\n")

    def _start_keyboard_listener(self) -> None:
        if self._keyboard_listener is not None:
            return
        self._keyboard_control = {"capture_pose": False, "end_selection": False, "quit": False}
        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self._keyboard_listener.start()

    def _stop_keyboard_listener(self) -> None:
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _on_key_press(self, key) -> None:
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "c":
                    self._keyboard_control["capture_pose"] = True
                elif key.char == "s":
                    self._keyboard_control["end_selection"] = True
                elif key.char == "q":
                    self._keyboard_control["quit"] = True
        except AttributeError:
            pass

    def _init_teleoperation(self) -> None:
        if self._teleop_agent is not None:
            return
        from franka_pipeline.agents.teleoperation_agent import TeleoperationAgent
        from franka_pipeline.input_controllers.spacemouse import SpaceMouseController
        if self._input_controller is None:
            self._input_controller = SpaceMouseController()
        self._teleop_agent = TeleoperationAgent(self._input_controller)

    # ------------------------------------------------------------------
    # Pose management
    # ------------------------------------------------------------------
    def add_calibration_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose).flatten()
        if pose.shape != (7,):
            raise ValueError(f"Pose must have 7 elements, got {pose.shape}")
        self.calibration_poses.append(pose.copy())
        logger.info(f"Added calibration pose {len(self.calibration_poses)}")

    def add_current_pose(self, robot_state: Dict[str, Any]) -> None:
        pos = robot_state["osc_position"].flatten()
        rot = robot_state["osc_rotation_quaternion"].flatten()
        self.add_calibration_pose(np.concatenate([pos, rot]))

    def load_poses_from_file(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        poses = np.load(str(filepath))
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        for pose in poses:
            self.add_calibration_pose(pose)
        logger.info(f"Loaded {len(poses)} poses from {filepath}")

    def save_poses_to_file(self, filepath: str | Path | None = None) -> Path:
        if filepath is None:
            filepath = self.config.output_dir / "calibration_poses.npy"
        else:
            filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(filepath), np.array(self.calibration_poses))
        logger.info(f"Saved {len(self.calibration_poses)} poses to {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._zmq_init()

        if self._use_manual_selection:
            self._start_manual_pose_selection()
        else:
            self._start_automatic_calibration()

    def _start_manual_pose_selection(self) -> None:
        self._print_controls()
        self._init_teleoperation()
        self._start_keyboard_listener()
        self.state = self.State.MANUAL_POSE_SELECTION

    def _start_automatic_calibration(self) -> None:
        if len(self.calibration_poses) < 3:
            logger.error("Need >= 3 poses. Falling back to manual selection.")
            self._use_manual_selection = True
            self._start_manual_pose_selection()
            return

        # Send INIT to the recording script
        logger.info(f"Sending INIT to recording script ({len(self.calibration_poses)} poses)...")
        try:
            reply = self._zmq_send({
                "cmd": "INIT",
                "num_poses": len(self.calibration_poses),
            })
            if reply.get("status") != "READY":
                logger.error(f"Recording script not ready: {reply}")
                self.state = self.State.IDLE
                return
        except zmq.error.Again:
            logger.error("Timeout waiting for recording script READY.  Is it running?")
            self.state = self.State.IDLE
            return

        self.current_pose_index = 0
        self.state = self.State.MOVING_TO_POSE
        logger.info(f"Recording script ready.  Starting calibration with {len(self.calibration_poses)} poses.")

    def stop(self) -> None:
        self._stop_keyboard_listener()
        if self._teleop_agent is not None:
            self._teleop_agent.reset()
            self._teleop_agent = None
        self._zmq_close()
        logger.info("MultiCameraCalibrationAgent stopped")

    def reset(self) -> None:
        self.current_pose_index = 0
        self.state = self.State.IDLE
        self._keyboard_control = {"capture_pose": False, "end_selection": False, "quit": False}

    def __del__(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # act()  - main state machine
    # ------------------------------------------------------------------
    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        metadata: Dict[str, Any] = {
            "action_type": self.action_type,
            "calibration_state": self.state,
            "current_pose_index": self.current_pose_index,
            "total_poses": len(self.calibration_poses),
        }

        # --- IDLE / COMPLETE ---
        if self.state in (self.State.IDLE, self.State.COMPLETE):
            if self.state == self.State.COMPLETE:
                metadata["quit"] = True
            return np.zeros(7, dtype=np.float32), metadata

        # --- MANUAL POSE SELECTION ---
        if self.state == self.State.MANUAL_POSE_SELECTION:
            return self._handle_manual_pose_selection(
                robot_state, observation, instruction, metadata
            )

        # --- MOVING TO POSE ---
        if self.state == self.State.MOVING_TO_POSE:
            target = self.calibration_poses[self.current_pose_index]
            action, at_target = self._move_to_pose(robot_state, target)
            if at_target:
                self.state = self.State.SETTLING
                self.wait_start_time = time.time()
                logger.info(
                    f"Reached pose {self.current_pose_index + 1}/"
                    f"{len(self.calibration_poses)}, settling..."
                )
            return action, metadata

        # --- SETTLING ---
        if self.state == self.State.SETTLING:
            if time.time() - self.wait_start_time >= self.config.settling_time:
                self.state = self.State.SIGNALLING_POSE_REACHED
            return np.zeros(7, dtype=np.float32), metadata

        # --- SIGNAL POSE_REACHED  ->  Docker B captures RGB ---
        if self.state == self.State.SIGNALLING_POSE_REACHED:
            T_base2ee = pos_rot_to_transformation_matrix(
                robot_state["osc_position"].flatten(),
                robot_state["osc_rotation_quaternion"].flatten(),
            )
            try:
                reply = self._zmq_send({
                    "cmd": "POSE_REACHED",
                    "index": self.current_pose_index,
                    "ee_pose": T_base2ee.reshape(-1).tolist(),
                })
                if reply.get("status") == "RGB_CAPTURED":
                    logger.info("Docker B captured RGB -> static event recording...")
                    self.state = self.State.STATIC_RECORDING
                    self.wait_start_time = time.time()
                else:
                    logger.warning(f"Unexpected reply: {reply}")
            except zmq.error.Again:
                logger.error("Timeout on POSE_REACHED.  Retrying next tick.")
            return np.zeros(7, dtype=np.float32), metadata

        # --- STATIC RECORDING ---
        if self.state == self.State.STATIC_RECORDING:
            if time.time() - self.wait_start_time >= self.config.static_event_duration:
                self.state = self.State.SIGNALLING_WIGGLE_DONE
            return np.zeros(7, dtype=np.float32), metadata

        # --- SIGNAL WIGGLE_DONE  ->  Docker B captures event frame ---
        if self.state == self.State.SIGNALLING_WIGGLE_DONE:
            try:
                reply = self._zmq_send({
                    "cmd": "WIGGLE_DONE",
                    "index": self.current_pose_index,
                })
                if reply.get("status") == "EVENT_CAPTURED":
                    logger.info(
                        f"Docker B captured event frame for pose {self.current_pose_index + 1}"
                    )
                else:
                    logger.warning(f"Unexpected reply: {reply}")

                self.current_pose_index += 1
                if self.current_pose_index >= len(self.calibration_poses):
                    self.state = self.State.WAITING_CALIBRATION
                    logger.info("All poses done -> telling Docker B to calibrate...")
                else:
                    self.state = self.State.MOVING_TO_POSE
            except zmq.error.Again:
                logger.error("Timeout on WIGGLE_DONE.  Retrying next tick.")
            return np.zeros(7, dtype=np.float32), metadata

        # --- WAITING FOR CALIBRATION ---
        if self.state == self.State.WAITING_CALIBRATION:
            try:
                # Long timeout: user may need several minutes to reposition the board
                self._zmq_req.setsockopt(zmq.RCVTIMEO, 600_000)
                reply = self._zmq_send({"cmd": "ALL_DONE"})

                if reply.get("more_rounds"):
                    round_done = reply.get("round", "?")
                    logger.info(
                        f"Round {round_done} complete. "
                        "Server waiting for board repositioning..."
                    )
                    # Send INIT for next round (server preserves data)
                    re_reply = self._zmq_send({
                        "cmd": "INIT",
                        "num_poses": len(self.calibration_poses),
                    })
                    if re_reply.get("status") == "READY":
                        self.current_pose_index = 0
                        self.state = self.State.MOVING_TO_POSE
                        logger.info("Starting next round of calibration poses...")
                    else:
                        logger.error(f"Unexpected reply to re-INIT: {re_reply}")
                        self.state = self.State.COMPLETE
                else:
                    success = reply.get("success", False)
                    if success:
                        logger.info("Calibration complete!  Results saved by Docker B.")
                    else:
                        logger.error(f"Calibration failed: {reply.get('error', 'unknown')}")
                    self.state = self.State.COMPLETE
            except zmq.error.Again:
                logger.error("Timeout waiting for calibration result.")
                self.state = self.State.COMPLETE
            return np.zeros(7, dtype=np.float32), metadata

        return np.zeros(7, dtype=np.float32), metadata

    # ------------------------------------------------------------------
    # Manual pose selection
    # ------------------------------------------------------------------
    def _handle_manual_pose_selection(
        self, robot_state, observation, instruction, metadata,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self._keyboard_control["quit"]:
            self._keyboard_control["quit"] = False
            self._stop_keyboard_listener()
            self.state = self.State.IDLE
            metadata["quit"] = True
            return np.zeros(7, dtype=np.float32), metadata

        if self._keyboard_control["capture_pose"]:
            self._keyboard_control["capture_pose"] = False
            self.add_current_pose(robot_state)
            print(f"  >> Captured pose {len(self.calibration_poses)}.  'c' more, 's' start.")

        if self._keyboard_control["end_selection"]:
            self._keyboard_control["end_selection"] = False
            if len(self.calibration_poses) < 3:
                print(f"  >> Need >= 3 poses (have {len(self.calibration_poses)}).")
            else:
                self._stop_keyboard_listener()
                path = self.save_poses_to_file()
                print(f"  >> Saved {len(self.calibration_poses)} poses to {path}")
                self._start_automatic_calibration()
                return self.act(robot_state, observation, instruction)

        if self._teleop_agent is not None:
            action, tm = self._teleop_agent.act(robot_state, observation, instruction)
            metadata.update(tm)
        else:
            action = np.zeros(7, dtype=np.float32)

        metadata["num_captured_poses"] = len(self.calibration_poses)
        return action, metadata

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def _move_to_pose(
        self, robot_state: Dict[str, Any], target_pose: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        pos = robot_state["osc_position"].flatten()
        rot = robot_state["osc_rotation_quaternion"].flatten()
        current_pose = np.concatenate([pos, rot])
        command = np.concatenate([target_pose, [self.gripper_action]])

        if self.osc_controller is None or not np.array_equal(
            self.osc_controller.target_pose, command
        ):
            self.osc_controller = OscPoseTargetController(
                target_pose=command,
                threshold_reach=self.config.position_tolerance,
                threshold_rotation=self.config.rotation_tolerance,
            )

        action, at_target = self.osc_controller.calculate_action(current_pose)
        return action, at_target
