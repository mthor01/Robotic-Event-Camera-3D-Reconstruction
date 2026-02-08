# TODO this is heavily vibe-coded and not yet checked.
"""This agent implements grasping arbitrary objects from user queries using MolMo for object identification in the image, and AnyGrasp for Grasp-Point detection"""

import base64
import io
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq
from pynput import keyboard
from scipy.spatial.transform import Rotation

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)
from franka_pipeline.utils import (
    T_from_t_R,
    transformation_matrix_to_pos_rot,
)

logger = get_logger(__name__)

# Constants
MOLMO_ZMQ_ADDR = "tcp://127.0.0.1:5583"
ANYGRASP_ZMQ_ADDR = "tcp://127.0.0.1:5562"
WAIT_STEPS_STABILIZE = 20  # ~1 second at 20Hz
GRASP_RADIUS_THRESHOLD = 0.1  # 10cm
DEPTH_MIN = 0.1
DEPTH_MAX = 2.0


class MolMoAnyGraspAgent(Agent):
    """Agent that uses MolMo and AnyGrasp to grasp objects based on natural language queries.

    This agent implements a state machine to:
    1. Wait for a grasp trigger ('g' key).
    2. Detect objects using MolMo and generate grasp candidates using AnyGrasp.
    3. Filter and select the best grasp candidate.
    4. Plan a sequence of poses (approach, reach, grasp, lift, handout, start).
    5. Execute the sequence upon trigger ('h' key).

    You can modify the MolMo object query at any time by pressing the 'i' key and entering the new query.
    """

    def __init__(
        self,
        T_ee2cam: Optional[np.ndarray] = None,
        best_grasp_color: List[int] = [0, 255, 0],
        all_grasps_color: List[int] = [100, 100, 100],
        zmq_timeout_ms: int = 10000,
    ) -> None:
        """Initialize the MolMoAnyGraspAgent.

        Args:
            T_ee2cam: 4x4 transformation matrix from camera to end-effector.
            best_grasp_color: RGB color for the best grasp visualization.
            all_grasps_color: RGB color for all other grasps visualization.
            zmq_timeout_ms: Timeout for ZMQ requests in milliseconds.
        """
        super().__init__(action_type="OSC_POSE")
        self.T_ee2cam = T_ee2cam
        self.best_grasp_color = best_grasp_color
        self.all_grasps_color = all_grasps_color
        self.object_query: Optional[str] = None
        self.state = "IDLE"

        self.grasp_sequence: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.current_target_idx = 0
        self.wait_steps = 0
        self.osc_controller: Optional[OscPoseTargetController] = None

        # ZMQ Setup
        self.context = zmq.Context()
        self.molmo_socket = self.context.socket(zmq.REQ)
        self.molmo_socket.setsockopt(zmq.RCVTIMEO, zmq_timeout_ms)
        self.molmo_socket.setsockopt(zmq.SNDTIMEO, zmq_timeout_ms)
        self.molmo_socket.setsockopt(zmq.LINGER, 0)
        self.molmo_socket.connect(MOLMO_ZMQ_ADDR)

        self.anygrasp_socket = self.context.socket(zmq.REQ)
        self.anygrasp_socket.setsockopt(zmq.RCVTIMEO, zmq_timeout_ms)
        self.anygrasp_socket.setsockopt(zmq.SNDTIMEO, zmq_timeout_ms)
        self.anygrasp_socket.setsockopt(zmq.LINGER, 0)
        self.anygrasp_socket.connect(ANYGRASP_ZMQ_ADDR)

        # Handout pose in front of the robot [x, y, z, qx, qy, qz, qw]
        self.handout_pose = np.array([0.7, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])

        # Starting pose to return to after execution [x, y, z, qx, qy, qz, qw]
        self.starting_pose = np.array([0.5, 0.0, 0.40, 1.0, 0.0, 0.0, 0.0])

        self.last_metadata: Dict[str, Any] = {}
        self._detection_thread: Optional[threading.Thread] = None
        self._trigger_grasp = False
        self._trigger_execution = False
        self._trigger_input = False
        self.is_input_active = False
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def __del__(self) -> None:
        """Cleanup resources."""
        self.close()

    def close(self) -> None:
        """Close ZMQ sockets and stop keyboard listener."""
        if hasattr(self, "_listener") and self._listener:
            self._listener.stop()
            self._listener = None

        if hasattr(self, "_detection_thread") and self._detection_thread:
            # Thread is daemon, so it will exit with the main process.
            # We don't join here to avoid blocking the close call if detection is stuck.
            self._detection_thread = None

        if hasattr(self, "molmo_socket") and self.molmo_socket:
            self.molmo_socket.close()
            self.molmo_socket = None

        if hasattr(self, "anygrasp_socket") and self.anygrasp_socket:
            self.anygrasp_socket.close()
            self.anygrasp_socket = None

        if hasattr(self, "context") and self.context:
            self.context.term()
            self.context = None

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle keyboard key press events."""
        try:
            if hasattr(key, "char"):
                if self.is_input_active:
                    # Ignore all agent-specific triggers while typing
                    return
                if key.char == "g":
                    self._trigger_grasp = True
                elif key.char == "h":
                    self._trigger_execution = True
                elif key.char == "i":
                    self._trigger_input = True
        except AttributeError:
            pass

    def _threaded_input(self) -> None:
        """Prompt user for input in a separate thread."""
        try:
            print("\n" + "=" * 40)
            print("INPUT MODE: Enter new object query")
            print("=" * 40)
            new_query = input("Query: ")
            if new_query.strip():
                self.set_object_to_grasp(new_query)
        except Exception as e:
            logger.error(f"Error in threaded input: {e}")
        finally:
            self.is_input_active = False
            print("\n" + "=" * 40)
            print("RESUMING NORMAL OPERATION")
            print("=" * 40)

    def set_object_to_grasp(self, query: str) -> None:
        """Set the natural language query for the object to grasp."""
        self.object_query = query
        logger.info(f"Object to grasp set to: {query}")

    def start_grasping(self) -> None:
        """Trigger the grasp detection sequence."""
        if self.object_query is None:
            logger.warning("No object query set. Call set_object_to_grasp first.")
            return
        self.state = "DETECTING"
        logger.info("Starting grasp detection sequence")

    def start_execution(self) -> None:
        """Trigger the grasp execution sequence."""
        if self.state != "WAITING_FOR_EXECUTION":
            logger.warning(
                f"Cannot start execution from state {self.state}. Must be in WAITING_FOR_EXECUTION."
            )
            return
        self.state = "EXECUTING"
        logger.info("Starting execution of grasp sequence")

    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate robot actions based on the current state machine."""

        if self._trigger_input and not self.is_input_active:
            self._trigger_input = False
            self.is_input_active = True
            # Use a thread for terminal input to avoid blocking the control loop
            threading.Thread(target=self._threaded_input, daemon=True).start()

        if self._trigger_grasp:
            self.start_grasping()
            self._trigger_grasp = False

        if self._trigger_execution:
            self.start_execution()
            self._trigger_execution = False

        metadata = {"action_type": self.action_type, "state": self.state}
        metadata.update(self.last_metadata)

        if self.is_input_active:
            metadata["is_input_active"] = True

        if self.state in ["IDLE", "WAITING_FOR_EXECUTION"]:
            return self._handle_idle_or_waiting(metadata)

        if self.state == "DETECTING":
            return self._handle_detecting(robot_state, observation, metadata)

        if self.state == "EXECUTING":
            # Get current pose for OSC control
            pos = robot_state["osc_position"]
            rot = robot_state["osc_rotation_quaternion"]
            current_pose = np.concatenate([pos.flatten(), rot.flatten()]).astype(float)
            return self._handle_executing(current_pose, metadata)

        return np.zeros(7, dtype=np.float32), metadata

    def _handle_idle_or_waiting(
        self, metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handle IDLE or WAITING_FOR_EXECUTION states."""
        # Stay at current pose, keep gripper open (-1.0)
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1.0
        return action, metadata

    def _handle_detecting(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handle the DETECTING state: MolMo + AnyGrasp."""
        if self._detection_thread is None:
            wrist_image = observation.get("wrist_image")
            wrist_depth = observation.get("wrist_depth")
            wrist_intrinsics = observation.get("wrist_intrinsics")
            T_world2ee = robot_state.get("osc_pose")

            if any(
                v is None
                for v in [wrist_image, wrist_depth, wrist_intrinsics, T_world2ee]
            ):
                logger.error("Missing data for detection")
                self.state = "IDLE"
                return self._handle_idle_or_waiting(metadata)

            if self.T_ee2cam is None:
                logger.error(
                    "T_ee2cam is not set. Cannot transform grasp to world frame."
                )
                self.state = "IDLE"
                return self._handle_idle_or_waiting(metadata)

            # Start detection thread
            self._detection_thread = threading.Thread(
                target=self._run_detection,
                args=(wrist_image, wrist_depth, wrist_intrinsics, T_world2ee),
                daemon=True,
            )
            self._detection_thread.start()
            logger.info("Detection thread started")

        if self._detection_thread.is_alive():
            return self._handle_idle_or_waiting(metadata)
        else:
            # Thread finished
            self._detection_thread.join()
            self._detection_thread = None
            # The thread updated self.state and self.last_metadata
            return self._handle_idle_or_waiting(metadata)

    def _run_detection(
        self,
        wrist_image: np.ndarray,
        wrist_depth: np.ndarray,
        wrist_intrinsics: Dict[str, float],
        T_world2ee: np.ndarray,
    ) -> None:
        """Run MolMo and AnyGrasp detection in a background thread."""
        try:
            # 1. Send to MolMo
            molmo_points = self._get_molmo_points(wrist_image)
            if not molmo_points:
                self.state = "IDLE"
                return

            # 2. Generate point cloud and send to AnyGrasp
            anygrasp_points = self._get_anygrasp_points(
                wrist_image, wrist_depth, wrist_intrinsics
            )
            if not anygrasp_points:
                self.state = "IDLE"
                return

            # 3. Filter grasp points near MolMo point
            best_grasp = self._filter_and_select_best_grasp(
                molmo_points[0], anygrasp_points, wrist_depth, wrist_intrinsics
            )
            if best_grasp is None:
                self.state = "IDLE"
                return

            # 4. Plan sequence in world frame
            self._plan_grasp_sequence(best_grasp, T_world2ee)

            # Update metadata for visualization (after planning to include potential flips)
            self.last_metadata = {
                "all_grasps": anygrasp_points,
                "best_grasp": best_grasp,
                "grasp_ee_pose": T_world2ee,
                "best_grasp_color": self.best_grasp_color,
                "all_grasps_color": self.all_grasps_color,
            }
            self.state = "WAITING_FOR_EXECUTION"
            logger.info("Grasp detection complete. Press 'h' to execute.")
        except Exception as e:
            logger.error(f"Error in detection thread: {e}")
            self.state = "IDLE"

    def _handle_executing(
        self, current_pose: np.ndarray, metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Handle the EXECUTING state: follow the planned sequence."""
        if not self.grasp_sequence:
            logger.warning("Grasp sequence is empty")
            self.state = "IDLE"
            return np.zeros(7, dtype=np.float32), metadata

        target_pos, target_rot, gripper = self.grasp_sequence[self.current_target_idx]
        target_pose = np.concatenate([target_pos, target_rot, [gripper]])

        if self.osc_controller is None or not np.array_equal(
            self.osc_controller.target_pose, target_pose
        ):
            self.osc_controller = OscPoseTargetController(
                target_pose=target_pose, max_steps=60
            )

        action, is_finished = self.osc_controller.calculate_action(current_pose)

        if is_finished:
            self.wait_steps += 1
            if self.wait_steps > WAIT_STEPS_STABILIZE:
                self.wait_steps = 0
                self.current_target_idx += 1
                if self.current_target_idx >= len(self.grasp_sequence):
                    logger.info("Grasp sequence completed")
                    self.state = "IDLE"
                    self.last_metadata = {}
                    self.osc_controller = None
                    return np.zeros(7, dtype=np.float32), metadata

                # Re-initialize controller for next target if target changed
                target_pos, target_rot, gripper = self.grasp_sequence[
                    self.current_target_idx
                ]
                target_pose = np.concatenate([target_pos, target_rot, [gripper]])
                self.osc_controller = OscPoseTargetController(
                    target_pose=target_pose, max_steps=60
                )
                action, is_finished = self.osc_controller.calculate_action(current_pose)
        else:
            self.wait_steps = 0

        return action, metadata

    def _get_molmo_points(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Send image to MolMo and return detected points."""
        if not self.object_query:
            logger.warning("No object query set.")
            return []

        logger.info(f"Sending request to MolMo with query: {self.object_query}")
        success, img_encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        if not success:
            logger.error("Failed to encode image for MolMo")
            return []

        image_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
        try:
            self.molmo_socket.send_json({"image": image_b64, "text": self.object_query})
            molmo_resp = self.molmo_socket.recv_json()
        except zmq.Again:
            logger.error("MolMo request timed out")
            return []
        except zmq.ZMQError as e:
            logger.error(f"ZMQ error communicating with MolMo: {e}")
            return []

        molmo_text = molmo_resp.get("generated_text", "")
        logger.info(f"MolMo response: {molmo_text}")

        points = self._parse_molmo_points(molmo_text, image.shape[1], image.shape[0])
        if not points:
            logger.warning("MolMo did not find any points for the query")
        return points

    def _get_anygrasp_points(
        self, image: np.ndarray, depth: np.ndarray, intrinsics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate point cloud and get grasp points from AnyGrasp."""
        points, colors = self._generate_point_cloud(image, depth, intrinsics)
        xyzrgb = np.concatenate([points, colors], axis=1).astype(np.float32)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, xyzrgb=xyzrgb)
        logger.info("Sending point cloud to AnyGrasp")
        try:
            self.anygrasp_socket.send(buffer.getvalue())
            anygrasp_resp_json = self.anygrasp_socket.recv_json()
        except zmq.Again:
            logger.error("AnyGrasp request timed out")
            return []
        except zmq.ZMQError as e:
            logger.error(f"ZMQ error communicating with AnyGrasp: {e}")
            return []

        try:
            anygrasp_points = json.loads(anygrasp_resp_json)
            logger.info(f"Received {len(anygrasp_points)} grasp points from AnyGrasp")
            return anygrasp_points
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to decode AnyGrasp response: {e}")
            return []

    def _filter_and_select_best_grasp(
        self,
        molmo_point: Tuple[int, int],
        anygrasp_points: List[Dict[str, Any]],
        depth: np.ndarray,
        intrinsics: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Filter AnyGrasp points based on MolMo detection and select the best one."""
        mx, my = molmo_point

        depth_val = depth[my, mx] / 1000.0
        if depth_val <= 0:
            logger.warning(f"MolMo point ({mx}, {my}) has invalid depth: {depth_val}")
            return None

        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        mz = depth_val
        mx_3d = (mx - cx) / fx * mz
        my_3d = (my - cy) / fy * mz
        molmo_point_3d = np.array([mx_3d, my_3d, mz])

        # logger.debug(f"AnyGrasp Points: {anygrasp_points}")

        filtered_grasps = [
            g
            for g in anygrasp_points
            if np.linalg.norm(np.array(g["translation"]) - molmo_point_3d)
            < GRASP_RADIUS_THRESHOLD
        ]

        if not filtered_grasps:
            logger.warning(
                f"No grasp points found within {GRASP_RADIUS_THRESHOLD}m of MolMo point."
            )
            return None

        return max(filtered_grasps, key=lambda x: x["score"])

    def _plan_grasp_sequence(
        self, best_grasp: Dict[str, Any], T_world2ee: np.ndarray
    ) -> None:
        """Plan the sequence of poses for grasping."""
        T_world2cam = T_world2ee @ self.T_ee2cam
        g_pos_cam = np.array(best_grasp["translation"])
        g_rot_cam = np.array(best_grasp["rotation_matrix"])

        # Rotation to match convention (AnyGrasp Z-approach -> Project X-approach)
        R_anygrasp2project = Rotation.from_euler(
            "YZ", [90, 90], degrees=True
        ).as_matrix()
        T_cam2grasp = T_from_t_R(g_pos_cam, g_rot_cam @ R_anygrasp2project)
        T_world2grasp = T_world2cam @ T_cam2grasp

        # For a two-finger gripper, any grasp can be rotated by 180 degrees around its Z-axis.
        # We flip the grasp if its y-axis points to the left of the robot (world Y < 0)
        # to make the approach easier for the robot.
        if T_world2grasp[1, 1] > 0:
            R_flip = Rotation.from_euler("z", 180, degrees=True).as_matrix()
            T_world2grasp[:3, :3] = T_world2grasp[:3, :3] @ R_flip

            # Update best_grasp rotation so visualization also shows the flip
            R_flip_cam = Rotation.from_euler("x", 180, degrees=True).as_matrix()
            best_grasp["rotation_matrix"] = (g_rot_cam @ R_flip_cam).tolist()
            logger.info(
                "Flipped grasp orientation by 180 degrees around Z-axis for easier grasp approach"
            )

        grasp_pos, grasp_rot = transformation_matrix_to_pos_rot(T_world2grasp)

        # Approach from 10cm above the grasp point in the world frame
        above_grasp_pos = grasp_pos + np.array([0, 0, 0.1])

        self.grasp_sequence = [
            (above_grasp_pos, grasp_rot, -1.0),  # Approach
            (grasp_pos, grasp_rot, -1.0),  # Reach
            (grasp_pos, grasp_rot, 1.0),  # Grasp
            (above_grasp_pos, grasp_rot, 1.0),  # Lift
            (self.handout_pose[:3], self.handout_pose[3:7], 1.0),  # Move to handout
            (self.handout_pose[:3], self.handout_pose[3:7], -1.0),  # Release
            (self.starting_pose[:3], self.starting_pose[3:7], -1.0),  # Return to start
        ]
        self.current_target_idx = 0

    def _parse_molmo_points(
        self, text: str, width: int, height: int
    ) -> List[Tuple[int, int]]:
        """Parse point coordinates from MolMo response text.

        Args:
            text: The generated text from MolMo.
            width: Image width.
            height: Image height.

        Returns:
            List of (x, y) pixel coordinates.
        """
        point_pattern = r'<point x="(\d+\.\d+)" y="(\d+\.\d+)"'
        matches = re.finditer(point_pattern, text)
        points = []
        for match in matches:
            x_percent = float(match.group(1))
            y_percent = float(match.group(2))
            x = int(x_percent * width / 100)
            y = int(y_percent * height / 100)
            points.append((x, y))
        return points

    def _generate_point_cloud(
        self, color: np.ndarray, depth: np.ndarray, intrinsics: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a point cloud from RGB-D data.

        Args:
            color: RGB image (H, W, 3).
            depth: Depth image in mm (H, W).
            intrinsics: Camera intrinsics dictionary.

        Returns:
            Tuple of (points, colors) arrays.
        """
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        height, width = depth.shape
        xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))

        # Depth is in mm, convert to meters
        z = depth.astype(np.float32) / 1000.0
        x = (xmap - cx) / fx * z
        y = (ymap - cy) / fy * z

        points = np.stack([x, y, z], axis=-1)
        colors = color.astype(np.float32) / 255.0

        # Filter invalid depth
        mask = (z > DEPTH_MIN) & (z < DEPTH_MAX)

        return points[mask], colors[mask]

    def reset(self) -> None:
        """Reset the agent state."""
        self.state = "IDLE"
        self.grasp_sequence = []
        self.current_target_idx = 0
        self.wait_steps = 0
        self.last_metadata = {}
        self._trigger_grasp = False
        self._trigger_execution = False
