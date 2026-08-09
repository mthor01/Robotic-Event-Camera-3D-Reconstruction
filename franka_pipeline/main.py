"""
Minimal Franka pipeline entrypoint for HalfSphereRecordingAgent + ZMQ streaming.

- Runs only the relevant parts for controlling the robot and executing the agent.
- Does NOT start or read any cameras (RealSense handled in another container).
- Publishes system time + end-effector pose (and optionally joints) over ZeroMQ.
- Sends episode control notifications (episode_start / episode_end / quit) over ZeroMQ.
- Optionally synchronizes with a recording script via REQ/REP socket.

Assumptions:
- robot_controller.get_state() returns robot_state with:
    - robot_state["osc_pose"] : 4x4 numpy array (EE pose in base/world frame)
    - robot_state["joint_position"] : (7,) array
    - robot_state["gripper_q"] : scalar-ish
- HalfSphereRecordingAgent returns metadata with "end_episode"/"reset_episode"/"quit" similarly.

Calibration mode:
python main.py --real-robot --agent-type multi_cam_calibrate --calibration-poses calibration_poses.npy

"""

import os
import time
import threading
from pathlib import Path
import robosuite

import numpy as np
import typer

# ZMQ deps
import zmq
import msgpack

from franka_pipeline.logging import get_logger, setup_logging
from franka_pipeline.agents.episode_control_wrapper_agent import EpisodeControlWrapperAgent
from franka_pipeline.agents.hemisphere_grid_agent import HemisphereGridAgent
from franka_pipeline.agents.random_hemisphere_agent import RandomHemisphereAgent
from franka_pipeline.agents.temporal_alignment_agent import TemporalAlignmentAgent
from franka_pipeline.robot_controllers.controller import (
    RealRobotController,
    SimulatedRobosuiteRobotController,
)
from franka_pipeline.sim.robosuite_env import RobosuiteSimEnv
from franka_pipeline.sim.empty_table_env import EmptyTableSimEnv, ObjectConfig
from franka_pipeline.synthetic_data import SyntheticDataRecorder
from franka_pipeline.synthetic_data.synthetic_recorder import SyntheticRecorderConfig
import config_defaults as cfg

app = typer.Typer(help="Half-sphere recording runner (no cameras) + ZMQ pose stream")
logger = get_logger(__name__)

from object_utils import (
    AVAILABLE_OBJECTS,
    DEFAULT_OBJECT_POSITION,
    get_object_height,
    create_object_config,
    list_available_objects,
)


class ZMQPosePublisher:
    """
    PUB socket that broadcasts:
      - topic b"pose": {t_ns, ep, step, ee_T(16 floats), q(7), gripper_q}
      - topic b"event": {t_ns, type, ep, ...}
    """

    def __init__(self, bind_addr: str):
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        # Keep latest if receiver is slow (live alignment use-case)
        #self.pub.setsockopt(zmq.SNDHWM, 1)
        self.pub.bind(bind_addr)
        self._lock = threading.Lock()
        logger.info(f"ZMQ PUB bound at {bind_addr}")

    def publish_pose(
        self,
        ep: int,
        step: int,
        ee_T: np.ndarray | None,
        q: np.ndarray | None,
        gripper_q: float | None,
        joint_velocity: np.ndarray | None = None,
    ) -> None:
        msg = {
            "t_ns": time.time_ns(),
            "ep": int(ep),
            "step": int(step),
        }
        if ee_T is not None:
            ee_T = np.asarray(ee_T, dtype=np.float64)
            if ee_T.shape != (4, 4):
                raise ValueError(f"osc_pose must be (4,4), got {ee_T.shape}")
            msg["ee_T"] = ee_T.reshape(-1).tolist()  # 16 floats, row-major
        if q is not None:
            msg["q"] = np.asarray(q, dtype=np.float64).reshape(-1).tolist()
        if gripper_q is not None:
            msg["gripper_q"] = float(gripper_q)
        if joint_velocity is not None:
            msg["jv"] = np.asarray(joint_velocity, dtype=np.float64).reshape(-1).tolist()

        payload = msgpack.packb(msg, use_bin_type=True)
        with self._lock:
            self.pub.send_multipart([b"pose", payload])

    def publish_event(self, event_type: str, ep: int, extra: dict | None = None) -> None:
        msg = {"t_ns": time.time_ns(), "type": event_type, "ep": int(ep)}
        if extra:
            msg.update(extra)
        payload = msgpack.packb(msg, use_bin_type=True)
        with self._lock:
            self.pub.send_multipart([b"event", payload])


class PosePublishThread:
    """Background thread that reads robot state and publishes poses at a
    fixed rate, independent of the (slower) control loop.

    This is necessary because ``robot_interface.control()`` blocks at the
    Deoxys policy rate (~20 Hz), whereas the Deoxys state buffer is
    updated at 100 Hz.  Running pose publishing in its own thread lets us
    capture every state update and stream it to the recording side.
    """

    def __init__(
        self,
        robot_controller,
        publisher: ZMQPosePublisher,
        publish_hz: float = 100.0,
    ):
        self.robot_controller = robot_controller
        self.pub = publisher
        self.period = 1.0 / publish_hz if publish_hz > 0 else 0.0
        self._ep = 0
        self._step = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # Allow the control loop to update ep/step counters
    def set_ep_step(self, ep: int, step: int) -> None:
        self._ep = ep
        self._step = step

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"PosePublishThread started ({1/self.period:.0f} Hz)" if self.period else "PosePublishThread started (unthrottled)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("PosePublishThread stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                robot_state, valid = self.robot_controller.get_state()
                if valid:
                    ee_pose = robot_state.get("osc_pose", None)
                    q = robot_state.get("joint_position", None)
                    joint_velocity = robot_state.get("joint_velocity", None)
                    gripper_q = None
                    if "gripper_q" in robot_state:
                        try:
                            gripper_q = float(np.asarray(robot_state["gripper_q"]).reshape(-1)[0])
                        except Exception:
                            pass
                    self.pub.publish_pose(
                        ep=self._ep,
                        step=self._step,
                        ee_T=ee_pose,
                        q=q,
                        gripper_q=gripper_q,
                        joint_velocity=joint_velocity,
                    )
            except Exception as e:
                logger.warning(f"PosePublishThread: {e}")
            # Sleep to hit desired rate
            elapsed = time.monotonic() - t0
            remaining = self.period - elapsed
            if remaining > 0:
                self._stop.wait(remaining)


class ZMQSyncServer:
    """
    REP socket to synchronize with the recording script.
    Waits for "ready" from recording script, then sends "start" to begin synchronized recording.
    """

    def __init__(self, bind_addr: str):
        self.bind_addr = bind_addr
        self.ctx = zmq.Context.instance()
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.setsockopt(zmq.RCVTIMEO, -1)  # Block indefinitely initially
        self.rep.bind(bind_addr)
        self.recording_ready = threading.Event()
        self.recording_done = threading.Event()
        logger.info(f"ZMQ SYNC REP bound at {bind_addr}")

    def wait_for_ready(self, timeout_sec: float = 120.0) -> bool:
        """
        Wait for "ready" message from recording script.
        Transparently handles any preceding "ping" messages used for RTT measurement.
        Returns True if ready received, False on timeout.
        """
        logger.info("Waiting for recording script 'ready' signal...")
        self.rep.setsockopt(zmq.RCVTIMEO, int(timeout_sec * 1000))

        while True:
            try:
                msg_bytes = self.rep.recv()
                msg = msgpack.unpackb(msg_bytes, raw=False)

                if msg.get("type") == "ping":
                    # Reply immediately — used by the recording script to measure
                    # one-way ZMQ transport delay before recording starts.
                    reply = {"type": "pong", "t1": msg.get("t1")}
                    self.rep.send(msgpack.packb(reply, use_bin_type=True))
                    continue

                if msg.get("type") == "ready":
                    logger.info(f"Recording script ready at t_ns={msg.get('t_ns')}")
                    self.recording_ready.set()
                    return True
                else:
                    logger.warning(f"Unexpected message type: {msg.get('type')}")
                    # Reply anyway to not block the client
                    reply = {"type": "error", "t_ns": time.time_ns(), "message": "Expected 'ready'"}
                    self.rep.send(msgpack.packb(reply, use_bin_type=True))
                    return False
            except zmq.error.Again:
                logger.warning("Timeout waiting for recording script 'ready'")
                return False

    def send_start(self) -> None:
        """Send 'start' signal to recording script."""
        reply = {"type": "start", "t_ns": time.time_ns()}
        self.rep.send(msgpack.packb(reply, use_bin_type=True))
        logger.info("Sent 'start' signal to recording script")

    def wait_for_done(self, timeout_sec: float = 300.0) -> bool:
        """
        Wait for "done" message from recording script.
        Returns True if done received, False on timeout.
        """
        logger.info("Waiting for recording script 'done' signal...")
        self.rep.setsockopt(zmq.RCVTIMEO, int(timeout_sec * 1000))

        try:
            msg_bytes = self.rep.recv()
            msg = msgpack.unpackb(msg_bytes, raw=False)

            if msg.get("type") == "done":
                logger.info(f"Recording script done at t_ns={msg.get('t_ns')}")
                self.recording_done.set()
                # Acknowledge
                reply = {"type": "ack", "t_ns": time.time_ns()}
                self.rep.send(msgpack.packb(reply, use_bin_type=True))
                return True
            else:
                logger.warning(f"Unexpected message type: {msg.get('type')}, expected 'done'")
                reply = {"type": "error", "t_ns": time.time_ns(), "message": "Expected 'done'"}
                self.rep.send(msgpack.packb(reply, use_bin_type=True))
                return False
        except zmq.error.Again:
            logger.warning("Timeout waiting for recording script 'done'")
            return False

    def close(self) -> None:
        """Close the socket."""
        self.rep.close()


def _run_single_object_recording(
    obj_config: ObjectConfig,
    obj_name: str,
    obj_dict: dict,  # Original object dictionary for getting height
    synthetic_output_dir: str,
    synthetic_camera_width: int,
    synthetic_camera_height: int,
    event_threshold_pos: float,
    event_threshold_neg: float,
    flip_vertical: bool,
    zmq_bind: str,
    publish_hz: float,
    headless: bool = False,
    agent_type: str = "random_hemisphere",
    sphere_radius: float = 0.3,
    num_poses: int = 20,
    wait_time: float = 0.0,
    random_seed: int = None,
    target_point: np.ndarray = None,
    center_z_offset: float = 0.0,
) -> None:
    """
    Run a single recording for one object.
    This is called by multi-object recording mode.
    """
    

    
    # Create environment with just this one object
    logger.info(f"Creating simulation environment with object: {obj_name}")
    logger.info(f"  - Type: {obj_config.type}")
    logger.info(f"  - Position: {obj_config.position}")
    if hasattr(obj_config, 'rgba') and obj_config.rgba:
        logger.info(f"  - Color (RGBA): {obj_config.rgba}")
    
    sim_env = EmptyTableSimEnv(
        controller_type="OSC_POSE",
        camera_width=synthetic_camera_width,
        camera_height=synthetic_camera_height,
        custom_objects=[obj_config],
        has_renderer=not headless,
    )
    
    # Print coordinate system debug info
    sim_env.print_coordinate_debug_info()
    
    # Log object positions
    positions = sim_env.get_object_positions()
    for name, pos in positions.items():
        logger.info(f"  Object '{name}': x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    
    robot_controller = SimulatedRobosuiteRobotController(
        sim_env=sim_env, controller_type="OSC_POSE"
    )
    robot_controller.reset_robot_joints()
    
    # ZMQ publisher (use a different port or reuse)
    pub = ZMQPosePublisher(bind_addr=zmq_bind)
    
    # Synthetic data recorder
    config = SyntheticRecorderConfig(
        output_dir=Path(synthetic_output_dir),
        camera_id="robot0_eye_in_hand",
        camera_width=synthetic_camera_width,
        camera_height=synthetic_camera_height,
        event_threshold_pos=event_threshold_pos,
        event_threshold_neg=event_threshold_neg,
        save_rgb=True,
        save_depth=True,
        save_events=True,
        save_poses=True,
        save_video=True,
        flip_vertical=flip_vertical,
    )
    synthetic_recorder = SyntheticDataRecorder(config)
    
    # Throttle publishing if desired
    min_period = (1.0 / publish_hz) if publish_hz and publish_hz > 0 else 0.0
    
    # Create agent based on type
    if agent_type.lower() == "hemisphere":
        agent = EpisodeControlWrapperAgent(HemisphereGridAgent(wait_time=wait_time))
    elif agent_type.lower() == "random_hemisphere":
        if target_point is None:
            target_point = np.array([0.35, 0.0, -0.1])
        # Adjust target z by half the object height so camera looks at object center
        obj_height = get_object_height(obj_dict) if obj_dict else 0.0
        adjusted_target = target_point.copy()
        adjusted_target[2] = target_point[2] + obj_height / 2
        logger.info(f"  Object height: {obj_height:.3f}m, adjusted target z: {target_point[2]:.3f} -> {adjusted_target[2]:.3f}")
        # For random_hemisphere, center equals adjusted target_point
        agent = EpisodeControlWrapperAgent(
            RandomHemisphereAgent(
                center=adjusted_target,
                radius=sphere_radius,
                inner_radius=cfg.INNER_RADIUS,
                num_poses=num_poses,
                wait_time=wait_time,
                seed=random_seed,
                target_point=adjusted_target,
                base_exclusion_radius=cfg.BASE_EXCLUSION_RADIUS,
                base_max_radius=cfg.BASE_MAX_RADIUS,
                min_z_height=cfg.MIN_Z_HEIGHT,
                hemisphere_top_cutoff=cfg.HEMISPHERE_TOP_CUTOFF,
                lock_rotation_horizontal=cfg.LOCK_ROTATION_HORIZONTAL,
                center_z_offset=center_z_offset,
            )
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")

    # Reset for recording
    robot_controller.reset_robot_joints()
    step_count = 0
    ep_count = 0
    next_pub_t = 0.0
    
    # Start synthetic recording with object name in the recording ID
    recording_id = obj_name
    synthetic_recorder.start_recording(recording_id)
    logger.info(f"Started synthetic recording: {recording_id}")
    
    pub.publish_event("episode_start", ep=ep_count)
    
    # Recording loop
    while True:
        robot_state, valid = robot_controller.get_state()
        if not valid:
            logger.warning("Invalid robot state, skipping iteration.")
            continue
        
        observation = {}
        instruction = ""
        
        action, metadata = agent.act(
            robot_state=robot_state,
            observation=observation,
            instruction=instruction,
        )
        robot_controller.control(command=action, controller_type=agent.action_type)
        
        # Record synthetic data
        if synthetic_recorder.is_recording:
            try:
                stats = synthetic_recorder.record_frame(
                    sim_env=sim_env,
                    robot_state=robot_state,
                    timestamp_ns=time.time_ns(),
                )
                if step_count % 100 == 0:
                    logger.debug(f"Synthetic recording step {step_count}: {stats.get('num_events', 0)} events")
            except Exception as e:
                logger.warning(f"Failed to record synthetic frame: {e}")
        
        # Publish pose
        now = time.time()
        if min_period == 0.0 or now >= next_pub_t:
            ee_pose = robot_state.get("osc_pose", None)
            q = robot_state.get("joint_position", None)
            joint_velocity = robot_state.get("joint_velocity", None)
            gripper_q = None
            if "gripper_q" in robot_state:
                try:
                    gripper_q = float(np.asarray(robot_state["gripper_q"]).reshape(-1)[0])
                except Exception:
                    gripper_q = None
            
            try:
                pub.publish_pose(
                    ep=ep_count,
                    step=step_count,
                    ee_T=ee_pose,
                    q=q,
                    gripper_q=gripper_q,
                    joint_velocity=joint_velocity,
                )
            except Exception as e:
                logger.warning(f"Failed to publish pose: {e}")
            
            next_pub_t = now + min_period
        
        step_count += 1
        
        # Episode control
        if metadata.get("end_episode", False):
            logger.info("Ending episode, as per agent request.")
            pub.publish_event("episode_end", ep=ep_count, extra={"step": step_count})
            robot_controller.reset_robot_joints()
            agent.reset()
            ep_count += 1
            step_count = 0
            pub.publish_event("episode_start", ep=ep_count)
        
        if metadata.get("reset_episode", False):
            logger.info("Resetting episode, as per agent request.")
            pub.publish_event("episode_reset", ep=ep_count, extra={"step": step_count})
            robot_controller.reset_robot_joints()
            agent.reset()
            step_count = 0
            pub.publish_event("episode_start", ep=ep_count)
        
        # Check if agent is complete (hemisphere or sphere)
        is_complete = metadata.get("hemisphere_complete", False) or metadata.get("sphere_complete", False)
        if is_complete:
            logger.info(f"Agent recording complete for object '{obj_name}'!")
            pub.publish_event("agent_complete", ep=ep_count, extra={
                "step": step_count,
                "total_poses": metadata.get("total_poses", 0),
                "object_name": obj_name,
            })
            
            # Stop synthetic recording and save data
            if synthetic_recorder.is_recording:
                summary = synthetic_recorder.stop_recording()
                logger.info(f"Synthetic recording saved: {summary.get('output_path')}")
                logger.info(f"  - Frames: {summary.get('num_frames', 0)}")
                logger.info(f"  - Events: {summary.get('total_events', 0)}")
            
            robot_controller.reset_robot_joints()
            if hasattr(agent, "stop"):
                agent.stop()
            break
        
        if metadata.get("quit", False):
            logger.info("Ending run, as per agent request (keyboard quit).")
            if synthetic_recorder.is_recording:
                summary = synthetic_recorder.stop_recording()
                logger.info(f"Synthetic recording saved on quit: {summary.get('output_path')}")
            robot_controller.reset_robot_joints()
            if hasattr(agent, "stop"):
                agent.stop()
            break
    
    # Cleanup - close the environment
    try:
        sim_env.close()
    except Exception as e:
        logger.warning(f"Error closing sim_env: {e}")


@app.command()
def main(
    simulated_robot: bool = typer.Option(
        False, "--simulated-robot/--real-robot", help="Run in robosuite sim"
    ),
    log_level: str = typer.Option(
        cfg.LOG_LEVEL, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging"),
    deps_log_level: str = typer.Option(
        cfg.DEPS_LOG_LEVEL, "--deps-log-level", help="Logging level for dependencies"
    ),
    # ZMQ
    zmq_bind: str = typer.Option(
        cfg.ZMQ_BIND,
        "--zmq-bind",
        help='ZMQ PUB bind address, e.g. "tcp://0.0.0.0:5556"',
    ),
    publish_hz: float = typer.Option(
        cfg.PUBLISH_HZ,
        "--publish-hz",
        help="Max publish rate for pose stream (0 = publish every loop iteration)",
    ),
    # Sync recording options
    sync_recording: bool = typer.Option(
        cfg.SYNC_RECORDING,
        "--sync-recording/--no-sync-recording",
        help="Enable synchronized recording mode. Waits for recording script before starting.",
    ),
    zmq_sync_bind: str = typer.Option(
        cfg.ZMQ_SYNC_BIND,
        "--zmq-sync-bind",
        help='ZMQ REP bind address for sync handshake, e.g. "tcp://0.0.0.0:5557"',
    ),
    # Synthetic data recording options
    synthetic_data: bool = typer.Option(
        cfg.SYNTHETIC_DATA,
        "--synthetic-data/--no-synthetic-data",
        help="Enable synthetic data recording (RGB, depth, events from simulation). Only works with --simulated-robot.",
    ),
    synthetic_output_dir: str = typer.Option(
        cfg.SYNTHETIC_OUTPUT_DIR,
        "--synthetic-output-dir",
        help="Output directory for synthetic data recordings.",
    ),
    synthetic_camera_width: int = typer.Option(
        cfg.SYNTHETIC_CAMERA_WIDTH,
        "--synthetic-camera-width",
        help="Width of synthetic camera images.",
    ),
    synthetic_camera_height: int = typer.Option(
        cfg.SYNTHETIC_CAMERA_HEIGHT,
        "--synthetic-camera-height",
        help="Height of synthetic camera images.",
    ),
    event_threshold_pos: float = typer.Option(
        cfg.EVENT_THRESHOLD_POS,
        "--event-threshold-pos",
        help="Positive contrast threshold for synthetic event generation.",
    ),
    event_threshold_neg: float = typer.Option(
        cfg.EVENT_THRESHOLD_NEG,
        "--event-threshold-neg",
        help="Negative contrast threshold for synthetic event generation.",
    ),
    flip_vertical: bool = typer.Option(
        cfg.FLIP_VERTICAL,
        "--flip-vertical/--no-flip-vertical",
        help="Flip all synthetic data (RGB, depth, events) vertically. Useful for camera mounting orientation.",
    ),
    object_filter: str = typer.Option(
        cfg.OBJECT_FILTER,
        "--object-filter",
        help="Filter objects by type. Comma-separated list (e.g., 'cube,sphere,milk'). If not specified, all objects are used.",
    ),
    list_objects: bool = typer.Option(
        False,
        "--list-objects",
        help="List all available objects for synthetic data generation and exit.",
    ),
    headless: bool = typer.Option(
        cfg.HEADLESS,
        "--headless/--no-headless",
        help="Run without rendering window (headless mode). Useful for batch processing.",
    ),
    # Agent selection options
    agent_type: str = typer.Option(
        cfg.AGENT_TYPE,
        "--agent-type",
        help="Agent type to use: 'hemisphere' or 'random_hemisphere'. Default: random_hemisphere",
    ),
    target_x: float = typer.Option(
        cfg.TARGET_X,
        "--target-x",
        help="X coordinate of target point (where camera looks).",
    ),
    target_y: float = typer.Option(
        cfg.TARGET_Y,
        "--target-y",
        help="Y coordinate of target point (where camera looks).",
    ),
    target_z: float = typer.Option(
        cfg.TARGET_Z,
        "--target-z",
        help="Z coordinate of target point (where camera looks).",
    ),
    sphere_radius: float = typer.Option(
        cfg.SPHERE_RADIUS,
        "--sphere-radius",
        help="Radius of the sphere in meters.",
    ),
    center_z_offset: float = typer.Option(
        cfg.CENTER_Z_OFFSET,
        "--center-z-offset",
        help="Z offset added to the hemisphere center only (target point unchanged).",
    ),
    num_poses: int = typer.Option(
        cfg.NUM_POSES,
        "--num-poses",
        help="Number of random poses to generate.",
    ),
    wait_time: float = typer.Option(
        cfg.WAIT_TIME,
        "--wait-time",
        help="Time to wait at each pose in seconds.",
    ),
    random_seed: int = typer.Option(
        cfg.RANDOM_SEED,
        "--random-seed",
        help="Random seed for reproducible pose generation. Default: None (random)",
    ),
    calibration_poses: str = typer.Option(
        "",
        "--calibration-poses",
        help="Path to .npy file with calibration poses (for multi_cam_calibrate agent type)",
    ),
) -> None:
    
    # Handle --list-objects early
    if list_objects:
        list_available_objects()
        raise typer.Exit(code=0)

    # Logging
    log_level = "DEBUG" if verbose else log_level
    setup_logging(level=log_level, deps_level=deps_log_level)

    # Mode selection: simulation only when explicitly requested.
    # If `--simulated-robot` is provided we run the simulator; otherwise
    # default to running on the real robot.
    if synthetic_data:
        logger.info("Using simulated robot for synthetic data generation")
    elif simulated_robot:
        logger.info("Using simulated robot")
    else:
        logger.info("Using real robot")

    

    # Filter objects if specified
    objects_to_record = AVAILABLE_OBJECTS.copy()
    if object_filter:
        filter_types = [t.strip().lower() for t in object_filter.split(",")]
        objects_to_record = [
            obj for obj in AVAILABLE_OBJECTS 
            if obj["type"].lower() in filter_types or obj["name"].lower() in filter_types
        ]
        logger.info(f"Filtered to {len(objects_to_record)} objects matching: {filter_types}")
        if not objects_to_record:
            typer.echo(f"Error: no objects match filter '{object_filter}'")
            raise typer.Exit(code=1)

    # Multi-object recording mode
    if synthetic_data:
        logger.info("=" * 70)
        logger.info("SYNTHETIC DATA RECORDING MODE")
        logger.info(f"Will generate synthetic data for {len(objects_to_record)} objects")
        logger.info("=" * 70)
        target_z = -0.1
        
        for obj_idx, obj_dict in enumerate(objects_to_record):
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"OBJECT {obj_idx + 1}/{len(objects_to_record)}: {obj_dict['name']} (type: {obj_dict['type']})")
            logger.info("=" * 70)
            
            # Create object config
            obj_config = create_object_config(obj_dict)
            
            # Run recording for this single object
            _run_single_object_recording(
                obj_config=obj_config,
                obj_name=obj_dict["name"],
                obj_dict=obj_dict,  # Pass object dict for height calculation
                synthetic_output_dir=synthetic_output_dir,
                synthetic_camera_width=synthetic_camera_width,
                synthetic_camera_height=synthetic_camera_height,
                event_threshold_pos=event_threshold_pos,
                event_threshold_neg=event_threshold_neg,
                flip_vertical=flip_vertical,
                zmq_bind=zmq_bind,
                publish_hz=publish_hz,
                headless=headless,
                agent_type=agent_type,
                sphere_radius=sphere_radius,
                num_poses=num_poses,
                wait_time=wait_time,
                random_seed=random_seed,
                target_point=np.array([target_x, target_y, target_z]),
                center_z_offset=center_z_offset,
            )
            
            logger.info(f"Completed recording for object: {obj_dict['name']}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"MULTI-OBJECT RECORDING COMPLETE: {len(objects_to_record)} objects recorded")
        logger.info("=" * 70)
        return

    # Single object / regular mode
    # Robot controller
    if simulated_robot:       
        sim_env = RobosuiteSimEnv(
            controller_type="OSC_POSE",
            has_renderer=not headless,
        )
        
        robot_controller = SimulatedRobosuiteRobotController(
            sim_env=sim_env, controller_type="OSC_POSE"
        )
    else:
        sim_env = None
        robot_controller = RealRobotController()

    robot_controller.reset_robot_joints()

    # ZMQ publisher
    pub = ZMQPosePublisher(bind_addr=zmq_bind)

    # Sync server (optional)
    sync_server = None
    if sync_recording:
        sync_server = ZMQSyncServer(bind_addr=zmq_sync_bind)
        logger.info("Sync recording mode enabled.")


    # For real robots, use a background thread to publish poses at full
    # Deoxys state rate (100 Hz), decoupled from the slower control loop.
    # For simulated robots, publish inline (control loop is fast enough).
    use_pose_thread = not simulated_robot
    pose_thread: PosePublishThread | None = None
    if use_pose_thread:
        pose_thread = PosePublishThread(robot_controller, pub, publish_hz=publish_hz)

    # Fallback inline throttle (only used when pose_thread is None)
    min_period = (1.0 / publish_hz) if publish_hz and publish_hz > 0 else 0.0
    recording_count = 0

    
    # Helper function to create the appropriate agent
    # obj parameter is optional - if provided, target z will be adjusted by half the object height
    def create_agent():
        if agent_type.lower() == "grid_hemisphere":
            logger.info("Using HemisphereGridAgent")
            return EpisodeControlWrapperAgent(HemisphereGridAgent(wait_time=wait_time))
        elif agent_type.lower() == "random_hemisphere":
            logger.info(f"Using RandomHemisphereAgent with {num_poses} poses")
            logger.info(f"  Hemisphere center = Target point: ({target_x}, {target_y}, {target_z})")
            logger.info(f"  Hemisphere radius: {sphere_radius}")
            logger.info(f"  Inner radius: {cfg.INNER_RADIUS}")
            logger.info(f"  Base exclusion radius: {cfg.BASE_EXCLUSION_RADIUS}")
            logger.info(f"  Min z-height: {cfg.MIN_Z_HEIGHT}")
            logger.info(f"  Hemisphere top cutoff: {cfg.HEMISPHERE_TOP_CUTOFF}")
            # For random_hemisphere, center equals target point
            return EpisodeControlWrapperAgent(
                RandomHemisphereAgent(
                    center=np.array([target_x, target_y, target_z]),
                    radius=sphere_radius,
                    inner_radius=cfg.INNER_RADIUS,
                    num_poses=num_poses,
                    wait_time=wait_time,
                    seed=random_seed,
                    target_point=np.array([target_x, target_y, target_z]),
                    base_exclusion_radius=cfg.BASE_EXCLUSION_RADIUS,
                    base_max_radius=cfg.BASE_MAX_RADIUS,
                    min_z_height=cfg.MIN_Z_HEIGHT,
                    hemisphere_top_cutoff=cfg.HEMISPHERE_TOP_CUTOFF,
                    lock_rotation_horizontal=cfg.LOCK_ROTATION_HORIZONTAL,
                    center_z_offset=center_z_offset,
                )
            )
        elif agent_type.lower() == "temporal_check":
            logger.info("Using TemporalAlignmentAgent (Z-rotation oscillation)")
            return EpisodeControlWrapperAgent(
                TemporalAlignmentAgent(
                    center_x=target_x,
                    center_y=target_y,
                    center_z=0.25,
                    rotation_deg=20.0,
                    num_sweeps=20,
                )
            )
        else:
            raise ValueError(f"Unknown agent_type: {agent_type!r}")

    # Special case: multi-camera calibration agent (own control loop)
    if calibration_poses or agent_type.lower() == "multi_cam_calibrate":
        from franka_pipeline.agents import (
            MultiCameraCalibrationAgent,
            MultiCameraCalibrationConfig,
        )

        logger.info("Using MultiCameraCalibrationAgent")
        config = MultiCameraCalibrationConfig()
        calib_agent = MultiCameraCalibrationAgent(
            config=config,
            poses_file=calibration_poses if calibration_poses else None,
        )
        calib_agent.start()

        step_count = 0

        if pose_thread is not None:
            pose_thread.set_ep_step(0, 0)
            pose_thread.start()

        next_pub_t = 0.0  # only used when pose_thread is None

        while True:
            robot_state, valid = robot_controller.get_state()
            if not valid:
                time.sleep(0.005)
                continue

            action, metadata = calib_agent.act(
                robot_state=robot_state,
                observation={},
                instruction="",
            )
            robot_controller.control(
                command=action, controller_type=calib_agent.action_type
            )

            step_count += 1
            if pose_thread is not None:
                pose_thread.set_ep_step(0, step_count)
            else:
                now = time.time()
                if min_period == 0.0 or now >= next_pub_t:
                    ee_pose = robot_state.get("osc_pose", None)
                    q = robot_state.get("joint_position", None)
                    joint_velocity = robot_state.get("joint_velocity", None)
                    try:
                        pub.publish_pose(
                            ep=0, step=step_count, ee_T=ee_pose, q=q, gripper_q=None,
                            joint_velocity=joint_velocity,
                        )
                    except Exception:
                        pass
                    next_pub_t = now + min_period

            if metadata.get("quit", False):
                logger.info("Multi-camera calibration finished.")
                if pose_thread is not None:
                    pose_thread.stop()
                robot_controller.reset_robot_joints()
                calib_agent.stop()
                break

        return

    # Outer loop for multiple recordings
    while True:
        # Create fresh agent for each recording
        agent = create_agent()
        
        # Wait for recording script to be ready (if in sync mode)
        if sync_server is not None:
            logger.info(f"Waiting for recording script ready signal (recording #{recording_count + 1})...")
            
            if not sync_server.wait_for_ready(timeout_sec=300.0):
                logger.info("Recording script did not send ready - assuming session ended.")
                break
            
            # Don't send 'start' yet - wait until the arm reaches the first pose
            logger.info("Recording script ready. Moving arm to first pose before starting recording...")

        # Reset for new recording
        robot_controller.reset_robot_joints()
        step_count = 0
        ep_count = 0
        next_pub_t = 0.0
        first_pose_reached = False  # Track whether we've signalled start to recording script

        pub.publish_event("episode_start", ep=ep_count)

        # Start background pose publisher for real robot
        if pose_thread is not None:
            pose_thread.set_ep_step(ep_count, step_count)
            pose_thread.start()

        # Inner loop for this recording's hemisphere traversal
        should_quit = False
        _invalid_state_start = time.time()
        while True:
            robot_state, valid = robot_controller.get_state()
            if not valid:
                if time.time() - _invalid_state_start > 10.0:
                    logger.error("Robot state never became valid after 10s. Is the robot control server running?")
                    return
                time.sleep(0.005)
                continue
            _invalid_state_start = time.time()  # reset on valid state

            # Minimal observation/instruction: agent should rely on robot_state for this task
            observation = {}
            instruction = ""

            action, metadata = agent.act(
                robot_state=robot_state,
                observation=observation,
                instruction=instruction,
            )
            robot_controller.control(command=action, controller_type=agent.action_type)

            step_count += 1
            if pose_thread is not None:
                pose_thread.set_ep_step(ep_count, step_count)
            else:
                # Inline publishing fallback (sim robot)
                now = time.time()
                if min_period == 0.0 or now >= next_pub_t:
                    ee_pose = robot_state.get("osc_pose", None)
                    q = robot_state.get("joint_position", None)
                    joint_velocity = robot_state.get("joint_velocity", None)
                    gripper_q = None
                    if "gripper_q" in robot_state:
                        try:
                            gripper_q = float(np.asarray(robot_state["gripper_q"]).reshape(-1)[0])
                        except Exception:
                            gripper_q = None

                    try:
                        pub.publish_pose(
                            ep=ep_count,
                            step=step_count,
                            ee_T=ee_pose,
                            q=q,
                            gripper_q=gripper_q,
                            joint_velocity=joint_velocity,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to publish pose: {e}")

                    next_pub_t = now + min_period

            # Send 'start' to recording script once the arm has reached its first pose
            if sync_server is not None and not first_pose_reached:
                if metadata.get("state") == "WAITING" and metadata.get("target_pose_idx", -1) == 0:
                    sync_server.send_start()
                    first_pose_reached = True
                    logger.info("First pose reached - recording started!")

            # Episode control mirrored from your main
            if metadata.get("end_episode", False):
                logger.info("Ending episode, as per agent request.")
                pub.publish_event("episode_end", ep=ep_count, extra={"step": step_count})

                robot_controller.reset_robot_joints()
                agent.reset()

                ep_count += 1
                step_count = 0

                if pose_thread is not None:
                    pose_thread.set_ep_step(ep_count, step_count)
                pub.publish_event("episode_start", ep=ep_count)

            if metadata.get("reset_episode", False):
                logger.info("Resetting episode, as per agent request.")
                pub.publish_event("episode_reset", ep=ep_count, extra={"step": step_count})

                robot_controller.reset_robot_joints()
                agent.reset()
                step_count = 0

                if pose_thread is not None:
                    pose_thread.set_ep_step(ep_count, step_count)
                # treat reset as continuing same episode id (matching your original behavior)
                pub.publish_event("episode_start", ep=ep_count)

            # Check completion signals emitted by the recording agents.
            is_complete = metadata.get("hemisphere_complete", False) or metadata.get("sphere_complete", False) or metadata.get("temporal_alignment_complete", False)
            if is_complete:
                logger.info("Agent recording complete! All poses visited.")
                if pose_thread is not None:
                    pose_thread.stop()
                pub.publish_event("agent_complete", ep=ep_count, extra={
                    "step": step_count,
                    "total_poses": metadata.get("total_poses", 0),
                })

                robot_controller.reset_robot_joints()
                if hasattr(agent, "stop"):
                    agent.stop()

                # Wait for recording script to finish this recording
                if sync_server is not None:
                    logger.info("Waiting for recording script to finish this recording...")
                    sync_server.wait_for_done(timeout_sec=60.0)
                    recording_count += 1
                    logger.info(f"Recording #{recording_count} complete. Waiting for next object...")
                
                # Break inner loop, continue to next recording
                break

            if metadata.get("quit", False):
                logger.info("Ending run, as per agent request (keyboard quit).")
                if pose_thread is not None:
                    pose_thread.stop()
                pub.publish_event("quit", ep=ep_count, extra={"step": step_count})

                robot_controller.reset_robot_joints()
                if hasattr(agent, "stop"):
                    agent.stop()

                # Wait for recording script to finish if in sync mode
                if sync_server is not None:
                    logger.info("Waiting for recording script to finish...")
                    sync_server.wait_for_done(timeout_sec=60.0)
                
                should_quit = True
                break
        
        # Check if we should exit the outer loop
        if should_quit or not sync_recording:
            # Either keyboard quit or not in multi-recording mode
            break

    # Cleanup
    if sync_server is not None:
        sync_server.close()
    logger.info(f"Session complete. Total recordings: {recording_count}")


if __name__ == "__main__":
    app()
