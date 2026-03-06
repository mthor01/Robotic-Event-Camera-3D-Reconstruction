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
"""

import os
import time
import threading
from pathlib import Path

import numpy as np
import typer

# ZMQ deps
import zmq
import msgpack

from franka_pipeline.logging import get_logger, setup_logging
from franka_pipeline.agents.episode_control_wrapper_agent import EpisodeControlWrapperAgent
from franka_pipeline.agents.hemisphere_grid_agent import HemisphereGridAgent
from franka_pipeline.agents.random_sphere_agent import RandomSphereAgent
from franka_pipeline.agents.random_hemisphere_agent import RandomHemisphereAgent
from franka_pipeline.robot_controllers.controller import (
    RealRobotController,
    SimulatedRobosuiteRobotController,
)
from franka_pipeline.sim.robosuite_env import RobosuiteSimEnv
from franka_pipeline.sim.custom_objects_env import CustomObjectsSimEnv, CustomObjectConfig
from franka_pipeline.sim.empty_table_env import EmptyTableSimEnv, ObjectConfig
from franka_pipeline.synthetic_data import SyntheticDataRecorder
from franka_pipeline.synthetic_data.synthetic_recorder import SyntheticRecorderConfig
import config_defaults as cfg

app = typer.Typer(help="Half-sphere recording runner (no cameras) + ZMQ pose stream")
logger = get_logger(__name__)

# ============================================================================
# Available objects for multi-object synthetic data generation
# ============================================================================
# These are all the object types supported by robosuite/EmptyTableSimEnv

AVAILABLE_OBJECTS = [
    # Primitive shapes (different sizes)
    {"name": "cube_small", "type": "cube", "size": 0.015, "rgba": (1.0, 0.0, 0.0, 1.0)},
    {"name": "cube_medium", "type": "cube", "size": 0.025, "rgba": (1.0, 0.0, 0.0, 1.0)},
    {"name": "cube_large", "type": "cube", "size": 0.035, "rgba": (1.0, 0.0, 0.0, 1.0)},
    
    {"name": "sphere_small", "type": "sphere", "size": [0.015], "rgba": (0.0, 1.0, 0.0, 1.0)},
    {"name": "sphere_medium", "type": "sphere", "size": [0.025], "rgba": (0.0, 1.0, 0.0, 1.0)},
    {"name": "sphere_large", "type": "sphere", "size": [0.04], "rgba": (0.0, 1.0, 0.0, 1.0)},
    
    {"name": "cylinder_small", "type": "cylinder", "size": [0.015, 0.02], "rgba": (0.0, 0.0, 1.0, 1.0)},
    {"name": "cylinder_medium", "type": "cylinder", "size": [0.02, 0.03], "rgba": (0.0, 0.0, 1.0, 1.0)},
    {"name": "cylinder_tall", "type": "cylinder", "size": [0.015, 0.05], "rgba": (0.0, 0.0, 1.0, 1.0)},
    
    {"name": "capsule_small", "type": "capsule", "size": [0.012, 0.025], "rgba": (1.0, 1.0, 0.0, 1.0)},
    {"name": "capsule_medium", "type": "capsule", "size": [0.015, 0.03], "rgba": (1.0, 1.0, 0.0, 1.0)},
    {"name": "capsule_large", "type": "capsule", "size": [0.02, 0.04], "rgba": (1.0, 1.0, 0.0, 1.0)},
    
    # Pre-made XML objects from robosuite
    {"name": "milk", "type": "milk"},
    {"name": "bread", "type": "bread"},
    {"name": "cereal", "type": "cereal"},
    {"name": "can", "type": "can"},
    {"name": "bottle", "type": "bottle"},
    {"name": "lemon", "type": "lemon"},
    {"name": "square_nut", "type": "square_nut"},
    {"name": "round_nut", "type": "round_nut"},
    
    # Composite objects from robosuite
    {"name": "pot", "type": "pot"},
    {"name": "hammer", "type": "hammer"},
    {"name": "hollow_cylinder", "type": "hollow_cylinder"},
    {"name": "cone", "type": "cone"},
]

# Robot base position in world frame (standard robosuite table setup)
ROBOT_BASE_POS_WORLD = np.array([-0.55, 0.0, 0.9])
TABLE_HEIGHT = 0.82

# Convert target position from robot base frame to world frame for object spawning
# Formula: pos_world = robot_base_pos + pos_base_frame
# Z coordinate uses table height since objects sit on the table
def get_object_spawn_position():
    """Calculate object spawn position in world frame from target in base frame."""
    world_x = ROBOT_BASE_POS_WORLD[0] + cfg.TARGET_X
    world_y = ROBOT_BASE_POS_WORLD[1] + cfg.TARGET_Y
    world_z = TABLE_HEIGHT  # Objects spawn on table surface
    return (world_x, world_y, world_z)

DEFAULT_OBJECT_POSITION = get_object_spawn_position()


# Approximate heights for pre-made XML objects (from robosuite)
# These are rough estimates - actual values may vary
PREMADE_OBJECT_HEIGHTS = {
    "milk": 0.14,         # Milk carton ~14cm tall
    "bread": 0.08,        # Bread loaf ~8cm tall
    "cereal": 0.22,       # Cereal box ~22cm tall
    "can": 0.12,          # Can ~12cm tall
    "bottle": 0.18,       # Bottle ~18cm tall
    "lemon": 0.05,        # Lemon ~5cm diameter
    "square_nut": 0.02,   # Square nut ~2cm tall
    "round_nut": 0.02,    # Round nut ~2cm tall
    "pot": 0.10,          # Pot ~10cm tall
    "hammer": 0.04,       # Hammer handle diameter ~4cm
    "hollow_cylinder": 0.08,  # Hollow cylinder ~8cm tall
    "cone": 0.08,         # Cone ~8cm tall
}


def get_object_height(obj) -> float:
    """
    Get the height of an object based on its type and size.
    Returns the height in meters.
    
    Accepts either a dict or an ObjectConfig instance.
    
    For primitive shapes:
    - cube: size is half-extent, so height = 2 * size
    - sphere: diameter = 2 * radius, so height = 2 * size[0]
    - cylinder: size = [radius, half_height], so height = 2 * size[1]
    - capsule: size = [radius, half_length], total height = 2 * (radius + half_length)
    
    For pre-made XML objects, use estimated heights.
    """
    # Handle both dict and ObjectConfig
    if isinstance(obj, dict):
        obj_type = obj.get("type", "")
        size = obj.get("size")
    else:
        # ObjectConfig dataclass
        obj_type = obj.type if obj.type else ""
        size = obj.size
    
    # Primitive shapes with explicit size
    if obj_type == "cube":
        # size is half-extent for cubes
        if isinstance(size, (int, float)):
            return 2 * size
        elif isinstance(size, (list, tuple)) and len(size) >= 1:
            return 2 * size[0]
        return 0.05  # Default cube height
    
    elif obj_type in ("sphere", "ball"):
        # size = [radius]
        if isinstance(size, (list, tuple)) and len(size) >= 1:
            return 2 * size[0]  # Diameter
        elif isinstance(size, (int, float)):
            return 2 * size
        return 0.05  # Default sphere diameter
    
    elif obj_type == "cylinder":
        # size = [radius, half_height]
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return 2 * size[1]  # Full height
        return 0.06  # Default cylinder height
    
    elif obj_type == "capsule":
        # size = [radius, half_length]
        # Total height = 2 * radius (for the caps) + 2 * half_length (cylinder part)
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            radius = size[0]
            half_length = size[1]
            return 2 * radius + 2 * half_length
        return 0.08  # Default capsule height
    
    # Pre-made XML objects
    if obj_type in PREMADE_OBJECT_HEIGHTS:
        return PREMADE_OBJECT_HEIGHTS[obj_type]
    
    # Default fallback
    return 0.05


def create_object_config(obj_dict: dict, position: tuple = DEFAULT_OBJECT_POSITION) -> ObjectConfig:
    """Create an ObjectConfig from a dictionary definition."""
    return ObjectConfig(
        name=obj_dict["name"],
        type=obj_dict["type"],
        position=position,
        rotation=obj_dict.get("rotation"),
        size=obj_dict.get("size"),
        scale=obj_dict.get("scale", 1.0),
        rgba=obj_dict.get("rgba", (1.0, 0.0, 0.0, 1.0)),
        density=obj_dict.get("density", 1000.0),
        material=obj_dict.get("material"),
    )


def list_available_objects() -> None:
    """Print all available objects for synthetic data generation."""
    print("\n" + "=" * 70)
    print("AVAILABLE OBJECTS FOR SYNTHETIC DATA GENERATION")
    print("=" * 70)
    spawn_pos = get_object_spawn_position()
    print(f"\nObject spawn position (world frame): {spawn_pos}")
    print(f"  Calculated from target position (base frame): ({cfg.TARGET_X}, {cfg.TARGET_Y}, {cfg.TARGET_Z})")
    print("\n--- PRIMITIVE SHAPES (different sizes) ---")
    primitives = [o for o in AVAILABLE_OBJECTS if o["type"] in ("cube", "box", "sphere", "ball", "cylinder", "capsule")]
    for obj in primitives:
        size_str = f", size={obj.get('size', 'default')}" if "size" in obj else ""
        print(f"  - {obj['name']} (type: {obj['type']}{size_str})")
    
    print("\n--- PRE-MADE XML OBJECTS (from robosuite) ---")
    premade = [o for o in AVAILABLE_OBJECTS if o["type"] in ("milk", "bread", "cereal", "can", "bottle", "lemon", "square_nut", "round_nut")]
    for obj in premade:
        print(f"  - {obj['name']} (type: {obj['type']})")
    
    print("\n--- COMPOSITE OBJECTS (from robosuite) ---")
    composite = [o for o in AVAILABLE_OBJECTS if o["type"] in ("pot", "hammer", "hollow_cylinder", "cone")]
    for obj in composite:
        print(f"  - {obj['name']} (type: {obj['type']})")
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {len(AVAILABLE_OBJECTS)} objects available")
    print("=" * 70 + "\n")


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
        self.pub.setsockopt(zmq.SNDHWM, 1)
        self.pub.bind(bind_addr)
        logger.info(f"ZMQ PUB bound at {bind_addr}")

    def publish_pose(
        self,
        ep: int,
        step: int,
        ee_T: np.ndarray | None,
        q: np.ndarray | None,
        gripper_q: float | None,
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

        payload = msgpack.packb(msg, use_bin_type=True)
        self.pub.send_multipart([b"pose", payload])

    def publish_event(self, event_type: str, ep: int, extra: dict | None = None) -> None:
        msg = {"t_ns": time.time_ns(), "type": event_type, "ep": int(ep)}
        if extra:
            msg.update(extra)
        payload = msgpack.packb(msg, use_bin_type=True)
        self.pub.send_multipart([b"event", payload])


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
        Returns True if ready received, False on timeout.
        """
        logger.info("Waiting for recording script 'ready' signal...")
        self.rep.setsockopt(zmq.RCVTIMEO, int(timeout_sec * 1000))

        try:
            msg_bytes = self.rep.recv()
            msg = msgpack.unpackb(msg_bytes, raw=False)

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


def _load_custom_objects_from_yaml(yaml_path: str) -> list[CustomObjectConfig]:
    """Load custom object configurations from a YAML file (legacy format)."""
    import yaml
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    objects = []
    for obj_config in config.get("objects", []):
        # Convert lists to tuples for dataclass
        if "position" in obj_config and isinstance(obj_config["position"], list):
            obj_config["position"] = tuple(obj_config["position"])
        if "rotation" in obj_config and isinstance(obj_config["rotation"], list):
            obj_config["rotation"] = tuple(obj_config["rotation"])
        if "rgba" in obj_config and isinstance(obj_config["rgba"], list):
            obj_config["rgba"] = tuple(obj_config["rgba"])
        
        objects.append(CustomObjectConfig(**obj_config))
    
    return objects


def _load_objects_from_yaml(yaml_path: str) -> list[ObjectConfig]:
    """Load object configurations from a YAML file (new format with types).
    
    YAML format:
        objects:
          - name: my_cube
            type: cube               # box, cube, ball, sphere, cylinder, capsule, milk, bread, cereal, can, bottle
            position: [0.0, 0.0, 0.82]
            rotation: [0, 0, 45]     # roll, pitch, yaw in degrees (optional)
            size: [0.02, 0.02, 0.02] # size params depend on type (optional)
            scale: 1.0               # for pre-made objects (optional)
            rgba: [1.0, 0.0, 0.0, 1.0]  # color (optional)
            density: 1000.0          # kg/m³ (optional)
            material: "WoodRed"      # texture name (optional)
    """
    import yaml
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    objects = []
    for obj_config in config.get("objects", []):
        # Convert lists to tuples for dataclass
        if "position" in obj_config and isinstance(obj_config["position"], list):
            obj_config["position"] = tuple(obj_config["position"])
        if "rotation" in obj_config and isinstance(obj_config["rotation"], list):
            obj_config["rotation"] = tuple(obj_config["rotation"])
        if "rgba" in obj_config and isinstance(obj_config["rgba"], list):
            obj_config["rgba"] = tuple(obj_config["rgba"])
        
        objects.append(ObjectConfig(**obj_config))
    
    return objects


def _has_type_field(yaml_path: str) -> bool:
    """Check if the YAML config uses the new format with 'type' field."""
    import yaml
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    objects = config.get("objects", [])
    if objects and "type" in objects[0]:
        return True
    return False


def _run_single_object_recording(
    obj_config: ObjectConfig,
    obj_name: str,
    obj_dict: dict,  # Original object dictionary for getting height
    synthetic_output_dir: str,
    synthetic_camera_id: str,
    synthetic_camera_width: int,
    synthetic_camera_height: int,
    event_threshold_pos: float,
    event_threshold_neg: float,
    flip_vertical: bool,
    zmq_bind: str,
    publish_hz: float,
    headless: bool = False,
    agent_type: str = "random_sphere",
    sphere_center: np.ndarray = None,
    sphere_radius: float = 0.3,
    num_poses: int = 20,
    wait_time: float = 0.0,
    random_seed: int = None,
    target_point: np.ndarray = None,
) -> None:
    """
    Run a single recording for one object.
    This is called by multi-object recording mode.
    """
    import robosuite
    from robosuite.controllers.composite.composite_controller_factory import (
        refactor_composite_controller_config,
    )
    
    cam_width = synthetic_camera_width
    cam_height = synthetic_camera_height
    
    # Create environment with just this one object
    logger.info(f"Creating simulation environment with object: {obj_name}")
    logger.info(f"  - Type: {obj_config.type}")
    logger.info(f"  - Position: {obj_config.position}")
    if hasattr(obj_config, 'rgba') and obj_config.rgba:
        logger.info(f"  - Color (RGBA): {obj_config.rgba}")
    
    sim_env = EmptyTableSimEnv(
        controller_type="OSC_POSE",
        camera_width=cam_width,
        camera_height=cam_height,
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
        camera_id=synthetic_camera_id,
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
                lock_rotation_horizontal=cfg.LOCK_ROTATION_HORIZONTAL,
            )
        )
    else:  # Default to random_sphere
        if sphere_center is None:
            sphere_center = np.array([0.4, 0.0, 0.0])
        if target_point is None:
            target_point = np.array([0.35, 0.0, -0.1])
        agent = EpisodeControlWrapperAgent(
            RandomSphereAgent(
                center=sphere_center,
                radius=sphere_radius,
                num_poses=num_poses,
                wait_time=wait_time,
                seed=random_seed,
                target_point=target_point,
            )
        )
    
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
        cfg.SIMULATED_ROBOT, "--simulated-robot/--no-simulated-robot", help="Run in robosuite sim"
    ),
    real_robot: bool = typer.Option(
        cfg.REAL_ROBOT, "--real-robot/--no-real-robot", help="Run on real robot"
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
    synthetic_camera_id: str = typer.Option(
        cfg.SYNTHETIC_CAMERA_ID,
        "--synthetic-camera-id",
        help="Camera ID to use for synthetic data (e.g., 'robot0_eye_in_hand', 'frontview').",
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
    # Custom objects options
    custom_objects_config: str = typer.Option(
        cfg.CUSTOM_OBJECTS_CONFIG,
        "--custom-objects-config",
        help="Path to YAML file defining custom objects to load (mesh paths, positions, scales).",
    ),
    # Multi-object synthetic data recording options
    multi_object_recording: bool = typer.Option(
        cfg.MULTI_OBJECT_RECORDING,
        "--multi-object-recording/--no-multi-object-recording",
        help="Generate synthetic data for ALL available objects automatically. Each object gets its own recording.",
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
        help="Agent type to use: 'hemisphere', 'random_sphere', or 'random_hemisphere'. Default: random_sphere",
    ),
    sphere_center_x: float = typer.Option(
        cfg.SPHERE_CENTER_X,
        "--sphere-center-x",
        help="X coordinate of sphere center (robot base frame).",
    ),
    sphere_center_y: float = typer.Option(
        cfg.SPHERE_CENTER_Y,
        "--sphere-center-y",
        help="Y coordinate of sphere center (robot base frame).",
    ),
    sphere_center_z: float = typer.Option(
        cfg.SPHERE_CENTER_Z,
        "--sphere-center-z",
        help="Z coordinate of sphere center (robot base frame).",
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
    # Agent params (you can expand as needed)
    # NOTE: If your HalfSphereRecordingAgent takes args, add them here.
) -> None:
    # Handle --list-objects early
    if list_objects:
        list_available_objects()
        raise typer.Exit(code=0)

    # Logging
    log_level = "DEBUG" if verbose else log_level
    setup_logging(level=log_level, deps_level=deps_log_level)

    # Mode selection
    if not simulated_robot and not real_robot:
        simulated_robot = True
        logger.info("No mode specified -> defaulting to simulated robot")

    if simulated_robot and real_robot:
        typer.echo("Error: cannot set both --simulated-robot and --real-robot")
        raise typer.Exit(code=1)

    # Validate synthetic data option
    if synthetic_data and not simulated_robot:
        typer.echo("Error: --synthetic-data requires --simulated-robot")
        raise typer.Exit(code=1)

    # Validate multi-object recording option
    if multi_object_recording and not synthetic_data:
        typer.echo("Error: --multi-object-recording requires --synthetic-data")
        raise typer.Exit(code=1)
    
    if multi_object_recording and custom_objects_config:
        typer.echo("Error: cannot use both --multi-object-recording and --custom-objects-config")
        raise typer.Exit(code=1)

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
    if multi_object_recording:
        logger.info("=" * 70)
        logger.info("MULTI-OBJECT SYNTHETIC DATA RECORDING MODE")
        logger.info(f"Will generate synthetic data for {len(objects_to_record)} objects")
        logger.info("=" * 70)
        
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
                synthetic_camera_id=synthetic_camera_id,
                synthetic_camera_width=synthetic_camera_width,
                synthetic_camera_height=synthetic_camera_height,
                event_threshold_pos=event_threshold_pos,
                event_threshold_neg=event_threshold_neg,
                flip_vertical=flip_vertical,
                zmq_bind=zmq_bind,
                publish_hz=publish_hz,
                headless=headless,
                agent_type=agent_type,
                sphere_center=np.array([sphere_center_x, sphere_center_y, sphere_center_z]),
                sphere_radius=sphere_radius,
                num_poses=num_poses,
                wait_time=wait_time,
                random_seed=random_seed,
                target_point=np.array([target_x, target_y, target_z]),
            )
            
            logger.info(f"Completed recording for object: {obj_dict['name']}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"MULTI-OBJECT RECORDING COMPLETE: {len(objects_to_record)} objects recorded")
        logger.info("=" * 70)
        return

    # Single object / regular mode
    # Robot controller
    custom_objects = None  # Will be populated if custom_objects_config is provided
    if simulated_robot:
        # Use higher resolution if synthetic data recording is enabled
        cam_width = synthetic_camera_width if synthetic_data else 64
        cam_height = synthetic_camera_height if synthetic_data else 64
        
        # Check if using custom objects
        if custom_objects_config:
            # Check if using new format (with 'type' field) or legacy format
            if _has_type_field(custom_objects_config):
                # New format: use EmptyTableSimEnv with proper object types
                custom_objects = _load_objects_from_yaml(custom_objects_config)
                logger.info(f"Loading {len(custom_objects)} objects from {custom_objects_config} (new format)")
                for obj in custom_objects:
                    logger.info(f"  - {obj.name} ({obj.type}) at {obj.position}")
                sim_env = EmptyTableSimEnv(
                    controller_type="OSC_POSE",
                    camera_width=cam_width,
                    camera_height=cam_height,
                    custom_objects=custom_objects,
                    has_renderer=not headless,
                )
            else:
                # Legacy format: use CustomObjectsSimEnv (just moves the cube)
                custom_objects = _load_custom_objects_from_yaml(custom_objects_config)
                logger.info(f"Loading {len(custom_objects)} custom objects from {custom_objects_config} (legacy format)")
                sim_env = CustomObjectsSimEnv(
                    controller_type="OSC_POSE",
                    camera_width=cam_width,
                    camera_height=cam_height,
                    custom_objects=custom_objects,
                    has_renderer=not headless,
                )
            # Print coordinate system debug info
            sim_env.print_coordinate_debug_info()
            # Log object positions
            positions = sim_env.get_object_positions()
            for name, pos in positions.items():
                logger.info(f"  Object '{name}': x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        else:
            sim_env = RobosuiteSimEnv(
                controller_type="OSC_POSE",
                camera_width=cam_width,
                camera_height=cam_height,
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

    # Synthetic data recorder (optional)
    synthetic_recorder = None
    if synthetic_data:
        config = SyntheticRecorderConfig(
            output_dir=Path(synthetic_output_dir),
            camera_id=synthetic_camera_id,
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
        logger.info(f"Synthetic data recording enabled. Output: {synthetic_output_dir}")
        if flip_vertical:
            logger.info("Vertical flip enabled for all synthetic data")

    # Throttle publishing if desired
    min_period = (1.0 / publish_hz) if publish_hz and publish_hz > 0 else 0.0

    recording_count = 0
    
    # Helper function to create the appropriate agent
    # obj parameter is optional - if provided, target z will be adjusted by half the object height
    def create_agent(obj=None):
        # Calculate height adjustment based on object
        obj_height = get_object_height(obj) if obj else 0.0
        adjusted_target_z = target_z + obj_height / 2
        
        if obj_height > 0:
            logger.info(f"  Object height: {obj_height:.3f}m, adjusting target z by {obj_height/2:.3f}m")
            logger.info(f"  Adjusted target z: {target_z:.3f} -> {adjusted_target_z:.3f}")
        
        if agent_type.lower() == "hemisphere":
            logger.info("Using HemisphereGridAgent")
            return EpisodeControlWrapperAgent(HemisphereGridAgent(wait_time=wait_time))
        elif agent_type.lower() == "random_hemisphere":
            logger.info(f"Using RandomHemisphereAgent with {num_poses} poses")
            logger.info(f"  Hemisphere center = Target point: ({target_x}, {target_y}, {adjusted_target_z})")
            logger.info(f"  Hemisphere radius: {sphere_radius}")
            logger.info(f"  Inner radius: {cfg.INNER_RADIUS}")
            logger.info(f"  Base exclusion radius: {cfg.BASE_EXCLUSION_RADIUS}")
            logger.info(f"  Min z-height: {cfg.MIN_Z_HEIGHT}")
            # For random_hemisphere, center equals target point
            return EpisodeControlWrapperAgent(
                RandomHemisphereAgent(
                    center=np.array([target_x, target_y, adjusted_target_z]),
                    radius=sphere_radius,
                    inner_radius=cfg.INNER_RADIUS,
                    num_poses=num_poses,
                    wait_time=wait_time,
                    seed=random_seed,
                    target_point=np.array([target_x, target_y, adjusted_target_z]),
                    base_exclusion_radius=cfg.BASE_EXCLUSION_RADIUS,
                    base_max_radius=cfg.BASE_MAX_RADIUS,
                    min_z_height=cfg.MIN_Z_HEIGHT,
                    lock_rotation_horizontal=cfg.LOCK_ROTATION_HORIZONTAL,
                )
            )
        else:  # Default to random_sphere
            logger.info(f"Using RandomSphereAgent with {num_poses} poses")
            logger.info(f"  Sphere center: ({sphere_center_x}, {sphere_center_y}, {sphere_center_z})")
            logger.info(f"  Target point: ({target_x}, {target_y}, {adjusted_target_z})")
            logger.info(f"  Sphere radius: {sphere_radius}")
            return EpisodeControlWrapperAgent(
                RandomSphereAgent(
                    center=np.array([sphere_center_x, sphere_center_y, sphere_center_z]),
                    radius=sphere_radius,
                    num_poses=num_poses,
                    wait_time=wait_time,
                    seed=random_seed,
                    target_point=np.array([target_x, target_y, adjusted_target_z]),
                )
            )
    
    # Outer loop for multiple recordings
    while True:
        # Create fresh agent for each recording
        # Pass first object if available for height adjustment
        first_obj = custom_objects[0] if custom_objects else None
        agent = create_agent(first_obj)
        
        # Wait for recording script to be ready (if in sync mode)
        if sync_server is not None:
            logger.info(f"Waiting for recording script ready signal (recording #{recording_count + 1})...")
            
            if not sync_server.wait_for_ready(timeout_sec=300.0):
                logger.info("Recording script did not send ready - assuming session ended.")
                break
            
            # Send start signal to begin synchronized recording
            sync_server.send_start()
            logger.info("Recording started - robot motion beginning!")

        # Reset for new recording
        robot_controller.reset_robot_joints()
        step_count = 0
        ep_count = 0
        next_pub_t = 0.0

        # Start synthetic recording for this session
        if synthetic_recorder is not None:
            recording_id = "synthetic"
            synthetic_recorder.start_recording(recording_id)
            logger.info(f"Started synthetic recording: {recording_id}")

        pub.publish_event("episode_start", ep=ep_count)

        # Inner loop for this recording's hemisphere traversal
        should_quit = False
        while True:
            robot_state, valid = robot_controller.get_state()
            if not valid:
                logger.warning("Invalid robot state, skipping iteration.")
                continue

            # Minimal observation/instruction: agent should rely on robot_state for this task
            observation = {}
            instruction = ""

            action, metadata = agent.act(
                robot_state=robot_state,
                observation=observation,
                instruction=instruction,
            )
            robot_controller.control(command=action, controller_type=agent.action_type)

            # Record synthetic data (if enabled)
            if synthetic_recorder is not None and synthetic_recorder.is_recording:
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

            # Publish pose at most publish_hz (or every loop if publish_hz=0)
            now = time.time()
            if min_period == 0.0 or now >= next_pub_t:
                ee_pose = robot_state.get("osc_pose", None)
                q = robot_state.get("joint_position", None)
                gripper_q = None
                if "gripper_q" in robot_state:
                    # your original code did robot_state["gripper_q"][np.newaxis], so it might be scalar-like
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
                    )
                except Exception as e:
                    logger.warning(f"Failed to publish pose: {e}")

                next_pub_t = now + min_period

            step_count += 1

            # Episode control mirrored from your main
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

                # treat reset as continuing same episode id (matching your original behavior)
                pub.publish_event("episode_start", ep=ep_count)

            # Check if agent is complete (from HemisphereGridAgent or RandomSphereAgent)
            is_complete = metadata.get("hemisphere_complete", False) or metadata.get("sphere_complete", False)
            if is_complete:
                logger.info("Agent recording complete! All poses visited.")
                pub.publish_event("agent_complete", ep=ep_count, extra={
                    "step": step_count,
                    "total_poses": metadata.get("total_poses", 0),
                })

                # Stop synthetic recording and save data
                if synthetic_recorder is not None and synthetic_recorder.is_recording:
                    summary = synthetic_recorder.stop_recording()
                    logger.info(f"Synthetic recording saved: {summary.get('output_path')}")
                    logger.info(f"  - Frames: {summary.get('num_frames', 0)}")
                    logger.info(f"  - Events: {summary.get('total_events', 0)}")

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
                pub.publish_event("quit", ep=ep_count, extra={"step": step_count})

                # Stop synthetic recording if active
                if synthetic_recorder is not None and synthetic_recorder.is_recording:
                    summary = synthetic_recorder.stop_recording()
                    logger.info(f"Synthetic recording saved on quit: {summary.get('output_path')}")

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
