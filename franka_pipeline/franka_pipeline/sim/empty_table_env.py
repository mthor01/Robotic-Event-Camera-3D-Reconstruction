"""
Empty Table Environment for robosuite.

This module creates a tabletop environment with no predefined task,
allowing you to place any combination of objects on the table.
Supports both primitive shapes (box, ball, cylinder, capsule) and
pre-made objects (milk, cereal, can, etc.).
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite
import robosuite.macros as macros
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import (
    # Primitives
    BoxObject,
    BallObject,
    CylinderObject,
    CapsuleObject,
    # XML Objects (pre-made)
    MilkObject,
    BreadObject,
    CerealObject,
    CanObject,
    BottleObject,
    LemonObject,
    SquareNutObject,
    RoundNutObject,
    # Composite Objects
    PotWithHandlesObject,
    HammerObject,
    HollowCylinderObject,
    ConeObject,
)
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

macros.IMAGE_CONVENTION = "opencv"

from .sim_env import SimEnv


# Available object types
PRIMITIVE_OBJECTS = {"box", "ball", "sphere", "cylinder", "capsule", "cube"}
PREMADE_OBJECTS = {"milk", "bread", "cereal", "can", "bottle", "lemon", "square_nut", "round_nut"}
COMPOSITE_OBJECTS = {"pot", "hammer", "hollow_cylinder", "cone"}
ALL_OBJECT_TYPES = PRIMITIVE_OBJECTS | PREMADE_OBJECTS | COMPOSITE_OBJECTS


@dataclass
class ObjectConfig:
    """Configuration for an object to place in the scene.
    
    Attributes:
        name: Unique name for this object instance
        type: Object type - one of: 'box'/'cube', 'ball'/'sphere', 'cylinder', 
              'capsule', 'milk', 'bread', 'cereal', 'can', 'bottle'
        position: (x, y, z) position in world frame. Table surface is at z≈0.8
        rotation: (roll, pitch, yaw) in degrees, or None for no rotation
        size: Size parameters (meaning depends on type):
              - box/cube: [half_x, half_y, half_z] or single value for cube
              - ball/sphere: [radius]
              - cylinder: [radius, half_height]
              - capsule: [radius, half_length]
              - premade objects: ignored (use scale instead)
        scale: Scale factor for premade objects (default 1.0)
        rgba: Color as (r, g, b, a) values 0-1
        density: Object density in kg/m³ (default 1000)
        material: Material name for texture (e.g., 'WoodRed', 'Steel', 'Brass')
    """
    name: str
    type: str = "box"
    position: tuple[float, float, float] = (0.0, 0.0, 0.85)
    rotation: tuple[float, float, float] | None = None
    size: list[float] | float | None = None
    scale: float = 1.0
    rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    density: float = 1000.0
    material: str | None = None


class EmptyTableEnv(ManipulationEnv):
    """
    Empty tabletop environment where you can place any objects.
    No predefined task - just a table with a robot and your custom objects.
    """
    
    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="frontview",
        control_freq=20,
        horizon=1000,
        seed=None,
        camera_names=None,
        camera_heights=480,
        camera_widths=640,
        camera_depths=True,
        custom_objects: list[ObjectConfig] | None = None,
        renderer="mjviewer",
        **kwargs
    ):
        """
        Args:
            custom_objects: List of ObjectConfig defining objects to place on the table.
            ... (other args are standard robosuite ManipulationEnv args)
        """
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)
        self.use_object_obs = use_object_obs
        self.custom_objects_configs = custom_objects or []
        
        # Default camera setup
        if camera_names is None:
            camera_names = ["frontview", "robot0_eye_in_hand"]

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            control_freq=control_freq,
            horizon=horizon,
            seed=seed,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            renderer=renderer,
            **kwargs
        )

    def reward(self, action=None):
        """No reward - this is a free-form environment."""
        return 0.0

    def _load_model(self):
        """Load the arena and objects into the simulation."""
        super()._load_model()

        # Adjust robot base pose for table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create empty table arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Create objects from config
        self.objects = []
        self._object_configs_map = {}
        
        for obj_config in self.custom_objects_configs:
            obj = self._create_object(obj_config)
            if obj is not None:
                self.objects.append(obj)
                self._object_configs_map[obj_config.name] = obj_config
                print(f"Created object '{obj_config.name}' (type: {obj_config.type})")

        # Create placement initializer (positions will be overridden in reset)
        if self.objects:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-0.15, 0.15],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        else:
            self.placement_initializer = None

        # Create the task (combines arena + robot + objects)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )
    
    def _create_object(self, config: ObjectConfig):
        """Create a robosuite object from config."""
        obj_type = config.type.lower()
        
        # Determine size
        if config.size is None:
            # Default sizes
            if obj_type in ("box", "cube"):
                size = [0.02, 0.02, 0.02]
            elif obj_type in ("ball", "sphere"):
                size = [0.025]
            elif obj_type == "cylinder":
                size = [0.02, 0.03]
            elif obj_type == "capsule":
                size = [0.02, 0.04]
            else:
                size = None
        elif isinstance(config.size, (int, float)):
            # Single value - interpret based on type
            if obj_type in ("box", "cube"):
                size = [config.size, config.size, config.size]
            else:
                size = [config.size]
        else:
            size = list(config.size)
        
        # Create material if specified
        material = None
        if config.material:
            tex_attrib = {"type": "cube"}
            mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
            material = CustomMaterial(
                texture=config.material,
                tex_name=f"{config.name}_tex",
                mat_name=f"{config.name}_mat",
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
        
        # Create primitive objects
        if obj_type in ("box", "cube"):
            return BoxObject(
                name=config.name,
                size=size,
                rgba=list(config.rgba),
                material=material,
                density=config.density,
            )
        elif obj_type in ("ball", "sphere"):
            return BallObject(
                name=config.name,
                size=size,
                rgba=list(config.rgba),
                material=material,
                density=config.density,
            )
        elif obj_type == "cylinder":
            return CylinderObject(
                name=config.name,
                size=size,
                rgba=list(config.rgba),
                material=material,
                density=config.density,
            )
        elif obj_type == "capsule":
            return CapsuleObject(
                name=config.name,
                size=size,
                rgba=list(config.rgba),
                material=material,
                density=config.density,
            )
        # Create pre-made objects
        elif obj_type == "milk":
            return MilkObject(name=config.name)
        elif obj_type == "bread":
            return BreadObject(name=config.name)
        elif obj_type == "cereal":
            return CerealObject(name=config.name)
        elif obj_type == "can":
            return CanObject(name=config.name)
        elif obj_type == "bottle":
            return BottleObject(name=config.name)
        elif obj_type == "lemon":
            return LemonObject(name=config.name)
        elif obj_type == "square_nut":
            return SquareNutObject(name=config.name)
        elif obj_type == "round_nut":
            return RoundNutObject(name=config.name)
        # Create composite objects
        elif obj_type == "pot":
            return PotWithHandlesObject(name=config.name)
        elif obj_type == "hammer":
            return HammerObject(name=config.name)
        elif obj_type == "hollow_cylinder":
            return HollowCylinderObject(name=config.name)
        elif obj_type == "cone":
            return ConeObject(name=config.name)
        else:
            print(f"Warning: Unknown object type '{obj_type}'. Available types: {ALL_OBJECT_TYPES}")
            return None

    def _setup_references(self):
        """Set up references to object body IDs."""
        super()._setup_references()
        self.object_body_ids = {}
        for obj in self.objects:
            self.object_body_ids[obj.name] = self.sim.model.body_name2id(obj.root_body)

    def _setup_observables(self):
        """Set up observation functions."""
        observables = super()._setup_observables()
        return observables

    def _reset_internal(self):
        """Reset the environment and place objects at configured positions."""
        super()._reset_internal()
        
        # Place each object at its configured position
        for obj in self.objects:
            if obj.name in self._object_configs_map:
                config = self._object_configs_map[obj.name]
                self._set_object_pose(obj, config.position, config.rotation)

    def _set_object_pose(self, obj, position, rotation_euler=None):
        """Set an object's position and rotation."""
        try:
            # Get joint name for this object
            joint_name = obj.joints[0]
            
            # Calculate quaternion from euler angles
            if rotation_euler is not None:
                r = Rotation.from_euler('xyz', rotation_euler, degrees=True)
                quat = r.as_quat()  # [x, y, z, w]
            else:
                quat = [0, 0, 0, 1]  # Identity quaternion
            
            # MuJoCo expects [w, x, y, z] for quaternion
            quat_mujoco = [quat[3], quat[0], quat[1], quat[2]]
            
            # Set the joint position (pos + quat)
            self.sim.data.set_joint_qpos(
                joint_name,
                np.concatenate([np.array(position), np.array(quat_mujoco)])
            )
            self.sim.forward()
        except Exception as e:
            print(f"Warning: Could not set pose for object '{obj.name}': {e}")

    def _check_success(self):
        """No success condition - free-form environment."""
        return False

    def visualize(self, vis_settings=None):
        """Visualize the environment."""
        super().visualize(vis_settings=vis_settings)


class EmptyTableSimEnv(SimEnv):
    """
    SimEnv wrapper around EmptyTableEnv for use with the franka_pipeline.
    
    This provides the same interface as RobosuiteSimEnv but with the custom
    EmptyTableEnv that supports arbitrary object placement.
    """
    
    TABLE_HEIGHT = 0.8
    
    def __init__(
        self,
        controller_type: str = "OSC_POSE",
        control_freq: int = 20,
        camera_width: int = 640,
        camera_height: int = 480,
        custom_objects: list[ObjectConfig] | None = None,
        has_renderer: bool = True,
    ) -> None:
        """
        Args:
            controller_type: Controller type for the robot arm.
            control_freq: Control loop frequency in Hz.
            camera_width: Width of the camera images.
            camera_height: Height of the camera images.
            custom_objects: List of ObjectConfig defining objects to load.
            has_renderer: Whether to show the rendering window (False for headless mode).
        """
        self.controller_type = controller_type
        self.control_freq = control_freq
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.custom_objects_configs = custom_objects or []
        
        # Set up controller
        arm_controller_config = robosuite.load_part_controller_config(
            default_controller=controller_type
        )
        controller_configs = refactor_composite_controller_config(
            arm_controller_config, "Panda", ["right"]
        )
        
        # Create the environment
        self.env = EmptyTableEnv(
            robots="Panda",
            controller_configs=controller_configs,
            has_renderer=has_renderer,
            ignore_done=True,
            control_freq=control_freq,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["frontview", "robot0_eye_in_hand"],
            camera_depths=True,
            camera_heights=camera_height,
            camera_widths=camera_width,
            renderer="mjviewer",
            hard_reset=False,
            custom_objects=custom_objects,
        )
        
        self._ee_correction_matrix = Rotation.from_euler(
            "z", 90, degrees=True
        ).as_matrix()
        
        self._last_obs = self.env.reset()
    
    def get_object_positions(self) -> dict[str, np.ndarray]:
        """Get positions of all objects in the scene."""
        positions = {}
        sim = self.env.sim
        
        for obj in self.env.objects:
            try:
                body_id = sim.model.body_name2id(obj.root_body)
                positions[obj.name] = sim.data.body_xpos[body_id].copy()
            except Exception:
                pass
        
        return positions
    
    def set_object_position(self, name: str, x: float, y: float, z: float | None = None) -> None:
        """Move an object to a specific position in WORLD frame."""
        if z is None:
            z = self.TABLE_HEIGHT + 0.02
        
        for obj in self.env.objects:
            if obj.name == name:
                config = self.env._object_configs_map.get(name)
                rotation = config.rotation if config else None
                self.env._set_object_pose(obj, (x, y, z), rotation)
                return
        
        print(f"Warning: Object '{name}' not found")
    
    def print_coordinate_debug_info(self) -> None:
        """Print debug information about coordinate systems."""
        robot = self.env.robots[0]
        sim = self.env.sim
        
        print("\n" + "="*60)
        print("COORDINATE SYSTEM DEBUG INFO")
        print("="*60)
        
        print(f"\nRobot base position (world frame): {robot.base_pos}")
        
        try:
            table_body_id = sim.model.body_name2id("table")
            table_pos = sim.data.body_xpos[table_body_id]
            print(f"Table center (world frame): {table_pos}")
        except Exception:
            print("Table position: Could not retrieve")
        
        # Print all object positions
        print("\nObject positions:")
        for obj in self.env.objects:
            try:
                body_id = sim.model.body_name2id(obj.root_body)
                pos = sim.data.body_xpos[body_id]
                print(f"  {obj.name}: {pos}")
            except Exception as e:
                print(f"  {obj.name}: Could not retrieve ({e})")
        
        print("="*60 + "\n")
    
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        self._last_obs = self.env.reset()
        return self._last_obs
    
    def step(self, action: Any) -> dict[str, Any]:
        import time
        start = time.perf_counter()
        
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        if self.env.viewer is not None:
            self.env.render()
        
        time_per_step = 1.0 / self.control_freq
        elapsed = time.perf_counter() - start
        if elapsed < time_per_step:
            time.sleep(time_per_step - elapsed)
        return obs
    
    def get_state(self) -> dict[str, Any]:
        """Get current robot state."""
        robot = self.env.robots[0]
        sim = self.env.sim
        
        arm_name = list(robot.eef_site_id.keys())[0]
        eef_site_id = robot.eef_site_id[arm_name]
        
        osc_position_world = sim.data.site_xpos[eef_site_id]
        osc_rotation_matrix_world = sim.data.site_xmat[eef_site_id].reshape(3, 3)
        
        T_world_ee = np.eye(4)
        T_world_ee[:3, :3] = osc_rotation_matrix_world
        T_world_ee[:3, 3] = osc_position_world
        
        T_world_base = np.eye(4)
        if robot.base_ori.ndim == 2:
            T_world_base[:3, :3] = robot.base_ori
        else:
            T_world_base[:3, :3] = Rotation.from_quat(robot.base_ori).as_matrix()
        T_world_base[:3, 3] = robot.base_pos
        
        T_base_ee = np.linalg.inv(T_world_base) @ T_world_ee
        T_base_ee[:3, :3] = T_base_ee[:3, :3] @ self._ee_correction_matrix
        
        osc_pose = T_base_ee
        osc_position = T_base_ee[:3, 3]
        osc_rotation_matrix = T_base_ee[:3, :3]
        osc_rotation_quaternion = Rotation.from_matrix(osc_rotation_matrix).as_quat()
        
        joint_positions_raw = robot._joint_positions
        joint_velocities_raw = robot._joint_velocities
        if isinstance(joint_positions_raw, dict):
            joint_position = np.array(joint_positions_raw[arm_name]).flatten()
            joint_velocity = np.array(joint_velocities_raw[arm_name]).flatten()
        else:
            joint_position = np.array(joint_positions_raw).flatten()
            joint_velocity = np.array(joint_velocities_raw).flatten()
        
        gripper_joints = robot.gripper_joints
        if gripper_joints is not None and hasattr(robot, "sim"):
            if isinstance(gripper_joints, dict):
                gripper_joint_names = gripper_joints[arm_name]
            else:
                gripper_joint_names = gripper_joints
            gripper_joint_positions = np.array([
                robot.sim.data.qpos[robot.sim.model.joint_name2id(joint)]
                for joint in gripper_joint_names
            ])
            gripper_q = np.array(np.sum(np.abs(gripper_joint_positions)))
        else:
            gripper_q = np.array(0.0)
        
        return {
            "osc_pose": osc_pose,
            "osc_position": osc_position,
            "osc_rotation_matrix": osc_rotation_matrix,
            "osc_rotation_quaternion": osc_rotation_quaternion,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_q": gripper_q,
        }
    
    def get_camera_intrinsics(self, camera_id: str) -> dict[str, float] | None:
        if not hasattr(self.env, "sim"):
            return None
        try:
            sim = self.env.sim
            cam_id = sim.model.camera_name2id(camera_id)
            fovy = sim.model.cam_fovy[cam_id]
            height = self.camera_height
            width = self.camera_width
            fovy_rad = np.deg2rad(fovy)
            f = (height / 2) / np.tan(fovy_rad / 2)
            return {"fx": f, "fy": f, "cx": width / 2, "cy": height / 2}
        except Exception:
            return None
    
    def get_camera_extrinsics(self, camera_id: str) -> np.ndarray | None:
        if not hasattr(self.env, "sim"):
            return None
        try:
            sim = self.env.sim
            robot = self.env.robots[0]
            
            arm_name = list(robot.eef_site_id.keys())[0]
            eef_site_id = robot.eef_site_id[arm_name]
            
            eef_pos = sim.data.site_xpos[eef_site_id]
            eef_mat = sim.data.site_xmat[eef_site_id].reshape(3, 3)
            
            T_world_ee = np.eye(4)
            T_world_ee[:3, :3] = eef_mat
            T_world_ee[:3, 3] = eef_pos
            
            cam_id = sim.model.camera_name2id(camera_id)
            cam_pos = sim.data.cam_xpos[cam_id]
            cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
            
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = cam_mat
            T_world_cam[:3, 3] = cam_pos
            
            R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            T_flip = np.eye(4)
            T_flip[:3, :3] = R_flip
            T_world_cam = T_world_cam @ T_flip
            
            T_ee_cam = np.linalg.inv(T_world_ee) @ T_world_cam
            T_ee_cam[:3, :3] = self._ee_correction_matrix.T @ T_ee_cam[:3, :3]
            T_ee_cam[:3, 3] = self._ee_correction_matrix.T @ T_ee_cam[:3, 3]
            
            return T_ee_cam
        except Exception:
            return None
    
    def get_image(
        self,
        camera_id: str,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
    ) -> Any:
        if segmentation:
            raise NotImplementedError("Segmentation not implemented.")
        
        suffix = "_image" if not depth else "_depth"
        obs_key = camera_id + suffix
        
        if obs_key and self._last_obs is not None and obs_key in self._last_obs:
            img = self._last_obs[obs_key]
            if depth and img is not None:
                sim = self.env.sim
                extent = sim.model.stat.extent
                znear = sim.model.vis.map.znear * extent
                zfar = sim.model.vis.map.zfar * extent
                d = img
                c = 1.0 - (znear / zfar)
                z_meters = znear / (1.0 - d * c)
                return z_meters
            return img
        return None
    
    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
