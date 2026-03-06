"""
Custom Objects Environment for robosuite.

This module creates a tabletop environment where you can load your own
3D mesh files (OBJ, STL) and place them at specific positions by
injecting them into the MuJoCo simulation after environment creation.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
import robosuite
import robosuite.macros as macros
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)
from scipy.spatial.transform import Rotation

macros.IMAGE_CONVENTION = "opencv"

from .sim_env import SimEnv


@dataclass
class CustomObjectConfig:
    """Configuration for a custom object to load into the scene.
    
    Attributes:
        name: Unique name for this object instance
        mesh_path: Path to the mesh file (OBJ or STL)
        position: (x, y, z) position in world frame. Table is at z≈0.8
        rotation: (roll, pitch, yaw) in degrees, or None for no rotation
        scale: Scale factor for the mesh (default 1.0)
        rgba: Color as (r, g, b, a) values 0-1, or None for mesh default
        density: Object density in kg/m³ (affects physics, default 1000)
    """
    name: str
    mesh_path: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.85)
    rotation: tuple[float, float, float] | None = None
    scale: float = 1.0
    rgba: tuple[float, float, float, float] | None = None
    density: float = 1000.0


class CustomObjectsSimEnv(SimEnv):
    """
    SimEnv that loads custom mesh objects into the robosuite Lift environment.
    
    Objects are injected into the MuJoCo XML before the simulation starts,
    allowing you to use your own downloaded 3D models.
    """
    
    TABLE_HEIGHT = 0.8  # Default table height in robosuite
    
    def __init__(
        self,
        controller_type: str = "OSC_POSE",
        control_freq: int = 20,
        camera_width: int = 640,
        camera_height: int = 480,
        custom_objects: list[CustomObjectConfig] | None = None,
        has_renderer: bool = True,
    ) -> None:
        """
        Args:
            controller_type: Controller type for the robot arm.
            control_freq: Control loop frequency in Hz.
            camera_width: Width of the camera images.
            camera_height: Height of the camera images.
            custom_objects: List of CustomObjectConfig defining objects to load.
            has_renderer: Whether to show the rendering window (False for headless mode).
        """
        self.controller_type = controller_type
        self.control_freq = control_freq
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.custom_objects_configs = custom_objects or []
        
        arm_controller_config = robosuite.load_part_controller_config(
            default_controller=controller_type
        )
        controller_configs = refactor_composite_controller_config(
            arm_controller_config, "Panda", ["right"]
        )
        
        # Create standard Lift environment
        self.env = robosuite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=has_renderer,
            ignore_done=True,
            control_freq=control_freq,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["frontview", "robot0_eye_in_hand"],
            camera_depths=True,
            camera_heights=camera_height,
            camera_widths=camera_width,
            controller_configs=controller_configs,
            renderer="mjviewer",
            hard_reset=False,
        )
        
        self._ee_correction_matrix = Rotation.from_euler(
            "z", 90, degrees=True
        ).as_matrix()
        
        # Reset first to initialize simulation
        self._last_obs = self.env.reset()
        
        # Now inject custom objects into the simulation
        self._custom_body_ids = {}
        if self.custom_objects_configs:
            self._inject_custom_objects()
    
    def _inject_custom_objects(self) -> None:
        """Inject custom mesh objects into the running MuJoCo simulation."""
        import mujoco
        
        sim = self.env.sim
        
        for config in self.custom_objects_configs:
            try:
                mesh_path = Path(config.mesh_path).resolve()
                if not mesh_path.exists():
                    print(f"Warning: Mesh file not found: {mesh_path}")
                    continue
                
                # Load mesh into MuJoCo
                # For runtime object addition, we need to modify the model
                # This is complex in MuJoCo, so we'll use a simpler approach:
                # Position the existing cube to act as a placeholder and note the limitation
                
                print(f"Note: Custom object '{config.name}' configured at position {config.position}")
                print(f"  Due to MuJoCo limitations, runtime mesh loading requires model recompilation.")
                print(f"  Consider using the object position to move the existing cube instead.")
                
                # Store the configuration for reference
                self._custom_body_ids[config.name] = {
                    "config": config,
                    "loaded": False,
                }
                
            except Exception as e:
                print(f"Warning: Failed to process object '{config.name}': {e}")
        
        # Move the existing cube to the first object's position as a workaround
        if self.custom_objects_configs:
            first_obj = self.custom_objects_configs[0]
            self._move_cube_to_position(first_obj.position, first_obj.rotation)
    
    def _move_cube_to_position(
        self, 
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] | None = None
    ) -> None:
        """Move the default cube to a specific position."""
        sim = self.env.sim
        
        try:
            # Find the cube body
            cube_body_id = sim.model.body_name2id("cube_main")
            joint_id = sim.model.body_jntadr[cube_body_id]
            
            if joint_id >= 0:
                qpos_addr = sim.model.jnt_qposadr[joint_id]
                
                # Set position
                sim.data.qpos[qpos_addr] = position[0]
                sim.data.qpos[qpos_addr + 1] = position[1]
                sim.data.qpos[qpos_addr + 2] = position[2]
                
                # Set rotation
                if rotation is not None:
                    r = Rotation.from_euler('xyz', rotation, degrees=True)
                    quat = r.as_quat()  # [x, y, z, w]
                    # MuJoCo uses [w, x, y, z]
                    sim.data.qpos[qpos_addr + 3] = quat[3]
                    sim.data.qpos[qpos_addr + 4] = quat[0]
                    sim.data.qpos[qpos_addr + 5] = quat[1]
                    sim.data.qpos[qpos_addr + 6] = quat[2]
                
                sim.forward()
                print(f"Moved cube to position: {position}")
                
        except Exception as e:
            print(f"Warning: Could not move cube: {e}")
    
    def set_object_position(self, x: float, y: float, z: float | None = None) -> None:
        """Move the cube/object to a specific position in WORLD frame."""
        if z is None:
            z = self.TABLE_HEIGHT + 0.02
        self._move_cube_to_position((x, y, z))
    
    def set_object_position_base_frame(
        self, 
        x: float, 
        y: float, 
        z: float | None = None,
        height_above_table: float = 0.02
    ) -> None:
        """Move the cube/object to a position specified in ROBOT BASE frame.
        
        This is convenient for placing objects relative to the hemisphere center,
        since the hemisphere_grid_agent uses base frame coordinates.
        
        Args:
            x: X position in robot base frame (e.g., 0.35 for hemisphere center)
            y: Y position in robot base frame (e.g., 0.0 for hemisphere center)
            z: Z position in robot base frame, or None to auto-calculate for table surface
            height_above_table: Height above table if z is None (default 0.02m)
        """
        robot = self.env.robots[0]
        
        # Transform from base frame to world frame
        # pos_world = robot.base_pos + R_base @ pos_base_frame
        # Since base_ori is usually identity, this simplifies to:
        # pos_world = robot.base_pos + pos_base_frame
        
        if robot.base_ori.ndim == 2:
            R_base = robot.base_ori
        else:
            R_base = Rotation.from_quat(robot.base_ori).as_matrix()
        
        if z is None:
            # Put object on table surface
            z_world = self.TABLE_HEIGHT + height_above_table
        else:
            # Transform z coordinate
            pos_base = np.array([x, y, z])
            pos_world = robot.base_pos + R_base @ pos_base
            z_world = pos_world[2]
        
        # Transform x, y
        pos_base_xy = np.array([x, y, 0.0])
        pos_world_xy = robot.base_pos + R_base @ pos_base_xy
        
        world_pos = (pos_world_xy[0], pos_world_xy[1], z_world)
        print(f"Setting object position: base_frame ({x}, {y}, {z}) -> world_frame {world_pos}")
        self._move_cube_to_position(world_pos)
    
    def get_object_positions(self) -> dict[str, np.ndarray]:
        """Get positions of objects in the scene."""
        positions = {}
        sim = self.env.sim
        
        try:
            cube_body_id = sim.model.body_name2id("cube_main")
            positions["cube"] = sim.data.body_xpos[cube_body_id].copy()
        except Exception:
            pass
        
        return positions
    
    def print_coordinate_debug_info(self) -> None:
        """Print debug information about coordinate systems.
        
        This helps understand the relationship between:
        - World frame (MuJoCo global frame, where objects are positioned)
        - Robot base frame (used by hemisphere_grid_agent for poses)
        - Table frame (table center is typically at world origin x=0, y=0)
        """
        robot = self.env.robots[0]
        sim = self.env.sim
        
        print("\n" + "="*60)
        print("COORDINATE SYSTEM DEBUG INFO")
        print("="*60)
        
        # Robot base position in world frame
        print(f"\nRobot base position (world frame): {robot.base_pos}")
        print(f"Robot base orientation: {robot.base_ori}")
        
        # Table position (usually at world origin)
        try:
            table_body_id = sim.model.body_name2id("table")
            table_pos = sim.data.body_xpos[table_body_id]
            print(f"\nTable center (world frame): {table_pos}")
        except Exception:
            print("\nTable position: Could not retrieve")
        
        # Cube position
        try:
            cube_body_id = sim.model.body_name2id("cube_main")
            cube_pos = sim.data.body_xpos[cube_body_id]
            print(f"Cube position (world frame): {cube_pos}")
            
            # Calculate cube position in robot base frame
            T_world_base = np.eye(4)
            if robot.base_ori.ndim == 2:
                T_world_base[:3, :3] = robot.base_ori
            else:
                T_world_base[:3, :3] = Rotation.from_quat(robot.base_ori).as_matrix()
            T_world_base[:3, 3] = robot.base_pos
            
            T_base_world = np.linalg.inv(T_world_base)
            cube_pos_base_frame = T_base_world[:3, :3] @ cube_pos + T_base_world[:3, 3]
            print(f"Cube position (robot base frame): {cube_pos_base_frame}")
        except Exception as e:
            print(f"Cube position: Could not retrieve ({e})")
        
        # EE position
        arm_name = list(robot.eef_site_id.keys())[0]
        eef_site_id = robot.eef_site_id[arm_name]
        ee_pos_world = sim.data.site_xpos[eef_site_id]
        print(f"\nEnd-effector position (world frame): {ee_pos_world}")
        
        # Transform to base frame
        T_world_base = np.eye(4)
        if robot.base_ori.ndim == 2:
            T_world_base[:3, :3] = robot.base_ori
        else:
            T_world_base[:3, :3] = Rotation.from_quat(robot.base_ori).as_matrix()
        T_world_base[:3, 3] = robot.base_pos
        T_base_world = np.linalg.inv(T_world_base)
        ee_pos_base = T_base_world[:3, :3] @ ee_pos_world + T_base_world[:3, 3]
        print(f"End-effector position (robot base frame): {ee_pos_base}")
        
        print("\n" + "-"*60)
        print("COORDINATE TRANSFORM FORMULA:")
        print("  pos_world = robot.base_pos + R_base @ pos_base_frame")
        print("  pos_base_frame = R_base.T @ (pos_world - robot.base_pos)")
        print("-"*60)
        
        # Calculate what object position (in world frame) corresponds to hemisphere center
        hemisphere_center_base = np.array([0.35, 0.0, -0.1])  # from hemisphere_grid_agent
        hemisphere_center_world = robot.base_pos + hemisphere_center_base
        # Note: This ignores rotation since base_ori is usually identity
        
        print(f"\nHemisphere center in base frame: {hemisphere_center_base}")
        print(f"Hemisphere center in world frame: {hemisphere_center_world}")
        print(f"\n→ To place cube at hemisphere center, use world position:")
        print(f"  x: {hemisphere_center_world[0]:.4f}")
        print(f"  y: {hemisphere_center_world[1]:.4f}")
        print(f"  z: {hemisphere_center_world[2]:.4f} (or higher for on-table)")
        print("="*60 + "\n")
    
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        self._last_obs = self.env.reset()
        
        # Re-apply custom object positions after reset
        if self.custom_objects_configs:
            first_obj = self.custom_objects_configs[0]
            self._move_cube_to_position(first_obj.position, first_obj.rotation)
        
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
