# TODO this is heavily vibe-coded and not yet checked.
# Esp. epsiode end and reset does not correctly work, it just stores into one consecutive rerun database.
# But for now, for visualization it suffices.

"""Live dataset visualization using Rerun.

This module provides a `LiveVisualizer` class to stream observations,
actions, and robot state to a Rerun viewer while recording.

Usage:
    visualizer = LiveVisualizer(repo_name="your_hf_username/my_dataset")
    visualizer.add_data(observation, action, robot_state, metadata)

The `add_data` method expects the same inputs and structure as
`DataCollector.collect(...)` in this project.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation


from franka_pipeline.logging import get_logger

logger = get_logger(__name__)


def _log_angle_rot(
    entity_to_transform: Dict[str, Tuple[np.ndarray, np.ndarray]],
    link: int,
    angle_rad: float,
    root_path: str = "",
    axis_length: float | None = None,
) -> None:
    """Log a joint angle rotation for the Franka Panda robot.

    All joint angles describe rotations around the transformed z-axis.

    Args:
        entity_to_transform: Mapping from entity path to (translation, rotation_matrix).
        link: The link index (1-indexed for Franka).
        angle_rad: Rotation angle in radians.
        root_path: Prefix for the entity path.
        axis_length: Optional length of the coordinate axes to display.
    """
    # Try different naming conventions
    # 1. Original panda convention: panda_link0/panda_link1/...
    path_panda = "/".join(f"panda_link{i}" for i in range(link + 1))

    # 2. New fer convention: base/fer_link0/fer_link1/...
    path_fer = "base/" + "/".join(f"fer_link{i}" for i in range(link + 1))

    full_path = root_path + path_panda
    if full_path not in entity_to_transform:
        full_path = root_path + path_fer
        if full_path not in entity_to_transform:
            # Could not find path, skip update
            return

    start_translation, start_rotation_mat = entity_to_transform[full_path]

    # Rotation is always around the Z axis in URDF joint frame
    vec = np.array([0, 0, 1]) * angle_rad
    rot = Rotation.from_rotvec(vec).as_matrix()
    rotation_mat = start_rotation_mat @ rot

    rr.log(
        full_path,
        rr.Transform3D(
            translation=start_translation, mat3x3=rotation_mat, axis_length=axis_length
        ),
    )


def _log_prismatic(
    entity_to_transform: Dict[str, Tuple[np.ndarray, np.ndarray]],
    full_path: str,
    position: float,
    axis: np.ndarray,
    axis_length: float | None = None,
) -> None:
    """Log a prismatic joint translation.

    Args:
        entity_to_transform: Mapping from entity path to (translation, rotation_matrix).
        full_path: Full entity path to log.
        position: Translation distance.
        axis: Translation axis in local frame.
        axis_length: Optional length of the coordinate axes to display.
    """
    if full_path not in entity_to_transform:
        return

    start_translation, start_rotation_mat = entity_to_transform[full_path]

    # Apply translation along the axis
    # The axis is in the local frame, so rotate it to parent frame
    translation = start_translation + (start_rotation_mat @ axis) * position

    rr.log(
        full_path,
        rr.Transform3D(
            translation=translation, mat3x3=start_rotation_mat, axis_length=axis_length
        ),
    )


def _log_gripper_state(
    entity_to_transform: Dict[str, Tuple[np.ndarray, np.ndarray]],
    width: float,
    axis_length: float | None = None,
) -> None:
    """Log the gripper state (finger positions).

    Args:
        entity_to_transform: Mapping from entity path to (translation, rotation_matrix).
        width: Total distance between fingers.
        axis_length: Optional length of the coordinate axes to display.
    """
    # Assuming width is the total distance between fingers.
    # Each finger moves by width / 2.
    finger_pos = width / 2.0

    # Search for finger entities
    # We search for keys ending with the finger link names
    for key in entity_to_transform:
        if key.endswith("fer_leftfinger"):
            _log_prismatic(
                entity_to_transform, key, finger_pos, np.array([0, 1, 0]), axis_length
            )
        elif key.endswith("fer_rightfinger"):
            _log_prismatic(
                entity_to_transform, key, finger_pos, np.array([0, 1, 0]), axis_length
            )


class URDFLogger:
    """Loads and logs a URDF robot model to Rerun for 3D visualization.

    This is a simplified version of the URDFLogger from the DROID dataset example.
    It parses the URDF file and logs the robot meshes and joint transforms.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        root_path: str = "",
        axis_length: Optional[float] = None,
    ) -> None:
        """Initialize the URDF logger.

        Args:
            filepath: Path to the URDF file.
            root_path: Prefix for all entity paths in Rerun.
            axis_length: Optional length of the coordinate axes to display.
        """
        try:
            from urdf_parser_py import urdf as urdf_parser
        except ImportError as e:
            raise ImportError(
                "urdf_parser_py is required for URDF visualization. "
                "Install it with: pip install urdf_parser_py"
            ) from e

        self.urdf = urdf_parser.URDF.from_xml_file(str(filepath))
        self.mat_name_to_mat = {mat.name: mat for mat in self.urdf.materials}
        self.entity_to_transform: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.root_path = root_path
        self._urdf_dir = Path(filepath).parent
        self.axis_length = axis_length

    def link_entity_path(self, link: Any) -> str:
        """Return the entity path for the URDF link."""
        root_name = self.urdf.get_root()
        link_names = self.urdf.get_chain(root_name, link.name)[0::2]  # skip the joints
        return "/".join(link_names)

    def joint_entity_path(self, joint: Any) -> str:
        """Return the entity path for the URDF joint."""
        root_name = self.urdf.get_root()
        link_names = self.urdf.get_chain(root_name, joint.child)[
            0::2
        ]  # skip the joints
        return "/".join(link_names)

    def log(self) -> None:
        """Log the URDF robot model to Rerun."""
        rr.log(self.root_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        for joint in self.urdf.joints:
            entity_path = self.joint_entity_path(joint)
            self._log_joint(entity_path, joint)

        for link in self.urdf.links:
            entity_path = self.link_entity_path(link)
            self._log_link(entity_path, link)

    def _log_joint(self, entity_path: str, joint: Any) -> None:
        """Log a joint transform."""
        translation = np.zeros(3)
        rotation = np.eye(3)

        if joint.origin is not None:
            if joint.origin.xyz is not None:
                translation = np.array(joint.origin.xyz)
            if joint.origin.rpy is not None:
                rotation = Rotation.from_euler("xyz", joint.origin.rpy).as_matrix()

        # Store the initial transform for later joint angle updates
        full_path = self.root_path + entity_path
        self.entity_to_transform[full_path] = (translation, rotation)

        rr.log(
            full_path,
            rr.Transform3D(
                translation=translation, mat3x3=rotation, axis_length=self.axis_length
            ),
        )

    def _log_link(self, entity_path: str, link: Any) -> None:
        """Log link visuals (meshes)."""
        for i, visual in enumerate(link.visuals):
            self._log_visual(f"{entity_path}/visual_{i}", visual)

    def _log_visual(self, entity_path: str, visual: Any) -> None:
        """Log a visual element (mesh, box, cylinder, sphere)."""
        try:
            import trimesh
        except ImportError:
            # trimesh not available, skip mesh visualization
            return

        from urdf_parser_py import urdf as urdf_parser

        material = None
        if visual.material is not None:
            if visual.material.color is None and visual.material.texture is None:
                material = self.mat_name_to_mat.get(visual.material.name)
            else:
                material = visual.material

        transform = np.eye(4)
        if visual.origin is not None:
            if visual.origin.xyz is not None:
                transform[:3, 3] = visual.origin.xyz
            if visual.origin.rpy is not None:
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", visual.origin.rpy
                ).as_matrix()

        mesh_or_scene = None
        if isinstance(visual.geometry, urdf_parser.Mesh):
            resolved_path = self._resolve_mesh_path(visual.geometry.filename)
            if resolved_path and os.path.exists(resolved_path):
                mesh_scale = visual.geometry.scale
                mesh_or_scene = trimesh.load(resolved_path)
                if mesh_scale is not None:
                    # Apply scale to the transform
                    scale_mat = np.diag([*mesh_scale, 1.0])
                    transform = transform @ scale_mat
        elif isinstance(visual.geometry, urdf_parser.Box):
            mesh_or_scene = trimesh.creation.box(extents=visual.geometry.size)
        elif isinstance(visual.geometry, urdf_parser.Cylinder):
            mesh_or_scene = trimesh.creation.cylinder(
                radius=visual.geometry.radius,
                height=visual.geometry.length,
            )
        elif isinstance(visual.geometry, urdf_parser.Sphere):
            mesh_or_scene = trimesh.creation.icosphere(
                radius=visual.geometry.radius,
            )

        if mesh_or_scene is None:
            return

        mesh_or_scene.apply_transform(transform)

        if isinstance(mesh_or_scene, trimesh.Scene):
            for i, mesh in enumerate(mesh_or_scene.dump()):
                self._apply_material_and_log(f"{entity_path}/{i}", mesh, material)
        else:
            self._apply_material_and_log(entity_path, mesh_or_scene, material)

    def _resolve_mesh_path(self, path: str) -> Optional[str]:
        """Resolve mesh file path, handling package:// URIs and relative paths."""
        if not path.startswith("package://"):
            # Try relative to URDF directory
            resolved = self._urdf_dir / path
            if resolved.exists():
                return str(resolved)

            # Try parent directories
            for parent in self._urdf_dir.parents:
                candidate = parent / path
                if candidate.exists():
                    return str(candidate)
            return None

        # TODO: there should only be one strategy to resolve the urdfs
        # Handle package:// URIs
        stripped_package = path.replace("package://", "")
        parts = stripped_package.split("/", 1)

        # Strategy 1: Try relative path without package name
        if len(parts) == 2:
            relative_path = parts[1]
            for parent in [self._urdf_dir] + list(self._urdf_dir.parents):
                candidate = parent / relative_path
                if candidate.exists():
                    return str(candidate)

        # Strategy 2: Try full path including package name
        for parent in [self._urdf_dir] + list(self._urdf_dir.parents):
            candidate = parent / stripped_package
            if candidate.exists():
                return str(candidate)

        # Strategy 3: Try stripping the first directory component if it matches URDF dir name
        path_parts = Path(stripped_package).parts
        if len(path_parts) > 1 and path_parts[0] == self._urdf_dir.name:
            stripped_path = Path(*path_parts[1:])
            resolved = self._urdf_dir / stripped_path
            if resolved.exists():
                return str(resolved)

        return None

    def _apply_material_and_log(
        self, entity_path: str, mesh: Any, material: Any
    ) -> None:
        """Apply material colors and log the mesh."""
        import trimesh

        vertex_colors = None
        if (
            material is not None
            and hasattr(material, "color")
            and material.color is not None
        ):
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = material.color.rgba

        if hasattr(mesh, "visual") and isinstance(
            mesh.visual, trimesh.visual.color.ColorVisuals
        ):
            vertex_colors = mesh.visual.vertex_colors

        rr.log(
            self.root_path + entity_path,
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=vertex_colors,
            ),
            static=True,
        )


class LiveVisualizer:
    """Encapsulates live visualization logic with Rerun.

    - Calls `rr.init(...)` automatically when the episode changes.
    - Logs images, actions, and observed state per frame.
    - Optionally visualizes a 3D robot model from a URDF file.
    """

    def __init__(
        self,
        repo_name: str,
        viewer_spawn: bool = True,
        urdf_path: Optional[Union[str, Path]] = None,
        robot_root_path: str = "",
        camera_extrinsics: Optional[np.ndarray] = None,
        axis_length: Optional[float] = None,
        camera_width: int = 640,
        camera_height: int = 480,
    ) -> None:
        """Initialize the live visualizer.

        Args:
            repo_name: Identifier used as the Rerun application ID.
            viewer_spawn: Whether to spawn the local viewer on the first episode.
            urdf_path: Optional path to a URDF file for 3D robot visualization.
            robot_root_path: Entity path prefix for the robot model in Rerun.
            camera_extrinsics: 4x4 matrix for camera-to-EE transform.
            axis_length: Optional length of the coordinate axes to display.
            camera_width: Width of the camera images.
            camera_height: Height of the camera images.
        """
        self.repo_name = repo_name
        self._current_episode: Optional[int] = None
        self._spawn_first_viewer = viewer_spawn
        self._robot_root_path = robot_root_path
        self._camera_extrinsics = camera_extrinsics
        self._axis_length = axis_length
        self.camera_width = camera_width
        self.camera_height = camera_height

        # URDF-based robot visualization
        self._urdf_path = Path(urdf_path) if urdf_path else None
        self._urdf_logger: Optional[URDFLogger] = None
        self._entity_to_transform: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Track plotted transforms for clearing and re-logging across episodes
        self._plotted_transforms: Dict[str, Tuple[np.ndarray, str, Any]] = {}

    def _ensure_episode_init(self, episode: int) -> None:
        """Initialize Rerun for the given episode if needed."""
        if self._current_episode == episode:
            return

        app_id = self.repo_name
        rec_id = f"episode_{episode}"

        if self._current_episode is None:
            # First episode: initialize and optionally spawn viewer
            rr.init(app_id, recording_id=rec_id)
            if self._spawn_first_viewer:
                try:
                    rr.spawn()
                except Exception:
                    pass
        else:
            # Subsequent episodes: start a fresh recording
            rr.new_recording(app_id, recording_id=rec_id)

        # Avoid potential blocking flush issues
        gc.collect()
        self._current_episode = episode

        # Log URDF robot model if configured
        self._log_urdf_if_needed()

    def start_new_episode(self, episode: int) -> None:
        """Force-start a new Rerun recording for a given episode."""
        self._ensure_episode_init(episode)

    def _log_urdf_if_needed(self) -> None:
        """Log the URDF robot model to Rerun if a URDF path is configured."""
        if self._urdf_path is None or not self._urdf_path.exists():
            return

        try:
            self._urdf_logger = URDFLogger(
                self._urdf_path,
                root_path=self._robot_root_path,
                axis_length=self._axis_length,
            )
            self._urdf_logger.log()
            self._entity_to_transform = self._urdf_logger.entity_to_transform
        except Exception as e:
            # Log warning but don't fail - robot visualization is optional
            rr.log("warnings", rr.TextLog(f"Failed to load URDF: {e}"))

        # Re-log any persistent transformation matrices for the new episode
        for entity_path, (matrix, _, color) in self._plotted_transforms.items():
            translation = matrix[:3, 3]
            rotation_mat = matrix[:3, :3]
            rr.log(
                entity_path,
                rr.Transform3D(
                    translation=translation,
                    mat3x3=rotation_mat,
                    axis_length=self._axis_length or 0.1,
                ),
            )
            if color is not None:
                rr.log(
                    f"{entity_path}/origin",
                    rr.Points3D([[0, 0, 0]], colors=[color], radii=[0.01]),
                )

    def log_robot_state(self, joint_positions: np.ndarray) -> None:
        """Update the 3D robot visualization with the current joint positions.

        Args:
            joint_positions: Array of joint angles in radians.
                Expected shape: (7,) for arm only, or (8,) for arm + gripper.
        """
        if not self._entity_to_transform:
            return

        joint_positions = np.asarray(joint_positions).flatten()
        for joint_idx, angle in enumerate(joint_positions):
            if joint_idx >= 7:  # Franka Panda has 7 joints
                break
            _log_angle_rot(
                self._entity_to_transform,
                joint_idx + 1,  # Links are 1-indexed
                float(angle),
                root_path=self._robot_root_path,
                axis_length=self._axis_length,
            )

        # Log gripper if present (8th element)
        if len(joint_positions) >= 8:
            gripper_width = float(joint_positions[7])
            _log_gripper_state(
                self._entity_to_transform, gripper_width, axis_length=self._axis_length
            )

    def plot_transformation_matrix(
        self,
        transformation_matrix: np.ndarray,
        origin_frame: str,
        color: Optional[Union[List[int], np.ndarray]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Plot a coordinate frame based on a 4x4 transformation matrix.

        Args:
            transformation_matrix: 4x4 transformation matrix.
            origin_frame: Entity path of the frame to which the transform is relative.
            color: Optional RGB color [R, G, B] to indicate the origin of the transform.
            name: Optional name for the transform.
        """
        # Ensure we have a valid episode initialized
        if self._current_episode is None:
            self._ensure_episode_init(0)

        # Create a unique entity path for this transform
        if name is None:
            transform_id = len(self._plotted_transforms)
            entity_path = f"{origin_frame}/plotted_transform_{transform_id}"
        else:
            entity_path = f"{origin_frame}/{name}"

        self._plotted_transforms[entity_path] = (
            transformation_matrix,
            origin_frame,
            color,
        )

        # Decompose 4x4 matrix
        translation = transformation_matrix[:3, 3]
        rotation_mat = transformation_matrix[:3, :3]

        # Log to Rerun
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation_mat,
                axis_length=self._axis_length or 0.1,
            ),
        )

        if color is not None:
            rr.log(
                f"{entity_path}/origin",
                rr.Points3D([[0, 0, 0]], colors=[color], radii=[0.01]),
            )

    def clear_all_transformation_matrix_plotting(self) -> None:
        """Clear all persistent transformation matrix plottings."""
        for entity_path in self._plotted_transforms:
            rr.log(entity_path, rr.Clear(recursive=True))
        self._plotted_transforms = {}

    @staticmethod
    def _to_uint8_image(img: np.ndarray) -> np.ndarray:
        """Convert image to HWC uint8 if needed."""
        if img.dtype == np.uint8:
            return img
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            return (img * 255.0).astype(np.uint8)
        return img.astype(np.uint8, copy=False)

    def _log_point_cloud(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: Dict[str, float],
        ee_pose: Optional[np.ndarray] = None,
    ) -> None:
        """Log point cloud from RGB-D data in world coordinates.

        Args:
            color: RGB image array (H, W, 3).
            depth: Depth image array (H, W) in mm.
            intrinsics: Camera intrinsics dictionary with fx, fy, cx, cy.
            ee_pose: Optional 4x4 end-effector pose matrix.
        """
        if depth is None or color is None:
            return

        # Parameters
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        scale = 1000.0  # Depth is in mm

        if color.shape[:2] != depth.shape[:2]:
            return

        # Normalize color
        colors = color.astype(np.float32) / 255.0

        # Get point cloud in camera coordinates
        height, width = depth.shape
        xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))

        points_z = depth / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # Filter valid depth points
        mask = (points_z > 0.1) & (points_z < 2.0)

        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        if len(points) == 0:
            return

        # Transform points from camera frame to world frame
        if self._camera_extrinsics is not None and ee_pose is not None:
            T_world2cam = ee_pose @ self._camera_extrinsics
            R = T_world2cam[:3, :3]
            t = T_world2cam[:3, 3]
            points = (R @ points.T).T + t
        elif self._camera_extrinsics is None or ee_pose is None:
            # If no extrinsics or pose, we can't log in world frame
            return

        rr.log("base/point_cloud", rr.Points3D(points, colors=colors))

    def add_data(
        self,
        obs: Dict[str, Any],
        action: Optional[np.ndarray],
        robot_state: Optional[Union[np.ndarray, Dict[str, Any]]],
        metadata: Dict[str, Any],
        ee_pose: Optional[np.ndarray] = None,
    ) -> None:
        """Add a new frame of data to the live visualization.

        Args:
            obs: Observation dictionary.
            action: Action array.
            robot_state: Robot state array or dictionary.
            metadata: Metadata dictionary.
            ee_pose: Optional 4x4 end-effector pose matrix.
        """
        episode = int(metadata.get("episode", 0))
        self._ensure_episode_init(episode)

        step = int(metadata.get("step", 0))
        timestamp = float(metadata.get("timestamp", time.time()))
        rr.set_time("frame_index", sequence=step)
        rr.set_time("timestamp", timestamp=timestamp)

        # Log images
        for key in ("image", "wrist_image"):
            img = obs.get(key)
            if img is not None:
                rr.log(key, rr.Image(self._to_uint8_image(np.asarray(img))))

        # Log point cloud
        if all(k in obs for k in ("wrist_image", "wrist_depth", "wrist_intrinsics")):
            self._log_point_cloud(
                obs["wrist_image"],
                obs["wrist_depth"],
                obs["wrist_intrinsics"],
                ee_pose=ee_pose,
            )

        # Log action
        if action is not None:
            for i, val in enumerate(np.asarray(action).flatten()):
                rr.log(f"action/{i}", rr.Scalars(float(val)))

        # Log robot state
        if robot_state is not None:
            state_arr = (
                robot_state if isinstance(robot_state, np.ndarray) else np.array([])
            )
            if isinstance(robot_state, dict):
                # Extract common state if it's a dict
                state_arr = robot_state.get("joint_positions", state_arr)

            if state_arr.size > 0:
                for i, val in enumerate(state_arr.flatten()):
                    rr.log(f"state/{i}", rr.Scalars(float(val)))
                self.log_robot_state(state_arr)

        # Log text metadata
        for key in ("agent", "instruction"):
            val = metadata.get(key)
            if val:
                rr.log(key, rr.TextLog(str(val)))

        # Log grasps
        if "all_grasps" in metadata:
            current_ee_pose = ee_pose
            if current_ee_pose is None and isinstance(robot_state, dict):
                current_ee_pose = robot_state.get("osc_pose")

            self._log_grasps(
                metadata["all_grasps"],
                metadata.get("best_grasp"),
                current_ee_pose,
                metadata=metadata,
            )

    def _log_grasps(
        self,
        all_grasps: List[Dict[str, Any]],
        best_grasp: Optional[Dict[str, Any]] = None,
        ee_pose: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log grasp points to Rerun.

        Args:
            all_grasps: List of grasp dictionaries.
            best_grasp: Optional best grasp dictionary.
            ee_pose: Optional 4x4 end-effector pose matrix.
            metadata: Optional metadata dictionary.
        """
        grasp_ee_pose = ee_pose
        if metadata and "grasp_ee_pose" in metadata:
            grasp_ee_pose = metadata["grasp_ee_pose"]

        if self._camera_extrinsics is None or grasp_ee_pose is None:
            return

        T_world2cam = grasp_ee_pose @ self._camera_extrinsics
        # Rotation to match convention (AnyGrasp Z-approach -> Project X-approach)
        R_anygrasp2project = Rotation.from_euler(
            "YZ", [90, 90], degrees=True
        ).as_matrix()

        # Rotation to rotate the strip to visualize the grasps correctly
        # This maps the XY-plane strip to the YZ-plane (finger plane in project convention)
        R_strip_fix = Rotation.from_euler("y", -90, degrees=True).as_matrix()

        strips, colors = [], []
        best_color = [0, 255, 0]
        others_color = [100, 100, 100]

        if metadata:
            best_color = metadata.get("best_grasp_color", best_color)
            others_color = metadata.get("all_grasps_color", others_color)

        def get_world_strip(grasp: Dict[str, Any]) -> np.ndarray:
            pos_cam = np.array(grasp["translation"])
            rot_cam = np.array(grasp["rotation_matrix"])

            T_cam2grasp = np.eye(4)
            T_cam2grasp[:3, :3] = rot_cam @ R_anygrasp2project
            T_cam2grasp[:3, 3] = pos_cam
            T_world2grasp = T_world2cam @ T_cam2grasp

            width = grasp["width"]
            depth = grasp["depth"]

            # logger.debug(f"Grasp width: {width}, depth: {depth}")

            local_strip = np.array(
                [
                    [0, -width / 2, 0],
                    [-depth, -width / 2, 0],
                    [-depth, width / 2, 0],
                    [0, width / 2, 0],
                ]
            )
            local_strip = (R_strip_fix @ local_strip.T).T
            return (T_world2grasp[:3, :3] @ local_strip.T).T + T_world2grasp[
                :3, 3
            ], T_world2grasp

        for g in all_grasps:
            if best_grasp and np.allclose(g["translation"], best_grasp["translation"]):
                continue
            strip, _ = get_world_strip(g)
            strips.append(strip)
            colors.append(others_color)

        if best_grasp:
            strip, T_world2grasp = get_world_strip(best_grasp)
            strips.append(strip)
            colors.append(best_color)
            self.plot_transformation_matrix(
                T_world2grasp, "base", color=best_color, name="best_grasp_pose"
            )

        if strips:
            rr.log("base/grasps", rr.LineStrips3D(strips, colors=colors, radii=0.001))
