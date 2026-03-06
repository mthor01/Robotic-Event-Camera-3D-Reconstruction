# Random Sphere Agent - samples random poses inside a sphere, always looking at target point

import time
from typing import Any, Dict, Tuple

import numpy as np

from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

from scipy.spatial.transform import Rotation

logger = get_logger(__name__)


class RandomSphereAgent(Agent):
    """
    Agent that moves to random positions inside a sphere, always looking at a target point.
    
    The camera (end-effector z-axis) always points toward the target point,
    which is where the object should be placed.
    
    During transitions between poses, intermediate waypoints are generated to keep
    the camera continuously pointing at the target.
    """

    def __init__(
        self,
        center: np.ndarray = None,
        radius: float = 0.2,
        num_poses: int = 20,
        wait_time: float = 0.0,
        seed: int = None,
        loop: bool = False,
        target_point: np.ndarray = None,
        waypoints_per_transition: int = 5,
        base_exclusion_radius: float = 0.3,
        base_max_radius: float = 0.65,
    ) -> None:
        """
        Args:
            center: Center of the sphere in robot base frame [x, y, z]. Default: [0.4, 0, 0.3]
            radius: Radius of the sphere in meters. Default: 0.3
            num_poses: Number of random poses to generate. Default: 20
            wait_time: Time to wait at each pose in seconds. Default: 0.0
            seed: Random seed for reproducibility. Default: None
            loop: If True, loop forever; if False, stop after visiting all poses.
            target_point: Point the camera looks at [x, y, z]. Default: [0.35, 0, -0.1]
            waypoints_per_transition: Number of intermediate waypoints between main poses. Default: 5
            base_exclusion_radius: Exclude poses within this x,y radius of robot base (0,0). Default: 0.25
            base_max_radius: Exclude poses beyond this x,y radius of robot base (0,0). Default: 0.65
        """
        super().__init__(action_type="OSC_POSE")
        
        if center is None:
            center = np.array([0.4, 0.0, 0.2])
        self.center = np.array(center, dtype=float)
        
        if target_point is None:
            target_point = np.array([0.35, 0.0, -0.1])
        self.target_point = np.array(target_point, dtype=float)
        
        self.radius = radius
        self.num_poses = num_poses
        self.wait_time = wait_time
        self.loop = loop
        self.waypoints_per_transition = waypoints_per_transition
        self.base_exclusion_radius = base_exclusion_radius
        self.base_max_radius = base_max_radius
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random poses inside the sphere
        self.poses = self._generate_random_sphere_poses()
        
        logger.info(f"Generated {len(self.poses)} random poses in sphere.")
        logger.info(f"  Center: {self.center}")
        logger.info(f"  Radius: {self.radius}")
        logger.info(f"  Target point: {self.target_point}")
        logger.info(f"  Waypoints per transition: {self.waypoints_per_transition}")
        logger.info(f"  Base exclusion radius: {self.base_exclusion_radius}")
        logger.info(f"  Base max radius: {self.base_max_radius}")

        self.current_pose_idx = 0
        self.state = "MOVING"  # States: MOVING, WAITING, COMPLETE
        self.wait_start_time: float | None = None
        self.osc_controller: OscPoseTargetController | None = None
        self.sphere_complete = False
        
        # Waypoint tracking for smooth transitions
        self.current_waypoints: list[np.ndarray] = []
        self.current_waypoint_idx = 0
        self.is_main_pose = True  # True when at a main pose (not intermediate waypoint)
        
        # Position logging
        self.last_position_print_time: float = 0.0

    def _generate_random_sphere_poses(self) -> list[np.ndarray]:
        """
        Generate random poses uniformly distributed inside a sphere (volume).
        Each pose has the camera (z-axis) pointing at the target point.
        """
        poses = []
        attempts = 0
        max_attempts = self.num_poses * 100  # Avoid infinite loop
        
        while len(poses) < self.num_poses and attempts < max_attempts:
            attempts += 1
            
            # Generate random point uniformly inside unit sphere
            # Method: rejection sampling or use r^(1/3) for uniform volume distribution
            # Using spherical coordinates with r^(1/3) scaling
            theta = np.random.uniform(0, 2 * np.pi)  # azimuthal angle
            phi = np.arccos(np.random.uniform(-1, 1))  # polar angle
            # For uniform distribution in volume, r should be sampled with cube root
            r = self.radius * (np.random.uniform(0, 1) ** (1/3))
            
            # Convert to Cartesian coordinates
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Translate to center
            position = self.center + np.array([x, y, z])
            
            # Filter out poses that are below the table
            if position[2] < 0.1:
                continue
            
            # Filter out poses too close to robot base (x,y distance from origin)
            xy_dist_from_base = np.sqrt(position[0]**2 + position[1]**2)
            if xy_dist_from_base < self.base_exclusion_radius:
                continue
            
            # Filter out poses too far from robot base (beyond reach)
            if xy_dist_from_base > self.base_max_radius:
                continue
            
            # Calculate orientation: z-axis points toward target point
            direction = self.target_point - position
            dist = np.linalg.norm(direction)
            if dist < 1e-6:  # Avoid division by zero if at target
                continue
            direction = direction / dist
            
            # Build rotation matrix with z pointing toward target
            quat = self._quat_from_z_direction(direction)
            
            # Create pose array [x, y, z, qx, qy, qz, qw]
            pose = np.zeros(7)
            pose[:3] = position
            pose[3:7] = quat
            
            poses.append(pose)
        
        logger.info(f"Generated {len(poses)} valid poses out of {self.num_poses} requested.")
        
        # Reorder poses using 2-opt TSP algorithm for shortest path
        poses = self._reorder_by_shortest_path(poses)
        
        return poses

    def _reorder_by_shortest_path(self, poses: list[np.ndarray]) -> list[np.ndarray]:
        """
        Reorder poses to minimize total travel distance using 2-opt algorithm.
        This finds a near-optimal solution to the Traveling Salesman Problem.
        """
        if len(poses) <= 2:
            return poses
        
        n = len(poses)
        
        # Extract positions for distance calculations
        positions = np.array([p[:3] for p in poses])
        
        # Compute distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        
        # Start with nearest-neighbor solution
        route = self._nearest_neighbor_route(dist_matrix)
        
        # Improve with 2-opt
        route = self._two_opt(route, dist_matrix)
        
        # Reorder poses according to optimized route
        return [poses[i] for i in route]
    
    def _nearest_neighbor_route(self, dist_matrix: np.ndarray) -> list[int]:
        """Generate initial route using nearest-neighbor heuristic."""
        n = len(dist_matrix)
        visited = [False] * n
        route = [0]
        visited[0] = True
        
        for _ in range(n - 1):
            current = route[-1]
            nearest = None
            nearest_dist = float('inf')
            for j in range(n):
                if not visited[j] and dist_matrix[current, j] < nearest_dist:
                    nearest = j
                    nearest_dist = dist_matrix[current, j]
            route.append(nearest)
            visited[nearest] = True
        
        return route
    
    def _two_opt(self, route: list[int], dist_matrix: np.ndarray) -> list[int]:
        """Improve route using 2-opt swaps until no improvement found."""
        n = len(route)
        improved = True
        
        while improved:
            improved = False
            for i in range(n - 2):
                for j in range(i + 2, n):
                    # Calculate current distance
                    d1 = dist_matrix[route[i], route[i + 1]]
                    if j == n - 1:
                        d2 = 0  # No edge after last node (open path)
                    else:
                        d2 = dist_matrix[route[j], route[j + 1]] if j + 1 < n else 0
                    
                    # Calculate new distance if we reverse segment [i+1, j]
                    d3 = dist_matrix[route[i], route[j]]
                    if j + 1 < n:
                        d4 = dist_matrix[route[i + 1], route[j + 1]]
                    else:
                        d4 = 0
                    
                    # If improvement, reverse the segment
                    if d3 + d4 < d1 + d2:
                        route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                        improved = True
        
        return route

    def _quat_from_z_direction(self, z_dir: np.ndarray) -> np.ndarray:
        """
        Create a quaternion where the z-axis points in the given direction.
        Uses the same convention as HemisphereGridAgent to ensure reachable orientations.
        
        Args:
            z_dir: Unit vector for the desired z-axis direction
            
        Returns:
            Quaternion [qx, qy, qz, qw]
        """
        z = z_dir / np.linalg.norm(z_dir)
        
        # Use world up to keep y-axis horizontal (same as HemisphereGridAgent)
        world_up = np.array([0.0, 0.0, 1.0])
        
        # Make EE y-axis horizontal and perpendicular to z
        y = np.cross(z, world_up)
        if np.linalg.norm(y) < 1e-8:
            y = np.array([0.0, 1.0, 0.0])
        y = y / np.linalg.norm(y)
        
        # Right-hand rule to get x
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        
        # Build rotation matrix [x, y, z] as columns
        R = np.column_stack([x, y, z])
        
        # CRITICAL: Apply the same axis flips as HemisphereGridAgent
        # This matches the Franka end-effector coordinate convention
        R[:, 0] *= -1
        R[:, 1] *= -1
        
        # Convert to quaternion
        quat = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
        
        # Normalize and ensure consistent hemisphere
        quat = quat / np.linalg.norm(quat)
        if quat[3] < 0:
            quat = -quat
            
        return quat

    def _generate_waypoints(self, start_pos: np.ndarray, end_pos: np.ndarray) -> list[np.ndarray]:
        """
        Generate intermediate waypoints between two positions.
        Each waypoint has orientation recalculated to point at target.
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            
        Returns:
            List of waypoint poses [x, y, z, qx, qy, qz, qw] (excluding start, including end)
        """
        waypoints = []
        
        for i in range(1, self.waypoints_per_transition + 1):
            t = i / self.waypoints_per_transition
            # Linear interpolation of position
            position = start_pos + t * (end_pos - start_pos)
            
            # Recalculate orientation to point at target from this position
            direction = self.target_point - position
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            direction = direction / dist
            quat = self._quat_from_z_direction(direction)
            
            pose = np.zeros(7)
            pose[:3] = position
            pose[3:7] = quat
            waypoints.append(pose)
        
        return waypoints

    def _init_osc_controller(self, target_pose: np.ndarray = None) -> None:
        """Initialize the OSC controller for the current target pose or waypoint."""
        if target_pose is None:
            target_pose = self.poses[self.current_pose_idx]
        # command is [x, y, z, qx, qy, qz, qw, gripper]
        command = np.concatenate([target_pose, [-1.0]])  # -1.0 = gripper open
        self.osc_controller = OscPoseTargetController(
            target_pose=command,
            threshold_reach=0.02,      # 5cm position tolerance (default 0.005)
            threshold_rotation=3.15,   # ~180 degrees - essentially ignore rotation
            max_steps=100,             # Give up after 100 steps to avoid getting stuck
        )
    
    def _setup_transition_to_pose(self, current_pos: np.ndarray, target_pose_idx: int) -> None:
        """Set up waypoints for transitioning to the next main pose."""
        target_pose = self.poses[target_pose_idx]
        target_pos = target_pose[:3]
        
        # Generate waypoints from current position to target
        self.current_waypoints = self._generate_waypoints(current_pos, target_pos)
        self.current_waypoint_idx = 0
        
        if len(self.current_waypoints) > 0:
            # Start with first waypoint
            self._init_osc_controller(self.current_waypoints[0])
            self.is_main_pose = False
        else:
            # No waypoints, go directly to target
            self._init_osc_controller(target_pose)
            self.is_main_pose = True

    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculate the next action to move toward the current target pose.
        Uses waypoints to keep camera pointed at target during transitions.
        """
        pos = robot_state["osc_position"]
        rot = robot_state["osc_rotation_quaternion"]
        current_pose = np.concatenate([pos.flatten(), rot.flatten()]).astype(float)
        current_pos = current_pose[:3]
        
        # Print position every second
        current_time = time.time()
        if current_time - self.last_position_print_time >= 1.0:
            print(f"Robot position: x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}")
            self.last_position_print_time = current_time

        if self.osc_controller is None:
            # First call - set up transition to first pose
            self._setup_transition_to_pose(current_pos, self.current_pose_idx)

        action, is_finished = self.osc_controller.calculate_action(current_pose)

        if self.state == "MOVING":
            if is_finished:
                if not self.is_main_pose:
                    # Finished a waypoint, move to next waypoint or main pose
                    self.current_waypoint_idx += 1
                    if self.current_waypoint_idx < len(self.current_waypoints):
                        # Move to next waypoint
                        self._init_osc_controller(self.current_waypoints[self.current_waypoint_idx])
                        logger.debug(f"Moving to waypoint {self.current_waypoint_idx + 1}/{len(self.current_waypoints)}")
                    else:
                        # All waypoints done, we're at main pose
                        self.is_main_pose = True
                        logger.info(
                            f"Reached pose {self.current_pose_idx + 1}/{len(self.poses)}. "
                            f"Waiting for {self.wait_time}s."
                        )
                        self.state = "WAITING"
                        self.wait_start_time = time.time()
                else:
                    # Reached main pose directly (no waypoints)
                    logger.info(
                        f"Reached pose {self.current_pose_idx + 1}/{len(self.poses)}. "
                        f"Waiting for {self.wait_time}s."
                    )
                    self.state = "WAITING"
                    self.wait_start_time = time.time()
            else:
                if self.is_main_pose:
                    logger.debug(f"Moving to pose {self.current_pose_idx + 1}/{len(self.poses)}.")
                else:
                    logger.debug(f"Moving through waypoint {self.current_waypoint_idx + 1}/{len(self.current_waypoints)} to pose {self.current_pose_idx + 1}")

        elif self.state == "WAITING":
            if self.wait_start_time is not None and (
                time.time() - self.wait_start_time > self.wait_time
            ):
                next_idx = self.current_pose_idx + 1
                
                # Check if we've completed all poses
                if next_idx >= len(self.poses):
                    if self.loop:
                        self.current_pose_idx = 0
                        logger.info("Sphere poses complete. Looping back to pose 1.")
                        self.state = "MOVING"
                        self._setup_transition_to_pose(current_pos, self.current_pose_idx)
                    else:
                        self.sphere_complete = True
                        self.state = "COMPLETE"
                        logger.info("Sphere poses complete! All poses visited.")
                else:
                    self.current_pose_idx = next_idx
                    logger.info(f"Moving to pose {self.current_pose_idx + 1}/{len(self.poses)}.")
                    self.state = "MOVING"
                    self._setup_transition_to_pose(current_pos, self.current_pose_idx)
                
                if self.state == "MOVING":
                    action, is_finished = self.osc_controller.calculate_action(current_pose)

        elif self.state == "COMPLETE":
            # Stay at current position
            pass

        metadata = {
            "action_type": self.action_type,
            "state": self.state,
            "target_pose_idx": self.current_pose_idx,
            "total_poses": len(self.poses),
            "is_finished": is_finished,
            "sphere_complete": self.sphere_complete,
            "is_main_pose": self.is_main_pose,
            "waypoint_idx": self.current_waypoint_idx if not self.is_main_pose else -1,
        }

        return action, metadata

    def reset(self) -> None:
        """Reset the agent to start from the first pose."""
        self.current_pose_idx = 0
        self.state = "MOVING"
        self.wait_start_time = None
        self.osc_controller = None
        self.sphere_complete = False
        self.current_waypoints = []
        self.current_waypoint_idx = 0
        self.is_main_pose = True
