import numpy as np
import time
import json
from scipy.spatial.transform import Rotation as R
from typing import Any, Dict, Tuple, List
from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

logger = get_logger(__name__)


class OscPoseTargetBenchmarkingAgent(Agent):
    """
    Agent to benchmark OSC pose target control parameters.
    It moves to a single target from various start positions in XY, XZ, and YZ planes.
    It can iterate through a grid of parameters.

    NOTE: if the deoxys C++ backend controller crashes due to limit violations, one can change the random seed `rng = np.random.default_rng(seed=44)` for better start poses.
    """

    def __init__(
        self,
        kp_pos_list: List[float],
        ki_pos_list: List[float],
        i_zone_list: List[float],
        kp_rot_list: List[float],
        ki_rot_list: List[float],
        max_steps: int = 50,
        starting_positions_on_circle: int = 2,
    ):
        super().__init__(action_type="OSC_POSE")

        # Full grid of all combinations
        self.configs = []
        for kp_p in kp_pos_list:
            for ki_p in ki_pos_list:
                for kp_r in kp_rot_list:
                    for ki_r in ki_rot_list:
                        for iz in i_zone_list:
                            self.configs.append(
                                {
                                    "kp_pos": kp_p,
                                    "ki_pos": ki_p,
                                    "kp_rot": kp_r,
                                    "ki_rot": ki_r,
                                    "i_zone": iz,
                                    "i_max": 0.1,
                                    "rot_i_max": 0.05,
                                }
                            )

        self.max_steps = max_steps

        # Fixed target for benchmarking (the target pose is chosen somewhat off center, as torque feed forward models become worse there)
        self.target_pos = np.array([0.5, -0.3, 0.4])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Downward facing
        self.target_pose = np.concatenate([self.target_pos, self.target_quat])

        # Generate unique start poses
        radii = [0.1, 0.3]
        angles = np.linspace(0, 2 * np.pi, starting_positions_on_circle, endpoint=False)

        potential_starts = []
        # XY Plane
        for i, a in enumerate(angles):
            r = radii[i % 2]
            potential_starts.append(
                self.target_pos + np.array([r * np.cos(a), r * np.sin(a), 0])
            )
        # XZ Plane
        for i, a in enumerate(angles):
            r = radii[i % 2]
            potential_starts.append(
                self.target_pos + np.array([r * np.sin(a), 0, r * np.cos(a)])
            )
        # YZ Plane
        for i, a in enumerate(angles):
            r = radii[i % 2]
            potential_starts.append(
                self.target_pos + np.array([0, r * np.cos(a), r * np.sin(a)])
            )

        self.start_poses = []
        # Use a fixed seed for benchmarking start poses so all parameter configs face the same trials
        rng = np.random.default_rng(seed=44)
        for p in potential_starts:
            if not any(
                np.allclose(p, existing[:3], atol=1e-4) for existing in self.start_poses
            ):
                # Randomized orientation in a safe way:
                # 1. half spin around Z-axis (pointing down)
                # 2. Small tilt around X/Y axes (+/- 20 degrees)
                spin = rng.uniform(0.5 * -np.pi, np.pi * 0.5)
                tilt_x = rng.uniform(-np.deg2rad(15), np.deg2rad(15))
                tilt_y = rng.uniform(-np.deg2rad(15), np.deg2rad(15))

                target_rot = R.from_quat(self.target_quat)
                rel_rot = R.from_euler("xyz", [tilt_x, tilt_y, spin])
                start_quat = (target_rot * rel_rot).as_quat()
                self.start_poses.append(np.concatenate([p, start_quat]))

        logger.info(
            f"Initialized benchmarking agent with {len(self.configs)} configurations and {len(self.start_poses)} unique start poses."
        )

        self.all_results = []  # Stores results for all configs
        self.current_config_idx = 0
        self.current_start_idx = 0
        self.current_config_results = []  # Results for current config

        self.state = "GO_TO_START"
        self.osc_controller = None
        self.measured_start_pose = None

    def _init_controller(self, target: np.ndarray, params: Dict[str, float] = None):
        """Initialize the OSC controller."""
        command = np.concatenate([target, [-1.0]])  # Keep gripper open

        if params:
            # Benchmark move
            self.osc_controller = OscPoseTargetController(
                target_pose=command,
                threshold_reach=0.00001,
                threshold_rotation=0.00001,
                kp_pos=params["kp_pos"],
                ki_pos=params["ki_pos"],
                kp_rot=params["kp_rot"],
                ki_rot=params["ki_rot"],
                i_zone=params["i_zone"],
                max_steps=self.max_steps,
            )
        else:
            # Setup move
            self.osc_controller = OscPoseTargetController(
                target_pose=command,
                threshold_reach=0.01,
                threshold_rotation=0.05,
                max_steps=40,
            )

    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pos = robot_state["osc_position"].flatten()
        rot = robot_state["osc_rotation_quaternion"].flatten()
        current_pose = np.concatenate([pos, rot]).astype(float)

        if self.state == "GO_TO_START":
            if self.osc_controller is None:
                start_pose = self.start_poses[self.current_start_idx]
                self._init_controller(start_pose)
                logger.info(
                    f"Config {self.current_config_idx+1}/{len(self.configs)}: Move to start {self.current_start_idx+1}/{len(self.start_poses)}"
                )

            action, is_finished = self.osc_controller.calculate_action(current_pose)

            if is_finished:
                self.measured_start_pose = current_pose.copy()
                self.state = "GO_TO_TARGET"
                self.osc_controller = None

        elif self.state == "GO_TO_TARGET":
            if self.osc_controller is None:
                config = self.configs[self.current_config_idx]
                self._init_controller(self.target_pose, params=config)
                logger.info(
                    f"Config {self.current_config_idx+1}: Benchmarking from start {self.current_start_idx+1}"
                )

            action, is_finished = self.osc_controller.calculate_action(current_pose)

            if is_finished:
                measured_end_pose = current_pose.copy()
                self.current_config_results.append(
                    {
                        "start_idx": self.current_start_idx,
                        "target_pose": self.target_pose.tolist(),
                        "start_pose_measured": self.measured_start_pose.tolist(),
                        "end_pose_measured": measured_end_pose.tolist(),
                    }
                )

                self.current_start_idx += 1
                if self.current_start_idx >= len(self.start_poses):
                    # Finished all starts for this config
                    self._summarize_current_config()
                    self.current_start_idx = 0
                    self.current_config_idx += 1

                    if self.current_config_idx >= len(self.configs):
                        self.state = "FINISHED"
                    else:
                        self.state = "GO_TO_START"
                else:
                    self.state = "GO_TO_START"
                self.osc_controller = None

        if self.state == "FINISHED":
            self.save_all_results()
            return np.zeros(7), {"quit": True}

        return action, {"state": self.state, "action_type": self.action_type}

    def _summarize_current_config(self):
        """Summarize results for the current parameter configuration."""
        pos_errors = []
        rot_errors = []

        for res in self.current_config_results:
            target_pos = np.array(res["target_pose"][:3])
            end_pos = np.array(res["end_pose_measured"][:3])
            pos_errors.append(np.linalg.norm(target_pos - end_pos))

            target_quat = np.array(res["target_pose"][3:])
            end_quat = np.array(res["end_pose_measured"][3:])
            dot = np.abs(np.dot(target_quat, end_quat))
            dot = np.clip(dot, 0.0, 1.0)
            rot_error = 2 * np.arccos(dot)
            rot_errors.append(rot_error)

        avg_pos_error = np.mean(pos_errors)
        avg_rot_error = np.mean(rot_errors)

        config = self.configs[self.current_config_idx]
        summary = {
            "params": config,
            "avg_pos_error": float(avg_pos_error),
            "avg_rot_error": float(avg_rot_error),
            "raw_trials": self.current_config_results,
        }
        self.all_results.append(summary)

        logger.info(
            f"DONE CONFIG {self.current_config_idx+1}/{len(self.configs)}: "
            f"Kp_p={config['kp_pos']}, Ki_p={config['ki_pos']}, Kp_r={config['kp_rot']}, Ki_r={config['ki_rot']}, "
            f"iz={config['i_zone']} | "
            f"Err: {avg_pos_error:.6f}m, {avg_rot_error:.6f}rad"
        )

        self.current_config_results = []

    def save_all_results(self):
        """Save all results to a JSON file."""
        timestamp = int(time.time())
        filename = f"benchmark_grid_search_{timestamp}.json"

        # Sort results by pos error for a quick overview
        sorted_results = sorted(self.all_results, key=lambda x: x["avg_pos_error"])

        logger.info("=" * 30)
        logger.info("BENCHMARK SUMMARY")
        for i, res in enumerate(sorted_results):
            p = res["params"]
            logger.info(
                f"{i+1}. PosErr: {res['avg_pos_error']:.6f}m, RotErr: {res['avg_rot_error']:.6f}rad | "
                f"Kp_p={p['kp_pos']}, Ki_p={p['ki_pos']}, kp_r={p['kp_rot']}, ki_r={p['ki_rot']}, "
                f"i_zone={p['i_zone']}"
            )
        logger.info("=" * 30)

        with open(filename, "w") as f:
            json.dump(self.all_results, f, indent=4)
        logger.info(f"Full grid search results saved to {filename}")

    def reset(self):
        self.current_start_idx = 0
        self.current_config_idx = 0
        self.all_results = []
        self.state = "GO_TO_START"
        self.osc_controller = None
