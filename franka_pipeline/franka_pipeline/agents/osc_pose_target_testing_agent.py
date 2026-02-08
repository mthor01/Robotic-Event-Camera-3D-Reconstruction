import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, List
from franka_pipeline.agents.agent import Agent
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)

logger = get_logger(__name__)


class OscPoseTargetTestingAgent(Agent):
    """
    Agent to test systematic errors in OSC pose target control.
    It moves to target poses from 8 different directions to see if the error is consistent.
    """

    def __init__(self, mode: str = "xy"):
        super().__init__(action_type="OSC_POSE")
        self.mode = mode.lower()
        if self.mode not in ["xy", "yz"]:
            logger.warning(f"Invalid mode {mode}, defaulting to xy")
            self.mode = "xy"

        # Base pose and targets as requested
        if self.mode == "xy":
            self.Z = 0.35
            base_pose = np.array([0.45, 0.0, self.Z, 1.0, 0.0, 0.0, 0.0])
            self.targets = [
                base_pose.copy() + np.array([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
                base_pose + np.array([0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
                base_pose + np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
                base_pose + np.array([-0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]
        else:  # mode == "yz"
            self.X = 0.60
            base_pose = np.array([self.X, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0])
            self.targets = [
                base_pose.copy() + np.array([0.0, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0]),
                base_pose + np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0]),
                base_pose + np.array([0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]),
                base_pose + np.array([0.0, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0]),
            ]

        # 8 start offsets on a circle of 10cm radius
        self.radius = 0.1
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self.start_offsets = []
        for a in angles:
            if self.mode == "xy":
                offset = np.array(
                    [self.radius * np.cos(a), self.radius * np.sin(a), 0, 0, 0, 0, 0]
                )
            else:  # yz
                offset = np.array(
                    [0, self.radius * np.cos(a), self.radius * np.sin(a), 0, 0, 0, 0]
                )
            self.start_offsets.append(offset)

        self.results = []  # List of dicts with target, start_measured, end_measured

        self.current_target_idx = 0
        self.current_offset_idx = 0

        self.state = "GO_TO_START"
        self.osc_controller = None
        self.start_pos_measured = None

    def _init_controller(
        self,
        target: np.ndarray,
        threshold_reach: float = 0.005,
        threshold_rotation: float = 0.05,
        max_steps: int = 40,
    ):
        """Initialize the OSC controller with a target pose."""
        # target is [x, y, z, qx, qy, qz, qw]
        # controller needs [x, y, z, qx, qy, qz, qw, gripper]
        command = np.concatenate([target, [-1.0]])  # Keep gripper open
        self.osc_controller = OscPoseTargetController(
            target_pose=command,
            threshold_reach=threshold_reach,
            threshold_rotation=threshold_rotation,
            max_steps=max_steps,
        )

    def act(
        self,
        robot_state: Dict[str, Any],
        observation: Dict[str, Any],
        instruction: str = "",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Get current pose
        pos = robot_state["osc_position"]
        rot = robot_state["osc_rotation_quaternion"]
        current_pose = np.concatenate([pos.flatten(), rot.flatten()]).astype(float)

        target = self.targets[self.current_target_idx]
        offset = self.start_offsets[self.current_offset_idx]

        action = np.zeros(7)
        metadata = {"state": self.state}

        if self.state == "GO_TO_START":
            if self.osc_controller is None:
                start_target = target + offset
                self._init_controller(
                    start_target,
                    threshold_reach=0.005,
                    threshold_rotation=0.05,
                    max_steps=20,
                )
                logger.info(
                    f"Moving to start position {self.current_offset_idx} for target {self.current_target_idx}"
                )

            action, is_finished = self.osc_controller.calculate_action(current_pose)

            if is_finished:
                self.start_pos_measured = current_pose.copy()
                self.state = "GO_TO_TARGET"
                self.osc_controller = None
                logger.debug(f"Reached start position {self.current_offset_idx}")

        elif self.state == "GO_TO_TARGET":
            if self.osc_controller is None:
                self._init_controller(
                    target,
                    threshold_reach=0.00001,
                    threshold_rotation=0.00001,
                    max_steps=100,
                )
                logger.info(
                    f"Moving to target {self.current_target_idx} from direction {self.current_offset_idx}"
                )

            action, is_finished = self.osc_controller.calculate_action(current_pose)

            if is_finished:
                end_pos_measured = current_pose.copy()
                self.results.append(
                    {
                        "target": target.copy(),
                        "start": self.start_pos_measured.copy(),
                        "end": end_pos_measured.copy(),
                    }
                )

                # Advance to next test
                self.current_offset_idx += 1
                if self.current_offset_idx >= len(self.start_offsets):
                    self.current_offset_idx = 0
                    self.current_target_idx += 1

                if self.current_target_idx >= len(self.targets):
                    self.state = "FINISHED"
                    logger.info("Testing complete. Preparing to plot results.")
                else:
                    self.state = "GO_TO_START"
                    self.osc_controller = None

        if self.state == "FINISHED":
            self.plot_results()
            return np.zeros(7), {"quit": True}

        metadata["action_type"] = self.action_type
        return action, metadata

    def plot_results(self):
        """Plot the target, start, and end positions on the chosen plane."""
        plt.figure(figsize=(10, 10))

        # Calculate statistics
        errors = [r["end"][:3] - r["target"][:3] for r in self.results]
        avg_dist = np.mean([np.linalg.norm(e) for e in errors])
        avg_x = np.mean([e[0] for e in errors])
        avg_y = np.mean([e[1] for e in errors])
        avg_z = np.mean([e[2] for e in errors])
        mae_x = np.mean([np.abs(e[0]) for e in errors])
        mae_y = np.mean([np.abs(e[1]) for e in errors])
        mae_z = np.mean([np.abs(e[2]) for e in errors])

        stats_text = (
            f"Avg Total Distance: {avg_dist:.4f}m\n"
            f"Mean Error (X/Y/Z): {avg_x:.4f}, {avg_y:.4f}, {avg_z:.4f}m\n"
            f"Mean Abs Error (X/Y/Z): {mae_x:.4f}, {mae_y:.4f}, {mae_z:.4f}m"
        )

        # Labels and data indices based on mode
        if self.mode == "xy":
            idx1, idx2 = 0, 1
            label1, label2 = "X", "Y"
            fix_param_text = f"Z={self.Z}m"
            plane_text = "X/Y Plane"
        else:  # yz
            idx1, idx2 = 1, 2
            label1, label2 = "Y", "Z"
            fix_param_text = f"X={self.X}m"
            plane_text = "Y/Z Plane"

        # Get unique targets
        targets_unique = []
        for r in self.results:
            t = r["target"][[idx1, idx2]]
            if not any(np.allclose(t, tu) for tu in targets_unique):
                targets_unique.append(t)

        for i, tu in enumerate(targets_unique):
            plt.plot(
                tu[0],
                tu[1],
                "rx",
                markersize=12,
                markeredgewidth=3,
                label="Target" if i == 0 else "",
            )

            # Plot results for this target
            first_entry = True
            for r in self.results:
                if np.allclose(r["target"][[idx1, idx2]], tu):
                    s = r["start"][[idx1, idx2]]
                    e = r["end"][[idx1, idx2]]

                    # Line from start to end
                    plt.plot([s[0], e[0]], [s[1], e[1]], "k-", alpha=0.2)
                    # Start point
                    plt.scatter(
                        s[0],
                        s[1],
                        c="blue",
                        s=20,
                        alpha=0.5,
                        label="Start Positions" if (i == 0 and first_entry) else "",
                    )
                    # End point
                    plt.scatter(
                        e[0],
                        e[1],
                        c="green",
                        s=40,
                        alpha=0.8,
                        label="End Positions" if (i == 0 and first_entry) else "",
                    )
                    first_entry = False

        plt.xlabel(f"{label1} (m)")
        plt.ylabel(f"{label2} (m)")
        plt.title(
            f"OSC Pose Target Systematic Error Analysis ({plane_text})\n{fix_param_text}, Radius={self.radius}m"
        )
        # Add stats text to plot
        plt.text(
            0.05,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.axis("equal")

        timestamp = int(time.time())
        save_path = f"osc_test_results_{self.mode}_{timestamp}.png"
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")

        # Also plot error vectors magnified
        plt.figure(figsize=(10, 10))
        magnification = 10.0
        for i, tu in enumerate(targets_unique):
            plt.plot(
                tu[0],
                tu[1],
                "rx",
                markersize=12,
                markeredgewidth=3,
                label="Target" if i == 0 else "",
            )

            for r in self.results:
                if np.allclose(r["target"][[idx1, idx2]], tu):
                    e = r["end"][[idx1, idx2]]
                    error = e - tu
                    # Plot magnified error from target
                    plt.arrow(
                        tu[0],
                        tu[1],
                        error[0] * magnification,
                        error[1] * magnification,
                        head_width=0.005,
                        head_length=0.005,
                        fc="g",
                        ec="g",
                        alpha=0.6,
                    )

        plt.xlabel(f"{label1} (m)")
        plt.ylabel(f"{label2} (m)")
        plt.title(
            f"Magnified Error Vectors (x{magnification})\n{plane_text}, {fix_param_text}, Radius={self.radius}m"
        )
        # Add stats text to plot here too
        plt.text(
            0.05,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        plt.grid(True)
        plt.axis("equal")
        plt.savefig(f"osc_test_errors_magnified_{self.mode}_{timestamp}.png")
        logger.info(
            f"Magnified error plot saved to osc_test_errors_magnified_{self.mode}_{timestamp}.png"
        )

        # Don't call plt.show() as it may block in non-GUI environments
        plt.close("all")

    def reset(self) -> None:
        """Reset the agent state."""
        self.current_target_idx = 0
        self.current_offset_idx = 0
        self.state = "GO_TO_START"
        self.osc_controller = None
        self.results = []
