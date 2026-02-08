"""Robot controllers for interfacing with real and simulated Franka robots.

This module provides controller classes for sending commands to and receiving
state from the Franka Emika Panda robot, both in simulation (robosuite) and
with real hardware (via deoxys).
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any

import deoxys
import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.yaml_config import YamlConfig

from franka_pipeline.logging import get_logger

logger = get_logger(__name__)


class Controller(ABC):
    """Abstract base class for robot controllers.

    Provides a unified interface for sending control commands to and
    receiving state from robot controllers.
    """

    @abstractmethod
    def control(self, command: np.ndarray, controller_type: str | None = None) -> None:
        """Send a control command to the robot.

        Args:
            command: Control command array (shape depends on controller_type).
            controller_type: Type of controller to use.
        """

    @abstractmethod
    def get_state(self) -> tuple[dict[str, Any], bool]:
        """Get the current robot state.

        Returns:
            Tuple of (state_dict, is_valid) where state_dict contains
            robot state information and is_valid indicates whether a valid state has been obtained.
        """

    @abstractmethod
    def reset_robot_joints(self) -> None:
        """Reset robot to a default joint configuration."""


# TODO: test all of them
_SUPPORTED_CONTROLLER_TYPES_SIM_ROBOT = [
    "OSC_POSE",
    "OSC_POSITION",
    "JOINT_POSITION",
    "JOINT_VELOCITY",
    "JOINT_TORQUE",
]


class SimulatedRobosuiteRobotController(Controller):
    """Controller for simulated Franka robot in robosuite.

    Note: Differences of SimulatedRobosuiteRobotController and RealRobotController:
    - Robosuite does not allow switching controller type after environment creation,
    unlike deoxys for real hardware, where the controller type may be switched at execution time.
    - TODO(check this): Robosuite does not allow partial gripper open/close commands, only full open/close. (in Robosuite: gripper_action < 0: open, >=0: closed; In deoxys: gripper_action in [-1, 0] allows continuous closing, gripper action > 0: open)
    - In robosuite, on reset_robo_joints() the whole environment is reset, while in deoxys only the robot is moved to the home position.

    Attributes:
        sim_env: The robosuite simulation environment.
        controller_type: The controller type used (fixed at creation).
        control_freq: Control loop frequency in Hz.
    """

    def __init__(self, sim_env: Any, controller_type: str = "OSC_POSE") -> None:
        """Initialize the simulated robot controller.

        Args:
            sim_env: Robosuite simulation environment instance.
            controller_type: Controller type to use (fixed for lifetime).
        """
        self.sim_env = sim_env
        assert (
            controller_type in _SUPPORTED_CONTROLLER_TYPES_SIM_ROBOT
        ), f"Unsupported controller type for SimulatedRobosuiteRobotController: {controller_type}"
        self.controller_type = controller_type
        self.control_freq = getattr(sim_env, "control_freq", 20)
        self.sim_env.reset()

    def control(self, command: np.ndarray, controller_type: str | None = None) -> None:
        """Send a control command to the simulated robot.

        Args:
            command: Control command array.
            controller_type: Controller type (must match init type).

        Raises:
            ValueError: If trying to change controller type.
        """
        if controller_type is not None and self.controller_type != controller_type:
            raise ValueError(
                "Changing controller type after environment startup is not supported for SimulatedRobotController using Robosuite."
            )

        self.sim_env.step(command)

    def reset_robot_joints(self) -> None:
        """Reset the simulation environment to default state."""
        self.sim_env.reset()

    def get_state(self) -> tuple[dict[str, Any], bool]:
        """Get the current state of the simulated robot.

        Returns:
            Tuple of (state_dict, is_valid) containing:
                - osc_pose: 4x4 transformation matrix
                - osc_position: (3,) position vector
                - osc_rotation_matrix: 3x3 rotation matrix
                - osc_rotation_quaternion: (4,) quaternion [x, y, z, w]
                - joint_position: (7,) joint positions
                - joint_velocity: (7,) joint velocities
                - # TODO: maybe also get other joint values (joint_torque, ...)?
                - gripper_q: gripper joint position
        """
        state = self.sim_env.get_state()
        is_valid = all(v is not None for v in state.values())
        return state, is_valid


# TODO: test all of them
_SUPPORTED_CONTROLLER_TYPES_REAL_ROBOT = [
    "OSC_POSE",
    "OSC_POSITION",
    "OSC_YAW",
    "JOINT_IMPEDANCE",
    "JOINT_POSITION",
    "CARTESIAN_VELOCITY",
]


class RealRobotController(Controller):
    """Controller for real Franka Emika Panda robot via deoxys.

    Attributes:
        robot_interface: The deoxys FrankaInterface instance.
        default_controller_type: Default controller type when not specified by control() calls.
        controller_configs: Loaded controller configurations by type.
    """

    def __init__(self, default_controller_type: str = "OSC_POSE") -> None:
        """Initialize the real robot controller.

        Args:
            default_controller_type: Controller type to use when not
                explicitly specified in control() calls.
        """
        self.robot_interface = FrankaInterface(
            os.path.join(deoxys.config_root, "charmander.yml")
        )

        # TODO: check, if connection is actually established

        assert (
            default_controller_type in _SUPPORTED_CONTROLLER_TYPES_REAL_ROBOT
        ), f"Unsupported default controller type for RealRobotController: {default_controller_type}"
        self.default_controller_type = default_controller_type

        # Load default controller configs
        self.controller_configs: dict[str, Any] = {}
        for ctrl_type in _SUPPORTED_CONTROLLER_TYPES_REAL_ROBOT:
            config = get_default_controller_config(ctrl_type)

            # # Optimize OSC controllers for higher precision and responsiveness
            # if "OSC_POSE" in ctrl_type:
            #     if "Kp" in config:
            #         # DEFAULT:
            #         # translation: 150.0
            #         # rotation: 250.0
            #         #     # config.Kp.translation = [500.0, 500, 500]
            #         #     # config.Kp.rotation = [500, 500, 500]
            #         config.Kp.translation = [800.0] * 3, 800.0, 800.0]
            #         config.Kp.rotation = [250] * 3
            #     #     # config.Kp.translation = [750.0, 750, 750]
            #     #     # config.Kp.rotation = [750, 750, 750]

            #     # # if "traj_interpolator_cfg" in config:
            #     #     # config.traj_interpolator_cfg.time_fraction = 0.1
            #     # TODO: For faster robot movement i would like to increase the action_scaling, however movement becomes mroe erratic
            #     # if "action_scale" in config:
            #     # Increase translation scale slightly to allow more authority
            #     # config.action_scale.translation = 0.1

            self.controller_configs[ctrl_type] = config

    def set_controller_config(self, control_interface: str, config_dir: str) -> None:
        """Set a custom controller configuration from a YAML file.

        Args:
            control_interface: Controller type to configure.
            config_dir: Path to the YAML configuration file.
        """
        self.controller_configs[control_interface] = YamlConfig(
            os.path.dirname(config_dir), os.path.basename(config_dir)
        )

    def control(self, command: np.ndarray, controller_type: str | None = None) -> None:
        """Send a control command to the real robot.

        Args:
            command: Control command array.
            controller_type: Controller type to use. If None, uses default.

        Raises:
            ValueError: If controller type is not supported.
        """
        if controller_type is None:
            controller_type = self.default_controller_type

        if controller_type not in _SUPPORTED_CONTROLLER_TYPES_REAL_ROBOT:
            raise ValueError(f"Unsupported controller type: {controller_type}")

        self.robot_interface.control(
            controller_type=controller_type,
            action=command,
            controller_cfg=self.controller_configs[controller_type],
        )

    def get_state(self) -> tuple[dict[str, Any], bool]:
        """Get the current state of the real robot.

        Returns:
            Tuple of (state_dict, is_valid) containing:
                - osc_pose: 4x4 transformation matrix
                - osc_position: (3,) position vector
                - osc_rotation_matrix: 3x3 rotation matrix
                - osc_rotation_quaternion: (4,) quaternion
                - joint_position: (7,) joint positions
                - joint_velocity: (7,) joint velocities
                - gripper_q: gripper joint positions
        """
        if self.robot_interface.state_buffer_size == 0:
            return {}, False

        osc_pose = self.robot_interface.last_eef_pose
        joint_position = self.robot_interface.last_q
        joint_velocity = self.robot_interface.last_dq
        osc_rotation_matrix, osc_position = self.robot_interface.last_eef_rot_and_pos
        osc_rotation_quaternion, _ = self.robot_interface.last_eef_quat_and_pos

        # TODO: For some reason this property does not exist
        # joint_torque = self.robot_interface.last_tau_J
        # In deoxys C++ code it does, so this should work:
        # joint_torque = self.robot_interface._state_buffer[-1].tau_J
        gripper_q = self.robot_interface.last_gripper_q

        state_values = [
            osc_pose,
            osc_position,
            osc_rotation_matrix,
            osc_rotation_quaternion,
            joint_position,
            joint_velocity,
            gripper_q,
        ]
        is_valid = all(v is not None for v in state_values)

        return {
            "osc_pose": osc_pose,
            "osc_position": (
                # TODO this check is still needed, to prevent errors when osc_position is None, but this is not very nice
                osc_position.flatten()
                if osc_position is not None
                else None
            ),
            "osc_rotation_matrix": osc_rotation_matrix,
            "osc_rotation_quaternion": osc_rotation_quaternion,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_q": gripper_q,
        }, is_valid

    def reset_robot_joints(self, time_limit: float = 5.0) -> None:
        """Reset robot to a default joint configuration.

        Args:
            time_limit: Maximum time in seconds to attempt reset.
        """
        # TODO: sometimes the reset does not finish due to the tolerances being too small
        # -> currently this is solved with the time-limit, but modifying the tolerances would be better
        # TODO for better generated training data we might want to add some random noise to the reset position, with an additional function parameter
        # TODO: allow time_limit = None for no time limit
        # TODO: make the tolerance a function parameter
        # Default home configuration
        reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

        action = np.array(reset_joint_positions + [-1.0])
        start_time = time.time()

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                current_q = np.array(self.robot_interface._state_buffer[-1].q)
                distance = np.max(np.abs(current_q - np.array(reset_joint_positions)))
                if distance < 1e-3:
                    break
                if time.time() - start_time > time_limit:
                    logger.warning("Joint Reset exceeded time limit. Stopping.")
                    break
            self.control(command=action, controller_type="JOINT_POSITION")
