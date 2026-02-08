import time
import numpy as np
from typing import Tuple, Optional
from deoxys.utils.transform_utils import quat2axisangle, quat_distance
from franka_pipeline.logging import get_logger

logger = get_logger(__name__)


class OscPoseTargetController:
    """
    Controller for moving to a target pose using OSC (Operational Space Control).

    The controller implements threshold_reach, threshold_rotation, min_steps, max_steps, and extra_steps to precisely control when the movement should be considered finished.

    - at least min_steps are executed, regardless of the thresholds
    - at most max_steps are executed, after which the movement is considered finished regardless of thresholds
    - once within thresholds, extra_steps are executed to improve convergence before considering the movement finished

    A user should probably consider the following default conditions:
    - Precise reaching of the target pose up to some threshold is required. In this case, the appropriate thresholds should be set and optionally extra_steps to improve convergence.
    - The user has an alotted time budget for the movement and wants to reach the target as good as possible. In this case max_steps should be set with very small thresholds.


    This implements a conditional saturated PI controller for both position and orientation. The additional integral term faciliates precise reaching of the target pose, as without it the real robot normally falls short of the target by ~2cm.

    A naive proportional controller would compute the action as the difference between the current pose and the target pose, scaled by a gain factor. However, such a controller is not capable of precise control on a real robot due to stiction and friction, so movements fall short of the target.

    This controller adds an integral term to the control law (-> PI controller). The integral term helps to eliminate steady-state error (i.e., the shortfall from the target due to friction).

    This implements a PI controller
        F = Kp * e + Ki * ∫e dt
    Where F is the force (here it is the output action), e is the error between current and target pose (i.e. the naive action to reach the target), ∫e dt is the integral of the error over time, and Kp and Ki are the proportional and integral gains respectively.

    Additionally, through i_max and i_rot_max integral windup is prevented (-> saturated PI). This prevents the integral term from accumulating excessively when the error is large for a prolonged period, which can lead to overshooting and instability, e.g. for unreachable targets or when the robot is temporarily blocked.

    the i_zone parameter defines the threshold within which the integral term is accumulated. This helps to prevent the integral term from accumulating when the system is far from the target, which can lead to instability.


    The parameters default values have been tuned to work well with the real robot with which I am working. They can be adjusted using the OscPoseTargetBenchmarkingAgent.


    NOTE: On movement to a new target, this class should be re-instantiated with the new target pose. (The self.target_pose of this class should never be updated). This is because the integral error terms are accumulated over time, and if the target pose is changed, the accumulated error terms would no longer be valid for the new target pose.

    NOTE: That this controller is not a robot controller in the sense of the franka_pipeline.robot_controllers.controller.Controller Abstract base class, but rather it is a helper class to compute OSC actions for a given target pose, which is used by agents (e.g. OscPoseTargetDemoAgent).
    NOTE: To sketch the full control stack: This controller produces cartesian actions, which are sent to the high-frequency 1000Hz robot controller (in this case implemented by deoxys) which calculates the appropriate joint torques by means of the jacobian and implements a PD controller for these joint torques. These joint torques are in turn executed by libfranka on the real robot (where there probably also is an additional control layer controlling the mismatch between desired torques and torques measured by the joint torque sensors).
    For this reason also, no D (derivative term) for a full PID controller is implemented here, as the derivative control is already implemented in the low-level robot controller.
    """

    def __init__(
        self,
        target_pose: np.ndarray,
        threshold_reach: float = 0.005,
        threshold_rotation: float = 0.05,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        extra_steps: Optional[int] = None,
        kp_pos: float = 20.0,
        kp_rot: float = 2.0,
        ki_pos: float = 18.0,
        ki_rot: float = 4.0,
        i_zone: float = 0.05,
        i_max: float = 0.1,
        rot_i_max: float = 0.05,
    ):
        """
        Initialize the OscPoseTargetController.

        Args:
            target_pose: Target pose of shape (8,) -> [x, y, z, qx, qy, qz, qw, gripper].
            threshold_reach: Distance threshold (meters) to consider the target position as reached.
            threshold_rotation: Rotation threshold (radians) to consider the target orientation as reached.
            min_steps: Minimum number of steps to run the controller for.
            max_steps: Maximum number of steps to run the controller for. If reached, is_finished will be set to True.
            extra_steps: Number of additional steps to run after reaching the target threshold to improve convergence.
            kp_pos: Proportional gain for position control.
            ki_pos: Integral gain for position control.
            kp_rot: Proportional gain for rotation control.
            ki_rot: Integral gain for rotation control.
            i_zone: Threshold within which to accumulate integral error (meters for position, radians for rotation).
            i_max: Maximum value for the accumulated integral error.
            rot_i_max: Maximum value for the accumulated rotation integral error.
        """
        self.target_pose = target_pose
        self.threshold_reach = threshold_reach
        self.threshold_rotation = threshold_rotation
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.extra_steps = extra_steps
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kp_rot = kp_rot
        self.ki_rot = ki_rot
        self.i_zone = i_zone
        self.i_max = i_max
        self.rot_i_max = rot_i_max

        self.steps = 0
        self.extra_steps_done = 0
        self.extra_steps_triggered = False

        self.pos_error_integral = np.zeros(3)
        self.rot_error_integral = np.zeros(3)
        self.last_time: Optional[float] = None

    def calculate_action(self, current_pose: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Calculate the OSC action to reach the target pose.

        Args:
            current_pose: Current pose of shape (7,) -> [x, y, z, qx, qy, qz, qw].

        Returns:
            Tuple of (action, is_finished) where action is [dx, dy, dz, dax, day, daz, gripper].
        """
        current_time = time.time()
        dt = (
            current_time - self.last_time
            if self.last_time is not None
            else 0.01  # Default dt if first call
        )
        self.last_time = current_time

        target_pos = self.target_pose[:3]
        target_quat = self.target_pose[3:7]
        gripper_action = self.target_pose[7]

        current_pos = current_pose[:3]
        current_quat = current_pose[3:7]

        # Ensure quaternions are in same hemisphere
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat

        # Position error
        pos_error = target_pos - current_pos
        dist = np.linalg.norm(pos_error)

        # Update integral action
        if dist < self.i_zone:
            self.pos_error_integral += pos_error * dt
        else:
            self.pos_error_integral[:] = 0.0

        self.pos_error_integral = np.clip(
            self.pos_error_integral, -self.i_max, self.i_max
        )

        # Scale position deltas
        action_pos = pos_error * self.kp_pos + self.ki_pos * self.pos_error_integral
        action_pos = np.clip(action_pos, -1.0, 1.0)

        # Rotation error
        quat_diff = quat_distance(target_quat, current_quat)
        axis_angle_diff = quat2axisangle(quat_diff).flatten()
        rot_dist = np.linalg.norm(axis_angle_diff)

        # Update integral action for rotation
        if rot_dist < self.i_zone:
            self.rot_error_integral += axis_angle_diff * dt
        else:
            self.rot_error_integral[:] = 0.0

        self.rot_error_integral = np.clip(
            self.rot_error_integral, -self.rot_i_max, self.rot_i_max
        )

        action_axis_angle = (
            axis_angle_diff * self.kp_rot + self.ki_rot * self.rot_error_integral
        )
        action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

        within_threshold = (dist < self.threshold_reach) and (
            rot_dist < self.threshold_rotation
        )

        self.steps += 1
        is_finished = False

        if self.extra_steps_triggered:
            self.extra_steps_done += 1
            if self.extra_steps_done >= self.extra_steps:
                is_finished = True
        elif (
            self.min_steps is None or self.steps >= self.min_steps
        ) and within_threshold:
            if self.extra_steps is None or self.extra_steps <= 0:
                is_finished = True
            else:
                self.extra_steps_triggered = True
                self.extra_steps_done = 0

        if self.max_steps is not None and self.steps >= self.max_steps:
            is_finished = True

        action = np.concatenate(
            [action_pos, action_axis_angle, [float(gripper_action)]]
        )

        logger.debug(
            f"OSC Action Calculation:\n"
            f"  Current Pos: {current_pos}, Target Pos: {target_pos}, Pos Error: {pos_error}, Dist: {dist}\n"
            f"  Current Quat: {current_quat}, Target Quat: {target_quat}, Rot Dist: {rot_dist}\n"
            f"  Pos Error Integral: {self.pos_error_integral}, Rot Error Integral: {self.rot_error_integral}\n"
            f"  Action Pos: {action_pos}, Action Axis-Angle: {action_axis_angle}\n"
            f"  Gripper Action: {gripper_action}, Within Threshold: {within_threshold}, "
            f"Extra Steps Done: {self.extra_steps_done}/{self.extra_steps if self.extra_steps is not None else 0}, "
            f"Steps: {self.steps}, Is Finished: {is_finished}"
        )

        return action.astype(np.float32), is_finished
