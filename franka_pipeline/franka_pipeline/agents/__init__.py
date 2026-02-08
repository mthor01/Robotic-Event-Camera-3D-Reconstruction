"""Agent modules for robot control.

This package provides various agent implementations for controlling
the Franka robot, including teleoperation and replay agents.
"""

from .agent import (
    Agent,
    DummyAgentOscPose,
    DummyAgentDoNothing,
    ReplayAgent,
)
from .camera_eye_in_hand_calibration_agent import (
    CalibrationConfig,
    CameraEyeInHandCalibrationAgent,
    CharucoProperties,
)
from .episode_control_wrapper_agent import EpisodeControlWrapperAgent
from .osc_pose_target_demo_agent import OscPoseTargetDemoAgent
from .teleoperation_agent import TeleoperationAgent

# from .remote_agent import RemoteAgent

__all__ = [
    "Agent",
    "CalibrationConfig",
    "CameraEyeInHandCalibrationAgent",
    "CharucoProperties",
    "EpisodeControlWrapperAgent",
    "DummyAgentOscPose",
    "DummyAgentDoNothing",
    "ReplayAgent",
    "TeleoperationAgent",
    # "RemoteAgent",
]
