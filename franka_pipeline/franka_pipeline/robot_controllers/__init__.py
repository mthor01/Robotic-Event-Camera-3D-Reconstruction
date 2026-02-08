"""Robot controller modules for Franka arm control.

This package provides controller interfaces for both real and simulated
Franka Emika Panda robots.
"""

from .controller import (
    Controller,
    RealRobotController,
    SimulatedRobosuiteRobotController,
)

__all__ = [
    "Controller",
    "RealRobotController",
    "SimulatedRobosuiteRobotController",
]
