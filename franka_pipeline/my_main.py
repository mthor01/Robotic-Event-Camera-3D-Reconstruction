"""
Minimal Franka pipeline entrypoint for HalfSphereRecordingAgent + ZMQ streaming.

- Runs only the relevant parts for controlling the robot and executing the agent.
- Does NOT start or read any cameras (RealSense handled in another container).
- Publishes system time + end-effector pose (and optionally joints) over ZeroMQ.
- Sends episode control notifications (episode_start / episode_end / quit) over ZeroMQ.

Assumptions:
- robot_controller.get_state() returns robot_state with:
    - robot_state["osc_pose"] : 4x4 numpy array (EE pose in base/world frame)
    - robot_state["joint_position"] : (7,) array
    - robot_state["gripper_q"] : scalar-ish
- HalfSphereRecordingAgent returns metadata with "end_episode"/"reset_episode"/"quit" similarly.
"""

import os
import time
from pathlib import Path

import numpy as np
import typer

# ZMQ deps
import zmq
import msgpack

from franka_pipeline.logging import get_logger, setup_logging
from franka_pipeline.agents.episode_control_wrapper_agent import EpisodeControlWrapperAgent
from franka_pipeline.agents.half_sphere_recording_agent import HalfSphereRecordingAgent
from franka_pipeline.robot_controllers.controller import (
    RealRobotController,
    SimulatedRobosuiteRobotController,
)
from franka_pipeline.sim.robosuite_env import RobosuiteSimEnv

app = typer.Typer(help="Half-sphere recording runner (no cameras) + ZMQ pose stream")
logger = get_logger(__name__)


class ZMQPosePublisher:
    """
    PUB socket that broadcasts:
      - topic b"pose": {t_ns, ep, step, ee_T(16 floats), q(7), gripper_q}
      - topic b"event": {t_ns, type, ep, ...}
    """

    def __init__(self, bind_addr: str):
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        # Keep latest if receiver is slow (live alignment use-case)
        self.pub.setsockopt(zmq.SNDHWM, 1)
        self.pub.bind(bind_addr)
        logger.info(f"ZMQ PUB bound at {bind_addr}")

    def publish_pose(
        self,
        ep: int,
        step: int,
        ee_T: np.ndarray | None,
        q: np.ndarray | None,
        gripper_q: float | None,
    ) -> None:
        msg = {
            "t_ns": time.time_ns(),
            "ep": int(ep),
            "step": int(step),
        }
        if ee_T is not None:
            ee_T = np.asarray(ee_T, dtype=np.float64)
            if ee_T.shape != (4, 4):
                raise ValueError(f"osc_pose must be (4,4), got {ee_T.shape}")
            msg["ee_T"] = ee_T.reshape(-1).tolist()  # 16 floats, row-major
        if q is not None:
            msg["q"] = np.asarray(q, dtype=np.float64).reshape(-1).tolist()
        if gripper_q is not None:
            msg["gripper_q"] = float(gripper_q)

        payload = msgpack.packb(msg, use_bin_type=True)
        self.pub.send_multipart([b"pose", payload])

    def publish_event(self, event_type: str, ep: int, extra: dict | None = None) -> None:
        msg = {"t_ns": time.time_ns(), "type": event_type, "ep": int(ep)}
        if extra:
            msg.update(extra)
        payload = msgpack.packb(msg, use_bin_type=True)
        self.pub.send_multipart([b"event", payload])


@app.command()
def main(
    simulated_robot: bool = typer.Option(
        False, "--simulated-robot/--no-simulated-robot", help="Run in robosuite sim"
    ),
    real_robot: bool = typer.Option(
        False, "--real-robot/--no-real-robot", help="Run on real robot"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable DEBUG logging"),
    deps_log_level: str = typer.Option(
        "INFO", "--deps-log-level", help="Logging level for dependencies"
    ),
    # ZMQ
    zmq_bind: str = typer.Option(
        os.getenv("POSE_PUB_BIND", "tcp://0.0.0.0:5556"),
        "--zmq-bind",
        help='ZMQ PUB bind address, e.g. "tcp://0.0.0.0:5556"',
    ),
    publish_hz: float = typer.Option(
        60.0,
        "--publish-hz",
        help="Max publish rate for pose stream (0 = publish every loop iteration)",
    ),
    # Agent params (you can expand as needed)
    # NOTE: If your HalfSphereRecordingAgent takes args, add them here.
) -> None:
    # Logging
    log_level = "DEBUG" if verbose else log_level
    setup_logging(level=log_level, deps_level=deps_log_level)

    # Mode selection
    if not simulated_robot and not real_robot:
        simulated_robot = True
        logger.info("No mode specified -> defaulting to simulated robot")

    if simulated_robot and real_robot:
        typer.echo("Error: cannot set both --simulated-robot and --real-robot")
        raise typer.Exit(code=1)

    # Robot controller
    if simulated_robot:
        # Minimal sim env config; camera settings irrelevant since we don't use images
        sim_env = RobosuiteSimEnv(controller_type="OSC_POSE", camera_width=64, camera_height=64)
        robot_controller = SimulatedRobosuiteRobotController(
            sim_env=sim_env, controller_type="OSC_POSE"
        )
    else:
        robot_controller = RealRobotController()

    robot_controller.reset_robot_joints()

    # Agent
    agent = EpisodeControlWrapperAgent(HalfSphereRecordingAgent())

    # ZMQ publisher
    pub = ZMQPosePublisher(bind_addr=zmq_bind)

    # Main loop
    step_count = 0
    ep_count = 0

    pub.publish_event("episode_start", ep=ep_count)

    # Throttle publishing if desired
    min_period = (1.0 / publish_hz) if publish_hz and publish_hz > 0 else 0.0
    next_pub_t = 0.0

    while True:
        robot_state, valid = robot_controller.get_state()
        if not valid:
            logger.warning("Invalid robot state, skipping iteration.")
            continue

        # Minimal observation/instruction: agent should rely on robot_state for this task
        observation = {}
        instruction = ""

        action, metadata = agent.act(
            robot_state=robot_state,
            observation=observation,
            instruction=instruction,
        )
        robot_controller.control(command=action, controller_type=agent.action_type)

        # Publish pose at most publish_hz (or every loop if publish_hz=0)
        now = time.time()
        if min_period == 0.0 or now >= next_pub_t:
            ee_pose = robot_state.get("osc_pose", None)
            q = robot_state.get("joint_position", None)
            gripper_q = None
            if "gripper_q" in robot_state:
                # your original code did robot_state["gripper_q"][np.newaxis], so it might be scalar-like
                try:
                    gripper_q = float(np.asarray(robot_state["gripper_q"]).reshape(-1)[0])
                except Exception:
                    gripper_q = None

            try:
                pub.publish_pose(
                    ep=ep_count,
                    step=step_count,
                    ee_T=ee_pose,
                    q=q,
                    gripper_q=gripper_q,
                )
            except Exception as e:
                logger.warning(f"Failed to publish pose: {e}")

            next_pub_t = now + min_period

        step_count += 1

        # Episode control mirrored from your main
        if metadata.get("end_episode", False):
            logger.info("Ending episode, as per agent request.")
            pub.publish_event("episode_end", ep=ep_count, extra={"step": step_count})

            robot_controller.reset_robot_joints()
            agent.reset()

            ep_count += 1
            step_count = 0

            pub.publish_event("episode_start", ep=ep_count)

        if metadata.get("reset_episode", False):
            logger.info("Resetting episode, as per agent request.")
            pub.publish_event("episode_reset", ep=ep_count, extra={"step": step_count})

            robot_controller.reset_robot_joints()
            agent.reset()
            step_count = 0

            # treat reset as continuing same episode id (matching your original behavior)
            pub.publish_event("episode_start", ep=ep_count)

        if metadata.get("quit", False):
            logger.info("Ending run, as per agent request.")
            pub.publish_event("quit", ep=ep_count, extra={"step": step_count})

            robot_controller.reset_robot_joints()
            if hasattr(agent, "stop"):
                agent.stop()
            break


if __name__ == "__main__":
    app()
