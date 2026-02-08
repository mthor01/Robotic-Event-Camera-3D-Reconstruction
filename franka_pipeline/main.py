"""Main entry point for the Franka pipeline.

This script manages the main control loop, component initialization,
and CLI argument parsing.
"""

#docker run -it -v $(pwd):/app --runtime=nvidia --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged openvla
#xhost +local:


import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import typer

from franka_pipeline.logging import get_logger, setup_logging

from franka_pipeline.agents.agent import (
    DummyAgentOscPose,
    DummyAgentDoNothing,
)
from franka_pipeline.agents.osc_pose_target_demo_agent import OscPoseTargetDemoAgent
from franka_pipeline.agents.osc_pose_target_testing_agent import (
    OscPoseTargetTestingAgent,
)
from franka_pipeline.agents.osc_pose_target_benchmarking_agent import (
    OscPoseTargetBenchmarkingAgent,
)
from franka_pipeline.agents.MolMoAnyGraspAgent import MolMoAnyGraspAgent
from franka_pipeline.agents.camera_eye_in_hand_calibration_agent import (
    CameraEyeInHandCalibrationAgent,
    CalibrationConfig,
)
from franka_pipeline.agents.episode_control_wrapper_agent import (
    EpisodeControlWrapperAgent,
)
from franka_pipeline.agents.teleoperation_agent import TeleoperationAgent
from franka_pipeline.agents.half_sphere_recording_agent import HalfSphereRecordingAgent
from franka_pipeline.agents.move_to_origin_agent import MoveToOriginAgent
from franka_pipeline.datacollector import DataCollector
from franka_pipeline.input_controllers.spacemouse import SpaceMouseController
from franka_pipeline.robot_controllers.controller import (
    RealRobotController,
    SimulatedRobosuiteRobotController,
)
from franka_pipeline.sensors import RealsenseCamera, DummyCamera
from franka_pipeline.sensors.robosuite_camera import RobosuiteCamera
from franka_pipeline.sim.robosuite_env import RobosuiteSimEnv
from franka_pipeline.visualization.live_visualizer import LiveVisualizer

from franka_pipeline.agents.half_sphere_recording_agent import HalfSphereRecordingAgent

app = typer.Typer(help="Modular framework for the Franka Emika Panda robot arm")

logger = get_logger(__name__)


# TODO move to utils
def _list_realsense_cameras() -> None:
    """Log connected RealSense camera serial numbers."""
    ctx = rs.context()
    devices = ctx.query_devices()
    logger.info("Connected RealSense devices:")
    for i, dev in enumerate(devices):
        logger.info(f"  Device {i}: {dev.get_info(rs.camera_info.name)}")
        logger.info(f"    Serial number: {dev.get_info(rs.camera_info.serial_number)}")


@app.command()
def main(
    simulated_robot: bool = typer.Option(
        False,
        "--simulated-robot/--no-simulated-robot",
        help="Run using the simulated robot",
    ),
    real_robot: bool = typer.Option(
        False, "--real-robot/--no-real-robot", help="Run using the real robot"
    ),
    live_viz: bool = typer.Option(
        True,
        "--live-viz/--no-live-viz",
        help="Enable live visualization with Rerun while recording",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    verbose: bool = typer.Option(
        False,  #  set to True for development, change to False for production
        "-v",
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
    deps_log_level: str = typer.Option(
        "INFO",
        "--deps-log-level",
        help="Logging level for dependencies (robosuite, deoxys) (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    # MolMo Agent parameters
    molmo_grasp: bool = typer.Option(
        False, "--molmo-grasp", help="Run the MolMoAnyGraspAgent"
    ),
    object_query: str = typer.Option(
        "Please point to the red block",
        "--object-query",
        help="Object to grasp for MolMoAnyGraspAgent",
    ),
    # OSC Pose Target Demo parameters
    osc_demo: bool = typer.Option(
        False, "--osc-demo", help="Run the OscPoseTargetDemoAgent"
    ),
    # OSC Pose Target Test parameters
    test_osc_pose_target: bool = typer.Option(
        False, "--test-osc-pose-target", help="Run the OscPoseTargetTestingAgent"
    ),
    test_osc_pose_target_mode: str = typer.Option(
        "xy",
        "--test-osc-pose-target-mode",
        help="Mode for OscPoseTargetTestingAgent (xy or yz)",
    ),
    # OSC Pose Target Benchmark parameters
    benchmark_osc_pose_target: bool = typer.Option(
        False,
        "--benchmark-osc-pose-target",
        help="Run the OscPoseTargetBenchmarkingAgent",
    ),
    benchmark_osc_pose_target_kp_pos: str = typer.Option(
        "20.0", "--benchmark-osc-pose-target-kp-pos", help="Kp values for position"
    ),
    benchmark_osc_pose_target_ki_pos: str = typer.Option(
        "14.0", "--benchmark-osc-pose-target-ki-pos", help="Ki values for position"
    ),
    benchmark_osc_pose_target_kp_rot: str = typer.Option(
        "1.0", "--benchmark-osc-pose-target-kp-rot", help="Kp values for rotation"
    ),
    benchmark_osc_pose_target_ki_rot: str = typer.Option(
        "6.0", "--benchmark-osc-pose-target-ki-rot", help="Ki values for rotation"
    ),
    benchmark_osc_pose_target_izone: str = typer.Option(
        "0.05",
        "--benchmark-osc-pose-target-izone",
        help="Comma-separated list of i_zone values",
    ),
    # Calibration parameters
    calibrate: bool = typer.Option(
        False, "--calibrate", help="Run the CameraEyeInHandCalibrationAgent"
    ),
    calibration_poses: str = typer.Option(
        "calibration_data/calibration_poses.npy",
        "--calibration-poses",
        help="Path to calibration poses file",
    ),
    camera_resolution: str = typer.Option(
        "640x480",
        "--camera-resolution",
        help="Camera resolution for the robot cameras (WxH)",
    ),
    camera_resolution_data_collector: str = typer.Option(
        "256x256",
        "--camera-resolution-data-collector",
        help="Camera resolution for the data collector (WxH)",
    ),
    # Half Sphere Recording Agent parameters
    half_sphere_record: bool = typer.Option(
        False, "--half-sphere-record", help="Run the HalfSphereRecordingAgent"
    ),
    half_sphere_radius: float = typer.Option(
        0.25, "--half-sphere-radius", help="Radius of the half-sphere"
    ),
    half_sphere_n_theta: int = typer.Option(
        8, "--half-sphere-n-theta", help="Number of azimuthal grid points"
    ),
    half_sphere_n_phi: int = typer.Option(
        5, "--half-sphere-n-phi", help="Number of elevation grid points"
    ),
    half_sphere_wait_time: float = typer.Option(
        2.0, "--half-sphere-wait-time", help="Wait time at each pose (seconds)"
    ),
    # Move To Origin Agent parameters
    move_to_origin: bool = typer.Option(
        False, "--move-to-origin", help="Move the arm to the origin and stop"
    ),
    origin_x: float = typer.Option(
        0.0, "--origin-x", help="Origin X coordinate"
    ),
    origin_y: float = typer.Option(
        0.0, "--origin-y", help="Origin Y coordinate"
    ),
    origin_z: float = typer.Option(
        0.0, "--origin-z", help="Origin Z coordinate"
    ),
    origin_wait_time: float = typer.Option(
        2.0, "--origin-wait-time", help="Wait time at origin (not used, for future)"
    ),
) -> None:
    """Run the Franka pipeline control loop.

    Args:
        simulated_robot: If True, use robosuite simulation.
        real_robot: If True, control the physical Franka robot.
        live_viz: If True, show live Rerun visualization.
        log_level: Logging level for franka_pipeline.
        verbose: If True, override log_level to DEBUG.
        deps_log_level: Logging level for dependency packages.
        camera_resolution: Resolution for robot cameras.
        camera_resolution_data_collector: Resolution for data collector.
    """
    # Configure logging first
    log_level = "DEBUG" if verbose else log_level
    setup_logging(level=log_level, deps_level=deps_log_level)

    # Parse camera resolutions
    try:
        cam_w, cam_h = map(int, camera_resolution.split("x"))
        data_w, data_h = map(int, camera_resolution_data_collector.split("x"))
    except ValueError:
        logger.error(
            f"Invalid camera resolution format. Expected 'WxH', got {camera_resolution} or {camera_resolution_data_collector}"
        )
        raise typer.Exit(code=1)

    if not simulated_robot and not real_robot:
        simulated_robot = True  # Default to simulated robot mode

    if simulated_robot and real_robot:
        typer.echo(
            "Error: Cannot use both simulated and real robot modes at the same time."
        )
        raise typer.Exit(code=1)

    # For convenience, list all connected RealSense camera serial numbers:
    # TODO, make CLI selection possible, for now just hard-coded script:
    _list_realsense_cameras()

    if simulated_robot:
        sim_env = RobosuiteSimEnv(
            controller_type="OSC_POSE", camera_width=cam_w, camera_height=cam_h
        )
        robot_controller = SimulatedRobosuiteRobotController(
            sim_env=sim_env, controller_type="OSC_POSE"
        )
    if real_robot:
        robot_controller = RealRobotController()

    robot_controller.reset_robot_joints()

    if simulated_robot:
        front_camera = RobosuiteCamera(sim_env=sim_env, camera_id="frontview")
        wrist_camera = RobosuiteCamera(sim_env=sim_env, camera_id="robot0_eye_in_hand")
    else:
        # front_camera = RealsenseCamera(
        #     serial_number="817412071429",
        #     enable_depth=True,
        #     width=cam_w,
        #     height=cam_h,
        # )
        front_camera = DummyCamera(width=cam_w, height=cam_h)
        wrist_camera = RealsenseCamera(
            serial_number="827112070121",
            enable_depth=True,
            width=cam_w,
            height=cam_h,
        )

    # Load calibration data
    calibration_file = Path("calibration_data/calibration_result.npz")
    T_ee2cam = None
    if simulated_robot:
        if hasattr(wrist_camera, "get_extrinsics"):
            T_ee2cam = wrist_camera.get_extrinsics()
            logger.info("Using ground truth simulation extrinsics for wrist camera")
    elif calibration_file.exists():
        try:
            calib_data = np.load(calibration_file)
            T_ee2cam = calib_data["T_ee2cam"]
            logger.info(f"Loaded calibration from {calibration_file}")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")

    if molmo_grasp:
        molmo_agent = MolMoAnyGraspAgent(
            T_ee2cam=T_ee2cam,
        )
        molmo_agent.set_object_to_grasp(object_query)
        agent = EpisodeControlWrapperAgent(molmo_agent)
    elif osc_demo:
        agent = EpisodeControlWrapperAgent(OscPoseTargetDemoAgent())
    elif test_osc_pose_target:
        agent = EpisodeControlWrapperAgent(
            OscPoseTargetTestingAgent(mode=test_osc_pose_target_mode)
        )
    elif benchmark_osc_pose_target:
        kp_pos_list = [float(x) for x in benchmark_osc_pose_target_kp_pos.split(",")]
        ki_pos_list = [float(x) for x in benchmark_osc_pose_target_ki_pos.split(",")]
        kp_rot_list = [float(x) for x in benchmark_osc_pose_target_kp_rot.split(",")]
        ki_rot_list = [float(x) for x in benchmark_osc_pose_target_ki_rot.split(",")]
        izone_list = [float(x) for x in benchmark_osc_pose_target_izone.split(",")]

        agent = EpisodeControlWrapperAgent(
            OscPoseTargetBenchmarkingAgent(
                kp_pos_list=kp_pos_list,
                ki_pos_list=ki_pos_list,
                kp_rot_list=kp_rot_list,
                ki_rot_list=ki_rot_list,
                i_zone_list=izone_list,
                max_steps=60,
            )
        )
    elif calibrate:
        calib_agent = CameraEyeInHandCalibrationAgent(
            poses_file=calibration_poses,
            camera_name="wrist_image",
        )
        agent = EpisodeControlWrapperAgent(calib_agent)
        calib_agent.start()
    elif half_sphere_record:
        agent = EpisodeControlWrapperAgent(
            HalfSphereRecordingAgent(
                #center=np.array([0.0, 0.0, 0.0]),
                #radius=half_sphere_radius,
                #n_theta=half_sphere_n_theta,
                #n_phi=half_sphere_n_phi,
                #wait_time=half_sphere_wait_time,
            )
        )
    elif move_to_origin:
        agent = EpisodeControlWrapperAgent(
            MoveToOriginAgent(
                origin=np.array([origin_x, origin_y, origin_z]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper=-1.0,
            )
        )
    else:
        agent = EpisodeControlWrapperAgent(
            TeleoperationAgent(input_controller=KeyboardController())
        )

    # datacollector = DataCollector(overwrite=True)
    # Don't collect data for calibration but for all other agents
    datacollector = None
    if not calibrate:
        datacollector = DataCollector(
            overwrite=True, camera_width=data_w, camera_height=data_h
        )

    visualizer = None
    if live_viz:
        # Path to URDF for 3D robot visualization
        urdf_path = (
            Path(__file__).parent
            / "franka_pipeline"
            / "visualization"
            / "urdfs"
            / "fer_franka_hand.urdf"
        )
        repo_name = datacollector.repo_name if datacollector else "franka_calibration"
        visualizer = LiveVisualizer(
            repo_name=repo_name,
            viewer_spawn=True,
            urdf_path=urdf_path,
            camera_extrinsics=T_ee2cam,
            camera_width=cam_w,
            camera_height=cam_h,
            # axis_length=None,  # Hide axes for all robot joints
            # axis_length=0.12,  # Show axes for all robot joints
        )

        # # Test plotting a transformation matrix
        # test_T = np.eye(4)
        # test_T[:3, 3] = [0.0, 0.0, 0.0]
        # # Add a small rotation
        # from scipy.spatial.transform import Rotation

        # test_T[:3, :3] = Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()

        # # Plot relative to the robot base
        # visualizer.plot_transformation_matrix(
        #     test_T, "base", color=[255, 0, 0], name="base_origin"
        # )

        # test_T = np.eye(4)
        # test_T[:3, 3] = [0.0, 0.2, 0.0]
        # # Add a small rotation
        # from scipy.spatial.transform import Rotation

        # test_T[:3, :3] = Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()

        # visualizer.plot_transformation_matrix(
        #     test_T, "base", color=[255, 0, 0], name="base_origin_offset_x"
        # )

        # # Test plotting a transformation matrix
        # test_T = np.eye(4)
        # test_T[:3, 3] = [0.2, 0.0, 0.0]  # Offset by 20cm in X, Y, Z
        # # Add a small rotation
        # from scipy.spatial.transform import Rotation

        # test_T[:3, :3] = Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()

        # # Plot relative to the robot base
        # visualizer.plot_transformation_matrix(
        #     test_T, "base/fer_link0", color=[255, 0, 0]
        # )
        # logger.info("Plotted test transformation matrix relative to base/fer_link0")

        # # Plot relative to the robot ee
        # visualizer.plot_transformation_matrix(
        #     test_T,
        #     "base/fer_link0/fer_link1/fer_link2/fer_link3/fer_link4/fer_link5/fer_link6/fer_link7/fer_link8",
        #     color=[0, 255, 0],
        # )

        # To clear all plottings, you can call:
        # visualizer.clear_all_transformation_matrix_plotting()

    front_camera.start()
    wrist_camera.start()

    front_intrinsics = None
    if hasattr(front_camera, "get_intrinsics"):
        front_intrinsics = front_camera.get_intrinsics()

    wrist_intrinsics = None
    if hasattr(wrist_camera, "get_intrinsics"):
        wrist_intrinsics = wrist_camera.get_intrinsics()

    # HACKY: Discard initial frames for camera warm-up to get a stable stream
    for _ in range(10):
        _ = front_camera.get_frames()
        _ = wrist_camera.get_frames()

    step_count = 0
    ep_count = 0

    while True:
        robot_state, valid = robot_controller.get_state()

        # logger.debug(f"robot_state: {robot_state}")
        # logger.debug(
        #     f"robot_state['osc_position']: {robot_state['osc_position']}, robot_state['osc_rotation_quaternion']: {robot_state['osc_rotation_quaternion']}"
        # )

        # print("robot_state:", robot_state)
        if not valid:
            logger.warning("Invalid robot state, skipping iteration.")
            continue

        color_image, front_depth = front_camera.get_frames()
        wrist_image, wrist_depth = wrist_camera.get_frames()

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
        # color_image = cv2.resize(
        #     color_image, (256, 256), interpolation=cv2.INTER_LINEAR
        # )
        # wrist_image = cv2.resize(
        #     wrist_image, (256, 256), interpolation=cv2.INTER_LINEAR
        # )

        observation = {"image": color_image, "wrist_image": wrist_image}
        if front_depth is not None:
            observation["front_depth"] = front_depth
        if front_intrinsics is not None:
            observation["front_intrinsics"] = front_intrinsics
        if wrist_depth is not None:
            observation["wrist_depth"] = wrist_depth
        if wrist_intrinsics is not None:
            observation["wrist_intrinsics"] = wrist_intrinsics
        instruction = ""

        action, metadata = agent.act(
            robot_state=robot_state,
            observation=observation,
            instruction=instruction,
        )
        robot_controller.control(command=action, controller_type=agent.action_type)

        step_metadata = {
            "step": step_count,
            "episode": ep_count,
            "action_type": agent.action_type,
            "agent": agent.__class__.__name__,
            "instruction": instruction,
            "timestamp": time.time(),
        }
        step_metadata.update(metadata)

        state_vec = np.concatenate(
            [
                robot_state["joint_position"],
                robot_state["gripper_q"][np.newaxis],
            ]
        ).astype(np.float32)

        if datacollector:

            datacollector_obs = observation.copy()
            resized_color_image = cv2.resize(
                color_image, (data_w, data_h), interpolation=cv2.INTER_LINEAR
            )
            resized_wrist_image = cv2.resize(
                wrist_image, (data_w, data_h), interpolation=cv2.INTER_LINEAR
            )
            datacollector_obs["image"] = resized_color_image
            datacollector_obs["wrist_image"] = resized_wrist_image

            datacollector.collect(
                datacollector_obs, action.astype(np.float32), state_vec, step_metadata
            )

        if visualizer:
            # # Plot the current end-effector pose for debugging
            ee_pose = robot_state.get("osc_pose")
            if ee_pose is not None:
                T_identity = np.eye(4)
                visualizer.plot_transformation_matrix(
                    T_identity,
                    "base/fer_link0",
                    color=[255, 255, 255],
                    name="robot_base",
                )

                visualizer.plot_transformation_matrix(
                    ee_pose, "base/fer_link0", color=[0, 0, 255], name="current_ee_pose"
                )

                # Plot the camera pose in world coordinates
                if T_ee2cam is not None:
                    camera_world_pose = ee_pose @ T_ee2cam
                    visualizer.plot_transformation_matrix(
                        camera_world_pose,
                        "base/fer_link0",
                        color=[255, 0, 0],  # Red
                        name="camera_world_pose",
                    )

            visualizer.add_data(
                obs=observation,
                action=action.astype(np.float32),
                robot_state=state_vec,
                metadata=step_metadata,
                ee_pose=ee_pose,
            )

        step_count += 1

        if metadata.get("end_episode", False):
            logger.info("Ending episode, as per agent request.")
            # TODO currently we are using robot_controller.reset_robot_joints() to represent resetting the whole environment.
            # In real, the user needs to do some stuff for reset
            # In sim, in fact it is not the robot_controller that is reset, but the whole environment
            # We could make this clearer in the future
            robot_controller.reset_robot_joints()
            if datacollector:
                datacollector.save()
            agent.reset()
            ep_count += 1
            step_count = 0
            if visualizer:
                visualizer.start_new_episode(ep_count)

        if metadata.get("reset_episode", False):
            logger.info("Resetting episode, as per agent request.")
            robot_controller.reset_robot_joints()
            if datacollector:
                datacollector.clear_episode_buffer()
            agent.reset()
            step_count = 0
            if visualizer:
                visualizer.start_new_episode(ep_count)

        if metadata.get("quit", False):
            logger.info("Ending run, as per agent request.")
            robot_controller.reset_robot_joints()
            if datacollector:
                datacollector.clear_episode_buffer()
                datacollector.finalize()
            # Clean up agents
            if hasattr(agent, "stop"):
                agent.stop()
            # if hasattr(calibration_agent, "stop"):
            # calibration_agent.stop()
            break

        # Check if calibration is complete
        if metadata.get("calibration_state") == "complete":
            logger.info("Calibration complete!")
            # if hasattr(calibration_agent, "save_calibration"):
            # calibration_path = calibration_agent.save_calibration()
            # logger.info(f"Calibration saved to {calibration_path}")
            robot_controller.reset_robot_joints()
            if hasattr(agent, "stop"):
                agent.stop()
            # if hasattr(calibration_agent, "stop"):
            # calibration_agent.stop()
            break


if __name__ == "__main__":
    app()
