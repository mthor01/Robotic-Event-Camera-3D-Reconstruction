"""
Synchronized multi-camera recording with ZeroMQ communication for robot arm coordination.

This script:
1. Initializes event cameras and RealSense depth camera
2. Sends a "ready" signal to my_main.py via ZMQ REQ/REP
3. Waits for a "start" signal from my_main.py to begin synchronized recording
4. Receives robot pose data over ZMQ PUB/SUB during recording
5. Stores all data (depth, poses, timestamps) in HDF5

Run this script first, then start my_main.py with --sync-recording flag.
"""
import argparse
import gc
import time
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
import h5py
from pathlib import Path
from multiprocessing import Process, Event as MPEvent, Queue
from collections import deque
from typing import Optional

import zmq
import msgpack

from metavision_hal import DeviceDiscovery
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm


# ================= CONFIG =================
FPS = 30

DATA_DIR = Path("data/real")

RS_WIDTH, RS_HEIGHT = 640, 480

BIAS_DIFF_ON = 10 #10 
BIAS_DIFF_OFF = 80 # 80
BIAS_FO = 0
BIAS_HPF = 50
BIAS_REFR = 150

# ZMQ addresses
ZMQ_SYNC_ADDR = "tcp://localhost:6001"  # REQ/REP for sync handshake
ZMQ_POSE_ADDR = "tcp://localhost:6000"  # PUB/SUB for pose streaming (from my_main)
# =========================================


class ZMQPoseReceiver:
    """
    Receives pose data from my_main.py over ZMQ PUB/SUB.
    Stores poses in a thread-safe deque.
    Also monitors for hemisphere_complete/quit events to signal recording stop.
    """

    def __init__(self, connect_addr: str = ZMQ_POSE_ADDR):
        self.connect_addr = connect_addr
        self.ctx = zmq.Context.instance()
        self.sub: Optional[zmq.Socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Thread-safe storage for pose data
        self.poses: deque = deque(maxlen=100000)  # Large buffer
        self.lock = threading.Lock()
        
        # Event to signal that recording should stop
        self.stop_recording_event = threading.Event()

    def start(self) -> None:
        """Start the receiver thread."""
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"pose")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"event")
        self.sub.setsockopt(zmq.RCVHWM, 10000)
        self.sub.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout for clean shutdown
        self.sub.connect(self.connect_addr)

        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[PoseReceiver] Started, connected to {self.connect_addr}")

    def _receive_loop(self) -> None:
        """Background thread that receives pose/event data."""
        while self.running:
            try:
                topic, payload = self.sub.recv_multipart()
                t_recv = time.time_ns()
                msg = msgpack.unpackb(payload, raw=False)

                with self.lock:
                    if topic == b"event":
                        event_type = msg.get("type", "")
                        if event_type == "agent_complete":
                            print(f"[PoseReceiver] Received '{event_type}' event - signaling stop")
                            self.stop_recording_event.set()
                    elif topic == b"pose":
                        msg["t_recv_ns"] = t_recv
                        self.poses.append(msg)
            except zmq.error.Again:
                # Timeout, check if we should keep running
                continue
            except Exception as e:
                if self.running:
                    print(f"[PoseReceiver] Error: {e}")

    def stop(self) -> None:
        """Stop the receiver thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.sub:
            self.sub.close()
            self.sub = None
        print("[PoseReceiver] Stopped")

    def should_stop_recording(self) -> bool:
        """Check if a stop signal has been received."""
        return self.stop_recording_event.is_set()

    def get_all_poses(self) -> list:
        """Get all received poses and clear the buffer."""
        with self.lock:
            poses = list(self.poses)
            self.poses.clear()
            return poses


class ZMQSyncClient:
    """
    REQ socket to synchronize with my_main.py.
    Sends "ready" when cameras are initialized, receives "start" to begin recording.
    """

    def __init__(self, connect_addr: str = ZMQ_SYNC_ADDR):
        self.connect_addr = connect_addr
        self.ctx = zmq.Context.instance()
        self.req: Optional[zmq.Socket] = None

    def connect(self) -> None:
        """Connect to the sync server (my_main.py)."""
        self.req = self.ctx.socket(zmq.REQ)
        self.req.connect(self.connect_addr)
        print(f"[SyncClient] Connected to {self.connect_addr}")

    def send_ready_wait_start(self, timeout_sec: float = 60.0) -> bool:
        """
        Send 'ready' signal and wait for 'start' response.
        Returns True if start signal received, False on timeout.
        """
        self.req.setsockopt(zmq.RCVTIMEO, int(timeout_sec * 1000))
        msg = {"type": "ready", "t_ns": time.time_ns()}
        self.req.send(msgpack.packb(msg, use_bin_type=True))
        print("[SyncClient] Sent 'ready', waiting for 'start'...")

        try:
            reply = msgpack.unpackb(self.req.recv(), raw=False)
            if reply.get("type") == "start":
                print(f"[SyncClient] Received 'start' signal at t_ns={reply.get('t_ns')}")
                return True
            else:
                print(f"[SyncClient] Unexpected reply: {reply}")
                return False
        except zmq.error.Again:
            print("[SyncClient] Timeout waiting for 'start'")
            return False

    def send_done(self) -> None:
        """Notify my_main.py that recording is complete."""
        self.req.setsockopt(zmq.RCVTIMEO, 5000)  # 5s is plenty for done ack
        msg = {"type": "done", "t_ns": time.time_ns()}
        self.req.send(msgpack.packb(msg, use_bin_type=True))
        try:
            reply = msgpack.unpackb(self.req.recv(), raw=False)
            print(f"[SyncClient] Recording done acknowledged: {reply.get('type')}")
        except zmq.error.Again:
            print("[SyncClient] Timeout on done acknowledgment")

    def close(self) -> None:
        """Close the socket."""
        if self.req:
            self.req.close()


def parse_raw_poses(poses: list) -> dict:
    """Parse a list of raw pose dicts (from ZMQ) into numpy arrays."""
    n = len(poses)
    t_ns = np.zeros(n, dtype=np.int64)
    ee_T = np.zeros((n, 4, 4), dtype=np.float64)
    joint_positions = np.zeros((n, 7), dtype=np.float64)
    gripper_q = np.zeros(n, dtype=np.float64)

    for i, pose in enumerate(poses):
        t_ns[i] = pose.get("t_ns", 0)

        if "ee_T" in pose:
            ee_T[i] = np.array(pose["ee_T"]).reshape(4, 4)
        else:
            ee_T[i] = np.eye(4)

        if "q" in pose:
            q = np.array(pose["q"])
            if len(q) >= 7:
                joint_positions[i] = q[:7]

        gripper_q[i] = pose.get("gripper_q", 0.0)

    return {
        "t_ns": t_ns,
        "ee_T": ee_T,
        "joint_positions": joint_positions,
        "gripper_q": gripper_q,
    }


def interpolate_poses_for_frames(
    frame_times_ns: np.ndarray,
    pose_times_ns: np.ndarray,
    pose_ee_T: np.ndarray,
    pose_joint_pos: np.ndarray,
    pose_gripper_q: np.ndarray,
) -> dict:
    """
    Interpolate robot poses to match each depth-frame timestamp.

    For each depth frame we find the two bracketing poses and linearly
    interpolate translation, joint positions, and gripper opening.
    Rotation matrices are blended linearly then re-orthogonalised via SVD
    (equivalent to SLERP for the small inter-pose angles typical of a
    robot control loop).

    Returns a dict of arrays aligned 1-to-1 with *frame_times_ns*:
        ee_T              (N, 4, 4) float64
        joint_positions   (N, 7)    float64
        gripper_q         (N,)      float64
        nearest_offset_ms (N,)      float64 – temporal gap to closest raw pose
    """
    N = len(frame_times_ns)
    interp_ee_T = np.zeros((N, 4, 4), dtype=np.float64)
    interp_joints = np.zeros((N, 7), dtype=np.float64)
    interp_gripper = np.zeros((N,), dtype=np.float64)
    nearest_offset_ms = np.zeros((N,), dtype=np.float64)

    # Sort poses by time (they should already be, but be safe)
    order = np.argsort(pose_times_ns)
    pose_times_ns = pose_times_ns[order]
    pose_ee_T = pose_ee_T[order]
    pose_joint_pos = pose_joint_pos[order]
    pose_gripper_q = pose_gripper_q[order]

    for i in range(N):
        t = frame_times_ns[i]
        idx = np.searchsorted(pose_times_ns, t)

        if idx == 0:
            # Frame is before first pose – use first pose
            interp_ee_T[i] = pose_ee_T[0]
            interp_joints[i] = pose_joint_pos[0]
            interp_gripper[i] = pose_gripper_q[0]
            nearest_offset_ms[i] = (t - pose_times_ns[0]) / 1e6
        elif idx >= len(pose_times_ns):
            # Frame is after last pose – use last pose
            interp_ee_T[i] = pose_ee_T[-1]
            interp_joints[i] = pose_joint_pos[-1]
            interp_gripper[i] = pose_gripper_q[-1]
            nearest_offset_ms[i] = (t - pose_times_ns[-1]) / 1e6
        else:
            t0, t1 = pose_times_ns[idx - 1], pose_times_ns[idx]
            alpha = float(t - t0) / float(t1 - t0) if t1 != t0 else 0.0

            T0, T1 = pose_ee_T[idx - 1], pose_ee_T[idx]

            # Translation: linear interpolation
            trans = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

            # Rotation: linear blend + SVD re-orthogonalisation
            R_blend = (1.0 - alpha) * T0[:3, :3] + alpha * T1[:3, :3]
            U, _, Vt = np.linalg.svd(R_blend)
            # Ensure proper rotation (det = +1)
            S = np.eye(3)
            S[2, 2] = np.linalg.det(U @ Vt)
            R_interp = U @ S @ Vt

            interp_ee_T[i, :3, :3] = R_interp
            interp_ee_T[i, :3, 3] = trans
            interp_ee_T[i, 3, 3] = 1.0

            # Joints & gripper: linear interpolation
            interp_joints[i] = (1.0 - alpha) * pose_joint_pos[idx - 1] + alpha * pose_joint_pos[idx]
            interp_gripper[i] = (1.0 - alpha) * pose_gripper_q[idx - 1] + alpha * pose_gripper_q[idx]

            nearest_offset_ms[i] = min(abs(t - t0), abs(t - t1)) / 1e6

    return {
        "ee_T": interp_ee_T,
        "joint_positions": interp_joints,
        "gripper_q": interp_gripper,
        "nearest_offset_ms": nearest_offset_ms,
    }


def get_object_name() -> str | None:
    """Prompt user for object name. Returns None if user wants to quit."""
    print("\n" + "=" * 50)
    print("Enter object name for this recording")
    print("(or 'q' to quit, 'l' to list existing recordings)")
    print("=" * 50)
    
    while True:
        name = input("Object name: ").strip()
        
        if name.lower() == 'q':
            return None
        
        if name.lower() == 'l':
            # List existing recordings
            if DATA_DIR.exists():
                recordings = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in ('videos', 'raw_event_data', 'hdf5')]
                if recordings:
                    print(f"Existing recordings: {', '.join(sorted(recordings))}")
                else:
                    print("No existing recordings found.")
            continue
        
        if not name:
            print("Please enter a valid name.")
            continue
        
        # Sanitize name (replace spaces with underscores, remove special chars)
        sanitized = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)
        
        # Check if recording already exists
        object_dir = DATA_DIR / sanitized
        if object_dir.exists():
            overwrite = input(f"Recording '{sanitized}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                continue
        
        return sanitized


def record_single_object(
    object_name: str,
    pipeline: rs.pipeline,
    pose_receiver: "ZMQPoseReceiver",
    sync_client: "ZMQSyncClient",
    num_event_cams: int = 1,
) -> bool:
    """
    Record a single object. Returns True if successful, False if should abort.
    """
    # Create object-specific directories
    object_dir = DATA_DIR / object_name
    object_video_dir = object_dir / "videos"
    object_raw_dir = object_dir / "raw_event_data"
    object_hdf5_dir = object_dir / "hdf5"
    
    object_video_dir.mkdir(parents=True, exist_ok=True)
    object_raw_dir.mkdir(parents=True, exist_ok=True)
    object_hdf5_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output files for this object
    event0_raw_file = object_raw_dir / "events_cam0.raw"
    event1_raw_file = object_raw_dir / "events_cam1.raw"
    rs_video_file = object_video_dir / "realsense_depth.mp4"
    rs_rgb_video_file = object_video_dir / "realsense_rgb.mp4"
    rs_h5_file = object_hdf5_dir / "realsense.h5"
    poses_h5_file = object_hdf5_dir / "poses.h5"
    raw_poses_h5_file = object_hdf5_dir / "raw_poses.h5"
    metadata_h5_file = object_hdf5_dir / "metadata.h5"
    
    print(f"\n[Recording] Starting recording for object: {object_name}")
    print(f"[Recording] Output directory: {object_dir}")
    
    # Initialize event cameras (in subprocess) if enabled
    event_proc = None
    stop_acquire_event = None
    event_done_event = None
    event_ready_event = None
    event_device_open_ns = 0  # system time when event sensor opened (clock origin)

    if num_event_cams > 0:
        stop_acquire_event = MPEvent()
        event_done_event = MPEvent()
        event_ready_event = MPEvent()
        start_logging_event = MPEvent()
        clock_sync_queue = Queue()

        event_proc = Process(
            target=event_drain_process,
            args=(stop_acquire_event, event_done_event, event_ready_event,
                  str(event0_raw_file), str(event1_raw_file), num_event_cams),
            kwargs={"clock_sync_queue": clock_sync_queue, "start_logging_event": start_logging_event},
        )
        event_proc.start()

        # Wait for event cameras to be ready
        print(f"[Recording] Waiting for {num_event_cams} event camera(s)...")
        event_ready_event.wait()
        try:
            event_device_open_ns = clock_sync_queue.get(timeout=5.0)
        except Exception:
            event_device_open_ns = 0
        print(f"[Recording] {num_event_cams} event camera(s) ready")
    else:
        print("[Recording] Event recording disabled for this run")
        start_logging_event = None
    
    
    # Reset and start pose receiver BEFORE the sync handshake so the ZMQ
    # subscription is fully established by the time my_main.py starts
    # publishing after "start" (avoids the ZMQ slow-joiner problem where
    # agent_complete is published before the SUB socket has connected).
    pose_receiver.stop_recording_event.clear()
    
    pose_receiver.start()

    # Send "ready" signal and wait for "start" (which arrives when the arm reaches its first pose)
    print("[Recording] Sending ready signal to robot controller...")
    if not sync_client.send_ready_wait_start(timeout_sec=120.0):
        print("[Recording] ERROR: Did not receive start signal, aborting")
        pose_receiver.stop()
        # Only attempt to stop event acquisition if it was started
        if stop_acquire_event is not None and event_done_event is not None and event_proc is not None:
            stop_acquire_event.set()
            event_done_event.wait()
            event_proc.join()
        return False

    # Signal event cameras to start logging (in sync with arm reaching first pose)
    event_logging_start_ns = 0
    if start_logging_event is not None:
        start_logging_event.set()
        event_logging_start_ns = time.time_ns()
        print("[Recording] Event cameras: started logging")
    
    # Initialize video writers and HDF5
    rs_video = cv2.VideoWriter(
        str(rs_video_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (RS_WIDTH, RS_HEIGHT),
        isColor=True
    )
    rs_rgb_video = cv2.VideoWriter(
        str(rs_rgb_video_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (RS_WIDTH, RS_HEIGHT),
        isColor=True
    )

    rs_h5 = h5py.File(rs_h5_file, "w")
    depth_ds = rs_h5.create_dataset(
        "depth",
        shape=(0, RS_HEIGHT, RS_WIDTH),
        maxshape=(None, RS_HEIGHT, RS_WIDTH),
        dtype=np.uint16,
        chunks=True
    )
    rgb_ds = rs_h5.create_dataset(
        "rgb",
        shape=(0, RS_HEIGHT, RS_WIDTH, 3),
        maxshape=(None, RS_HEIGHT, RS_WIDTH, 3),
        dtype=np.uint8,
        chunks=True
    )
    t_sys_ds = rs_h5.create_dataset(
        "t_sys_ns",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64
    )
    t_hw_ds = rs_h5.create_dataset(
        "t_hw_ms",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float64
    )
    frame_num_ds = rs_h5.create_dataset(
        "frame_number",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64
    )
    
    recording_start_ns = time.time_ns()
    rs_idx = 0
    accumulated_poses = []
    frame_times_ns_list = []
    rs_timestamp_domain = ""
    timestamp_domain_recorded = False
    
    print(f"[Recording] Recording '{object_name}' (waiting for agent_complete signal)...")
    
    try:
        while True:
            # Stop when robot signals completion
            if pose_receiver.should_stop_recording():
                print("[Recording] Received stop signal from robot - ending recording")
                break

            # Capture depth + color frame (blocking, 100 ms timeout)
            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                new_poses = pose_receiver.get_all_poses()
                accumulated_poses.extend(new_poses)
                continue

            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                new_poses = pose_receiver.get_all_poses()
                accumulated_poses.extend(new_poses)
                continue

            depth_img = np.asanyarray(depth.get_data())
            color_img = np.asanyarray(color.get_data())

            # Resize HDF5 datasets
            depth_ds.resize(rs_idx + 1, axis=0)
            rgb_ds.resize(rs_idx + 1, axis=0)
            t_sys_ds.resize(rs_idx + 1, axis=0)
            t_hw_ds.resize(rs_idx + 1, axis=0)
            frame_num_ds.resize(rs_idx + 1, axis=0)

            # Store data
            t_ns = time.time_ns()
            depth_ds[rs_idx] = depth_img
            rgb_ds[rs_idx] = color_img
            t_sys_ds[rs_idx] = t_ns
            t_hw_ds[rs_idx] = depth.get_timestamp()
            frame_num_ds[rs_idx] = depth.get_frame_number()
            frame_times_ns_list.append(t_ns)

            # Record RealSense timestamp domain once for diagnostics
            if not timestamp_domain_recorded:
                rs_timestamp_domain = str(depth.get_frame_timestamp_domain())
                timestamp_domain_recorded = True

            depth_colorized = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_TURBO
            )
            rs_video.write(depth_colorized)
            rs_rgb_video.write(color_img)
            rs_idx += 1
            
            # Collect poses
            accumulated_poses.extend(pose_receiver.get_all_poses())
    
    finally:
        # Stop event camera acquisition (only if we started it)
        if event_proc is not None and stop_acquire_event is not None and event_done_event is not None:
            stop_acquire_event.set()
            event_done_event.wait()
            event_proc.join()
        
        # Get remaining poses
        pose_receiver.stop()
        accumulated_poses.extend(pose_receiver.get_all_poses())
        
        recording_end_ns = time.time_ns()
        raw_poses_received = len(accumulated_poses)
        
        # Close realsense HDF5 and video writers
        rs_video.release()
        rs_rgb_video.release()
        rs_h5.close()
        
        # Interpolate raw poses to per-depth-frame timestamps → poses.h5
        frame_times_ns = np.array(frame_times_ns_list, dtype=np.int64)
        
        if raw_poses_received > 1 and rs_idx > 0:
            try:
                parsed = parse_raw_poses(accumulated_poses)
                # Save all raw poses into their own HDF5
                with h5py.File(raw_poses_h5_file, "w") as rpf:
                    rpf.create_dataset("t_ns", data=parsed["t_ns"])
                    rpf.create_dataset("ee_T", data=parsed["ee_T"])
                    rpf.create_dataset("joint_positions", data=parsed["joint_positions"])
                    rpf.create_dataset("gripper_q", data=parsed["gripper_q"])
                print(f"[Recording] Raw poses saved: {len(parsed['t_ns'])} poses → {raw_poses_h5_file}")
                interp = interpolate_poses_for_frames(
                    frame_times_ns=frame_times_ns,
                    pose_times_ns=parsed["t_ns"],
                    pose_ee_T=parsed["ee_T"],
                    pose_joint_pos=parsed["joint_positions"],
                    pose_gripper_q=parsed["gripper_q"],
                )
                with h5py.File(poses_h5_file, "w") as pf:
                    pf.create_dataset("ee_T", data=interp["ee_T"])
                    pf.create_dataset("joint_positions", data=interp["joint_positions"])
                    pf.create_dataset("gripper_q", data=interp["gripper_q"])
                    pf.create_dataset("nearest_offset_ms", data=interp["nearest_offset_ms"])
                median_off = np.median(interp["nearest_offset_ms"])
                max_off = np.max(interp["nearest_offset_ms"])
                print(
                    f"[Recording] Per-frame pose interpolation: "
                    f"median offset {median_off:.1f} ms, max {max_off:.1f} ms"
                )
            except Exception as e:
                print(f"[Recording] Warning: per-frame pose interpolation failed: {e}")
        else:
            print(f"[Recording] Warning: not enough poses ({raw_poses_received}) to interpolate")
        
        # Write metadata
        with h5py.File(metadata_h5_file, "w") as mf:
            mf.attrs["object_name"] = object_name
            mf.attrs["recording_start_ns"] = recording_start_ns
            mf.attrs["recording_end_ns"] = recording_end_ns
            mf.attrs["fps"] = FPS
            mf.attrs["event_device_open_ns"] = event_device_open_ns
            mf.attrs["event_logging_start_ns"] = event_logging_start_ns
            mf.attrs["rs_timestamp_domain"] = rs_timestamp_domain
            mf.attrs["depth_frames_recorded"] = rs_idx
            mf.attrs["raw_poses_received"] = raw_poses_received

        # Notify robot controller that recording is done
        sync_client.send_done()

    print(f"\n[Recording] Finished '{object_name}'!")
    print(f"  RealSense frames: {rs_idx}")
    print(f"  Raw poses received: {raw_poses_received}")
    print(f"  Output directory: {object_dir}")

    # Generate event videos from raw files, then align to depth frames
    if num_event_cams > 0:
        print("[Recording] Generating event camera video(s) from raw data...")
        generate_event_videos(object_raw_dir, object_video_dir, object_hdf5_dir, num_event_cams)
        print("[Recording] Aligning event frames to depth timestamps...")
        align_event_frames_to_depth(
            object_hdf5_dir, num_event_cams, frame_times_ns, event_logging_start_ns
        )

    return True


def generate_event_videos(
    object_raw_dir: Path,
    object_video_dir: Path,
    object_hdf5_dir: Path,
    num_cameras: int,
) -> None:
    """Generate event frame MP4 videos and HDF5 files from raw event recordings."""
    delta_t_us = int(1e6 / FPS)

    for cam_idx in range(num_cameras):
        raw_file = object_raw_dir / f"events_cam{cam_idx}.raw"
        video_file = object_video_dir / f"events_cam{cam_idx}.mp4"
        h5_file = object_hdf5_dir / f"events_cam{cam_idx}.h5"

        if not raw_file.exists():
            print(f"[EventVideo] Raw file not found: {raw_file}, skipping")
            continue

        print(f"[EventVideo] Processing cam{cam_idx}: {raw_file} ...")
        ev_it = EventsIterator(input_path=str(raw_file), delta_t=delta_t_us)
        height, width = ev_it.get_size()

        video = cv2.VideoWriter(
            str(video_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (width, height),
            isColor=False
        )

        h5 = h5py.File(h5_file, "w")
        grp = h5.create_group("events")
        frames_ds = grp.create_dataset(
            "frames",
            shape=(0, height, width),
            maxshape=(None, height, width),
            dtype=np.uint8,
            chunks=True
        )
        t_start_ds = grp.create_dataset(
            "t_ev_start_us", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        t_end_ds = grp.create_dataset(
            "t_ev_end_us", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        grp.attrs.update({"fps": FPS, "delta_t_us": delta_t_us, "width": width, "height": height})

        frame_gen = PeriodicFrameGenerationAlgorithm(width, height, delta_t_us)
        idx = 0

        def on_frame(ts, frame):
            nonlocal idx
            # Rotate 180 degrees so output videos/frames are upright
            # (rotation is the correct fix when the video is upside-down,
            # instead of mirroring/flipping vertically)
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            frames_ds.resize(idx + 1, axis=0)
            t_start_ds.resize(idx + 1, axis=0)
            t_end_ds.resize(idx + 1, axis=0)
            frames_ds[idx] = gray
            t_start_ds[idx] = ts - delta_t_us
            t_end_ds[idx] = ts
            video.write(gray)
            idx += 1

        frame_gen.set_output_callback(on_frame)
        for evs in ev_it:
            frame_gen.process_events(evs)

        video.release()
        h5.close()
        print(f"[EventVideo] cam{cam_idx}: {idx} frames written to {video_file}")


def align_event_frames_to_depth(
    hdf5_dir: Path,
    num_cameras: int,
    depth_times_ns: np.ndarray,
    event_logging_start_ns: int,
) -> None:
    """
    Align event camera HDF5 frames to depth frame timestamps.

    Both event and depth recordings start at approximately the same system
    time (``event_logging_start_ns``).  We align by *elapsed time* from that
    common origin so the result is independent of whether the raw-file
    timestamps are absolute camera-internal values or rebased to zero.

    The unaligned events_cam{i}.h5 (M frames) is replaced with an aligned
    version (N frames) where N = len(depth_times_ns), so that
    events[i] corresponds to depth[i] and poses[i].
    """
    N = len(depth_times_ns)
    if N == 0:
        return

    for cam_idx in range(num_cameras):
        h5_path = hdf5_dir / f"events_cam{cam_idx}.h5"
        if not h5_path.exists():
            print(f"[Align] events_cam{cam_idx}.h5 not found, skipping")
            continue

        # Read the unaligned event frames and timestamps
        with h5py.File(h5_path, "r") as f:
            ev_frames = f["events/frames"][:]        # (M, H, W)
            ev_t_end_us = f["events/t_ev_end_us"][:]  # (M,)
            ev_t_start_us = f["events/t_ev_start_us"][:]
            attrs = dict(f["events"].attrs)

        M = len(ev_t_end_us)
        if M == 0:
            print(f"[Align] cam{cam_idx}: no event frames, skipping")
            continue

        # Elapsed time (ns) from each recording's own start.
        # This is immune to raw-file timestamp rebasing and clock-sync
        # uncertainty between the event sub-process and the main process.
        ev_elapsed_ns = (ev_t_end_us - ev_t_end_us[0]).astype(np.int64) * 1000
        depth_elapsed_ns = (depth_times_ns - event_logging_start_ns).astype(np.int64)

        # For each depth frame, find the nearest event frame
        indices = np.searchsorted(ev_elapsed_ns, depth_elapsed_ns, side="left")
        indices = np.clip(indices, 0, M - 1)

        # Check if the left neighbour is actually closer
        left = np.clip(indices - 1, 0, M - 1)
        d_right = np.abs(ev_elapsed_ns[indices] - depth_elapsed_ns)
        d_left = np.abs(ev_elapsed_ns[left] - depth_elapsed_ns)
        use_left = d_left < d_right
        indices[use_left] = left[use_left]

        offset_ms = np.abs(ev_elapsed_ns[indices] - depth_elapsed_ns) / 1e6

        # Select the aligned frames
        aligned_frames = ev_frames[indices]          # (N, H, W)
        aligned_t_start = ev_t_start_us[indices]
        aligned_t_end = ev_t_end_us[indices]

        # Overwrite the events .h5 with the aligned version
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("events")
            grp.create_dataset("frames", data=aligned_frames)
            grp.create_dataset("t_ev_start_us", data=aligned_t_start)
            grp.create_dataset("t_ev_end_us", data=aligned_t_end)
            grp.create_dataset("alignment_offset_ms", data=offset_ms)
            for k, v in attrs.items():
                grp.attrs[k] = v

        print(
            f"[Align] cam{cam_idx}: {M} → {N} frames, "
            f"median offset {np.median(offset_ms):.1f} ms, "
            f"max {np.max(offset_ms):.1f} ms"
        )


def event_drain_process(
    stop_acquire_event,
    done_event,
    ready_event,
    event0_path: str,
    event1_path: str,
    num_cameras: int = 1,
    flush_seconds: float = 0.5,
    clock_sync_queue: Optional[Queue] = None,
    start_logging_event=None,
) -> None:
    """Process that handles event camera acquisition with custom output paths."""
    devices = DeviceDiscovery.list()
    device0 = DeviceDiscovery.open(devices[0])
    device_open_ns = time.time_ns()  # system time ≈ camera-clock origin
    device1 = DeviceDiscovery.open(devices[1]) if num_cameras >= 2 else None

    # Apply biases
    for device in filter(None, (device0, device1)):
        biases = device.get_i_ll_biases()
        biases.set("bias_diff_on", BIAS_DIFF_ON)
        biases.set("bias_diff_off", BIAS_DIFF_OFF)
        biases.set("bias_fo", BIAS_FO)
        biases.set("bias_hpf", BIAS_HPF)
        biases.set("bias_refr", BIAS_REFR)

    raw0 = device0.get_i_events_stream()
    raw0.start()

    if device1 is not None:
        raw1 = device1.get_i_events_stream()
        raw1.start()
    else:
        raw1 = None

    it0 = iter(EventsIterator.from_device(device0))
    it1 = iter(EventsIterator.from_device(device1)) if device1 is not None else None

    ready_event.set()

    # Report device-open system timestamp so the main process can map
    # event-camera microseconds → system nanoseconds
    if clock_sync_queue is not None:
        clock_sync_queue.put(device_open_ns)

    # Drain events (without logging) until start_logging_event is set,
    # so that the event recording starts in sync with the RealSense recording.
    if start_logging_event is not None:
        while not start_logging_event.is_set() and not stop_acquire_event.is_set():
            next(it0)
            if it1 is not None:
                next(it1)

    logging_active = False
    try:
        if not stop_acquire_event.is_set():
            # Start file logging now that the arm has reached the first pose
            raw0.log_raw_data(event0_path)
            if raw1 is not None:
                raw1.log_raw_data(event1_path)
            logging_active = True

            while not stop_acquire_event.is_set():
                next(it0)
                if it1 is not None:
                    next(it1)

            # Flush remaining events
            flush_start = time.time()
            while time.time() - flush_start < flush_seconds:
                next(it0)
                if it1 is not None:
                    next(it1)

    finally:
        if logging_active:
            raw0.stop_log_raw_data()
            if raw1 is not None:
                raw1.stop_log_raw_data()
        done_event.set()


def main(
    zmq_sync_addr: str = ZMQ_SYNC_ADDR,
    zmq_pose_addr: str = ZMQ_POSE_ADDR,
    num_event_cams: int = 1,
    temporal_check: bool = False,
) -> None:
    """
    Main synchronized recording function with multi-object support.
    
    Loop:
    1. Ask for object name
    2. Initialize cameras, send "ready", wait for "start"
    3. Record until hemisphere complete
    4. Send "done", ask for next object name
    5. Repeat until user quits
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize ZMQ sync client
    sync_client = ZMQSyncClient(zmq_sync_addr)
    sync_client.connect()
    
    # Initialize pose receiver
    pose_receiver = ZMQPoseReceiver(zmq_pose_addr)
    
    # Initialize RealSense (keep running across recordings)
    pipeline = None
    cfg = None
    profile = None
    depth_sensor = None

    print("[Recording] Initializing RealSense...")
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, FPS)

    try:
        profile = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.laser_power, 360)

        print("[Recording] Warming up RealSense...")
        for _ in range(30):
            pipeline.wait_for_frames(timeout_ms=1000)

    except Exception as e:
        print(f"[Recording] RealSense startup failed: {e}")

        try:
            pipeline.stop()
        except Exception:
            pass

        # Hardware reset
        ctx = rs.context()
        for dev in ctx.query_devices():
            print(f"[Recording] Resetting device: {dev.get_info(rs.camera_info.name)}")
            dev.hardware_reset()

        time.sleep(3.0)

        # Retry once
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.disable_all_streams()
        cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, FPS)

        profile = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.laser_power, 360)

        print("[Recording] Warming up RealSense after reset...")
        for _ in range(30):
            pipeline.wait_for_frames(timeout_ms=1000)

    print("[Recording] RealSense initialized")
    
    recording_count = 0
    
    try:
        while True:
            # In temporal-check mode, use a fixed name and record once
            if temporal_check:
                object_name = "temporal_check"
                object_dir = DATA_DIR / object_name
                if object_dir.exists():
                    import shutil
                    shutil.rmtree(object_dir)
                    print(f"[Recording] Removed previous temporal_check recording")
            else:
                # Ask for object name
                object_name = get_object_name()
            
            if object_name is None:
                print("\n[Recording] User requested quit. Exiting...")
                break
            
            # Record this object
            success = record_single_object(
                object_name=object_name,
                pipeline=pipeline,
                pose_receiver=pose_receiver,
                sync_client=sync_client,
                num_event_cams=num_event_cams,
            )
            
            if success:
                recording_count += 1
                print(f"\n[Recording] Completed {recording_count} recording(s) so far.")
                if temporal_check:
                    print("\n[Recording] Running temporal alignment analysis...")
                    import sys, importlib.util
                    _script = Path(__file__).parent / "viz_and_tests" / "analyze_temporal_alignment.py"
                    _spec = importlib.util.spec_from_file_location("analyze_temporal_alignment", _script)
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    _mod.analyze_temporal_alignment(DATA_DIR / object_name)
                    break
            else:
                print("\n[Recording] Recording failed or was aborted.")
                break
    
    finally:
        print("[Recording] Shutting down RealSense...")
        try:
            pipeline.stop()
        except Exception as e:
            print(f"[Recording] Warning during pipeline.stop(): {e}")

        # Explicitly release SDK objects
        try:
            del depth_sensor
        except Exception:
            pass
        try:
            del profile
        except Exception:
            pass
        try:
            del cfg
        except Exception:
            pass
        try:
            del pipeline
        except Exception:
            pass

        # Give librealsense / USB stack a moment to settle
        gc.collect()
        time.sleep(1.0)

        sync_client.close()
        print(f"\n[Recording] Session complete. Total recordings: {recording_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synchronized multi-camera recording")
    parser.add_argument(
        "--zmq-sync-addr",
        default=ZMQ_SYNC_ADDR,
        help=f"ZMQ sync address (default: {ZMQ_SYNC_ADDR})"
    )
    parser.add_argument(
        "--zmq-pose-addr",
        default=ZMQ_POSE_ADDR,
        help=f"ZMQ pose subscription address (default: {ZMQ_POSE_ADDR})"
    )
    parser.add_argument(
        "--event-cameras",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Number of event cameras to use: 0 (disabled), 1 (default), or 2"
    )
    parser.add_argument(
        "--temporal-check",
        action="store_true",
        help="Run temporal alignment check after recording. "
             "Use with --agent-type temporal_check on the robot side.",
    )

    args = parser.parse_args()

    main(
        zmq_sync_addr=args.zmq_sync_addr,
        zmq_pose_addr=args.zmq_pose_addr,
        num_event_cams=args.event_cameras,
        temporal_check=args.temporal_check,
    )
