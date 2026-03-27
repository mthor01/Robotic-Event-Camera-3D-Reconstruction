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
TARGET_SECONDS = 20
FPS = 30

DATA_DIR = Path("data")
VIDEO_DIR = DATA_DIR / "videos"
RAW_DIR = DATA_DIR / "raw_event_data"
HDF5_DIR = DATA_DIR / "hdf5"

EVENT0_RAW_FILE = RAW_DIR / "events_cam0.raw"
EVENT1_RAW_FILE = RAW_DIR / "events_cam1.raw"

RS_VIDEO_FILE = VIDEO_DIR / "realsense.mp4"
RS_H5_FILE = HDF5_DIR / "synchronized_recording.h5"

RS_WIDTH, RS_HEIGHT = 640, 480

BIAS_DIFF_ON = 10
BIAS_DIFF_OFF = 80
BIAS_FO = 0
BIAS_HPF = 50
BIAS_REFR = 150

# ZMQ addresses
ZMQ_SYNC_ADDR = "tcp://localhost:6001"  # REQ/REP for sync handshake
ZMQ_POSE_ADDR = "tcp://localhost:6000"  # PUB/SUB for pose streaming (from my_main)
# =========================================


def event_drain_process(
    stop_acquire_event: MPEvent,
    done_event: MPEvent,
    ready_event: MPEvent,
    flush_seconds: float = 0.5
) -> None:
    """Process that handles event camera acquisition."""
    devices = DeviceDiscovery.list()
    device0 = DeviceDiscovery.open(devices[0])
    device1 = DeviceDiscovery.open(devices[1])

    # Apply biases
    for device in (device0, device1):
        biases = device.get_i_ll_biases()
        biases.set("bias_diff_on", BIAS_DIFF_ON)
        biases.set("bias_diff_off", BIAS_DIFF_OFF)
        biases.set("bias_fo", BIAS_FO)
        biases.set("bias_hpf", BIAS_HPF)
        biases.set("bias_refr", BIAS_REFR)

    raw0 = device0.get_i_events_stream()
    raw1 = device1.get_i_events_stream()

    raw0.start()
    raw1.start()

    raw0.log_raw_data(str(EVENT0_RAW_FILE))
    raw1.log_raw_data(str(EVENT1_RAW_FILE))

    it0 = iter(EventsIterator.from_device(device0))
    it1 = iter(EventsIterator.from_device(device1))

    ready_event.set()

    try:
        while not stop_acquire_event.is_set():
            next(it0)
            next(it1)

        # Flush remaining events
        flush_start = time.time()
        while time.time() - flush_start < flush_seconds:
            next(it0)
            next(it1)

    finally:
        raw0.stop_log_raw_data()
        raw1.stop_log_raw_data()
        done_event.set()


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
        self.events: deque = deque(maxlen=1000)
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
                msg = msgpack.unpackb(payload, raw=False)

                with self.lock:
                    if topic == b"event":
                        self.events.append(msg)
                        # Check for stop signals — include events published by my_main
                        event_type = msg.get("type", "")
                        print(event_type)
                        if event_type in ("agent_complete"):
                            print(f"[PoseReceiver] Received '{event_type}' event - signaling stop")
                            self.stop_recording_event.set()
                    elif topic == b"pose":
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

    def get_all_events(self) -> list:
        """Get all received events and clear the buffer."""
        with self.lock:
            events = list(self.events)
            self.events.clear()
            return events


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
        self.req.setsockopt(zmq.RCVTIMEO, 60000)  # 60s timeout for waiting
        self.req.connect(self.connect_addr)
        print(f"[SyncClient] Connected to {self.connect_addr}")

    def send_ready_wait_start(self, timeout_sec: float = 60.0) -> bool:
        """
        Send 'ready' signal and wait for 'start' response.
        Returns True if start signal received, False on timeout.
        """
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


def save_poses_to_hdf5(h5_file: h5py.File, poses: list) -> None:
    """Save accumulated pose data to HDF5."""
    if not poses:
        return

    # Create poses group if it doesn't exist
    if "poses" not in h5_file:
        poses_grp = h5_file.create_group("poses")

        # Create datasets
        poses_grp.create_dataset(
            "t_ns", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        poses_grp.create_dataset(
            "episode", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        poses_grp.create_dataset(
            "step", shape=(0,), maxshape=(None,), dtype=np.int32
        )
        poses_grp.create_dataset(
            "ee_T", shape=(0, 4, 4), maxshape=(None, 4, 4), dtype=np.float64
        )
        poses_grp.create_dataset(
            "joint_positions", shape=(0, 7), maxshape=(None, 7), dtype=np.float64
        )
        poses_grp.create_dataset(
            "gripper_q", shape=(0,), maxshape=(None,), dtype=np.float64
        )
    else:
        poses_grp = h5_file["poses"]

    # Append poses
    current_len = poses_grp["t_ns"].shape[0]
    new_len = current_len + len(poses)

    # Resize all datasets
    for key in ["t_ns", "episode", "step", "ee_T", "joint_positions", "gripper_q"]:
        poses_grp[key].resize(new_len, axis=0)

    for i, pose in enumerate(poses):
        idx = current_len + i
        poses_grp["t_ns"][idx] = pose.get("t_ns", 0)
        poses_grp["episode"][idx] = pose.get("ep", 0)
        poses_grp["step"][idx] = pose.get("step", 0)

        if "ee_T" in pose:
            ee_T = np.array(pose["ee_T"]).reshape(4, 4)
            poses_grp["ee_T"][idx] = ee_T
        else:
            poses_grp["ee_T"][idx] = np.eye(4)

        if "q" in pose:
            q = np.array(pose["q"])
            if len(q) >= 7:
                poses_grp["joint_positions"][idx] = q[:7]
            else:
                poses_grp["joint_positions"][idx] = np.zeros(7)
        else:
            poses_grp["joint_positions"][idx] = np.zeros(7)

        poses_grp["gripper_q"][idx] = pose.get("gripper_q", 0.0)


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
    target_seconds: float,
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
    rs_h5_file = object_hdf5_dir / "synchronized_recording.h5"
    
    print(f"\n[Recording] Starting recording for object: {object_name}")
    print(f"[Recording] Output directory: {object_dir}")
    
    # Initialize event cameras (in subprocess) if enabled
    event_proc = None
    stop_acquire_event = None
    event_done_event = None
    event_ready_event = None

    if num_event_cams > 0:
        stop_acquire_event = MPEvent()
        event_done_event = MPEvent()
        event_ready_event = MPEvent()

        event_proc = Process(
            target=event_drain_process_with_paths,
            args=(stop_acquire_event, event_done_event, event_ready_event,
                  str(event0_raw_file), str(event1_raw_file), num_event_cams)
        )
        event_proc.start()

        # Wait for event cameras to be ready
        print(f"[Recording] Waiting for {num_event_cams} event camera(s)...")
        event_ready_event.wait()
        print(f"[Recording] {num_event_cams} event camera(s) ready")
    else:
        print("[Recording] Event recording disabled for this run")
    
    
    # Reset and start pose receiver BEFORE the sync handshake so the ZMQ
    # subscription is fully established by the time my_main.py starts
    # publishing after "start" (avoids the ZMQ slow-joiner problem where
    # agent_complete is published before the SUB socket has connected).
    pose_receiver.stop_recording_event.clear()
    
    pose_receiver.start()

    # Send "ready" signal and wait for "start"
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
    
    # Initialize video writers and HDF5
    rs_video = cv2.VideoWriter(
        str(rs_video_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (RS_WIDTH, RS_HEIGHT),
        isColor=False
    )
    rs_rgb_video = cv2.VideoWriter(
        str(rs_rgb_video_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (RS_WIDTH, RS_HEIGHT),
        isColor=True
    )

    rs_h5 = h5py.File(rs_h5_file, "w")

    # Create realsense group
    rs_grp = rs_h5.create_group("realsense")
    depth_ds = rs_grp.create_dataset(
        "depth",
        shape=(0, RS_HEIGHT, RS_WIDTH),
        maxshape=(None, RS_HEIGHT, RS_WIDTH),
        dtype=np.uint16,
        chunks=True
    )
    rgb_ds = rs_grp.create_dataset(
        "rgb",
        shape=(0, RS_HEIGHT, RS_WIDTH, 3),
        maxshape=(None, RS_HEIGHT, RS_WIDTH, 3),
        dtype=np.uint8,
        chunks=True
    )
    t_sys_ds = rs_grp.create_dataset(
        "t_sys_ns",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64
    )
    
    # Recording metadata
    meta_grp = rs_h5.create_group("metadata")
    meta_grp.attrs["object_name"] = object_name
    meta_grp.attrs["recording_start_ns"] = time.time_ns()
    meta_grp.attrs["target_seconds"] = target_seconds
    meta_grp.attrs["fps"] = FPS
    
    rs_idx = 0
    accumulated_poses = []
    
    print(f"[Recording] Recording '{object_name}' (max {target_seconds}s, or until hemisphere complete)...")
    start_time = time.time()
    stop_reason = "timeout"
    
    try:
        while time.time() - start_time < target_seconds:
            # Check if robot signaled stop (hemisphere complete or quit)
            if pose_receiver.should_stop_recording():
                stop_reason = "hemisphere_complete"
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

            # Store data
            depth_ds[rs_idx] = depth_img
            rgb_ds[rs_idx] = color_img
            t_sys_ds[rs_idx] = time.time_ns()

            rs_video.write(cv2.convertScaleAbs(depth_img, alpha=0.03))
            rs_rgb_video.write(color_img)
            rs_idx += 1
            
            # Collect poses periodically
            new_poses = pose_receiver.get_all_poses()
            accumulated_poses.extend(new_poses)
            
            # Save poses to HDF5 in batches
            if len(accumulated_poses) >= 100:
                save_poses_to_hdf5(rs_h5, accumulated_poses)
                accumulated_poses.clear()
    
    finally:
        # Stop event camera acquisition (only if we started it)
        if event_proc is not None and stop_acquire_event is not None and event_done_event is not None:
            stop_acquire_event.set()
            event_done_event.wait()
            event_proc.join()
        
        # Get remaining poses
        pose_receiver.stop()
        remaining_poses = pose_receiver.get_all_poses()
        accumulated_poses.extend(remaining_poses)
        
        # Save final poses
        save_poses_to_hdf5(rs_h5, accumulated_poses)
        
        # Update metadata
        meta_grp.attrs["recording_end_ns"] = time.time_ns()
        meta_grp.attrs["depth_frames_recorded"] = rs_idx
        meta_grp.attrs["stop_reason"] = stop_reason
        if "poses" in rs_h5:
            poses_recorded = rs_h5["poses"]["t_ns"].shape[0]
            meta_grp.attrs["poses_recorded"] = poses_recorded
        else:
            poses_recorded = 0
            meta_grp.attrs["poses_recorded"] = 0
        
        # Cleanup
        rs_video.release()
        rs_rgb_video.release()
        rs_h5.close()

        # Notify robot controller that recording is done
        sync_client.send_done()

    print(f"\n[Recording] Finished '{object_name}'! (stop reason: {stop_reason})")
    print(f"  RealSense frames: {rs_idx}")
    print(f"  Robot poses: {poses_recorded}")
    print(f"  Output directory: {object_dir}")

    # Generate event videos from raw files
    if num_event_cams > 0:
        print("[Recording] Generating event camera video(s) from raw data...")
        generate_event_videos(object_raw_dir, object_video_dir, object_hdf5_dir, num_event_cams)

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


def event_drain_process_with_paths(
    stop_acquire_event,
    done_event,
    ready_event,
    event0_path: str,
    event1_path: str,
    num_cameras: int = 1,
    flush_seconds: float = 0.5
) -> None:
    """Process that handles event camera acquisition with custom output paths."""
    devices = DeviceDiscovery.list()
    device0 = DeviceDiscovery.open(devices[0])
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
    raw0.log_raw_data(event0_path)

    if device1 is not None:
        raw1 = device1.get_i_events_stream()
        raw1.start()
        raw1.log_raw_data(event1_path)
    else:
        raw1 = None

    it0 = iter(EventsIterator.from_device(device0))
    it1 = iter(EventsIterator.from_device(device1)) if device1 is not None else None

    ready_event.set()

    try:
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
        raw0.stop_log_raw_data()
        if raw1 is not None:
            raw1.stop_log_raw_data()
        done_event.set()


def main(
    zmq_sync_addr: str = ZMQ_SYNC_ADDR,
    zmq_pose_addr: str = ZMQ_POSE_ADDR,
    target_seconds: float = TARGET_SECONDS,
    num_event_cams: int = 1,
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
            # Ask for object name
            object_name = get_object_name()
            
            if object_name is None:
                print("\n[Recording] User requested quit. Exiting...")
                # Send a quit signal to my_main if it's waiting
                break
            
            # Record this object
            success = record_single_object(
                object_name=object_name,
                pipeline=pipeline,
                pose_receiver=pose_receiver,
                sync_client=sync_client,
                target_seconds=target_seconds,
                num_event_cams=num_event_cams,
            )
            
            if success:
                recording_count += 1
                print(f"\n[Recording] Completed {recording_count} recording(s) so far.")
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
        import gc
        gc.collect()
        time.sleep(1.0)

        sync_client.close()
        print(f"\n[Recording] Session complete. Total recordings: {recording_count}")


if __name__ == "__main__":
    import argparse

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
        "--duration",
        type=float,
        default=TARGET_SECONDS,
        help=f"Recording duration in seconds (default: {TARGET_SECONDS})"
    )
    parser.add_argument(
        "--event-cameras",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Number of event cameras to use: 0 (disabled), 1 (default), or 2"
    )

    args = parser.parse_args()

    main(
        zmq_sync_addr=args.zmq_sync_addr,
        zmq_pose_addr=args.zmq_pose_addr,
        target_seconds=args.duration,
        num_event_cams=args.event_cameras,
    )
