"""
Synchronized multi-camera recording with ZeroMQ communication for robot arm coordination.

Per-recording workflow:
  1. Start event-camera drain subprocess and initialize RealSense (both kept alive
     across multiple recordings in the same session).
  2. Send a "ready" signal to my_main.py via ZMQ REQ/REP and block until "start"
     arrives (my_main.py sends it when the arm reaches its first pose).
  3. Open event-camera file logging in sync with the "start" signal.
  4. Capture RealSense depth + colour frames; collect robot poses from ZMQ PUB/SUB.
  5. Stop when the agent publishes an "agent_complete" event.
  6. Align event frames to depth timestamps, interpolate poses per frame, write HDF5.
  7. Send "done" to my_main.py, then loop for the next object.

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
import zmq
import msgpack
from pathlib import Path
from multiprocessing import Process, Event as MPEvent, Queue
from collections import deque
from typing import Optional
from metavision_hal import DeviceDiscovery
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm

from config import (
    FPS, RS_WIDTH, RS_HEIGHT,
    BIAS_DIFF_ON, BIAS_DIFF_OFF, BIAS_FO, BIAS_HPF, BIAS_REFR,
    ZMQ_SYNC_ADDR, ZMQ_POSE_ADDR, DATA_ROOT, TEMPORAL_CHECK_ROOT, LIGHT_CHECK_ROOT,
    POSE_TIME_OFFSET_MS,
    DEPTH_EVENT_ALIGN_OFFSET_FRAMES,
    DEPTH_VIZ_MIN, DEPTH_VIZ_MAX,
)

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

    def measure_transport_delay(self, n_rounds: int = 20) -> int:
        """
        Measure one-way ZMQ transport delay via ping-pong RTT.

        Sends ``n_rounds`` ping messages to the sync server (my_main.py) and
        measures the round-trip time for each.  The server must be ready to
        receive (i.e. blocked in ``wait_for_ready``), which handles pings
        transparently before the "ready" handshake.

        Returns the median one-way delay in nanoseconds (RTT / 2).
        This can be subtracted from ``t_recv_ns`` pose timestamps to get a
        better estimate of when the robot was actually at each pose.
        """
        self.req.setsockopt(zmq.RCVTIMEO, 2000)
        rtts_ns = []
        for _ in range(n_rounds):
            t1 = time.time_ns()
            msg = {"type": "ping", "t1": t1}
            self.req.send(msgpack.packb(msg, use_bin_type=True))
            try:
                reply = msgpack.unpackb(self.req.recv(), raw=False)
                t4 = time.time_ns()
                if reply.get("type") == "pong":
                    rtts_ns.append(t4 - t1)
            except zmq.error.Again:
                pass  # skip failed rounds

        if not rtts_ns:
            print("[ClockSync] WARNING: all ping rounds failed, transport delay set to 0")
            return 0

        median_rtt_ns = int(np.median(rtts_ns))
        one_way_ns = median_rtt_ns // 2
        print(
            f"[ClockSync] Transport delay: {one_way_ns / 1e6:.2f} ms "
            f"(median RTT {median_rtt_ns / 1e6:.2f} ms over {len(rtts_ns)}/{n_rounds} rounds)"
        )
        return one_way_ns

    def close(self) -> None:
        """Close the socket."""
        if self.req:
            self.req.close()


def parse_raw_poses(poses: list) -> dict:
    """Parse a list of raw pose dicts (from ZMQ) into numpy arrays."""
    n = len(poses)
    t_ns = np.zeros(n, dtype=np.int64)
    t_recv_ns = np.zeros(n, dtype=np.int64)
    ee_T = np.zeros((n, 4, 4), dtype=np.float64)
    joint_velocity = np.zeros((n, 7), dtype=np.float64)

    for i, pose in enumerate(poses):
        t_ns[i] = pose.get("t_ns", 0)
        t_recv_ns[i] = pose.get("t_recv_ns", 0)

        if "ee_T" in pose:
            ee_T[i] = np.array(pose["ee_T"]).reshape(4, 4)
        else:
            ee_T[i] = np.eye(4)

        if "jv" in pose:
            jv = np.array(pose["jv"], dtype=np.float64).reshape(-1)
            joint_velocity[i, : len(jv)] = jv

    return {
        "t_ns": t_ns,
        "t_recv_ns": t_recv_ns,
        "ee_T": ee_T,
        "joint_velocity": joint_velocity,
    }


def assign_poses_to_frames(
    frame_times_ns: np.ndarray,
    pose_times_ns: np.ndarray,
    pose_ee_T: np.ndarray,
) -> dict:
    """
    Assign the nearest robot pose to each depth-frame timestamp.

    At 100 Hz pose rate and 30 fps depth frames the nearest pose is always
    within ~5 ms.

    Returns a dict of arrays aligned 1-to-1 with *frame_times_ns*:
        ee_T              (N, 4, 4) float64
        nearest_offset_ms (N,)      float64 – signed temporal gap to chosen pose
    """
    # Sort poses by time (they should already be, but be safe)
    order = np.argsort(pose_times_ns)
    pose_times_ns = pose_times_ns[order]
    pose_ee_T = pose_ee_T[order]

    # For each frame, find the insertion point and compare left/right neighbour
    indices = np.searchsorted(pose_times_ns, frame_times_ns, side="left")
    indices = np.clip(indices, 0, len(pose_times_ns) - 1)
    left = np.clip(indices - 1, 0, len(pose_times_ns) - 1)
    use_left = np.abs(pose_times_ns[left] - frame_times_ns) < np.abs(pose_times_ns[indices] - frame_times_ns)
    indices[use_left] = left[use_left]

    nearest_offset_ms = (pose_times_ns[indices] - frame_times_ns).astype(np.float64) / 1e6

    return {
        "ee_T": pose_ee_T[indices],
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
            if DATA_ROOT.exists():
                recordings = [d.name for d in DATA_ROOT.iterdir() if d.is_dir() and d.name not in ('videos', 'raw_event_data', 'hdf5')]
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
        object_dir = DATA_ROOT / sanitized
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
    data_root: Path = DATA_ROOT,
    transport_delay_ns: int = 0,
    hw_trigger_sync: bool = False,
    debug: bool = False,
    ) -> bool:
    """
    Record a single object. Returns True if successful, False if should abort.

    data_root: root directory under which <object_name>/ is created.
    transport_delay_ns: measured one-way ZMQ delay (ns). Subtracted from
        t_recv_ns pose timestamps to compensate for message-passing latency.
    hw_trigger_sync: when True, the RealSense GPIO trigger output is used to
        record precise per-frame timestamps on the event camera for alignment.
    """
    # Create object-specific sub-directories for this recording
    object_dir = data_root / object_name
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
    logging_start_queue = None
    hw_trigger_times_us = np.array([], dtype=np.int64)

    if num_event_cams > 0:
        stop_acquire_event = MPEvent()
        event_done_event = MPEvent()
        event_ready_event = MPEvent()
        start_logging_event = MPEvent()
        clock_sync_queue = Queue()
        logging_start_queue = Queue()

        event_proc = Process(
            target=event_drain_process,
            args=(stop_acquire_event, event_done_event, event_ready_event,
                  str(event0_raw_file), str(event1_raw_file), num_event_cams),
            kwargs={"clock_sync_queue": clock_sync_queue, "start_logging_event": start_logging_event,
                    "logging_start_queue": logging_start_queue,
                    "hw_trigger_sync": hw_trigger_sync},
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
    t_global_ds = rs_h5.create_dataset(
        "t_global_ms",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float64
    )
    t_rgb_ds = rs_h5.create_dataset(
        "t_rgb_ms",
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
     
    rs_idx = 0
    accumulated_poses = []
    frame_hw_ms_list = []
    rs_timestamp_domain = ""
    timestamp_domain_recorded = False
    
    print(f"[Recording] Recording '{object_name}' (waiting for agent_complete signal)...")

    recording_start_ns = time.time_ns()
    
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
            t_global_ds.resize(rs_idx + 1, axis=0)
            t_rgb_ds.resize(rs_idx + 1, axis=0)
            frame_num_ds.resize(rs_idx + 1, axis=0)

            # Store data — t_global_ms is the RealSense global timestamp, which
            # librealsense maps to the host PC system clock via RS2_OPTION_GLOBAL_TIME_ENABLED.
            # t_rgb_ms is the color sensor's own global timestamp from the same frameset;
            # it may differ from t_global_ms by a few ms due to inter-sensor timing.
            t_global_ms = depth.get_timestamp()
            t_rgb_ms = color.get_timestamp()
            depth_ds[rs_idx] = depth_img
            rgb_ds[rs_idx] = color_img
            t_global_ds[rs_idx] = t_global_ms
            t_rgb_ds[rs_idx] = t_rgb_ms
            frame_num_ds[rs_idx] = depth.get_frame_number()
            frame_hw_ms_list.append(t_global_ms)

            # Record RealSense timestamp domain once for diagnostics
            if not timestamp_domain_recorded:
                rs_timestamp_domain = str(depth.get_frame_timestamp_domain())
                timestamp_domain_recorded = True

            _d_min_mm = DEPTH_VIZ_MIN * 1000.0
            _d_max_mm = DEPTH_VIZ_MAX * 1000.0
            _d_norm = np.clip(
                (depth_img.astype(np.float32) - _d_min_mm) / (_d_max_mm - _d_min_mm),
                0.0, 1.0,
            )
            _d_u8 = (_d_norm * 255).astype(np.uint8)
            _d_u8[depth_img == 0] = 0
            depth_colorized = cv2.applyColorMap(_d_u8, cv2.COLORMAP_TURBO)
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

        # Refine event_logging_start_ns using the exact time captured in the subprocess
        # right before raw0.log_raw_data() — eliminates inter-process scheduling jitter.
        if logging_start_queue is not None:
            try:
                event_logging_start_ns = logging_start_queue.get_nowait()
            except Exception:
                pass  # keep the fallback value captured in the main process

        # Collect hardware trigger timestamps from the raw event file.
        # The triggers are stored in the .raw file as EventExtTrigger records
        # (the same mechanism verified by test_rs_trigger.py); reading them
        # back after recording is more reliable than a real-time callback.
        if hw_trigger_sync and event0_raw_file.exists():
            try:
                from metavision_core.event_io import RawReader
                _rr = RawReader(str(event0_raw_file))
                while not _rr.is_done():
                    _rr.load_delta_t(100_000)
                _trig = _rr.get_ext_trigger_events()
                if _trig is not None and len(_trig) > 0:
                    # Keep only rising edges (p == 1) — one per depth frame
                    _rising = _trig[_trig["p"] == 1]
                    hw_trigger_times_us = _rising["t"].astype(np.int64)
                    print(f"[HWSync] Read {len(hw_trigger_times_us)} rising-edge trigger timestamps from raw file")
                else:
                    print("[HWSync] WARNING: No trigger events found in raw file")
            except Exception as _e:
                print(f"[HWSync] WARNING: could not read trigger timestamps from raw file: {_e}")

        # Get remaining poses
        pose_receiver.stop()
        accumulated_poses.extend(pose_receiver.get_all_poses())
        
        recording_end_ns = time.time_ns()
        raw_poses_received = len(accumulated_poses)
        
        # Close realsense HDF5 and video writers
        rs_video.release()
        rs_rgb_video.release()
        rs_h5.close()
        
        # t_global_ms is already system-clock-referenced (RS2_OPTION_GLOBAL_TIME_ENABLED).
        # Convert ms → ns for downstream alignment and pose assignment.
        frame_times_ns = (np.array(frame_hw_ms_list) * 1e6).astype(np.int64)

        if raw_poses_received > 1 and rs_idx > 0:
            try:
                parsed = parse_raw_poses(accumulated_poses)
                # Save all raw poses into their own HDF5
                with h5py.File(raw_poses_h5_file, "w") as rpf:
                    rpf.create_dataset("t_ns", data=parsed["t_ns"])
                    rpf.create_dataset("t_recv_ns", data=parsed["t_recv_ns"])
                    rpf.create_dataset("ee_T", data=parsed["ee_T"])
                    rpf.create_dataset("joint_velocity", data=parsed["joint_velocity"])
                print(f"[Recording] Raw poses saved: {len(parsed['t_ns'])} poses → {raw_poses_h5_file}")
                # Subtract the measured ZMQ one-way transport delay so that
                # the corrected timestamp approximates when the robot was
                # actually at each pose, not when the message arrived.
                # Also apply any manual offset from reconstruction_config.
                manual_offset_ns = int(POSE_TIME_OFFSET_MS * 1e6)
                corrected_pose_times_ns = parsed["t_recv_ns"] - transport_delay_ns + manual_offset_ns
                if manual_offset_ns != 0:
                    print(f"[Recording] Manual pose time offset applied: {POSE_TIME_OFFSET_MS:+.1f} ms")
                interp = assign_poses_to_frames(
                    frame_times_ns=frame_times_ns,
                    pose_times_ns=corrected_pose_times_ns,
                    pose_ee_T=parsed["ee_T"],
                )
                with h5py.File(poses_h5_file, "w") as pf:
                    pf.create_dataset("ee_T", data=interp["ee_T"])
                    pf.create_dataset("nearest_offset_ms", data=interp["nearest_offset_ms"])
                median_off = np.median(interp["nearest_offset_ms"])
                max_off = np.max(interp["nearest_offset_ms"])
                print(
                    f"[Recording] Per-frame pose assignment: "
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
            mf.attrs["transport_delay_ns"] = transport_delay_ns
            mf.attrs["pose_time_offset_ms"] = float(POSE_TIME_OFFSET_MS)
            mf.attrs["hw_trigger_sync"] = hw_trigger_sync
            mf.attrs["hw_trigger_count"] = len(hw_trigger_times_us)

        # Notify robot controller that recording is done
        sync_client.send_done()

    print(f"\n[Recording] Finished '{object_name}'!")
    print(f"  RealSense frames: {rs_idx}")
    print(f"  Raw poses received: {raw_poses_received}")
    print(f"  Output directory: {object_dir}")

    # Generate event frames and align to depth frames
    if num_event_cams > 0:
        # Determine whether direct trigger-aligned generation is possible
        use_hw_align = False
        _aligned_triggers = np.array([], dtype=np.int64)
        if hw_trigger_sync and len(hw_trigger_times_us) > 0:
            n_trig = len(hw_trigger_times_us)
            tol = max(5, int(rs_idx * 0.05))
            if abs(n_trig - rs_idx) <= tol:
                use_hw_align = True
                _aligned_triggers = hw_trigger_times_us[:rs_idx]
            else:
                print(
                    f"[HWSync] WARNING: trigger count ({n_trig}) differs from depth frame count "
                    f"({rs_idx}) by more than tolerance ({tol}). Falling back to elapsed-time alignment."
                )
        elif hw_trigger_sync:
            print("[HWSync] WARNING: No trigger timestamps received. Falling back to elapsed-time alignment.")

        if use_hw_align:
            print("[Recording] Generating trigger-aligned event frames from raw data...")
            generate_event_frames_hw_triggered(
                object_raw_dir, object_video_dir, object_hdf5_dir,
                num_event_cams, _aligned_triggers,
            )
        else:
            print("[Recording] Generating event camera video(s) from raw data...")
            generate_event_videos(object_raw_dir, object_video_dir, object_hdf5_dir, num_event_cams)
            print("[Recording] Aligning event frames to depth timestamps (using global timestamps)...")
            align_event_frames_to_depth(
                object_hdf5_dir, num_event_cams, frame_times_ns, event_logging_start_ns,
                depth_frame_offset=DEPTH_EVENT_ALIGN_OFFSET_FRAMES,
            )

        if debug and hw_trigger_sync:
            generate_hw_sync_debug_plots(
                object_dir=object_dir,
                hdf5_dir=object_hdf5_dir,
                hw_trigger_times_us=hw_trigger_times_us,
                frame_times_ns=frame_times_ns,
                num_event_cams=num_event_cams,
                use_hw_align=use_hw_align,
                event_device_open_ns=event_device_open_ns,
            )

    return True


def generate_hw_sync_debug_plots(
    object_dir: Path,
    hdf5_dir: Path,
    hw_trigger_times_us: np.ndarray,
    frame_times_ns: np.ndarray,
    num_event_cams: int,
    use_hw_align: bool,
    event_device_open_ns: int = 0,
) -> None:
    """
    Save a set of matplotlib figures that help diagnose hardware-trigger sync issues.

    Plots saved to <object_dir>/debug/hw_sync/:

    1. trigger_intervals.png  — inter-trigger intervals (µs) over time.
       Expected: flat line at ~1e6/FPS µs.  Gaps/spikes = missing/double triggers.

    2. trigger_vs_depth_count.png — bar chart comparing n_triggers vs n_depth_frames.
       Should be equal (or within the 5% tolerance) for hw-align to succeed.

    3. depth_frame_intervals.png — inter-depth-frame intervals (ms) over time.
       Any irregular gaps here show RealSense pipeline drops.

    4. alignment_offsets_cam{k}.png (per camera) — signed offset (ms) between the
       aligned event-frame centre and its trigger timestamp after alignment.
       Sub-millisecond is good; large offsets = wrong event frame was picked.

    5. trigger_vs_event_frame_centers.png — scatter of trigger timestamps (µs) vs
       event-frame centre timestamps (µs) for each camera.  Should lie on y=x.

    6. system_time_comparison.png — triggers converted to system clock (via
       event_device_open_ns) vs RealSense depth-frame system timestamps.
       Ideally the two traces overlap; any constant offset = clock registration
       error; growing drift = rate mismatch between event-camera µs clock and
       host system clock.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debug_dir = object_dir / "debug" / "hw_sync"
    debug_dir.mkdir(parents=True, exist_ok=True)

    expected_interval_us = 1e6 / FPS
    n_trig = len(hw_trigger_times_us)
    n_depth = len(frame_times_ns)

    # ------------------------------------------------------------------ #
    # 1. Inter-trigger intervals
    # ------------------------------------------------------------------ #
    if n_trig > 1:
        intervals_us = np.diff(hw_trigger_times_us.astype(np.float64))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(intervals_us, linewidth=0.8, color="steelblue")
        ax.axhline(expected_interval_us, color="red", linestyle="--",
                   label=f"expected {expected_interval_us:.1f} µs")
        ax.set_xlabel("Trigger index")
        ax.set_ylabel("Interval (µs)")
        ax.set_title(
            f"Inter-trigger intervals  |  n_triggers={n_trig}, "
            f"FPS={FPS}, expected={expected_interval_us:.1f} µs\n"
            f"mean={intervals_us.mean():.1f} µs  std={intervals_us.std():.1f} µs  "
            f"min={intervals_us.min():.1f} µs  max={intervals_us.max():.1f} µs"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(debug_dir / "trigger_intervals.png", dpi=150)
        plt.close(fig)
        print(f"[Debug] trigger_intervals.png saved")
    else:
        print(f"[Debug] Skipping trigger_intervals.png: only {n_trig} trigger(s)")

    # ------------------------------------------------------------------ #
    # 2. Trigger count vs depth frame count
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Triggers\n(event cam)", "Depth frames\n(RealSense)"],
                  [n_trig, n_depth],
                  color=["steelblue", "darkorange"])
    for bar, val in zip(bars, [n_trig, n_depth]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=12)
    diff = n_trig - n_depth
    tol = max(5, int(n_depth * 0.05))
    status = "OK (hw-align used)" if use_hw_align else f"MISMATCH (diff={diff:+d}, tol±{tol})"
    ax.set_title(f"Trigger vs depth frame count\n{status}")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(debug_dir / "trigger_vs_depth_count.png", dpi=150)
    plt.close(fig)
    print(f"[Debug] trigger_vs_depth_count.png saved")

    # ------------------------------------------------------------------ #
    # 3. Depth frame inter-frame intervals
    # ------------------------------------------------------------------ #
    if n_depth > 1:
        depth_intervals_ms = np.diff(frame_times_ns.astype(np.float64)) / 1e6
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(depth_intervals_ms, linewidth=0.8, color="darkorange")
        ax.axhline(1000.0 / FPS, color="red", linestyle="--",
                   label=f"expected {1000.0/FPS:.1f} ms")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Interval (ms)")
        ax.set_title(
            f"Depth frame inter-frame intervals  |  n_frames={n_depth}\n"
            f"mean={depth_intervals_ms.mean():.2f} ms  std={depth_intervals_ms.std():.2f} ms  "
            f"max={depth_intervals_ms.max():.2f} ms"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(debug_dir / "depth_frame_intervals.png", dpi=150)
        plt.close(fig)
        print(f"[Debug] depth_frame_intervals.png saved")

    # ------------------------------------------------------------------ #
    # 4. Alignment offsets (read from HDF5)
    # ------------------------------------------------------------------ #
    for cam_idx in range(num_event_cams):
        h5_path = hdf5_dir / f"events_cam{cam_idx}.h5"
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                if "events/alignment_offset_ms" not in f:
                    print(f"[Debug] cam{cam_idx}: no alignment_offset_ms in HDF5, skipping offset plot")
                    continue
                offset_ms = f["events/alignment_offset_ms"][:]
                ev_t_start = f["events/t_ev_start_us"][:]
                ev_t_end   = f["events/t_ev_end_us"][:]
        except Exception as e:
            print(f"[Debug] cam{cam_idx}: could not read HDF5 for offset plot: {e}")
            continue

        ev_center_us = (ev_t_start.astype(np.float64) + ev_t_end.astype(np.float64)) / 2.0

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))

        # Offset over time
        ax = axes[0]
        ax.plot(offset_ms, linewidth=0.8, color="purple")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Offset (ms)")
        ax.set_title(
            f"Alignment offset: event-frame-center − trigger  |  cam{cam_idx}\n"
            f"{'hw-trigger aligned' if use_hw_align else 'elapsed-time aligned'}  "
            f"median={np.median(offset_ms):.3f} ms  "
            f"mean={np.mean(offset_ms):.3f} ms  "
            f"max_abs={np.max(np.abs(offset_ms)):.3f} ms"
        )

        # Offset histogram
        ax = axes[1]
        ax.hist(offset_ms, bins=min(50, max(10, len(offset_ms) // 5)),
                color="purple", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Offset (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Offset histogram")

        fig.tight_layout()
        fig.savefig(debug_dir / f"alignment_offsets_cam{cam_idx}.png", dpi=150)
        plt.close(fig)
        print(f"[Debug] alignment_offsets_cam{cam_idx}.png saved")

        # ------------------------------------------------------------------ #
        # 5. Trigger timestamps vs event-frame-centre scatter (only for hw-align)
        # ------------------------------------------------------------------ #
        if use_hw_align and n_trig >= n_depth:
            aligned_triggers = hw_trigger_times_us[:n_depth].astype(np.float64)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(aligned_triggers / 1000.0, ev_center_us / 1000.0,
                       s=4, alpha=0.5, color="teal")
            # y = x reference line
            t_min = min(aligned_triggers.min(), ev_center_us.min()) / 1000.0
            t_max = max(aligned_triggers.max(), ev_center_us.max()) / 1000.0
            ax.plot([t_min, t_max], [t_min, t_max], "r--", linewidth=1, label="y = x (perfect)")
            ax.set_xlabel("Trigger timestamp (ms)")
            ax.set_ylabel("Event frame centre (ms)")
            ax.set_title(
                f"Trigger vs event-frame-centre  |  cam{cam_idx}\n"
                f"Points on y=x → perfect sync"
            )
            ax.legend()
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(debug_dir / f"trigger_vs_event_frame_centers_cam{cam_idx}.png", dpi=150)
            plt.close(fig)
            print(f"[Debug] trigger_vs_event_frame_centers_cam{cam_idx}.png saved")

    # ------------------------------------------------------------------ #
    # 6. System-time comparison: triggers (event-cam µs → system ns via
    #    event_device_open_ns) vs RealSense depth-frame system timestamps.
    # ------------------------------------------------------------------ #
    if event_device_open_ns > 0 and n_trig > 0 and n_depth > 0:
        # Convert event-camera µs → system ns using the device-open anchor.
        # The event camera's internal µs clock starts at device open, so:
        #   system_ns ≈ event_device_open_ns + trigger_time_us * 1000
        trigger_sys_ns = event_device_open_ns + hw_trigger_times_us.astype(np.float64) * 1000.0

        # Align both sequences to a common relative start (elapsed ms from the
        # first available reference in each series) so they can be overlaid.
        n_common = min(n_trig, n_depth)
        trig_elapsed_ms  = (trigger_sys_ns[:n_common] - trigger_sys_ns[0]) / 1e6
        depth_elapsed_ms = (frame_times_ns[:n_common].astype(np.float64) - frame_times_ns[0]) / 1e6

        # Per-pair difference: trigger_sys - depth_frame (ns → ms)
        # Positive = trigger is *later* than the depth timestamp (unexpected lag)
        # Negative = trigger arrived *before* depth timestamp was recorded
        diff_ms = (trigger_sys_ns[:n_common] - frame_times_ns[:n_common].astype(np.float64)) / 1e6

        fig, axes = plt.subplots(3, 1, figsize=(13, 11))

        # --- (a) Both timelines overlaid ---
        ax = axes[0]
        ax.plot(trig_elapsed_ms,  label="trigger (event cam → sys clock)", linewidth=0.9, color="steelblue")
        ax.plot(depth_elapsed_ms, label="depth frame (RealSense sys clock)", linewidth=0.9,
                color="darkorange", linestyle="--")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Elapsed time (ms)")
        ax.set_title(
            f"System-time comparison: triggers vs depth frames  (n={n_common})\n"
            f"Trigger origin: event_device_open_ns + trigger_us × 1000"
        )
        ax.legend()

        # --- (b) Per-frame difference ---
        ax = axes[1]
        ax.plot(diff_ms, linewidth=0.9, color="purple")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.fill_between(range(n_common), diff_ms, 0,
                        where=(np.array(diff_ms) >= 0), alpha=0.25, color="red",
                        label="trigger later than depth")
        ax.fill_between(range(n_common), diff_ms, 0,
                        where=(np.array(diff_ms) < 0), alpha=0.25, color="green",
                        label="trigger earlier than depth")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Δ system time (ms)\n(trigger_sys − depth_frame)")
        ax.set_title(
            f"Per-frame system-time offset  |  "
            f"mean={np.mean(diff_ms):.2f} ms  "
            f"std={np.std(diff_ms):.2f} ms  "
            f"median={np.median(diff_ms):.2f} ms  "
            f"max_abs={np.max(np.abs(diff_ms)):.2f} ms"
        )
        ax.legend(fontsize=8)

        # --- (c) Histogram of differences ---
        ax = axes[2]
        ax.hist(diff_ms, bins=min(60, max(10, n_common // 5)),
                color="purple", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, label="zero")
        ax.axvline(np.mean(diff_ms), color="red", linestyle="-", linewidth=1.2,
                   label=f"mean {np.mean(diff_ms):.2f} ms")
        ax.set_xlabel("Δ system time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of per-frame system-time offset")
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(debug_dir / "system_time_comparison.png", dpi=150)
        plt.close(fig)
        print(f"[Debug] system_time_comparison.png saved")
    else:
        print(f"[Debug] Skipping system_time_comparison.png: "
              f"event_device_open_ns={event_device_open_ns}, "
              f"n_trig={n_trig}, n_depth={n_depth}")

    print(f"[Debug] HW-sync debug plots saved to: {debug_dir}")


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
    depth_frame_offset: int = 0,
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

    ``depth_frame_offset`` (signed int, in frames): manual correction applied
    before the nearest-neighbour search.  Negative values shift the event
    lookup backward (use this when depth is behind events); positive shifts
    it forward.  Corresponds to subtracting
    ``depth_frame_offset * (1e9 / FPS)`` ns from depth_elapsed_ns.
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
        # Use frame-center timestamps: each event frame covers [t_start, t_end]
        # and the midpoint is the most representative time for that frame.
        ev_t_center_us = (ev_t_start_us.astype(np.int64) + ev_t_end_us.astype(np.int64)) // 2
        ev_elapsed_ns = (ev_t_center_us - ev_t_center_us[0]) * 1000
        depth_elapsed_ns = (depth_times_ns - event_logging_start_ns).astype(np.int64)

        # Apply manual frame offset (e.g. to correct for observed systematic lag)
        if depth_frame_offset != 0:
            frame_duration_ns = int(round(1e9 / FPS))
            depth_elapsed_ns = depth_elapsed_ns - depth_frame_offset * frame_duration_ns
            print(f"[Align] cam{cam_idx}: applying depth_frame_offset={depth_frame_offset:+d} "
                  f"({-depth_frame_offset * frame_duration_ns / 1e6:+.1f} ms shift on depth_elapsed)")

        # For each depth frame, find the nearest event frame
        indices = np.searchsorted(ev_elapsed_ns, depth_elapsed_ns, side="left")
        indices = np.clip(indices, 0, M - 1)

        # Check if the left neighbour is actually closer
        left = np.clip(indices - 1, 0, M - 1)
        d_right = np.abs(ev_elapsed_ns[indices] - depth_elapsed_ns)
        d_left = np.abs(ev_elapsed_ns[left] - depth_elapsed_ns)
        use_left = d_left < d_right
        indices[use_left] = left[use_left]

        offset_ms = (ev_elapsed_ns[indices] - depth_elapsed_ns).astype(np.float64) / 1e6

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
            f"median signed offset {np.median(offset_ms):.1f} ms, "
            f"mean {np.mean(offset_ms):.1f} ms, "
            f"max abs {np.max(np.abs(offset_ms)):.1f} ms"
        )


def align_event_frames_to_depth_hw_trigger(
    hdf5_dir: Path,
    num_cameras: int,
    trigger_times_us: np.ndarray,
) -> None:
    """
    Align event camera HDF5 frames to depth frames using hardware trigger timestamps.

    The RealSense camera sends a GPIO pulse for every depth frame
    (``rs.option.output_trigger_enabled``).  The event camera records each
    pulse as an external trigger event whose timestamp is in the event
    camera's own internal microsecond clock — the same domain as
    ``t_ev_start_us`` / ``t_ev_end_us``.  This eliminates the elapsed-time
    drift assumptions required by ``align_event_frames_to_depth``.

    trigger_times_us : 1-D int64 array, length = number of depth frames,
        containing the event-camera µs timestamp of each RealSense trigger pulse.
    """
    N = len(trigger_times_us)
    if N == 0:
        print("[AlignHW] No trigger timestamps provided, skipping")
        return

    for cam_idx in range(num_cameras):
        h5_path = hdf5_dir / f"events_cam{cam_idx}.h5"
        if not h5_path.exists():
            print(f"[AlignHW] events_cam{cam_idx}.h5 not found, skipping")
            continue

        with h5py.File(h5_path, "r") as f:
            ev_frames     = f["events/frames"][:]
            ev_t_start_us = f["events/t_ev_start_us"][:]
            ev_t_end_us   = f["events/t_ev_end_us"][:]
            attrs = dict(f["events"].attrs)

        M = len(ev_t_end_us)
        if M == 0:
            print(f"[AlignHW] cam{cam_idx}: no event frames, skipping")
            continue

        # Frame-centre timestamps (event-camera µs) — same clock as trigger_times_us.
        ev_t_center_us  = (ev_t_start_us.astype(np.int64) + ev_t_end_us.astype(np.int64)) // 2
        trigger_i64     = trigger_times_us.astype(np.int64)

        # Nearest-neighbour search: for each depth frame, find the closest event frame.
        indices = np.searchsorted(ev_t_center_us, trigger_i64, side="left")
        indices = np.clip(indices, 0, M - 1)
        left    = np.clip(indices - 1, 0, M - 1)
        d_right = np.abs(ev_t_center_us[indices] - trigger_i64)
        d_left  = np.abs(ev_t_center_us[left]    - trigger_i64)
        use_left = d_left < d_right
        indices[use_left] = left[use_left]

        offset_ms = (ev_t_center_us[indices] - trigger_i64).astype(np.float64) / 1000.0

        aligned_frames  = ev_frames[indices]
        aligned_t_start = ev_t_start_us[indices]
        aligned_t_end   = ev_t_end_us[indices]

        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("events")
            grp.create_dataset("frames",          data=aligned_frames)
            grp.create_dataset("t_ev_start_us",   data=aligned_t_start)
            grp.create_dataset("t_ev_end_us",     data=aligned_t_end)
            grp.create_dataset("alignment_offset_ms", data=offset_ms)
            grp.create_dataset("hw_trigger_times_us", data=trigger_times_us)
            for k, v in attrs.items():
                grp.attrs[k] = v
            grp.attrs["hw_trigger_sync"] = True

        print(
            f"[AlignHW] cam{cam_idx}: {M} event frames → {N} aligned, "
            f"median offset {np.median(offset_ms):.3f} ms, "
            f"max abs {np.max(np.abs(offset_ms)):.3f} ms"
        )


def generate_event_frames_hw_triggered(
    object_raw_dir: Path,
    object_video_dir: Path,
    object_hdf5_dir: Path,
    num_cameras: int,
    trigger_times_us: np.ndarray,
) -> None:
    """
    Generate one event frame per depth frame using hardware trigger intervals.

    For depth frame i, accumulates all events in the interval
    [trigger[i-1], trigger[i]).  Frame 0 uses a synthetic start of
    trigger[0] - mean_period.  This eliminates both the fixed-FPS binning
    of generate_event_videos() and the subsequent nearest-neighbour search
    of align_event_frames_to_depth_hw_trigger(), replacing them with a
    direct trigger-to-frame mapping.

    Output: events_cam{k}.h5 and events_cam{k}.mp4 written to the usual
    locations, format identical to generate_event_videos().
    """
    from metavision_core.event_io import RawReader as _RawReader

    N = len(trigger_times_us)
    if N == 0:
        print("[EventHW] No trigger timestamps, skipping")
        return

    trigger_i64 = trigger_times_us.astype(np.int64)
    mean_period_us = int(np.mean(np.diff(trigger_i64))) if N > 1 else int(1e6 / FPS)

    # Window boundaries: event_frame[i] covers [t_starts[i], t_ends[i])
    t_ends   = trigger_i64
    t_starts = np.empty(N, dtype=np.int64)
    t_starts[0]  = trigger_i64[0] - mean_period_us
    t_starts[1:] = trigger_i64[:-1]

    for cam_idx in range(num_cameras):
        raw_file   = object_raw_dir  / f"events_cam{cam_idx}.raw"
        video_file = object_video_dir / f"events_cam{cam_idx}.mp4"
        h5_file    = object_hdf5_dir  / f"events_cam{cam_idx}.h5"

        if not raw_file.exists():
            print(f"[EventHW] Raw file not found: {raw_file}, skipping")
            continue

        print(f"[EventHW] cam{cam_idx}: reading all events from {raw_file} ...")
        rr = _RawReader(str(raw_file))
        height, width = rr.get_size()

        chunks = []
        while not rr.is_done():
            chunk = rr.load_delta_t(100_000)
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)

        if not chunks:
            print(f"[EventHW] cam{cam_idx}: no events found, skipping")
            continue

        all_events = np.concatenate(chunks)
        all_t = all_events["t"].astype(np.int64)

        print(f"[EventHW][DEBUG] cam{cam_idx}")
        print(f"  CD events: count={len(all_events)}, range=[{all_t[0]}, {all_t[-1]}] us")
        print(f"  Triggers : count={len(trigger_i64)}, range=[{trigger_i64[0]}, {trigger_i64[-1]}] us")
        print(f"  First 10 CD times: {all_t[:10]}")
        print(f"  First 10 triggers: {trigger_i64[:10]}")
        print(f"  Last 10 CD times: {all_t[-10:]}")
        print(f"  Last 10 triggers: {trigger_i64[-10:]}")

        video = cv2.VideoWriter(
            str(video_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (width, height),
            isColor=False,
        )

        with h5py.File(h5_file, "w") as h5:
            grp = h5.create_group("events")
            frames_ds  = grp.create_dataset("frames",        shape=(N, height, width), dtype=np.uint8)
            t_start_ds = grp.create_dataset("t_ev_start_us", shape=(N,), dtype=np.int64)
            t_end_ds   = grp.create_dataset("t_ev_end_us",   shape=(N,), dtype=np.int64)
            grp.create_dataset("hw_trigger_times_us", data=trigger_i64)
            grp.attrs.update({
                "fps": FPS, "delta_t_us": mean_period_us,
                "width": width, "height": height,
                "hw_trigger_sync": True,
            })

            for i in range(N):
                t0, t1 = t_starts[i], t_ends[i]
                lo = int(np.searchsorted(all_t, t0, side="left"))
                hi = int(np.searchsorted(all_t, t1, side="left"))
                evs = all_events[lo:hi]

                # Grey background (128), positive events → white (255),
                # negative events → black (0).  Last event at each pixel wins.
                frame = np.full((height, width), 128, dtype=np.uint8)
                if len(evs) > 0:
                    xs = evs["x"].astype(np.int32)
                    ys = evs["y"].astype(np.int32)
                    ps = evs["p"].astype(bool)
                    frame[ys[~ps], xs[~ps]] = 0    # negative first
                    frame[ys[ps],  xs[ps]]  = 255  # positive last (wins on overlap)

                # Rotate 180° to match generate_event_videos() orientation
                frame = cv2.rotate(frame, cv2.ROTATE_180)

                frames_ds[i]  = frame
                t_start_ds[i] = t0
                t_end_ds[i]   = t1
                video.write(frame)

        video.release()
        print(f"[EventHW] cam{cam_idx}: {N} trigger-aligned frames written → {h5_file}")


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
    logging_start_queue: Optional[Queue] = None,
    hw_trigger_sync: bool = False,
    ) -> None:
    """
    Subprocess entry point for event camera acquisition.

    Lifecycle:
      1. Open device(s), apply biases; if hw_trigger_sync, enable trigger input
         so RealSense pulses are recorded as EventExtTrigger records in the .raw file.
      2. Start raw streaming; signal ready_event; push device_open_ns to
         clock_sync_queue so the main process can map event µs → system ns.
      3. Drain events without storing until start_logging_event is set.
      4. Call log_raw_data(); push the precise start timestamp to
         logging_start_queue for downstream clock alignment.
      5. Drain events until stop_acquire_event is set (robot finished).
      6. Flush for flush_seconds; stop.
    """
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

    # Enable hardware trigger input and register a callback so trigger timestamps
    # are captured in real-time and stored alongside the raw event data.

    if hw_trigger_sync:
        from metavision_hal import I_TriggerIn
        for _dev in filter(None, (device0, device1)):
            _trig_in = _dev.get_i_trigger_in()
            if _trig_in is not None:
                _trig_in.enable(I_TriggerIn.Channel.MAIN)  # channel 0 = rising edge
                print(f"[HWSync] Trigger input enabled on device")

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

    # Actively consume events without writing to disk until the start gate fires.
    # This ensures the raw file only contains events from the moment the arm
    # reached its first pose — aligned with when RealSense recording starts.
    if start_logging_event is not None:
        while not start_logging_event.is_set() and not stop_acquire_event.is_set():
            next(it0)
            if it1 is not None:
                next(it1)

    logging_active = False
    try:
        if not stop_acquire_event.is_set():
            # Start file logging now that the arm has reached the first pose.
            # Capture time immediately before calling log_raw_data so the main
            # process has a precise anchor for the event-camera clock origin.
            log_start_ns = time.time_ns()
            raw0.log_raw_data(event0_path)
            if raw1 is not None:
                raw1.log_raw_data(event1_path)
            logging_active = True
            if logging_start_queue is not None:
                logging_start_queue.put(log_start_ns)

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
    light_check: bool = False,
    hw_trigger_sync: bool = False,
    debug: bool = False,
    ) -> None:
    """
    Main synchronized recording function with multi-object support.

    When ``light_check=True`` a single recording is made with object name
    "light_check".  The robot arm stays still (use ``--agent-type light_check``
    in my_main.py) while the cameras record a flashing screen.

    Normal loop:
    1. Ask for object name
    2. Initialize cameras, send "ready", wait for "start"
    3. Record until hemisphere complete
    4. Send "done", ask for next object name
    5. Repeat until user quits
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    sync_client = None
    pose_receiver = None
    transport_delay_ns = 0

    # Initialize ZMQ sync client
    sync_client = ZMQSyncClient(zmq_sync_addr)
    sync_client.connect()

    # Measure one-way ZMQ transport delay before any recording starts.
    # my_main.py's wait_for_ready() handles ping messages transparently.
    print("[ClockSync] Measuring ZMQ transport delay...")
    transport_delay_ns = sync_client.measure_transport_delay(n_rounds=20)

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
        # Enable hardware GPIO trigger output: fires a pulse for each depth frame.
        # The event camera records these pulses as external trigger events, enabling
        # precise per-frame alignment without inter-camera clock assumptions.
        if hw_trigger_sync:
            if depth_sensor.supports(rs.option.output_trigger_enabled):
                depth_sensor.set_option(rs.option.output_trigger_enabled, 1)
                print("[HWSync] RealSense hardware trigger output (output_trigger_enabled) enabled")
            else:
                print("[HWSync] WARNING: depth sensor does not support output_trigger_enabled; "
                      "hardware trigger sync will be unavailable")
        # Enable system-clock timestamps on every sensor that supports it
        # (depth + color) so both t_global_ms and t_rgb_ms are wall-clock referenced.
        for _sensor in profile.get_device().query_sensors():
            if _sensor.supports(rs.option.global_time_enabled):
                _sensor.set_option(rs.option.global_time_enabled, 1)

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
        if hw_trigger_sync:
            if depth_sensor.supports(rs.option.output_trigger_enabled):
                depth_sensor.set_option(rs.option.output_trigger_enabled, 1)
                print("[HWSync] RealSense hardware trigger output enabled (after reset)")
            else:
                print("[HWSync] WARNING: depth sensor does not support output_trigger_enabled")
        for _sensor in profile.get_device().query_sensors():
            if _sensor.supports(rs.option.global_time_enabled):
                _sensor.set_option(rs.option.global_time_enabled, 1)

        print("[Recording] Warming up RealSense after reset...")
        for _ in range(30):
            pipeline.wait_for_frames(timeout_ms=1000)

    print("[Recording] RealSense initialized")

    recording_count = 0

    try:
        while True:
            # In light-check mode, use a fixed name and record once
            if light_check:
                object_name = "light_check"
                rec_dir = LIGHT_CHECK_ROOT.parent
                if LIGHT_CHECK_ROOT.exists():
                    import shutil
                    shutil.rmtree(LIGHT_CHECK_ROOT)
                    print("[Recording] Removed previous light_check recording")
            # In temporal-check mode, use a fixed name and record once
            elif temporal_check:
                object_name = "temporal_check"
                rec_dir = TEMPORAL_CHECK_ROOT.parent
                if TEMPORAL_CHECK_ROOT.exists():
                    import shutil
                    shutil.rmtree(TEMPORAL_CHECK_ROOT)
                    print(f"[Recording] Removed previous temporal_check recording")
            else:
                rec_dir = DATA_ROOT
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
                data_root=rec_dir,
                transport_delay_ns=transport_delay_ns,
                hw_trigger_sync=hw_trigger_sync,
                debug=debug,
            )
            
            if success:
                recording_count += 1
                print(f"\n[Recording] Completed {recording_count} recording(s) so far.")
                if light_check:
                    break
                elif temporal_check:
                    print("\n[Recording] Running temporal alignment analysis...")
                    import sys, importlib.util
                    # TODO: 'analyze_temporal_alignment.py' does not exist in viz_and_tests/.
                    # Update this path to point to the correct analysis script.
                    _script = Path(__file__).parent / "viz_and_tests" / "analyze_temporal_alignment.py"
                    _spec = importlib.util.spec_from_file_location("analyze_temporal_alignment", _script)
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    _mod.analyze_temporal_alignment(TEMPORAL_CHECK_ROOT)
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

        if sync_client is not None:
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
    parser.add_argument(
        "--light-check",
        action="store_true",
        help="Record a static light-check sequence (arm holds its current position for 20 s). "
             "Point cameras at a periodically flashing screen to verify RealSense/event "
             "temporal alignment. Start my_main.py with --sync-recording --agent-type light_check.",
    )

    parser.add_argument(
        "--no-hw-trigger-sync",
        action="store_true",
        dest="no_hw_trigger_sync",
        help="Disable hardware trigger synchronisation (enabled by default). "
             "By default the RealSense depth sensor outputs a GPIO pulse for every depth "
             "frame (output_trigger_enabled) and the event camera records each pulse as an "
             "external trigger event, giving sub-frame alignment precision. "
             "Pass this flag to fall back to elapsed-time estimation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="After each recording, save hardware-trigger sync debug plots to "
             "<object_dir>/debug/hw_sync/ for diagnosing trigger timing problems.",
    )

    args = parser.parse_args()

    main(
        zmq_sync_addr=args.zmq_sync_addr,
        zmq_pose_addr=args.zmq_pose_addr,
        num_event_cams=args.event_cameras,
        temporal_check=args.temporal_check,
        light_check=args.light_check,
        hw_trigger_sync=not args.no_hw_trigger_sync,
        debug=args.debug,
    )