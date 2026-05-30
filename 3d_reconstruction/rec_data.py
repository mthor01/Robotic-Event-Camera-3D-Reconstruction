"""
Simplified single-camera recording with ZeroMQ communication for robot arm coordination.

Per-recording workflow:
  1. Initialize RealSense (kept alive across multiple recordings in the same session).
  2. Start event-camera drain subprocess.
  3. Send a "ready" signal to my_main.py via ZMQ REQ/REP and block until "start"
     arrives (my_main.py sends it when the arm reaches its first pose).
  4. Open event-camera file logging in sync with the "start" signal.
  5. Capture RealSense depth + colour frames; collect robot poses from ZMQ PUB/SUB.
  6. Stop when the agent publishes an "agent_complete" event.
  7. Align event frames to depth timestamps via elapsed-time, interpolate poses per
     frame, write HDF5.
  8. Send "done" to my_main.py, then loop for the next object.

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
    ZMQ_SYNC_ADDR, ZMQ_POSE_ADDR, DATA_ROOT,
    POSE_TIME_OFFSET_MS,
    DEPTH_EVENT_ALIGN_OFFSET_FRAMES,
    DEPTH_VIZ_MIN, DEPTH_VIZ_MAX,
)


class ZMQPoseReceiver:
    """
    Receives pose data from my_main.py over ZMQ PUB/SUB.
    Stores poses in a thread-safe deque.
    Also monitors for agent_complete events to signal recording stop.
    """

    def __init__(self, connect_addr: str = ZMQ_POSE_ADDR):
        self.connect_addr = connect_addr
        self.ctx = zmq.Context.instance()
        self.sub: Optional[zmq.Socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.poses: deque = deque(maxlen=100000)
        self.lock = threading.Lock()
        self.stop_recording_event = threading.Event()

    def start(self) -> None:
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"pose")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"event")
        self.sub.setsockopt(zmq.RCVHWM, 10000)
        self.sub.setsockopt(zmq.RCVTIMEO, 100)
        self.sub.connect(self.connect_addr)

        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[PoseReceiver] Started, connected to {self.connect_addr}")

    def _receive_loop(self) -> None:
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
                continue
            except Exception as e:
                if self.running:
                    print(f"[PoseReceiver] Error: {e}")

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.sub:
            self.sub.close()
            self.sub = None
        print("[PoseReceiver] Stopped")

    def should_stop_recording(self) -> bool:
        return self.stop_recording_event.is_set()

    def get_all_poses(self) -> list:
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
        self.req = self.ctx.socket(zmq.REQ)
        self.req.connect(self.connect_addr)
        print(f"[SyncClient] Connected to {self.connect_addr}")

    def send_ready_wait_start(self, timeout_sec: float = 60.0) -> bool:
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
        self.req.setsockopt(zmq.RCVTIMEO, 5000)
        msg = {"type": "done", "t_ns": time.time_ns()}
        self.req.send(msgpack.packb(msg, use_bin_type=True))
        try:
            reply = msgpack.unpackb(self.req.recv(), raw=False)
            print(f"[SyncClient] Recording done acknowledged: {reply.get('type')}")
        except zmq.error.Again:
            print("[SyncClient] Timeout on done acknowledgment")

    def measure_transport_delay(self, n_rounds: int = 20) -> int:
        """Measure one-way ZMQ transport delay via ping-pong RTT."""
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
                pass

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

    Returns a dict of arrays aligned 1-to-1 with *frame_times_ns*:
        ee_T              (N, 4, 4) float64
        nearest_offset_ms (N,)      float64
    """
    order = np.argsort(pose_times_ns)
    pose_times_ns = pose_times_ns[order]
    pose_ee_T = pose_ee_T[order]

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

        sanitized = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)

        object_dir = DATA_ROOT / sanitized
        if object_dir.exists():
            overwrite = input(f"Recording '{sanitized}' already exists. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                continue

        return sanitized


def event_drain_process(
    stop_acquire_event,
    done_event,
    ready_event,
    event0_path: str,
    flush_seconds: float = 0.5,
    clock_sync_queue: Optional[Queue] = None,
    start_logging_event=None,
    logging_start_queue: Optional[Queue] = None,
    hw_sync: bool = False,
) -> None:
    """
    Subprocess entry point for single event camera acquisition.

    Normal-mode lifecycle:
      1. Open device, apply biases.
      2. Start raw streaming; signal ready_event; push device_open_ns to
         clock_sync_queue so the main process can map event µs → system ns.
      3. Drain events without storing until start_logging_event is set.
      4. Call log_raw_data(); push the precise start timestamp to
         logging_start_queue for downstream clock alignment.
      5. Drain events until stop_acquire_event is set (robot finished).
      6. Flush for flush_seconds; stop.

    HW-sync mode lifecycle (hw_sync=True):
      1. Open device, apply biases, enable trigger input channel MAIN.
      2. Call log_raw_data() immediately; alignment is done via hardware
         trigger timestamps so the preamble before the first trigger is
         automatically discarded.
      3. Signal ready_event; push device_open_ns to clock_sync_queue.
      4. Drain events until stop_acquire_event is set.
      5. Flush for flush_seconds; stop.
    """
    devices = DeviceDiscovery.list()
    device0 = DeviceDiscovery.open(devices[0])
    device_open_ns = time.time_ns()

    biases = device0.get_i_ll_biases()
    biases.set("bias_diff_on", BIAS_DIFF_ON)
    biases.set("bias_diff_off", BIAS_DIFF_OFF)
    biases.set("bias_fo", BIAS_FO)
    biases.set("bias_hpf", BIAS_HPF)
    biases.set("bias_refr", BIAS_REFR)

    if hw_sync:
        try:
            from metavision_hal import I_TriggerIn
            trig_in = device0.get_i_trigger_in()
            trig_in.enable(I_TriggerIn.Channel.MAIN)
            print("[EventDrain] HW trigger input channel MAIN enabled")
        except Exception as e:
            print(f"[EventDrain] WARNING: could not enable HW trigger input: {e}")

    raw0 = device0.get_i_events_stream()
    raw0.start()

    it0 = iter(EventsIterator.from_device(device0))

    logging_active = False
    try:
        if hw_sync:
            # Start logging immediately — pre-trigger frames are discarded
            # automatically during trigger-based alignment.
            log_start_ns = time.time_ns()
            raw0.log_raw_data(event0_path)
            logging_active = True
            if logging_start_queue is not None:
                logging_start_queue.put(log_start_ns)
            if clock_sync_queue is not None:
                clock_sync_queue.put(device_open_ns)
            ready_event.set()

            while not stop_acquire_event.is_set():
                next(it0)

            flush_start = time.time()
            while time.time() - flush_start < flush_seconds:
                next(it0)

        else:
            ready_event.set()
            if clock_sync_queue is not None:
                clock_sync_queue.put(device_open_ns)

            # Drain events without writing until recording starts
            if start_logging_event is not None:
                while not start_logging_event.is_set() and not stop_acquire_event.is_set():
                    next(it0)

            if not stop_acquire_event.is_set():
                log_start_ns = time.time_ns()
                raw0.log_raw_data(event0_path)
                logging_active = True
                if logging_start_queue is not None:
                    logging_start_queue.put(log_start_ns)

                while not stop_acquire_event.is_set():
                    next(it0)

                flush_start = time.time()
                while time.time() - flush_start < flush_seconds:
                    next(it0)

    finally:
        if logging_active:
            raw0.stop_log_raw_data()
        done_event.set()


def generate_event_video(
    object_raw_dir: Path,
    object_hdf5_dir: Path,
) -> None:
    """Decode raw event recording into per-frame HDF5 (no video — written after alignment)."""
    delta_t_us = int(1e6 / FPS)

    raw_file = object_raw_dir / "events_cam0.raw"
    h5_file = object_hdf5_dir / "events_cam0.h5"

    if not raw_file.exists():
        print(f"[EventH5] Raw file not found: {raw_file}, skipping")
        return

    print(f"[EventH5] Decoding cam0: {raw_file} ...")
    ev_it = EventsIterator(input_path=str(raw_file), delta_t=delta_t_us)
    height, width = ev_it.get_size()

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
        rotated = cv2.rotate(frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        frames_ds.resize(idx + 1, axis=0)
        t_start_ds.resize(idx + 1, axis=0)
        t_end_ds.resize(idx + 1, axis=0)
        frames_ds[idx] = gray
        t_start_ds[idx] = ts - delta_t_us
        t_end_ds[idx] = ts
        idx += 1

    frame_gen.set_output_callback(on_frame)
    for evs in ev_it:
        frame_gen.process_events(evs)

    h5.close()
    print(f"[EventH5] cam0: {idx} frames decoded")


def align_event_frames_to_depth(
    hdf5_dir: Path,
    depth_times_ns: np.ndarray,
    event_logging_start_ns: int,
    depth_frame_offset: int = 0,
) -> None:
    """
    Align event camera HDF5 frames to depth frame timestamps via elapsed time.

    Both event and depth recordings start at approximately the same system
    time (event_logging_start_ns). We align by elapsed time from that common
    origin so the result is independent of whether the raw-file timestamps are
    absolute camera-internal values or rebased to zero.

    The unaligned events_cam0.h5 (M frames) is replaced with an aligned
    version (N frames) where N = len(depth_times_ns), so that
    events[i] corresponds to depth[i] and poses[i].
    """
    N = len(depth_times_ns)
    if N == 0:
        return

    h5_path = hdf5_dir / "events_cam0.h5"
    if not h5_path.exists():
        print(f"[Align] events_cam0.h5 not found, skipping")
        return

    with h5py.File(h5_path, "r") as f:
        ev_frames = f["events/frames"][:]
        ev_t_end_us = f["events/t_ev_end_us"][:]
        ev_t_start_us = f["events/t_ev_start_us"][:]
        attrs = dict(f["events"].attrs)

    M = len(ev_t_end_us)
    if M == 0:
        print(f"[Align] cam0: no event frames, skipping")
        return

    ev_t_center_us = (ev_t_start_us.astype(np.int64) + ev_t_end_us.astype(np.int64)) // 2
    ev_elapsed_ns = (ev_t_center_us - ev_t_center_us[0]) * 1000
    depth_elapsed_ns = (depth_times_ns - event_logging_start_ns).astype(np.int64)

    if depth_frame_offset != 0:
        frame_duration_ns = int(round(1e9 / FPS))
        depth_elapsed_ns = depth_elapsed_ns - depth_frame_offset * frame_duration_ns
        print(f"[Align] cam0: applying depth_frame_offset={depth_frame_offset:+d} "
              f"({-depth_frame_offset * frame_duration_ns / 1e6:+.1f} ms shift on depth_elapsed)")

    indices = np.searchsorted(ev_elapsed_ns, depth_elapsed_ns, side="left")
    indices = np.clip(indices, 0, M - 1)

    left = np.clip(indices - 1, 0, M - 1)
    d_right = np.abs(ev_elapsed_ns[indices] - depth_elapsed_ns)
    d_left = np.abs(ev_elapsed_ns[left] - depth_elapsed_ns)
    use_left = d_left < d_right
    indices[use_left] = left[use_left]

    offset_ms = (ev_elapsed_ns[indices] - depth_elapsed_ns).astype(np.float64) / 1e6

    aligned_frames = ev_frames[indices]
    aligned_t_start = ev_t_start_us[indices]
    aligned_t_end = ev_t_end_us[indices]

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("events")
        grp.create_dataset("frames", data=aligned_frames)
        grp.create_dataset("t_ev_start_us", data=aligned_t_start)
        grp.create_dataset("t_ev_end_us", data=aligned_t_end)
        grp.create_dataset("alignment_offset_ms", data=offset_ms)
        for k, v in attrs.items():
            grp.attrs[k] = v

    print(
        f"[Align] cam0: {M} → {N} frames, "
        f"median signed offset {np.median(offset_ms):.1f} ms, "
        f"mean {np.mean(offset_ms):.1f} ms, "
        f"max abs {np.max(np.abs(offset_ms)):.1f} ms"
    )


def extract_hw_triggers(raw_file: Path) -> np.ndarray:
    """
    Extract rising-edge hardware trigger timestamps (µs) from a .raw event file.

    The RealSense GPIO fires once per depth frame when output_trigger_enabled=1.
    These are stored as EventExtTrigger records in the .raw file.
    Returns a sorted int64 array of trigger times in event-camera µs.
    """
    from metavision_core.event_io import RawReader
    rr = RawReader(str(raw_file))
    while not rr.is_done():
        rr.load_delta_t(100_000)  # process in 100 ms chunks
    trig = rr.get_ext_trigger_events()
    if trig is None or len(trig) == 0:
        print("[HWTrigger] No trigger events found in raw file")
        return np.array([], dtype=np.int64)
    rising = trig[trig["p"] == 1]
    if len(rising) == 0:
        print("[HWTrigger] No rising-edge trigger events found in raw file")
        return np.array([], dtype=np.int64)
    ts = np.sort(rising["t"].astype(np.int64))
    span_s = (ts[-1] - ts[0]) / 1e6
    print(
        f"[HWTrigger] {len(ts)} rising-edge triggers: "
        f"first={ts[0]} µs, last={ts[-1]} µs, span={span_s:.3f} s"
    )

    # First 10 trigger timestamps in ms
    n_show = min(10, len(ts))
    first_ms = ts[:n_show] / 1e3
    print(f"[HWTrigger] First {n_show} trigger timestamps (ms): "
          + ", ".join(f"{t:.3f}" for t in first_ms))

    # 10 largest gaps between consecutive triggers
    if len(ts) >= 2:
        gaps_us = np.diff(ts)
        top_n = min(10, len(gaps_us))
        top_idx = np.argsort(gaps_us)[-top_n:][::-1]
        print(f"[HWTrigger] {top_n} largest inter-trigger gaps:")
        for rank, i in enumerate(top_idx, 1):
            print(f"  #{rank:2d}: {gaps_us[i] / 1e3:.3f} ms  "
                  f"(between trigger {i} @ {ts[i]/1e3:.3f} ms "
                  f"and trigger {i+1} @ {ts[i+1]/1e3:.3f} ms)")

    return ts


def align_event_frames_to_depth_hw(
    hdf5_dir: Path,
    trigger_times_us: np.ndarray,
    n_depth_frames: int,
) -> None:
    """
    Align event camera HDF5 frames to depth frames using hardware trigger timestamps.

    trigger_times_us[i] is the event-camera timestamp (µs) of the RealSense GPIO
    pulse that fired in sync with depth frame i.  For each trigger we find the
    event accumulation frame whose window center is closest to that timestamp.

    The existing events_cam0.h5 is replaced with an aligned version (one event
    frame per depth frame).  The trigger timestamps and per-frame alignment
    offsets (in µs) are stored alongside the frames.
    """
    N = min(len(trigger_times_us), n_depth_frames)
    if N == 0:
        print("[AlignHW] No trigger-to-frame pairs available, skipping alignment")
        return

    if len(trigger_times_us) != n_depth_frames:
        print(
            f"[AlignHW] WARNING: {len(trigger_times_us)} triggers vs "
            f"{n_depth_frames} depth frames — using first {N} pairs"
        )

    h5_path = hdf5_dir / "events_cam0.h5"
    if not h5_path.exists():
        print("[AlignHW] events_cam0.h5 not found, skipping")
        return

    with h5py.File(h5_path, "r") as f:
        ev_frames = f["events/frames"][:]
        ev_t_end_us = f["events/t_ev_end_us"][:]
        ev_t_start_us = f["events/t_ev_start_us"][:]
        attrs = dict(f["events"].attrs)

    M = len(ev_t_end_us)
    if M == 0:
        print("[AlignHW] No event frames found, skipping")
        return

    trig_us = trigger_times_us[:N]
    ev_t_center_us = (ev_t_start_us.astype(np.int64) + ev_t_end_us.astype(np.int64)) // 2

    # For each trigger find the event frame whose center is nearest
    indices = np.searchsorted(ev_t_center_us, trig_us, side="left")
    indices = np.clip(indices, 0, M - 1)
    left = np.clip(indices - 1, 0, M - 1)
    d_right = np.abs(ev_t_center_us[indices] - trig_us)
    d_left = np.abs(ev_t_center_us[left] - trig_us)
    use_left = d_left < d_right
    indices[use_left] = left[use_left]

    # Signed offset: positive means event frame is ahead of the trigger
    offset_us = (ev_t_center_us[indices] - trig_us).astype(np.float64)

    aligned_frames = ev_frames[indices]
    aligned_t_start = ev_t_start_us[indices]
    aligned_t_end = ev_t_end_us[indices]

    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("events")
        grp.create_dataset("frames", data=aligned_frames)
        grp.create_dataset("t_ev_start_us", data=aligned_t_start)
        grp.create_dataset("t_ev_end_us", data=aligned_t_end)
        grp.create_dataset("alignment_offset_us", data=offset_us)
        grp.create_dataset("hw_trigger_times_us", data=trig_us)
        for k, v in attrs.items():
            grp.attrs[k] = v
        grp.attrs["alignment_mode"] = "hw_trigger"

    print(
        f"[AlignHW] {M} event frames → {N} aligned frames, "
        f"median signed offset {np.median(offset_us):.1f} µs, "
        f"mean {np.mean(offset_us):.1f} µs, "
        f"max abs {np.max(np.abs(offset_us)):.1f} µs"
    )


def write_event_video_from_h5(hdf5_dir: Path, video_dir: Path) -> None:
    """Overwrite events_cam0.mp4 with the frames stored in the (aligned) events_cam0.h5."""
    h5_path = hdf5_dir / "events_cam0.h5"
    video_path = video_dir / "events_cam0.mp4"

    if not h5_path.exists():
        print("[EventVideo] events_cam0.h5 not found, skipping video regeneration")
        return

    with h5py.File(h5_path, "r") as f:
        frames = f["events/frames"][:]
        fps = f["events"].attrs.get("fps", FPS)

    N, height, width = frames.shape
    video = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        isColor=False,
    )
    for frame in frames:
        video.write(frame)
    video.release()
    print(f"[EventVideo] Aligned video written: {N} frames → {video_path}")


def record_single_object(
    object_name: str,
    pipeline: rs.pipeline,
    pose_receiver: "ZMQPoseReceiver",
    sync_client: "ZMQSyncClient",
    data_root: Path = DATA_ROOT,
    transport_delay_ns: int = 0,
    hw_sync: bool = False,
    depth_sensor=None,
) -> bool:
    """
    Record a single object. Returns True if successful, False if should abort.

    With hw_sync=True the event camera starts logging before the RealSense
    recording loop begins.  The RealSense GPIO trigger output is enabled for
    the duration of the loop so that each captured depth frame fires an
    EventExtTrigger pulse into the event stream, enabling sample-accurate
    frame alignment via align_event_frames_to_depth_hw().
    """
    object_dir = data_root / object_name
    object_video_dir = object_dir / "videos"
    object_raw_dir = object_dir / "raw_event_data"
    object_hdf5_dir = object_dir / "hdf5"

    object_video_dir.mkdir(parents=True, exist_ok=True)
    object_raw_dir.mkdir(parents=True, exist_ok=True)
    object_hdf5_dir.mkdir(parents=True, exist_ok=True)

    event0_raw_file = object_raw_dir / "events_cam0.raw"
    rs_video_file = object_video_dir / "realsense_depth.mp4"
    rs_rgb_video_file = object_video_dir / "realsense_rgb.mp4"
    rs_h5_file = object_hdf5_dir / "realsense.h5"
    poses_h5_file = object_hdf5_dir / "poses.h5"
    raw_poses_h5_file = object_hdf5_dir / "raw_poses.h5"
    metadata_h5_file = object_hdf5_dir / "metadata.h5"

    print(f"\n[Recording] Starting recording for object: {object_name}")
    print(f"[Recording] Output directory: {object_dir}")

    # Start event camera subprocess
    stop_acquire_event = MPEvent()
    event_done_event = MPEvent()
    event_ready_event = MPEvent()
    start_logging_event = MPEvent()
    clock_sync_queue = Queue()
    logging_start_queue = Queue()

    event_proc = Process(
        target=event_drain_process,
        args=(stop_acquire_event, event_done_event, event_ready_event, str(event0_raw_file)),
        kwargs={
            "clock_sync_queue": clock_sync_queue,
            "start_logging_event": start_logging_event,
            "logging_start_queue": logging_start_queue,
            "hw_sync": hw_sync,
        },
    )
    event_proc.start()

    print("[Recording] Waiting for event camera...")
    event_ready_event.wait()
    try:
        event_device_open_ns = clock_sync_queue.get(timeout=5.0)
    except Exception:
        event_device_open_ns = 0
    print("[Recording] Event camera ready")

    # Reset pose receiver and start it before the sync handshake to avoid
    # the ZMQ slow-joiner problem (agent_complete published before SUB connects).
    pose_receiver.stop_recording_event.clear()
    pose_receiver.start()

    print("[Recording] Sending ready signal to robot controller...")
    if not sync_client.send_ready_wait_start(timeout_sec=120.0):
        print("[Recording] ERROR: Did not receive start signal, aborting")
        pose_receiver.stop()
        stop_acquire_event.set()
        event_done_event.wait()
        event_proc.join()
        return False

    if hw_sync:
        # Flush any RS frames buffered during the ZMQ handshake so the first
        # frame captured in the recording loop is the one that fires the first
        # trigger pulse into the event stream.
        if depth_sensor is not None and depth_sensor.supports(rs.option.output_trigger_enabled):
            depth_sensor.set_option(rs.option.output_trigger_enabled, 0)
        print("[HWSync] Flushing stale RS frames before enabling triggers...")
        flushed = 0
        flush_deadline = time.time() + 0.5  # drain at most ~0.5 s of buffered frames
        while time.time() < flush_deadline:
            try:
                pipeline.wait_for_frames(timeout_ms=50)
                flushed += 1
            except RuntimeError:
                break
        print(f"[HWSync] Flushed {flushed} stale RS frames")
        if depth_sensor is not None and depth_sensor.supports(rs.option.output_trigger_enabled):
            depth_sensor.set_option(rs.option.output_trigger_enabled, 1)
            print("[HWSync] RS trigger output enabled — recording loop starting")
        else:
            print("[HWSync] WARNING: depth_sensor does not support output_trigger_enabled")
        event_logging_start_ns = time.time_ns()
    else:
        # Normal mode: signal event camera to start logging
        start_logging_event.set()
        event_logging_start_ns = time.time_ns()
        print("[Recording] Event camera: started logging")

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
            if pose_receiver.should_stop_recording():
                print("[Recording] Received stop signal from robot - ending recording")
                break

            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                accumulated_poses.extend(pose_receiver.get_all_poses())
                continue

            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                accumulated_poses.extend(pose_receiver.get_all_poses())
                continue

            depth_img = np.asanyarray(depth.get_data())
            color_img = np.asanyarray(color.get_data())

            depth_ds.resize(rs_idx + 1, axis=0)
            rgb_ds.resize(rs_idx + 1, axis=0)
            t_global_ds.resize(rs_idx + 1, axis=0)
            t_rgb_ds.resize(rs_idx + 1, axis=0)
            frame_num_ds.resize(rs_idx + 1, axis=0)

            t_global_ms = depth.get_timestamp()
            t_rgb_ms = color.get_timestamp()
            depth_ds[rs_idx] = depth_img
            rgb_ds[rs_idx] = color_img
            t_global_ds[rs_idx] = t_global_ms
            t_rgb_ds[rs_idx] = t_rgb_ms
            frame_num_ds[rs_idx] = depth.get_frame_number()
            frame_hw_ms_list.append(t_global_ms)

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

            accumulated_poses.extend(pose_receiver.get_all_poses())

    finally:
        if hw_sync and depth_sensor is not None:
            if depth_sensor.supports(rs.option.output_trigger_enabled):
                try:
                    depth_sensor.set_option(rs.option.output_trigger_enabled, 0)
                    print("[HWSync] RS trigger output disabled")
                except Exception as e:
                    print(f"[HWSync] Warning: could not disable trigger output: {e}")
        stop_acquire_event.set()
        event_done_event.wait()
        event_proc.join()

        # Refine event_logging_start_ns using the exact time captured in the subprocess
        try:
            event_logging_start_ns = logging_start_queue.get_nowait()
        except Exception:
            pass

        pose_receiver.stop()
        accumulated_poses.extend(pose_receiver.get_all_poses())

        recording_end_ns = time.time_ns()
        raw_poses_received = len(accumulated_poses)

        rs_video.release()
        rs_rgb_video.release()
        rs_h5.close()

        frame_times_ns = (np.array(frame_hw_ms_list) * 1e6).astype(np.int64)

        if raw_poses_received > 1 and rs_idx > 0:
            try:
                parsed = parse_raw_poses(accumulated_poses)
                with h5py.File(raw_poses_h5_file, "w") as rpf:
                    rpf.create_dataset("t_ns", data=parsed["t_ns"])
                    rpf.create_dataset("t_recv_ns", data=parsed["t_recv_ns"])
                    rpf.create_dataset("ee_T", data=parsed["ee_T"])
                    rpf.create_dataset("joint_velocity", data=parsed["joint_velocity"])
                print(f"[Recording] Raw poses saved: {len(parsed['t_ns'])} poses → {raw_poses_h5_file}")

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
            mf.attrs["hw_sync"] = hw_sync

        sync_client.send_done()

    print(f"\n[Recording] Finished '{object_name}'!")
    print(f"  RealSense frames: {rs_idx}")
    print(f"  Raw poses received: {raw_poses_received}")
    print(f"  Output directory: {object_dir}")

    print("[Recording] Decoding raw event data to HDF5...")
    generate_event_video(object_raw_dir, object_hdf5_dir)

    if hw_sync:
        print("[Recording] Extracting hardware trigger timestamps from raw event file...")
        triggers_us = extract_hw_triggers(event0_raw_file)
        print("[Recording] Aligning event frames to depth frames using hardware triggers...")
        align_event_frames_to_depth_hw(object_hdf5_dir, triggers_us, rs_idx)
    else:
        print("[Recording] Aligning event frames to depth timestamps (using global timestamps)...")
        align_event_frames_to_depth(
            object_hdf5_dir, frame_times_ns, event_logging_start_ns,
            depth_frame_offset=DEPTH_EVENT_ALIGN_OFFSET_FRAMES,
        )

    print("[Recording] Regenerating event video from aligned frames...")
    write_event_video_from_h5(object_hdf5_dir, object_video_dir)

    return True


def main(
    zmq_sync_addr: str = ZMQ_SYNC_ADDR,
    zmq_pose_addr: str = ZMQ_POSE_ADDR,
    hw_sync: bool = False,
) -> None:
    """
    Main recording loop with multi-object support.

    1. Ask for object name
    2. Initialize cameras, send "ready", wait for "start"
    3. Record until hemisphere complete
    4. Send "done", ask for next object name
    5. Repeat until user quits

    With hw_sync=True the event camera's trigger input is enabled and the
    RealSense GPIO trigger output drives per-frame alignment via hardware
    EventExtTrigger pulses.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    sync_client = ZMQSyncClient(zmq_sync_addr)
    sync_client.connect()

    print("[ClockSync] Measuring ZMQ transport delay...")
    transport_delay_ns = sync_client.measure_transport_delay(n_rounds=20)

    pose_receiver = ZMQPoseReceiver(zmq_pose_addr)

    print("[Recording] Initializing RealSense...")
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, FPS)

    depth_sensor = None
    profile = None

    try:
        profile = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.laser_power, 360)
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

        ctx = rs.context()
        for dev in ctx.query_devices():
            print(f"[Recording] Resetting device: {dev.get_info(rs.camera_info.name)}")
            dev.hardware_reset()

        time.sleep(3.0)

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.disable_all_streams()
        cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, FPS)

        profile = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.laser_power, 360)
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
            object_name = get_object_name()

            if object_name is None:
                print("\n[Recording] User requested quit. Exiting...")
                break

            success = record_single_object(
                object_name=object_name,
                pipeline=pipeline,
                pose_receiver=pose_receiver,
                sync_client=sync_client,
                data_root=DATA_ROOT,
                transport_delay_ns=transport_delay_ns,
                hw_sync=hw_sync,
                depth_sensor=depth_sensor,
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

        gc.collect()
        time.sleep(1.0)

        sync_client.close()
        print(f"\n[Recording] Session complete. Total recordings: {recording_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified single-camera recording")
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
        "--hw-sync",
        action="store_true",
        default=False,
        help=(
            "Use RealSense hardware trigger output (output_trigger_enabled) for "
            "event camera frame alignment. The event camera starts logging first, "
            "then the RealSense sends a GPIO pulse per depth frame that is recorded "
            "as an EventExtTrigger in the .raw file."
        ),
    )

    args = parser.parse_args()

    main(
        zmq_sync_addr=args.zmq_sync_addr,
        zmq_pose_addr=args.zmq_pose_addr,
        hw_sync=args.hw_sync,
    )
