#!/usr/bin/env python3
"""Multi-camera calibration with optional robot arm data collection.

Two modes:
  1. Calibration-only mode (default): Uses pre-collected frames from
     rgb_frames/ and event_frames/ in --output-dir. Requires RealSense
     to be connected for SDK intrinsics/extrinsics extraction.

  2. Robot arm collection mode (--collect-data): Runs ZMQ server to collect
     new frames via robot arm movements. Overwrites existing frames.

Usage (calibration-only, default):
    python data_recording/calibration.py --output-dir camera_data

Usage (collect new data with robot arm):
    python data_recording/calibration.py --collect-data \
        [--zmq-bind tcp://0.0.0.0:6002] [--output-dir camera_data]

Protocol for --collect-data mode (this script = REP,  agent in Docker A = REQ):
    INIT          ->  initialise, reply READY (data preserved between rounds)
    POSE_REACHED  ->  capture RS RGB, start event accumulation, reply RGB_CAPTURED
    WIGGLE_DONE   ->  stop event accumulation (static window), reply EVENT_CAPTURED
    ALL_DONE      ->  if more rounds: wait for user to reposition board,
                      reply ROUND_COMPLETE with more_rounds=True;
                      if final round: run calibration, reply CALIBRATION_COMPLETE

Outputs (saved to --output-dir):
    rgb_frames/rgb_XXX.png                       RGB images
    event_frames/event_XXX.png                   Accumulated event frames
    rs_depth_intrinsics.npz                      SDK factory depth K + distortion
    rs_rgb_intrinsics.npz                        ChArUco-calibrated RGB K + dist
    event_intrinsics.npz                         ChArUco-calibrated event K + dist
    T_color_from_depth.npz                       SDK depth -> color 4x4
    T_event_from_rgb.npz                         Stereo RGB -> event 4x4
    T_event_from_depth.npz                       Composed depth -> event 4x4
    depth_scale.npz                              RealSense depth scale
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq
import msgpack

from metavision_hal import DeviceDiscovery
from metavision_core.event_io import EventsIterator

from config import (
    FPS, RS_WIDTH, RS_HEIGHT,
    BIAS_DIFF_ON, BIAS_DIFF_OFF, BIAS_FO, BIAS_HPF, BIAS_REFR,
    CHARUCO_SQUARES_H, CHARUCO_SQUARES_V, CHARUCO_SQUARE_LEN, CHARUCO_MARKER_LEN,
)
from helpers import create_charuco_board as _make_charuco, detect_charuco


# ── ChArUco board defaults (must match the physical board) ──────────────
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_H = CHARUCO_SQUARES_H
SQUARES_V = CHARUCO_SQUARES_V
SQUARE_LEN = CHARUCO_SQUARE_LEN
MARKER_LEN = CHARUCO_MARKER_LEN

# ── RealSense stream config ───────────────────────────────────────────
RS_W, RS_H, RS_FPS = RS_WIDTH, RS_HEIGHT, FPS


# ═══════════════════════════════════════════════════════════════════════
#  Event-camera accumulator (background thread)
# ═══════════════════════════════════════════════════════════════════════
class EventAccumulator(threading.Thread):
    """Drains events continuously; collects raw events on demand."""

    def __init__(self, device):
        super().__init__(daemon=True)
        self._device = device
        geom = device.get_i_geometry()
        self.width = geom.get_width()
        self.height = geom.get_height()
        self._lock = threading.Lock()
        self._accum = False
        self._events_buffer: list[np.ndarray] = []
        self._alive = True

    def run(self):
        for events in EventsIterator.from_device(self._device, delta_t=10_000):
            if not self._alive:
                break
            if events is None or len(events) == 0:
                continue
            with self._lock:
                if self._accum:
                    self._events_buffer.append(events.copy())

    def start_accumulation(self):
        with self._lock:
            self._events_buffer.clear()
            self._accum = True

    def stop_accumulation(self) -> np.ndarray:
        """Stop accumulating and return all raw events as a structured array."""
        with self._lock:
            self._accum = False
            if self._events_buffer:
                return np.concatenate(self._events_buffer)
            return np.empty(0)

    def events_to_frame(self, raw_events: np.ndarray) -> np.ndarray:
        """Convert raw events into a normalised uint8 accumulated frame."""
        if len(raw_events) == 0:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        np.add.at(frame, (raw_events["y"], raw_events["x"]), 1)
        mx = frame.max()
        if mx > 0:
            return (frame / mx * 255).astype(np.uint8)
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def shutdown(self):
        self._alive = False


# ═══════════════════════════════════════════════════════════════════════
#  ChArUco helpers
# ═══════════════════════════════════════════════════════════════════════
def _detect(image, detector):
    """Detect ChArUco corners. Returns (corners, ids) or (None, None)."""
    return detect_charuco(image, detector, min_corners=8)


# ═══════════════════════════════════════════════════════════════════════
#  Calibration math
# ═══════════════════════════════════════════════════════════════════════
def calibrate_intrinsics(images, board, detector, label="camera"):
    """Intrinsic calibration from ChArUco detections.

    Returns (K, dist, (w, h), rms) or None.
    """
    all_c, all_i = [], []
    for i, img in enumerate(images):
        c, ids = _detect(img, detector)
        if c is not None:
            all_c.append(c)
            all_i.append(ids)
            print(f"  [{label}] image {i:3d}: {len(c)} corners detected")
        else:
            print(f"  [{label}] image {i:3d}: no detection")

    if len(all_c) < 3:
        print(f"[{label}] Too few detections for intrinsics: "
              f"{len(all_c)}/{len(images)}")
        return None

    h, w = images[0].shape[:2]
    obj_pts = [board.getChessboardCorners()[ids.flatten()] for ids in all_i]
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, all_c, (w, h), None, None
    )
    print(f"[{label}] Intrinsics  RMS={rms:.4f}  "
          f"({len(all_c)}/{len(images)} images used)")
    return K, dist, (w, h), rms


def calibrate_stereo(rgb_imgs, ev_imgs, K_rgb, d_rgb, K_ev, d_ev,
                     board, detector):
    """Stereo calibration.

    Returns (R, T) such that  p_event = R @ p_rgb + T,  or None.
    Camera 1 = RGB,  Camera 2 = event.
    """
    obj_list, pts1, pts2 = [], [], []

    for i, (rgb, ev) in enumerate(zip(rgb_imgs, ev_imgs)):
        c1, i1 = _detect(rgb, detector)
        c2, i2 = _detect(ev, detector)
        if c1 is None or c2 is None:
            continue

        common = np.intersect1d(i1.flatten(), i2.flatten())
        if len(common) < 8:
            continue

        m1 = np.isin(i1.flatten(), common)
        m2 = np.isin(i2.flatten(), common)
        o1 = np.argsort(i1.flatten()[m1])
        o2 = np.argsort(i2.flatten()[m2])

        obj_all = board.getChessboardCorners()
        obj_list.append(obj_all[np.sort(common)].astype(np.float32))
        pts1.append(c1[m1][o1].reshape(-1, 1, 2).astype(np.float32))
        pts2.append(c2[m2][o2].reshape(-1, 1, 2).astype(np.float32))

    if len(obj_list) < 3:
        print(f"[stereo] Too few matched pairs: {len(obj_list)}")
        return None

    h_rgb, w_rgb = rgb_imgs[0].shape[:2]
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_list, pts1, pts2,
        K_rgb, d_rgb, K_ev, d_ev, (w_rgb, h_rgb),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"[stereo] Stereo  RMS={rms:.4f}  ({len(obj_list)} pairs)")
    return R, T


# ── pyrealsense2 conversion helpers ──────────────────────────────────
def _rs_extrinsics_to_4x4(extr) -> np.ndarray:
    """Convert RealSense extrinsics to 4x4 transformation matrix.
    
    CRITICAL: RealSense stores rotation as column-major 3x3 matrix in a
    9-element array. Must use Fortran-order reshape or transpose.
    """
    T = np.eye(4)
    # RealSense rotation is column-major, so use order='F' for correct reshape
    T[:3, :3] = np.array(extr.rotation).reshape(3, 3, order='F')
    T[:3, 3] = np.array(extr.translation).flatten()
    return T


def _rs_intrinsics_to_Kd(intr):
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    return K, dist, (intr.width, intr.height)


# ─── Hand-eye helpers ────────────────────────────────────────────────
def _pose7_to_4x4(pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to a 4x4 transformation matrix."""
    x, y, z, qx, qy, qz, qw = pose
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1 - 2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1 - 2*(qx*qx+qy*qy)],
    ])
    T[:3, 3] = [x, y, z]
    return T


def _run_hand_eye_calibration(
    rgb_images: list[np.ndarray],
    ee_poses: list[np.ndarray],
    K_rgb: np.ndarray,
    d_rgb: np.ndarray,
    board,
    det,
) -> np.ndarray | None:
    """Compute T_rgb_from_ee via cv2.calibrateHandEye.

    All ee_poses must be 4x4 base->EE transformation matrices.
    Returns 4x4 T such that  p_rgb = T @ p_ee,  or None on failure.
    """
    n = min(len(rgb_images), len(ee_poses))
    if n == 0:
        print("[hand-eye] No EE poses – skipping hand-eye calibration")
        return None
    print(f"[hand-eye] Using {n} image/pose pairs for hand-eye")

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for i in range(n):
        corners, ids = _detect(rgb_images[i], det)
        if corners is None:
            continue
        obj_pts = board.getChessboardCorners()[ids.flatten()].astype(np.float32)
        img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_rgb, d_rgb)
        if not ok:
            continue
        R_board2cam, _ = cv2.Rodrigues(rvec)
        T_ee = ee_poses[i]
        R_g2b.append(T_ee[:3, :3].copy())
        t_g2b.append(T_ee[:3, 3:4].copy())
        R_t2c.append(R_board2cam)
        t_t2c.append(tvec)

    if len(R_g2b) < 3:
        print(f"[hand-eye] Too few valid pairs: {len(R_g2b)} (need >= 3) – skipping")
        return None

    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
    )
    T_cam2ee = np.eye(4)
    T_cam2ee[:3, :3] = R_cam2ee
    T_cam2ee[:3, 3] = t_cam2ee.flatten()
    T_rgb_from_ee = np.linalg.inv(T_cam2ee)
    t_cm = t_cam2ee.flatten() * 100
    print(
        f"[hand-eye] {len(R_g2b)} pairs  |  "
        f"cam-in-EE: [{t_cm[0]:+.2f}, {t_cm[1]:+.2f}, {t_cm[2]:+.2f}] cm"
    )
    return T_rgb_from_ee


def _load_ee_poses(
    output_dir: Path,
    calibration_poses_file: Path | None,
) -> list[np.ndarray]:
    """Load EE poses for hand-eye calibration.

    Priority:
      1. output_dir/ee_poses.npy  – actual measured poses saved by --collect-data
         (shape N x 4 x 4)
      2. calibration_poses_file   – commanded target poses [x,y,z,qx,qy,qz,qw]
         (shape N x 7, lower accuracy than measured poses)

    Returns a list of 4x4 base->EE transformation matrices.
    """
    p = output_dir / "ee_poses.npy"
    if p.exists():
        poses = np.load(str(p))  # (N, 4, 4)
        print(f"[hand-eye] Loaded {len(poses)} actual EE poses from {p}")
        return list(poses)

    if calibration_poses_file is not None:
        cp = Path(calibration_poses_file)
        if cp.exists():
            raw = np.load(str(cp))  # (N, 7)
            poses = [_pose7_to_4x4(row) for row in raw]
            print(
                f"[hand-eye] Loaded {len(poses)} commanded poses from {cp}  "
                f"(approximate – actual measured poses preferred)"
            )
            return poses
        else:
            print(f"[hand-eye] WARNING: --calibration-poses file not found: {cp}")

    return []


# ═══════════════════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════════════════
# Duration of the server-side event accumulation window (robot stays still).
EVENT_CAPTURE_DURATION = 0.100  # 100 ms


class CalibrationRecordingServer:
    def __init__(self, bind_addr: str, output_dir: str):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

        # Clear stale data from previous runs
        for npz in self.output.glob("*.npz"):
            npz.unlink()
        for subdir in ("rgb_frames", "event_frames"):
            p = self.output / subdir
            if p.exists():
                shutil.rmtree(p)
        print(f"[server] Cleared old calibration data in {self.output}")

        self._img_idx = 0

        # ZMQ REP
        self._ctx = zmq.Context()
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(bind_addr)

        # Cameras
        self._rs_pipe: rs.pipeline | None = None
        self._event_acc: EventAccumulator | None = None

        # SDK calibration data
        self._K_depth_sdk = None
        self._d_depth_sdk = None
        self._sz_depth = None
        self._K_color_sdk = None
        self._d_color_sdk = None
        self._sz_color = None
        self._T_color_from_depth = None
        self._depth_scale = None

        # Collected per-pose data
        self._rgb_images: list[np.ndarray] = []
        self._ev_frames: list[np.ndarray] = []
        self._ee_poses: list[np.ndarray] = []

        # ChArUco
        self._board, self._det = _make_charuco()

        # Output directories
        self._rgb_dir = self.output / "rgb_frames"
        self._event_frames_dir = self.output / "event_frames"
        self._rgb_dir.mkdir(parents=True, exist_ok=True)
        self._event_frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"[server] ZMQ REP bound on {bind_addr}")
        print(f"[server] Output directory: {self.output}")

    # ── camera init ─────────────────────────────────────────────────
    def _start_realsense(self):
        """Start RealSense pipeline with warmup; hardware-reset and retry once on failure."""
        import time as _time

        def _try_start():
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.disable_all_streams()
            cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
            cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
            profile = pipe.start(cfg)
            print("[RS] Warming up...")
            for _ in range(30):
                pipe.wait_for_frames(timeout_ms=1000)
            return pipe, profile

        print("[RS] Initializing RealSense...")
        try:
            pipe, profile = _try_start()
        except Exception as e:
            print(f"[RS] Startup failed: {e} – performing hardware reset and retrying...")
            try:
                rs.pipeline().stop()
            except Exception:
                pass
            ctx = rs.context()
            for dev in ctx.query_devices():
                print(f"[RS] Resetting: {dev.get_info(rs.camera_info.name)}")
                dev.hardware_reset()
            _time.sleep(3.0)
            pipe, profile = _try_start()

        c_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        d_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        self._K_color_sdk, self._d_color_sdk, self._sz_color = (
            _rs_intrinsics_to_Kd(c_prof.get_intrinsics())
        )
        self._K_depth_sdk, self._d_depth_sdk, self._sz_depth = (
            _rs_intrinsics_to_Kd(d_prof.get_intrinsics())
        )
        self._T_color_from_depth = _rs_extrinsics_to_4x4(
            d_prof.get_extrinsics_to(c_prof)
        )
        self._depth_scale = (
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        self._rs_pipe = pipe
        print(
            f"[RS]  color {self._sz_color}  depth {self._sz_depth}  "
            f"scale={self._depth_scale:.6f}"
        )

    def _setup_cameras(self):
        # --- RealSense ---
        self._start_realsense()

        # --- Event camera ---
        devs = DeviceDiscovery.list()
        if not devs:
            print("[event] WARNING: no event camera – event calibration will be skipped")
            return
        dev = DeviceDiscovery.open(devs[0])
        biases = dev.get_i_ll_biases()
        for name, val in [
            ("bias_diff_on", BIAS_DIFF_ON),
            ("bias_diff_off", BIAS_DIFF_OFF),
            ("bias_fo", BIAS_FO),
            ("bias_hpf", BIAS_HPF),
            ("bias_refr", BIAS_REFR),
        ]:
            try:
                biases.set(name, val)
            except Exception:
                pass
        self._event_acc = EventAccumulator(dev)
        self._event_acc.start()
        print(
            f"[event] {self._event_acc.width}x{self._event_acc.height}  "
            f"biases applied"
        )

    # ── RS capture ─────────────────────────────────────────────────
    def _capture_rgb(self) -> np.ndarray:
        for _ in range(5):
            self._rs_pipe.wait_for_frames()
        frames = self._rs_pipe.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy()  # BGR uint8 – copy out of SDK buffer

    # ── ZMQ reply helper ───────────────────────────────────────────
    def _reply(self, msg: dict):
        self._rep.send(msgpack.packb(msg, use_bin_type=True))

    # ── main serve loop ────────────────────────────────────────────
    def serve(self):
        self._setup_cameras()
        print("[server] Cameras ready – waiting for agent commands ...")

        while True:
            raw = self._rep.recv()
            msg = msgpack.unpackb(raw, raw=False)
            cmd = msg.get("cmd", "")

            if cmd == "INIT":
                n = msg.get("num_poses", "?")
                print(f"\n[server] INIT  ({n} poses)")
                self._rgb_images.clear()
                self._ev_frames.clear()
                self._ee_poses.clear()
                self._img_idx = 0
                self._reply({"status": "READY"})

            elif cmd == "POSE_REACHED":
                idx = msg.get("index", -1)
                ee_flat = msg.get("ee_pose")
                if ee_flat is not None:
                    self._ee_poses.append(
                        np.array(ee_flat, dtype=np.float64).reshape(4, 4)
                    )

                # Capture RGB
                bgr = self._capture_rgb()
                self._rgb_images.append(bgr)
                rgb_out = self._rgb_dir / f"rgb_{self._img_idx:03d}.png"
                cv2.imwrite(str(rgb_out), bgr)
                c, _ = _detect(bgr, self._det)
                rgb_found = c is not None

                # Accumulate events over fixed server-side window (robot stays still)
                if self._event_acc is not None:
                    self._event_acc.start_accumulation()
                    time.sleep(EVENT_CAPTURE_DURATION)
                    raw_events = self._event_acc.stop_accumulation()
                    n_events = len(raw_events)
                    ev_frame = self._event_acc.events_to_frame(raw_events)
                    ev_frame = cv2.rotate(ev_frame, cv2.ROTATE_180)  # camera mounted upside-down
                    self._ev_frames.append(ev_frame)
                    img_out = self._event_frames_dir / f"event_{self._img_idx:03d}.png"
                    cv2.imwrite(str(img_out), ev_frame)
                    c_ev, _ = _detect(ev_frame, self._det)
                    ev_found = c_ev is not None
                    print(
                        f"[server] POSE_REACHED #{idx}  "
                        f"RGB={'yes' if rgb_found else 'NO'}  "
                        f"{n_events} events  event={'yes' if ev_found else 'NO'}"
                    )
                else:
                    print(f"[server] POSE_REACHED #{idx}  RGB={'yes' if rgb_found else 'NO'}  (no event camera)")

                self._img_idx += 1
                self._reply({"status": "CAPTURE_COMPLETE", "charuco_rgb": rgb_found})

            elif cmd == "ALL_DONE":
                print(f"\n[server] ALL_DONE — {len(self._rgb_images)} RGB, {len(self._ev_frames)} event images")
                print("[server] Running calibration pipeline ...")
                success, error = self._run_calibration()
                self._reply({
                    "status": "CALIBRATION_COMPLETE",
                    "success": success,
                    "error": error,
                })
                break

            else:
                print(f"[server] Unknown command: {cmd}")
                self._reply({"status": "ERROR", "error": f"unknown cmd: {cmd}"})

        self._cleanup()

    # ── calibration pipeline ───────────────────────────────────────
    def _run_calibration(self) -> tuple[bool, str]:
        try:
            ok = self._calibrate()
            return ok, "" if ok else "calibration step failed (see logs)"
        except Exception as e:
            traceback.print_exc()
            return False, str(e)

    def _calibrate(self) -> bool:
        out = self.output

        # 1. SDK depth intrinsics + depth->color extrinsics + depth scale
        np.savez(
            str(out / "rs_depth_intrinsics.npz"),
            camera_matrix=self._K_depth_sdk,
            dist_coeffs=self._d_depth_sdk,
            image_size=np.array(self._sz_depth),
        )
        np.savez(
            str(out / "T_color_from_depth.npz"),
            T=self._T_color_from_depth,
        )
        np.savez(str(out / "depth_scale.npz"), scale=self._depth_scale)
        print("[cal] Saved SDK depth intrinsics + T_color_from_depth + depth_scale")

        # 2. RS RGB intrinsics (ChArUco)
        print("\n=== RGB intrinsic calibration ===")
        rgb_res = calibrate_intrinsics(
            self._rgb_images, self._board, self._det, label="RS-RGB"
        )
        if rgb_res is None:
            print("[cal] RGB intrinsic calibration failed")
            return False
        K_rgb, d_rgb, sz_rgb, rms_rgb = rgb_res
        np.savez(
            str(out / "rs_rgb_intrinsics.npz"),
            camera_matrix=K_rgb,
            dist_coeffs=d_rgb,
            image_size=np.array(sz_rgb),
            rms=rms_rgb,
        )
        print(f"[cal] Saved rs_rgb_intrinsics.npz  (RMS={rms_rgb:.4f})")

        # 2b. Eye-in-hand calibration (RGB camera → end-effector)
        print("\n=== Eye-in-hand calibration ===")
        # Persist actual measured poses so standalone recalibration can use them
        if self._ee_poses:
            np.save(str(out / "ee_poses.npy"), np.array(self._ee_poses))
            print(f"[cal] Saved ee_poses.npy  ({len(self._ee_poses)} actual measured poses)")
        T_rgb_from_ee = self._calibrate_hand_eye(K_rgb, d_rgb)
        if T_rgb_from_ee is not None:
            np.savez(str(out / "T_rgb_from_ee.npz"), T=T_rgb_from_ee)
            print("[cal] Saved T_rgb_from_ee.npz")

        # 3. Event-camera intrinsics (ChArUco)
        if not self._ev_frames:
            print("[cal] No event frames – skipping event + stereo calibration")
            try:
                save_calibration_visualizations(out)
            except Exception:
                traceback.print_exc()
            return True

        print("\n=== Event intrinsic calibration ===")
        ev_res = calibrate_intrinsics(
            self._ev_frames, self._board, self._det, label="Event"
        )
        if ev_res is None:
            print("[cal] Event intrinsic calibration failed")
            return False
        K_ev, d_ev, sz_ev, rms_ev = ev_res
        np.savez(
            str(out / "event_intrinsics.npz"),
            camera_matrix=K_ev,
            dist_coeffs=d_ev,
            image_size=np.array(sz_ev),
            rms=rms_ev,
        )
        print(f"[cal] Saved event_intrinsics.npz  (RMS={rms_ev:.4f})")

        # 4. Stereo RGB <-> event
        n_pairs = min(len(self._rgb_images), len(self._ev_frames))
        print(f"\n=== Stereo calibration ({n_pairs} pairs) ===")
        stereo = calibrate_stereo(
            self._rgb_images[:n_pairs], self._ev_frames[:n_pairs],
            K_rgb, d_rgb, K_ev, d_ev,
            self._board, self._det,
        )
        if stereo is None:
            print("[cal] Stereo calibration failed")
            return False
        R, T = stereo
        T_event_from_rgb = np.eye(4)
        T_event_from_rgb[:3, :3] = R
        T_event_from_rgb[:3, 3] = T.flatten()
        np.savez(
            str(out / "T_event_from_rgb.npz"),
            T=T_event_from_rgb,
            R=R,
            t=T,
        )
        print("[cal] Saved T_event_from_rgb.npz")

        # 5. Compose T_event_from_depth = T_event_from_rgb @ T_color_from_depth
        T_event_from_depth = T_event_from_rgb @ self._T_color_from_depth
        np.savez(
            str(out / "T_event_from_depth.npz"), T=T_event_from_depth
        )
        print("[cal] Saved T_event_from_depth.npz  (composed)")

        try:
            save_calibration_visualizations(out)
        except Exception:
            traceback.print_exc()
        return True

    # ── eye-in-hand calibration ──────────────────────────────────────
    def _calibrate_hand_eye(
        self, K_rgb: np.ndarray, d_rgb: np.ndarray
    ) -> np.ndarray | None:
        return _run_hand_eye_calibration(
            self._rgb_images, self._ee_poses, K_rgb, d_rgb, self._board, self._det
        )

    # ── cleanup ─────────────────────────────────────────────────────
    def _cleanup(self):
        if self._rs_pipe:
            self._rs_pipe.stop()
        if self._event_acc:
            self._event_acc.shutdown()
        if hasattr(self, '_rep') and self._rep:
            self._rep.close()
        print("[server] Cleanup done")


# ═══════════════════════════════════════════════════════════════════════
#  Standalone calibration (no robot arm)
# ═══════════════════════════════════════════════════════════════════════
def load_frames_from_disk(output_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load pre-collected RGB and event frames from disk.
    
    Returns (rgb_images, event_frames) as lists of numpy arrays.
    """
    rgb_dir = output_dir / "rgb_frames"
    event_dir = output_dir / "event_frames"
    
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB frames directory not found: {rgb_dir}")
    
    rgb_images = []
    event_frames = []
    
    # Load RGB frames
    rgb_files = sorted(rgb_dir.glob("rgb_*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    
    print(f"[load] Loading {len(rgb_files)} RGB frames from {rgb_dir}")
    for rgb_file in rgb_files:
        img = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
        if img is not None:
            rgb_images.append(img)
        else:
            print(f"[load] WARNING: Failed to load {rgb_file}")
    
    # Load event frames (optional)
    if event_dir.exists():
        event_files = sorted(event_dir.glob("event_*.png"))
        if event_files:
            print(f"[load] Loading {len(event_files)} event frames from {event_dir}")
            for event_file in event_files:
                img = cv2.imread(str(event_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    event_frames.append(img)
                else:
                    print(f"[load] WARNING: Failed to load {event_file}")
        else:
            print(f"[load] No event frames found in {event_dir}")
    else:
        print(f"[load] Event frames directory not found: {event_dir} (skipping event calibration)")
    
    return rgb_images, event_frames


def run_standalone_calibration(
    output_dir: Path,
    max_images: int | None = None,
    calibration_poses_file: Path | None = None,
) -> bool:
    """Run calibration using pre-collected frames (no robot arm required).

    Requires RealSense to be connected for SDK intrinsics/extrinsics extraction.
    """
    print("\n" + "="*60)
    print("CALIBRATION-ONLY MODE")
    print("="*60)
    print(f"Output directory: {output_dir}\n")
    
    # Load frames from disk
    try:
        rgb_images, event_frames = load_frames_from_disk(output_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nTo collect calibration data, run with --collect-data flag.")
        return False

    if max_images is not None:
        rgb_images = rgb_images[:max_images]
        event_frames = event_frames[:max_images]
        print(f"[cal] Using first {max_images} images (--max-images)")
    
    print(f"[cal] Loaded {len(rgb_images)} RGB images, {len(event_frames)} event frames\n")
    
    # Initialize RealSense to get SDK intrinsics/extrinsics
    print("[RS] Connecting to RealSense for SDK calibration data...")
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.disable_all_streams()
        cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
        cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
        profile = pipe.start(cfg)
        
        # Extract SDK calibration data
        c_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        d_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        
        K_depth_sdk, d_depth_sdk, sz_depth = _rs_intrinsics_to_Kd(d_prof.get_intrinsics())
        K_color_sdk, d_color_sdk, sz_color = _rs_intrinsics_to_Kd(c_prof.get_intrinsics())
        T_color_from_depth = _rs_extrinsics_to_4x4(d_prof.get_extrinsics_to(c_prof))
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        pipe.stop()
        print(f"[RS] Extracted SDK calibration data (depth scale={depth_scale:.6f})")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to connect to RealSense: {e}")
        print("RealSense camera must be connected for calibration (SDK intrinsics/extrinsics needed).")
        return False
    
    # Clear stale .npz outputs before writing new ones
    for npz in output_dir.glob("*.npz"):
        npz.unlink()
    print("[cal] Cleared old .npz files")

    # Save SDK data
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_dir / "rs_depth_intrinsics.npz"),
        camera_matrix=K_depth_sdk,
        dist_coeffs=d_depth_sdk,
        image_size=np.array(sz_depth),
    )
    np.savez(str(output_dir / "T_color_from_depth.npz"), T=T_color_from_depth)
    np.savez(str(output_dir / "depth_scale.npz"), scale=depth_scale)
    print("[cal] Saved SDK depth intrinsics + T_color_from_depth + depth_scale")
    
    # ChArUco board setup
    board, det = _make_charuco()
    
    # RGB intrinsic calibration
    print("\n=== RGB intrinsic calibration ===")
    rgb_res = calibrate_intrinsics(rgb_images, board, det, label="RS-RGB")
    if rgb_res is None:
        print("[cal] RGB intrinsic calibration failed")
        return False
    K_rgb, d_rgb, sz_rgb, rms_rgb = rgb_res
    np.savez(
        str(output_dir / "rs_rgb_intrinsics.npz"),
        camera_matrix=K_rgb,
        dist_coeffs=d_rgb,
        image_size=np.array(sz_rgb),
        rms=rms_rgb,
    )
    print(f"[cal] Saved rs_rgb_intrinsics.npz  (RMS={rms_rgb:.4f})")

    # 2b. Eye-in-hand calibration (RGB camera → end-effector)
    print("\n=== Eye-in-hand calibration ===")
    ee_poses = _load_ee_poses(output_dir, calibration_poses_file)
    if max_images is not None:
        ee_poses = ee_poses[:max_images]
    if ee_poses:
        T_rgb_from_ee = _run_hand_eye_calibration(rgb_images, ee_poses, K_rgb, d_rgb, board, det)
        if T_rgb_from_ee is not None:
            np.savez(str(output_dir / "T_rgb_from_ee.npz"), T=T_rgb_from_ee)
            print("[cal] Saved T_rgb_from_ee.npz")
    else:
        print("[hand-eye] No EE poses found – run with --collect-data first, or pass --calibration-poses")

    # Event camera intrinsic calibration (if event frames exist)
    if event_frames:
        print("\n=== Event intrinsic calibration ===")
        ev_res = calibrate_intrinsics(event_frames, board, det, label="Event")
        if ev_res is None:
            print("[cal] Event intrinsic calibration failed")
            return False
        K_ev, d_ev, sz_ev, rms_ev = ev_res
        np.savez(
            str(output_dir / "event_intrinsics.npz"),
            camera_matrix=K_ev,
            dist_coeffs=d_ev,
            image_size=np.array(sz_ev),
            rms=rms_ev,
        )
        print(f"[cal] Saved event_intrinsics.npz  (RMS={rms_ev:.4f})")
        
        # Stereo calibration RGB <-> event
        n_pairs = min(len(rgb_images), len(event_frames))
        print(f"\n=== Stereo calibration ({n_pairs} pairs) ===")
        stereo = calibrate_stereo(
            rgb_images[:n_pairs], event_frames[:n_pairs],
            K_rgb, d_rgb, K_ev, d_ev,
            board, det,
        )
        if stereo is None:
            print("[cal] Stereo calibration failed")
            return False
        R, T = stereo
        T_event_from_rgb = np.eye(4)
        T_event_from_rgb[:3, :3] = R
        T_event_from_rgb[:3, 3] = T.flatten()
        np.savez(
            str(output_dir / "T_event_from_rgb.npz"),
            T=T_event_from_rgb,
            R=R,
            t=T,
        )
        print("[cal] Saved T_event_from_rgb.npz")
        
        # Compose T_event_from_depth
        T_event_from_depth = T_event_from_rgb @ T_color_from_depth
        np.savez(str(output_dir / "T_event_from_depth.npz"), T=T_event_from_depth)
        print("[cal] Saved T_event_from_depth.npz  (composed)")
    else:
        print("\n[cal] No event frames – skipping event + stereo calibration")

    try:
        save_calibration_visualizations(output_dir)
    except Exception:
        traceback.print_exc()
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Calibration visualization
# ═══════════════════════════════════════════════════════════════════════
def save_calibration_visualizations(output_dir: Path) -> None:
    """Save 3D and top-down views of calibrated camera / EE positions.

    All positions are expressed in the **RGB camera frame**
    (OpenCV convention: X right, Y down, Z forward).

    Cameras drawn
    -------------
    RGB   – identity (reference frame)
    Depth – T_color_from_depth  (depth → RGB)
    Event – inv(T_event_from_rgb)  (event pose in RGB frame)
    EE    – T_rgb_from_ee  (EE origin in RGB frame)

    Missing transforms are silently skipped.

    Outputs
    -------
    output_dir/calibration_visualization/camera_geometry_3d.png
    output_dir/calibration_visualization/camera_geometry_topdown.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    vis_dir = output_dir / "calibration_visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── collect available transforms (all in RGB camera frame) ──────
    cameras: list[tuple[str, np.ndarray, str]] = []

    # RGB camera = reference frame (identity)
    cameras.append(("RGB", np.eye(4), "#2196F3"))

    # Depth camera: T_color_from_depth maps depth→RGB
    p = output_dir / "T_color_from_depth.npz"
    if p.exists():
        cameras.append(("Depth", np.load(str(p))["T"], "#FF9800"))

    # Event camera: inv(T_event_from_rgb) gives event pose in RGB frame
    p = output_dir / "T_event_from_rgb.npz"
    if p.exists():
        T_evfr = np.load(str(p))["T"]
        cameras.append(("Event", np.linalg.inv(T_evfr), "#4CAF50"))

    # EE: T_rgb_from_ee maps EE → RGB
    p = output_dir / "T_rgb_from_ee.npz"
    if p.exists():
        cameras.append(("EE", np.load(str(p))["T"], "#F44336"))

    if len(cameras) < 2:
        print("[viz] Not enough transforms available – skipping visualization")
        return

    positions = np.array([T[:3, 3] for _, T, _ in cameras])

    # Arrow length = 25% of bounding-box span, at least 1 cm
    span = float(np.ptp(positions, axis=0).max())
    arrow_len = max(span * 0.25, 0.01)

    # ── 3D figure ────────────────────────────────────────────────────
    def _draw_frame_3d(ax, T, label, color):
        o = T[:3, 3]
        R = T[:3, :3]
        for i, ac in enumerate(["#c0392b", "#27ae60", "#2980b9"]):  # X Y Z
            e = o + R[:, i] * arrow_len
            ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color=ac, lw=1.8)
        ax.scatter(*o, color=color, s=80, zorder=5, depthshade=False)
        ax.text(o[0], o[1], o[2], f"  {label}", color=color,
                fontsize=10, fontweight="bold")

    fig3d = plt.figure(figsize=(9, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")
    for label, T, color in cameras:
        _draw_frame_3d(ax3d, T, label, color)

    center = positions.mean(axis=0)
    half = max(span * 0.65, arrow_len * 2)
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_xlabel("X  (m)")
    ax3d.set_ylabel("Y  (m)")
    ax3d.set_zlabel("Z  (m)")
    ax3d.set_title(
        "Camera geometry – 3D view  (RGB camera frame)\n"
        "frame axes:  X = red   Y = green   Z = blue"
    )
    ax3d.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=10, label=lbl)
            for lbl, _, c in cameras
        ],
        loc="upper left",
    )
    fig3d.tight_layout()
    out3d = vis_dir / "camera_geometry_3d.png"
    fig3d.savefig(str(out3d), dpi=150)
    plt.close(fig3d)
    print(f"[viz] Saved 3D view      → {out3d}")

    # ── top-down figure (XZ plane, Y axis suppressed) ─────────────
    def _draw_frame_2d(ax, T, label, color):
        """Project onto XY plane (X right, Y up/down)."""
        o = T[:3, 3]
        R = T[:3, :3]
        x, y = o[0], o[1]
        # X column of rotation (red)
        ax.annotate("", xy=(x + R[0, 0] * arrow_len, y + R[1, 0] * arrow_len),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.8))
        # Y column of rotation (green)
        ax.annotate("", xy=(x + R[0, 1] * arrow_len, y + R[1, 1] * arrow_len),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.8))
        ax.scatter(x, y, color=color, s=100, zorder=5)
        ax.text(x, y, f"  {label}", color=color,
                fontsize=10, fontweight="bold")

    fig2d, ax2d = plt.subplots(figsize=(8, 7))
    for label, T, color in cameras:
        _draw_frame_2d(ax2d, T, label, color)

    ax2d.set_xlabel("X  (m)  →  right")
    ax2d.set_ylabel("Y  (m)  →  down")
    ax2d.set_title(
        "Camera geometry – top-down view  (XY plane, RGB camera frame)\n"
        "frame arrows:  X = red   Y = green"
    )
    ax2d.set_aspect("equal", adjustable="datalim")
    ax2d.grid(True, alpha=0.3)
    ax2d.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markersize=10, label=lbl)
            for lbl, _, c in cameras
        ],
        loc="best",
    )
    fig2d.tight_layout()
    out2d = vis_dir / "camera_geometry_topdown.png"
    fig2d.savefig(str(out2d), dpi=150)
    plt.close(fig2d)
    print(f"[viz] Saved top-down view → {out2d}")

    save_reprojection_visualizations(output_dir)


# ═══════════════════════════════════════════════════════════════════════
#  Per-image reprojection error visualization
# ═══════════════════════════════════════════════════════════════════════
def _draw_reprojection_on_image(
    img: np.ndarray,
    detected: np.ndarray,
    projected: np.ndarray,
    rms: float,
    label: str,
) -> np.ndarray:
    """Return a BGR image with detected (green dots) and projected (red circles) corners.

    Yellow lines connect each detected–projected pair to show displacement.
    An error badge is drawn in the top-right corner.
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    det = detected.reshape(-1, 2)
    proj = projected.reshape(-1, 2)

    # Yellow displacement lines
    for d, p in zip(det, proj):
        cv2.line(out, (int(d[0]), int(d[1])), (int(p[0]), int(p[1])),
                 (0, 220, 220), 1)
    # Green filled dots – detected
    for pt in det:
        cv2.circle(out, (int(pt[0]), int(pt[1])), 5, (0, 210, 0), -1)
    # Red open circles – projected
    for pt in proj:
        cv2.circle(out, (int(pt[0]), int(pt[1])), 5, (0, 0, 220), 2)

    # Error badge top-right
    h, w = out.shape[:2]
    text = f"{label}  RMS={rms:.3f}px"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    x1 = w - tw - 2 * pad
    y1 = 0
    cv2.rectangle(out, (x1, y1), (w, th + 2 * pad + baseline), (30, 30, 30), -1)
    cv2.putText(out, text, (x1 + pad, th + pad), font, scale, (255, 255, 255), thick)
    return out


def save_reprojection_visualizations(output_dir: Path) -> None:
    """Save a side-by-side reprojection image for every captured pose.

    For each frame index, draws detected ChArUco corners (green) and
    projected corners (red) using the calibrated intrinsics, with
    per-image RMS reprojection error shown in the top-right corner.

    Reads from:
        output_dir/rs_rgb_intrinsics.npz
        output_dir/event_intrinsics.npz   (optional)
        output_dir/rgb_frames/rgb_NNN.png
        output_dir/event_frames/event_NNN.png  (optional)

    Saves to:
        output_dir/calibration_visualization/reprojection_NNN.png
        (side-by-side RGB | Event when both cameras available)
    """
    vis_dir = output_dir / "calibration_visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Intrinsics ───────────────────────────────────────────────────
    p = output_dir / "rs_rgb_intrinsics.npz"
    if not p.exists():
        print("[viz] rs_rgb_intrinsics.npz missing – skipping reprojection viz")
        return
    d = np.load(str(p))
    K_rgb, dist_rgb = d["camera_matrix"], d["dist_coeffs"]

    K_ev = dist_ev = None
    p = output_dir / "event_intrinsics.npz"
    if p.exists():
        d = np.load(str(p))
        K_ev, dist_ev = d["camera_matrix"], d["dist_coeffs"]

    # ── Image lists ───────────────────────────────────────────────────
    rgb_dir = output_dir / "rgb_frames"
    ev_dir = output_dir / "event_frames"
    rgb_files = sorted(rgb_dir.glob("rgb_*.png")) if rgb_dir.exists() else []
    ev_files = sorted(ev_dir.glob("event_*.png")) if (ev_dir.exists() and K_ev is not None) else []

    if not rgb_files:
        print("[viz] No RGB frames – skipping reprojection viz")
        return

    board, det_board = _make_charuco()

    def _reproject_one(img: np.ndarray, K: np.ndarray, dist: np.ndarray,
                       label: str) -> np.ndarray:
        """Detect corners, solvePnP, projectPoints; return annotated BGR frame."""
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        corners, ids = _detect(img, det_board)
        if corners is None:
            out = bgr.copy()
            h, w = out.shape[:2]
            cv2.rectangle(out, (w - 240, 0), (w, 30), (30, 30, 30), -1)
            cv2.putText(out, f"{label}  no detection",
                        (w - 235, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (100, 100, 255), 2)
            return out

        obj_pts = board.getChessboardCorners()[ids.flatten()].astype(np.float32)
        img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        if not ok:
            return bgr

        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        errs = np.linalg.norm(
            img_pts.reshape(-1, 2) - projected.reshape(-1, 2), axis=1
        )
        rms = float(np.sqrt(np.mean(errs ** 2)))
        return _draw_reprojection_on_image(bgr, img_pts, projected, rms, label)

    n_total = max(len(rgb_files), len(ev_files))
    n_saved = 0
    for i in range(n_total):
        panels = []

        if i < len(rgb_files):
            rgb_img = cv2.imread(str(rgb_files[i]), cv2.IMREAD_COLOR)
            if rgb_img is not None:
                panels.append(_reproject_one(rgb_img, K_rgb, dist_rgb, "RGB"))

        if i < len(ev_files):
            ev_img = cv2.imread(str(ev_files[i]), cv2.IMREAD_GRAYSCALE)
            if ev_img is not None:
                panels.append(_reproject_one(ev_img, K_ev, dist_ev, "Event"))

        if not panels:
            continue

        if len(panels) == 2:
            h0, h1 = panels[0].shape[0], panels[1].shape[0]
            if h0 != h1:
                scale = h0 / h1
                panels[1] = cv2.resize(
                    panels[1], (int(panels[1].shape[1] * scale), h0)
                )
            frame = np.concatenate(panels, axis=1)
        else:
            frame = panels[0]

        # Divider line between panels
        if len(panels) == 2:
            mid = panels[0].shape[1]
            cv2.line(frame, (mid, 0), (mid, frame.shape[0]), (200, 200, 200), 2)

        out_path = vis_dir / f"reprojection_{i:03d}.png"
        cv2.imwrite(str(out_path), frame)
        n_saved += 1

    print(f"[viz] Saved {n_saved} reprojection image(s) → {vis_dir}/reprojection_NNN.png")


# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Multi-camera calibration with optional robot arm data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate using existing frames (default)
  python data_recording/calibration.py --output-dir camera_data

  # Collect new data with robot arm (overwrites existing frames)
  python data_recording/calibration.py --collect-data --output-dir camera_data
        """
    )
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Enable robot arm data collection mode (requires ZMQ backend). "
             "Overwrites existing frames in output-dir.",
    )
    parser.add_argument(
        "--zmq-bind",
        default="tcp://0.0.0.0:6002",
        help="ZMQ REP bind address for --collect-data mode (default: tcp://0.0.0.0:6002)",
    )
    parser.add_argument(
        "--output-dir",
        default=_PROJECT_DIR / "camera_data",
        help="Directory to save/load calibration data (default: camera_data)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N image pairs for calibration (default: use all)",
    )
    parser.add_argument(
        "--calibration-poses",
        default=None,
        metavar="FILE",
        help="Path to calibration_poses.npy for hand-eye calibration in standalone mode. "
             "Used as fallback when ee_poses.npy is not present in --output-dir. "
             "Note: commanded poses are less accurate than the actual measured poses "
             "saved automatically by --collect-data (default: None)",
    )
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = _PROJECT_DIR / output_path
    calibration_poses_path = None
    if args.calibration_poses:
        calibration_poses_path = Path(args.calibration_poses)
        if not calibration_poses_path.is_absolute():
            calibration_poses_path = _PROJECT_DIR / calibration_poses_path
    
    if args.collect_data:
        # Robot arm collection mode
        print("\n" + "="*60)
        print("ROBOT ARM DATA COLLECTION MODE")
        print("="*60)
        print(f"Output directory: {output_path}")
        print(f"ZMQ bind address: {args.zmq_bind}\n")
        
        server = CalibrationRecordingServer(
            bind_addr=args.zmq_bind,
            output_dir=str(output_path),
        )
        server.serve()
    else:
        # Standalone calibration mode (default)
        success = run_standalone_calibration(
            output_path,
            max_images=args.max_images,
            calibration_poses_file=calibration_poses_path,
        )
        if not success:
            exit(1)


if __name__ == "__main__":
    main()
