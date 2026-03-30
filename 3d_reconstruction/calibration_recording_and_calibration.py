#!/usr/bin/env python3
"""Combined calibration recording + calibration server (Docker B / metavision).

Records RGB images and accumulated event frames, then runs the full
intrinsic + extrinsic calibration pipeline in one go.

Usage:
    python calibration_recording_and_calibration.py \
        [--zmq-bind tcp://0.0.0.0:6002] [--output-dir camera_data]

Protocol  (this script = REP,  agent in Docker A = REQ):
    INIT          ->  initialise, reply READY
    POSE_REACHED  ->  capture RS RGB, start event accumulation, reply RGB_CAPTURED
    WIGGLE_DONE   ->  stop event accumulation, reply EVENT_CAPTURED
    ALL_DONE      ->  run calibration pipeline, reply success / fail

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
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq
import msgpack

from metavision_hal import DeviceDiscovery
from metavision_core.event_io import EventsIterator


# ── ChArUco board defaults (must match the physical board) ──────────────
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_H = 6
SQUARES_V = 9
SQUARE_LEN = 0.03     # metres
MARKER_LEN = 0.015    # metres

# ── Event camera biases ────────────────────────────────────────────────
BIAS_DIFF_ON = 10
BIAS_DIFF_OFF = 80
BIAS_FO = 0
BIAS_HPF = 50
BIAS_REFR = 150

# ── RealSense stream config ───────────────────────────────────────────
RS_W, RS_H, RS_FPS = 640, 480, 30


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
def _make_charuco():
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_H, SQUARES_V), SQUARE_LEN, MARKER_LEN, d
    )
    det = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters()
    )
    return board, det


def _detect(image, detector):
    """Detect ChArUco corners. Returns (corners, ids) or (None, None)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is not None and len(corners) >= 4:
        return corners, ids
    return None, None


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
        if len(common) < 6:
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
    T = np.eye(4)
    print(extr.rotation)
    print(extr.translation)
    T[:3, :3] = np.array(extr.rotation).reshape(3, 3)
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


# ═══════════════════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════════════════
class CalibrationRecordingServer:
    def __init__(self, bind_addr: str, output_dir: str):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

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
        return np.asanyarray(color.get_data())  # BGR uint8

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
                self._reply({"status": "READY"})

            elif cmd == "POSE_REACHED":
                idx = msg.get("index", -1)
                ee_flat = msg.get("ee_pose")
                if ee_flat is not None:
                    self._ee_poses.append(
                        np.array(ee_flat, dtype=np.float64).reshape(4, 4)
                    )
                bgr = self._capture_rgb()
                self._rgb_images.append(bgr)
                c, _ = _detect(bgr, self._det)
                rgb_found = c is not None
                print(
                    f"[server] POSE_REACHED #{idx}  "
                    f"RGB charuco={'yes' if rgb_found else 'NO'}"
                )
                # Save RGB image
                rgb_out = self._rgb_dir / f"rgb_{idx:03d}.png"
                cv2.imwrite(str(rgb_out), bgr)
                # Start event accumulation
                if self._event_acc is not None:
                    self._event_acc.start_accumulation()
                self._reply({"status": "RGB_CAPTURED", "charuco_found": rgb_found})

            elif cmd == "WIGGLE_DONE":
                idx = msg.get("index", -1)
                if self._event_acc is not None:
                    raw_events = self._event_acc.stop_accumulation()
                    n_events = len(raw_events)
                    # Build accumulated event frame
                    ev_frame = self._event_acc.events_to_frame(raw_events)
                    # Rotate 180° – event camera is mounted upside-down
                    ev_frame = cv2.rotate(ev_frame, cv2.ROTATE_180)
                    self._ev_frames.append(ev_frame)
                    # Save event frame image
                    img_out = self._event_frames_dir / f"event_{idx:03d}.png"
                    cv2.imwrite(str(img_out), ev_frame)
                    # ChArUco detection on event frame
                    c, _ = _detect(ev_frame, self._det)
                    ev_found = c is not None
                    print(
                        f"[server] WIGGLE_DONE #{idx}  "
                        f"{n_events} events  "
                        f"Event charuco={'yes' if ev_found else 'NO'}"
                    )
                else:
                    print(f"[server] WIGGLE_DONE #{idx}  (no event camera)")
                self._reply({"status": "EVENT_CAPTURED"})

            elif cmd == "ALL_DONE":
                print("\n[server] ALL_DONE – running calibration pipeline ...")
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
        T_rgb_from_ee = self._calibrate_hand_eye(K_rgb, d_rgb)
        if T_rgb_from_ee is not None:
            np.savez(str(out / "T_rgb_from_ee.npz"), T=T_rgb_from_ee)
            print("[cal] Saved T_rgb_from_ee.npz")

        # 3. Event-camera intrinsics (ChArUco)
        if not self._ev_frames:
            print("[cal] No event frames – skipping event + stereo calibration")
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

        return True

    # ── eye-in-hand calibration ──────────────────────────────────────
    def _calibrate_hand_eye(
        self, K_rgb: np.ndarray, d_rgb: np.ndarray
    ) -> np.ndarray | None:
        """Compute T_rgb_from_ee via cv2.calibrateHandEye.

        Returns 4×4 T such that  p_rgb = T @ p_ee,  or None on failure.
        """
        n = min(len(self._rgb_images), len(self._ee_poses))
        if n == 0:
            print("[hand-eye] No EE poses recorded – skipping hand-eye calibration")
            return None

        R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
        for i in range(n):
            corners, ids = _detect(self._rgb_images[i], self._det)
            if corners is None:
                continue
            obj_pts = (
                self._board.getChessboardCorners()[ids.flatten()].astype(np.float32)
            )
            img_pts = corners.reshape(-1, 1, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_rgb, d_rgb)
            if not ok:
                continue
            R_board2cam, _ = cv2.Rodrigues(rvec)
            T_ee = self._ee_poses[i]
            R_g2b.append(T_ee[:3, :3].copy())
            t_g2b.append(T_ee[:3, 3:4].copy())
            R_t2c.append(R_board2cam)
            t_t2c.append(tvec)

        if len(R_g2b) < 3:
            print(
                f"[hand-eye] Too few valid pairs: {len(R_g2b)} (need >= 3) – skipping"
            )
            return None

        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
            R_gripper2base=R_g2b,
            t_gripper2base=t_g2b,
            R_target2cam=R_t2c,
            t_target2cam=t_t2c,
        )

        # T_cam2ee: maps camera points into EE frame  (p_ee = T_cam2ee @ p_cam)
        T_cam2ee = np.eye(4)
        T_cam2ee[:3, :3] = R_cam2ee
        T_cam2ee[:3, 3] = t_cam2ee.flatten()

        # Invert → T_rgb_from_ee: maps EE points into RGB camera frame
        T_rgb_from_ee = np.linalg.inv(T_cam2ee)
        t_cm = t_cam2ee.flatten() * 100
        print(
            f"[hand-eye] {len(R_g2b)} pairs  |  "
            f"cam-in-EE: [{t_cm[0]:+.2f}, {t_cm[1]:+.2f}, {t_cm[2]:+.2f}] cm"
        )
        return T_rgb_from_ee

    # ── cleanup ─────────────────────────────────────────────────────
    def _cleanup(self):
        if self._rs_pipe:
            self._rs_pipe.stop()
        if self._event_acc:
            self._event_acc.shutdown()
        self._rep.close()
        print("[server] Cleanup done")


# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Combined calibration recording + calibration server (Docker B)"
    )
    parser.add_argument(
        "--zmq-bind",
        default="tcp://0.0.0.0:6002",
        help="ZMQ REP bind address (default: tcp://0.0.0.0:6002)",
    )
    parser.add_argument(
        "--output-dir",
        default="camera_data",
        help="Directory to save calibration results (default: camera_data)",
    )
    args = parser.parse_args()

    server = CalibrationRecordingServer(
        bind_addr=args.zmq_bind, output_dir=args.output_dir
    )
    server.serve()


if __name__ == "__main__":
    main()
