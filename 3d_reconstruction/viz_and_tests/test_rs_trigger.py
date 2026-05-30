#!/usr/bin/env python3
"""test_rs_trigger.py — Drive the RealSense strobe/trigger GPIO for oscilloscope testing,
optionally recording raw event-camera data at the same time.

The D4xx series exposes a 9-pin "External Sensor Sync" connector.
Relevant pins (from the datasheet Table 3-35):
  Pin 5 – Z_VSYNC   (Depth VSYNC — the strobe output we care about)
  Pin 8 – VDD33V    (3.3 V supply — reference only, do NOT load)
  Pin 9 – GND

When `output_trigger_enabled = 1` the depth sensor pulses Z_VSYNC once
per frame, aligned with the start of each exposure.  The pulse width
equals the configured exposure time, so fixing the exposure gives a
clean, stable square wave.

Typical usage (oscilloscope probe to pin 5, scope GND to pin 9):

    python3 test_rs_trigger.py                          # 30 Hz, auto exposure
    python3 test_rs_trigger.py --fps 15                 # 15 Hz
    python3 test_rs_trigger.py --exposure 8000          # 8 ms exposure → 8 ms pulse
    python3 test_rs_trigger.py --serial 827112070121
    python3 test_rs_trigger.py --list                   # list connected cameras and exit
    python3 test_rs_trigger.py --output /tmp/test_rec   # also record raw event data
    python3 test_rs_trigger.py --no-event               # disable event recording
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import time
from multiprocessing import Process, Event as MPEvent, Queue
from typing import Optional

import pyrealsense2 as rs

# Allow imports from the parent 3d_reconstruction directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metavision_hal import DeviceDiscovery
from metavision_core.event_io import EventsIterator
from config import (
    BIAS_DIFF_ON, BIAS_DIFF_OFF, BIAS_FO, BIAS_HPF, BIAS_REFR,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def event_drain_process(
    stop_event: MPEvent,
    done_event: MPEvent,
    ready_event: MPEvent,
    start_log_event: MPEvent,
    event0_path: str,
    event1_path: str,
    num_cameras: int = 1,
    flush_seconds: float = 0.5,
) -> None:
    """Subprocess: open event camera(s), apply biases, start streaming.
    Drains events (discarding them) until *start_log_event* is set by the
    parent (meaning the RealSense is fully stable), then begins writing raw
    files.  Stops when *stop_event* is set, flushes, and exits.
    """
    import signal
    # Ignore SIGINT so Ctrl+C in the parent doesn't kill this process
    # mid-drain — the parent sets stop_event to trigger a clean shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    devices = DeviceDiscovery.list()
    device0 = DeviceDiscovery.open(devices[0])
    device1 = DeviceDiscovery.open(devices[1]) if num_cameras >= 2 else None

    for device in filter(None, (device0, device1)):
        biases = device.get_i_ll_biases()
        biases.set("bias_diff_on",  BIAS_DIFF_ON)
        biases.set("bias_diff_off", BIAS_DIFF_OFF)
        biases.set("bias_fo",       BIAS_FO)
        biases.set("bias_hpf",      BIAS_HPF)
        biases.set("bias_refr",     BIAS_REFR)

    raw0 = device0.get_i_events_stream()
    raw0.start()
    # log_raw_data() is called later, after the parent signals RS is stable.

    # Enable external trigger input so RealSense Z_VSYNC pulses are logged
    # as EventExtTrigger records (channel 0) in the .raw file.
    trig0 = device0.get_i_trigger_in()
    if trig0 is not None:
        for ch in trig0.get_available_channels().values():
            trig0.enable(trig0.Channel(ch))

    raw1 = None
    if device1 is not None:
        raw1 = device1.get_i_events_stream()
        raw1.start()
        trig1 = device1.get_i_trigger_in()
        if trig1 is not None:
            for ch in trig1.get_available_channels().values():
                trig1.enable(trig1.Channel(ch))

    it0 = iter(EventsIterator.from_device(device0))
    it1 = iter(EventsIterator.from_device(device1)) if device1 is not None else None

    ready_event.set()

    # Drain and discard events until the parent confirms RS is in steady state.
    print("[EventProc] Waiting for RealSense to reach steady state...", flush=True)
    try:
        while not start_log_event.is_set():
            next(it0)
            if it1 is not None:
                next(it1)
    except Exception as exc:
        print(f"[EventProc] ERROR during pre-log drain: {exc}", flush=True)

    # RealSense is now stable — start writing raw files.
    raw0.log_raw_data(event0_path)
    if raw1 is not None:
        raw1.log_raw_data(event1_path)
    print("[EventProc] Logging started.", flush=True)

    try:
        while not stop_event.is_set():
            next(it0)
            if it1 is not None:
                next(it1)

        flush_start = time.time()
        while time.time() - flush_start < flush_seconds:
            next(it0)
            if it1 is not None:
                next(it1)
    except Exception as exc:
        print(f"[EventProc] ERROR in drain loop: {exc}", flush=True)
    finally:
        try:
            raw0.stop_log_raw_data()
            print(f"[EventProc] Finalized {event0_path}", flush=True)
        except Exception as exc:
            print(f"[EventProc] ERROR stopping log cam0: {exc}", flush=True)
        if raw1 is not None:
            try:
                raw1.stop_log_raw_data()
                print(f"[EventProc] Finalized {event1_path}", flush=True)
            except Exception as exc:
                print(f"[EventProc] ERROR stopping log cam1: {exc}", flush=True)
        done_event.set()


def list_cameras() -> None:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
    print(f"Found {len(devices)} RealSense device(s):")
    for dev in devices:
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw     = dev.get_info(rs.camera_info.firmware_version)
        print(f"  {name}  serial={serial}  fw={fw}")


def _set_if_supported(sensor: rs.sensor, option: rs.option, value: float, label: str) -> bool:
    if sensor.supports(option):
        sensor.set_option(option, value)
        print(f"  [ok] {label} = {value}")
        return True
    else:
        print(f"  [--] {label} not supported on this sensor")
        return False


def _find_device(serial: str | None) -> rs.device:
    """Return the device matching *serial*, or the first device if *serial* is None."""
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense devices found.")
    if serial is None:
        return devices[0]
    for dev in devices:
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return dev
    raise RuntimeError(f"Device with serial {serial} not found.")


def _hardware_reset(serial: str | None = None) -> None:
    """Hardware-reset the target device (or all devices when *serial* is None)."""
    ctx = rs.context()
    for dev in ctx.query_devices():
        if serial is None or dev.get_info(rs.camera_info.serial_number) == serial:
            print(f"  Resetting device: {dev.get_info(rs.camera_info.name)}")
            dev.hardware_reset()


def _start_pipeline(args: argparse.Namespace) -> tuple[rs.pipeline, rs.sensor]:
    """Configure inter_cam_sync_mode (pre-stream), start the pipeline, apply strobe
    options, and return *(pipeline, depth_sensor)*.

    On failure mirrors synchronised_recording.py: hardware-reset all devices,
    sleep 3 s, then retry once before propagating the exception.
    """
    def _do_start() -> tuple[rs.pipeline, rs.sensor]:
        # Set inter_cam_sync_mode BEFORE the pipeline starts — it cannot be
        # changed while streaming.
        if not args.no_master:
            dev = _find_device(args.serial)
            _set_if_supported(
                dev.first_depth_sensor(),
                rs.option.inter_cam_sync_mode, 1.0,
                "inter_cam_sync_mode (1 = master)",
            )

        pipeline = rs.pipeline()
        config   = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(
            rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
        )

        profile      = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()

        print("\nConfiguring strobe output:")
        _set_if_supported(
            depth_sensor, rs.option.output_trigger_enabled, 1.0,
            "output_trigger_enabled"
        )
        # inter_cam_sync_mode was already set before streaming (see above)

        if args.exposure is not None:
            _set_if_supported(
                depth_sensor, rs.option.enable_auto_exposure, 0.0,
                "enable_auto_exposure (0 = manual)"
            )
            _set_if_supported(
                depth_sensor, rs.option.exposure, float(args.exposure),
                f"depth exposure ({args.exposure} µs)"
            )

        return pipeline, depth_sensor

    try:
        return _do_start()
    except Exception as e:
        print(f"\n[WARN] Pipeline start failed: {e}")
        print("[WARN] Performing hardware reset and retrying once...")
        _hardware_reset(args.serial)
        time.sleep(3.0)
        return _do_start()   # propagate if it fails again


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RealSense GPIO strobe/trigger signal test for oscilloscope"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second — sets strobe frequency (default: 30)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Depth stream width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Depth stream height (default: 480)",
    )
    parser.add_argument(
        "--serial", type=str, default=None,
        help="Camera serial number (omit to use first available device)",
    )
    parser.add_argument(
        "--exposure", type=int, default=None,
        metavar="US",
        help="Fix depth exposure in microseconds for a stable pulse width. "
             "Omit for auto-exposure (pulse width varies per frame).",
    )
    parser.add_argument(
        "--no-master", action="store_true",
        help="Skip setting inter_cam_sync_mode=1 (master). "
             "Use if you only want the strobe output without the sync-master role.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List connected RealSense devices and exit.",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).resolve().parent / "test_recording"),
        metavar="DIR",
        help="Directory to write raw event files into. "
             "Creates events_cam0.raw (and events_cam1.raw for 2 cameras). "
             "(default: viz_and_tests/test_recording)",
    )
    parser.add_argument(
        "--num-event-cams", type=int, default=1, metavar="N",
        help="Number of event cameras to record (1 or 2, default: 1).",
    )
    parser.add_argument(
        "--no-event", action="store_true",
        help="Disable event-camera recording even when --output is given.",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, metavar="SEC",
        help="Stop automatically after this many seconds (default: 5). "
             "Use 0 for unlimited (Ctrl+C to stop).",
    )
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    pulse_info = f"fixed {args.exposure} µs" if args.exposure is not None else "variable (auto-exposure)"
    record_events = not args.no_event

    # ── event cameras first (matches synchronised_recording.py init order) ───────
    event_proc    = None
    stop_evt      = None
    done_evt      = None
    start_log_evt = None

    if record_events:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        event0_path = str(out_dir / "events_cam0.raw")
        event1_path = str(out_dir / "events_cam1.raw")

        stop_evt      = MPEvent()
        done_evt      = MPEvent()
        ready_evt     = MPEvent()
        start_log_evt = MPEvent()

        event_proc = Process(
            target=event_drain_process,
            args=(stop_evt, done_evt, ready_evt, start_log_evt,
                  event0_path, event1_path, args.num_event_cams),
        )
        event_proc.start()
        print(f"\nWaiting for {args.num_event_cams} event camera(s)...")
        ready_evt.wait()
        print(f"  Event camera(s) ready — writing to {out_dir}")

    # ── RealSense ─────────────────────────────────────────────────────────────
    print(f"\nStarting RealSense  {args.width}x{args.height} @ {args.fps} fps ...")
    pipeline, depth_sensor = _start_pipeline(args)

    # ── print oscilloscope connection instructions ──────────────────────────────
    duration_info = f"{args.duration:.0f} s" if args.duration > 0 else "unlimited (Ctrl+C to stop)"
    print(f"""
Oscilloscope setup
──────────────────
  Signal  →  pin 5  Z_VSYNC  (Depth VSYNC)
  GND     →  pin 9  GND
  (pin 8 = 3.3 V power — do not connect to scope)
  Expected:  ~{args.fps} Hz square wave, pulse width {pulse_info}
  Duration:  {duration_info}

Streaming — press Ctrl+C to stop.
""")

    # ── streaming loop ──────────────────────────────────────────────────────────
    # Use a short timeout so stalls are detected quickly.  10 consecutive
    # timeouts (≈ 5 s) trigger a hardware reset + pipeline restart, mirroring
    # the recovery logic in synchronised_recording.py.
    FRAME_TIMEOUT_MS      = 500
    MAX_CONSECUTIVE_STALL = 10
    RS_WARMUP_S           = 3.0   # seconds to wait for AE to converge before logging

    frame_count          = 0
    consecutive_timeouts = 0
    t0                   = time.monotonic()
    t_warmup             = None   # set on first successful RS frame
    t_start              = None   # set after RS warmup; starts the duration clock

    try:
        while True:
            if args.duration > 0 and t_start is not None and (time.monotonic() - t_start) >= args.duration:
                print(f"\nDuration of {args.duration:.0f} s reached — stopping.")
                break
            try:
                pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
                consecutive_timeouts = 0
                frame_count += 1
                if t_warmup is None:
                    t_warmup = time.monotonic()
                    print("  First RS frame received — waiting for steady state...", flush=True)
                # After the warmup period, signal the event process to start logging
                # and start the duration countdown.
                if t_start is None and (time.monotonic() - t_warmup) >= RS_WARMUP_S:
                    t_start = time.monotonic()
                    if start_log_evt is not None:
                        start_log_evt.set()
                    if args.duration > 0:
                        print(f"  RealSense stable — logging started, recording for {args.duration:.0f} s...", flush=True)
                    else:
                        print("  RealSense stable — logging started.", flush=True)
            except RuntimeError:
                consecutive_timeouts += 1
                if consecutive_timeouts >= MAX_CONSECUTIVE_STALL:
                    stall_s = FRAME_TIMEOUT_MS * MAX_CONSECUTIVE_STALL / 1000
                    print(
                        f"\n[WARN] No frames received for {stall_s:.0f} s — "
                        "performing hardware reset and restarting pipeline..."
                    )
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    _hardware_reset(args.serial)
                    time.sleep(3.0)
                    print(f"\nRestarting RealSense  {args.width}x{args.height} @ {args.fps} fps ...")
                    pipeline, depth_sensor = _start_pipeline(args)
                    consecutive_timeouts = 0
                    frame_count = 0
                    t_warmup = None
                    t_start  = None
                    t0       = time.monotonic()
                continue

            elapsed = time.monotonic() - t0
            if elapsed >= 3.0:
                measured_fps = frame_count / elapsed
                period_ms    = 1000.0 / measured_fps if measured_fps > 0 else float("inf")
                print(
                    f"  {frame_count:5d} frames  |  {measured_fps:.2f} fps  "
                    f"|  period ≈ {period_ms:.1f} ms",
                    flush=True,
                )
                frame_count = 0
                t0          = time.monotonic()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()
        print("Pipeline stopped.")

        if event_proc is not None and stop_evt is not None and done_evt is not None:
            print("Stopping event camera(s)...")
            stop_evt.set()
            done_evt.wait(timeout=10.0)
            event_proc.join(timeout=11.0)
            print("Event camera(s) stopped.")

        # Verify output files
        if record_events:
            import numpy as np
            import matplotlib.pyplot as plt
            from metavision_core.event_io import RawReader

            plots_dir = Path(__file__).resolve().parent / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            print("\nOutput file check:")
            for cam_idx, path in enumerate(
                [event0_path] + ([event1_path] if args.num_event_cams >= 2 else [])
            ):
                p = Path(path)
                if p.exists():
                    size_kb = p.stat().st_size / 1024
                    if size_kb > 0:
                        print(f"  [OK]  {path}  ({size_kb:.1f} KB)")
                        try:
                            r = RawReader(path)
                            total_events = 0
                            while not r.is_done():
                                evs = r.load_delta_t(100000)
                                if evs is not None:
                                    total_events += len(evs)
                            # get_ext_trigger_events() is cumulative — call it once
                            # after reading all data to get the correct total count.
                            t = r.get_ext_trigger_events()
                            total_triggers = len(t) if t is not None else 0
                            trigger_times = []
                            if t is not None and len(t) > 0:
                                # only keep rising edges (p==1) for interval analysis
                                rising = t[t["p"] == 1]
                                trigger_times.extend(rising["t"].tolist())
                            print(f"        events:   {total_events:,}")
                            expected_triggers = int(args.fps * args.duration * 2)  # rising + falling edge per frame
                            print(f"        triggers: {total_triggers:,}  (expected ~{expected_triggers} for {args.duration:.0f} s @ {args.fps} fps, both edges)")

                            # Plot inter-trigger intervals (skip first — RealSense fires one
                            # startup trigger then pauses ~2-3 s before the regular stream begins)
                            if len(trigger_times) >= 3:
                                ts = np.array(sorted(trigger_times), dtype=np.float64)
                                ts = ts[1:]  # drop first trigger to remove startup artifact
                                intervals_ms = np.diff(ts) / 1000.0  # µs → ms
                                expected_ms  = 1000.0 / args.fps

                                fig, axes = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True)

                                axes[0].plot(intervals_ms, linewidth=0.8, color="steelblue")
                                axes[0].axhline(expected_ms, color="red", linestyle="--",
                                                linewidth=1, label=f"expected {expected_ms:.2f} ms")
                                axes[0].set_xlabel("Trigger index")
                                axes[0].set_ylabel("Interval (ms)")
                                axes[0].set_title(f"Inter-trigger intervals — cam{cam_idx}  "
                                                  f"(mean {intervals_ms.mean():.3f} ms, "
                                                  f"std {intervals_ms.std():.3f} ms)")
                                axes[0].legend()

                                axes[1].hist(intervals_ms, bins=60, color="steelblue", edgecolor="white")
                                axes[1].axvline(expected_ms, color="red", linestyle="--",
                                                linewidth=1, label=f"expected {expected_ms:.2f} ms")
                                axes[1].set_xlabel("Interval (ms)")
                                axes[1].set_ylabel("Count")
                                axes[1].set_title("Interval histogram")
                                axes[1].legend()

                                plot_path = plots_dir / f"trigger_intervals_cam{cam_idx}.png"
                                fig.savefig(plot_path, dpi=150)
                                plt.close(fig)
                                print(f"        plot:     {plot_path}")
                            else:
                                print(f"        [WARN] Not enough trigger rising edges to plot ({len(trigger_times)} found)")
                        except Exception as e:
                            print(f"        [WARN] Could not read file for stats: {e}")
                    else:
                        print(f"  [WARN] {path} exists but is EMPTY (0 bytes)")
                else:
                    print(f"  [FAIL] {path} was NOT created")


if __name__ == "__main__":
    main()
