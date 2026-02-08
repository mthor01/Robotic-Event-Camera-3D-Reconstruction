import time
import cv2
import numpy as np
import pyrealsense2 as rs
import h5py
from pathlib import Path
from multiprocessing import Process, Event

from metavision_hal import DeviceDiscovery
from metavision_core.event_io import EventsIterator

# ================= CONFIG =================
TARGET_SECONDS = 10
FPS = 30

DATA_DIR = Path("data")
VIDEO_DIR = DATA_DIR / "videos"
RAW_DIR   = DATA_DIR / "raw_event_data"
HDF5_DIR  = DATA_DIR / "hdf5"

EVENT0_RAW_FILE = RAW_DIR / "events_cam0.raw"
EVENT1_RAW_FILE = RAW_DIR / "events_cam1.raw"

RS_VIDEO_FILE = VIDEO_DIR / "realsense.mp4"
RS_H5_FILE    = HDF5_DIR / "realsense.h5"

RS_WIDTH, RS_HEIGHT = 640, 480

BIAS_DIFF_ON  = 10
BIAS_DIFF_OFF = 80
BIAS_FO       = 0
BIAS_HPF      = 50
BIAS_REFR     = 150

# =========================================


def event_drain_process(
    stop_acquire_event: Event,
    done_event: Event,
    ready_event: Event,
    flush_seconds=0.5
):
    devices = DeviceDiscovery.list()
    device0 = DeviceDiscovery.open(devices[0])
    device1 = DeviceDiscovery.open(devices[1])

    # --------- APPLY BIASES (added) ---------
    for device in (device0, device1):
        biases = device.get_i_ll_biases()
        biases.set("bias_diff_on", BIAS_DIFF_ON)
        biases.set("bias_diff_off", BIAS_DIFF_OFF)
        biases.set("bias_fo", BIAS_FO)
        biases.set("bias_hpf", BIAS_HPF)
        biases.set("bias_refr", BIAS_REFR)
    # ---------------------------------------

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

        flush_start = time.time()
        while time.time() - flush_start < flush_seconds:
            next(it0)
            next(it1)

    finally:
        raw0.stop_log_raw_data()
        raw1.stop_log_raw_data()
        done_event.set()





def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    HDF5_DIR.mkdir(parents=True, exist_ok=True)

    stop_event = Event()

    # ---------- RealSense ----------
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, FPS)
    pipeline.start(cfg)

    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.laser_power, 360)  # or small value



    stop_acquire_event = Event()
    event_done_event = Event()
    event_ready_event = Event()

    event_proc = Process(
        target=event_drain_process,
        args=(stop_acquire_event, event_done_event, event_ready_event)
    )
    event_proc.start()

    

    rs_video = cv2.VideoWriter(
        str(RS_VIDEO_FILE),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (RS_WIDTH, RS_HEIGHT),
        isColor=False
    )

    rs_h5 = h5py.File(RS_H5_FILE, "w")
    rs_grp = rs_h5.create_group("realsense")

    depth_ds = rs_grp.create_dataset(
        "depth",
        shape=(0, RS_HEIGHT, RS_WIDTH),
        maxshape=(None, RS_HEIGHT, RS_WIDTH),
        dtype=np.uint16,
        chunks=True
    )
    t_sys_ds = rs_grp.create_dataset(
        "t_sys_ns",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64
    )

    rs_idx = 0
    # ---- wait until event cameras are ready ----
    event_ready_event.wait()
    print("Event cameras ready, starting recording")

    start_time = time.time()

    try:
        while time.time() - start_time < TARGET_SECONDS:
            frames = pipeline.poll_for_frames()
            if not frames:
                continue

            depth = frames.get_depth_frame()
            if not depth:
                continue

            depth_img = np.asanyarray(depth.get_data())

            depth_ds.resize(rs_idx + 1, axis=0)
            t_sys_ds.resize(rs_idx + 1, axis=0)

            depth_ds[rs_idx] = depth_img
            t_sys_ds[rs_idx] = time.time_ns()

            rs_video.write(
                cv2.convertScaleAbs(depth_img, alpha=0.03)
            )
            rs_idx += 1

    finally:
        # Signal end of acquisition
        stop_acquire_event.set()

        # Wait until event buffers are fully flushed
        event_done_event.wait()
        event_proc.join()

        rs_video.release()
        rs_h5.close()
        pipeline.stop()


    print("Recording finished")
    print(f"RealSense frames recorded: {rs_idx}")


if __name__ == "__main__":
    main()
