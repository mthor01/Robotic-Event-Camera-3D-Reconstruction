import time
import cv2
import numpy as np
import h5py
from pathlib import Path

from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm

from reconstruction_config import FPS, DELTA_T_US

# ================= CONFIG =================
DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "raw_event_data"
VIDEO_DIR = DATA_DIR / "videos"
HDF5_DIR  = DATA_DIR / "hdf5"

EVENT0_RAW_FILE = RAW_DIR / "events_cam0.raw"
EVENT1_RAW_FILE = RAW_DIR / "events_cam1.raw"

EVENT0_VIDEO_FILE = VIDEO_DIR / "events_cam0.mp4"
EVENT1_VIDEO_FILE = VIDEO_DIR / "events_cam1.mp4"

EVENT0_H5_FILE = HDF5_DIR / "events_cam0.h5"
EVENT1_H5_FILE = HDF5_DIR / "events_cam1.h5"
# =========================================


def process_raw(raw_file, video_file, h5_file):
    ev_it = EventsIterator(input_path=str(raw_file), delta_t=DELTA_T_US)
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

    grp.attrs.update({
        "fps": FPS,
        "delta_t_us": DELTA_T_US,
        "width": width,
        "height": height
    })

    frame_gen = PeriodicFrameGenerationAlgorithm(width, height, DELTA_T_US)
    idx = 0

    def on_frame(ts, frame):
        nonlocal idx
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames_ds.resize(idx + 1, axis=0)
        t_start_ds.resize(idx + 1, axis=0)
        t_end_ds.resize(idx + 1, axis=0)

        frames_ds[idx] = gray
        t_start_ds[idx] = ts - DELTA_T_US
        t_end_ds[idx] = ts

        video.write(gray)
        idx += 1

    frame_gen.set_output_callback(on_frame)

    for evs in ev_it:
        frame_gen.process_events(evs)

    video.release()
    h5.close()

    return idx  # number of frames



def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    HDF5_DIR.mkdir(parents=True, exist_ok=True)

    print("Processing cam0")
    n0 = process_raw(EVENT0_RAW_FILE, EVENT0_VIDEO_FILE, EVENT0_H5_FILE)
    print(f"cam0 frames: {n0}")

    print("Processing cam1")
    n1 = process_raw(EVENT1_RAW_FILE, EVENT1_VIDEO_FILE, EVENT1_H5_FILE)
    print(f"cam1 frames: {n1}")

    print("Done")



if __name__ == "__main__":
    main()
