import time
import sys
import cv2
import h5py
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconstruction_config import DEPTH_VIZ_P_LOW, DEPTH_VIZ_P_HIGH

# ================= CONFIG =================
DATA_DIR = Path("data") / "hdf5"

EVENT0_H5 = DATA_DIR / "events_cam0.h5"
EVENT1_H5 = DATA_DIR / "events_cam1.h5"
RS_H5     = DATA_DIR / "realsense.h5"

SLOWDOWN = 4.0          # 4x slower than real time
WINDOW_NAME = "Alignment check (ESC to quit)"
# =========================================


def normalize_uint8(img):
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return (img * 255).astype(np.uint8)

def depth_to_uint8(depth, p_low=DEPTH_VIZ_P_LOW, p_high=DEPTH_VIZ_P_HIGH):
    depth = depth.astype(np.float32)

    lo = np.percentile(depth, p_low)
    hi = np.percentile(depth, p_high)

    depth = np.clip(depth, lo, hi)
    depth = (depth - lo) / (hi - lo + 1e-6)

    return (depth * 255).astype(np.uint8)


def main():
    h5_e0 = h5py.File(EVENT0_H5, "r")
    h5_e1 = h5py.File(EVENT1_H5, "r")
    h5_rs = h5py.File(RS_H5, "r")

    ev0 = h5_e0["events/frames"]
    ev1 = h5_e1["events/frames"]
    rs  = h5_rs["realsense/depth"]

    fps = h5_e0["events"].attrs["fps"]
    delay = (1.0 / fps) * SLOWDOWN

    n = min(len(ev0), len(ev1), len(rs))
    print(f"Playing {n} frames")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    for i in range(n):
        f0 = ev0[i]
        f1 = ev1[i]
        d  = rs[i]

        # ensure uint8 for display
        f0 = normalize_uint8(f0)
        f1 = normalize_uint8(f1)
        d  = depth_to_uint8(d)

        # convert to BGR so we can stack
        f0 = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR)
        f1 = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
        d  = cv2.cvtColor(d,  cv2.COLOR_GRAY2BGR)

        # resize depth to event cam size if needed
        if d.shape[:2] != f0.shape[:2]:
            d = cv2.resize(d, (f0.shape[1], f0.shape[0]))

        vis = np.hstack([f0, f1, d])
        cv2.putText(
            vis, f"frame {i}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        cv2.imshow(WINDOW_NAME, vis)
        if cv2.waitKey(1) == 27:  # ESC
            break

        time.sleep(delay)

    cv2.destroyAllWindows()
    h5_e0.close()
    h5_e1.close()
    h5_rs.close()


if __name__ == "__main__":
    main()
