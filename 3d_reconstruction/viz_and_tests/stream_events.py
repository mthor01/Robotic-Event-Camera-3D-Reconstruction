from metavision_core.event_io import EventsIterator
import numpy as np
import cv2

# 100ms per frame (~10 fps) instead of 10ms — drastically reduces rendering load.
# Further reduce events per frame by keeping only 1 in every N events (subsampling).
SUBSAMPLE = 500  # keep 1/10 of events

it = EventsIterator(input_path="", delta_t=30000)
height, width = it.get_size()

for evs in it:
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if len(evs):
        # Subsample to cap the number of events rendered
        evs = evs[::SUBSAMPLE]
        x = evs["x"]
        y = evs["y"]
        p = evs["p"]

        img[y[p == 1], x[p == 1]] = (255, 255, 255)
        img[y[p == 0], x[p == 0]] = (100, 100, 100)

    cv2.imshow("events", img)

    if cv2.waitKey(1) == 27:
        break