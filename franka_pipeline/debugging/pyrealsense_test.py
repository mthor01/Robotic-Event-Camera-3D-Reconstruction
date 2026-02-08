# A minimal test script to verify multiple RealSense cameras are working correctly


import pyrealsense2 as rs
import cv2
import numpy as np

# Get all connected RealSense devices
context = rs.context()
devices = context.query_devices()

if len(devices) == 0:
    print("No RealSense devices found!")
    exit()

print(f"Found {len(devices)} camera(s)")

pipes = []
# for idx, device in enumerate(devices[1:2]):
for idx, device in enumerate(devices):
    pipe = rs.pipeline(context)
    config = rs.config()
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(config)
    pipes.append(pipe)
    print(f"Started camera {idx}: {device.get_info(rs.camera_info.name)}")

try:
    for i in range(100000):
        for idx, pipe in enumerate(pipes):
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame:
                image = np.asanyarray(color_frame.get_data())
                cv2.imshow(f"RealSense Camera {idx} - Color", image)

            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image_normalized = cv2.normalize(
                    depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                cv2.imshow(f"RealSense Camera {idx} - Depth", depth_image_normalized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    for pipe in pipes:
        pipe.stop()
    cv2.destroyAllWindows()
