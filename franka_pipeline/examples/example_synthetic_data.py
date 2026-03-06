#!/usr/bin/env python3
"""
Example: Generate synthetic event camera + depth data using simulation.

This script demonstrates how to use the franka_pipeline to generate
synthetic training data that matches the format of real recordings
from multi_recording.py.

Usage:
    # Basic synthetic data recording
    python my_main.py --simulated-robot --synthetic-data

    # With custom output directory and thresholds
    python my_main.py --simulated-robot --synthetic-data \
        --synthetic-output-dir /path/to/output \
        --event-threshold-pos 0.2 \
        --event-threshold-neg 0.3

    # Using frontview camera instead of eye-in-hand
    python my_main.py --simulated-robot --synthetic-data \
        --synthetic-camera-id frontview

Output Structure:
    synthetic_data/
    └── synthetic_0000_1234567890/
        ├── hdf5/
        │   ├── depth.h5       # Depth frames (compatible with multi_recording.py format)
        │   └── rgb.h5         # RGB frames for reference
        ├── events/
        │   └── events.npy     # Synthetic events (structured array, Metavision-compatible)
        ├── poses/
        │   └── poses.npy      # Robot poses with camera intrinsics/extrinsics
        └── videos/
            ├── rgb.mp4        # RGB video
            └── events.mp4     # Event visualization (red=ON, blue=OFF)

Notes:
    - Events are generated using v2e-style log intensity change detection
    - Depth is saved in millimeters (uint16) matching RealSense format
    - Poses include camera intrinsics (fx, fy, cx, cy) and extrinsics (T_ee2cam)
    - For real recordings, use --real-robot --sync-recording with multi_recording.py
"""

# For programmatic use:
if __name__ == "__main__":
    print(__doc__)
    print("\nRun 'python my_main.py --help' for all available options.")
