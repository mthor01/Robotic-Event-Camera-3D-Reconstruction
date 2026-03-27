"""
Default configuration values for my_main.py command options.

Change values here to modify the defaults for all command-line options.
These can still be overridden via command-line arguments.
"""

import os

# =============================================================================
# ROBOT MODE
# =============================================================================
SIMULATED_ROBOT = True
REAL_ROBOT = False

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
DEPS_LOG_LEVEL = "INFO"

# =============================================================================
# ZMQ COMMUNICATION
# =============================================================================
ZMQ_BIND = os.getenv("POSE_PUB_BIND", "tcp://0.0.0.0:6000")
ZMQ_SYNC_BIND = os.getenv("SYNC_BIND", "tcp://0.0.0.0:6001")
PUBLISH_HZ = 60.0

# =============================================================================
# SYNC RECORDING
# =============================================================================
SYNC_RECORDING = True

# =============================================================================
# SYNTHETIC DATA RECORDING
# =============================================================================
SYNTHETIC_DATA = False
SYNTHETIC_OUTPUT_DIR = "../3d_reconstruction/data/synthetic_data"
SYNTHETIC_CAMERA_ID = "robot0_eye_in_hand"
SYNTHETIC_CAMERA_WIDTH = 346
SYNTHETIC_CAMERA_HEIGHT = 260

# Event camera simulation thresholds
EVENT_THRESHOLD_POS = 0.25
EVENT_THRESHOLD_NEG = 0.25

# Flip synthetic data vertically (for camera mounting orientation)
FLIP_VERTICAL = True

# =============================================================================
# CUSTOM OBJECTS
# =============================================================================
CUSTOM_OBJECTS_CONFIG = None  # Path to YAML file, or None

# =============================================================================
# MULTI-OBJECT RECORDING
# =============================================================================
OBJECT_FILTER = None  # Comma-separated string like "cube,sphere,milk", or None for all

# =============================================================================
# DISPLAY
# =============================================================================
HEADLESS = False

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================
# Agent type: "hemisphere", "random_sphere", or "random_hemisphere"
AGENT_TYPE = "random_hemisphere"

# Sphere center (robot base frame) - only used by RandomSphereAgent
# For RandomHemisphereAgent, the center is always equal to TARGET_X/Y/Z
SPHERE_CENTER_X = 0.4
SPHERE_CENTER_Y = 0.0
SPHERE_CENTER_Z = 0.4

# Target point (where the camera looks at)
# For RandomHemisphereAgent, this is also the center of the hemisphere
TARGET_X = 0.35
TARGET_Y = 0.0
TARGET_Z = 0.15

# Sphere/Hemisphere radius
SPHERE_RADIUS = 0.3

# Inner radius for hollow hemisphere (RandomHemisphereAgent only)
INNER_RADIUS = 0.05

# Number of random poses to generate
NUM_POSES = 15

# Time to wait at each pose (seconds)
WAIT_TIME = 0.0

# Random seed for reproducibility (None = random)
RANDOM_SEED = None

# =============================================================================
# SAFETY ZONE PARAMETERS (for RandomSphereAgent and RandomHemisphereAgent)
# =============================================================================
# Exclude poses within this x,y radius of robot base (0,0)
BASE_EXCLUSION_RADIUS = 0.35

# Exclude poses beyond this x,y radius of robot base (0,0)
BASE_MAX_RADIUS = 0.5

# Minimum z-height for poses (table level)
MIN_Z_HEIGHT = 0.2

# =============================================================================
# WAYPOINT INTERPOLATION
# =============================================================================
# Number of intermediate waypoints between main poses
WAYPOINTS_PER_TRANSITION = 1

# =============================================================================
# ROTATION CONTROL
# =============================================================================
# If True, lock Y-axis to be parallel to the table (horizontal)
# If False, allow random roll rotation around the viewing direction
LOCK_ROTATION_HORIZONTAL = True
