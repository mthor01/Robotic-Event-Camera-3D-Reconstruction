"""
Centralized configuration for the 3D reconstruction pipeline.

All shared constants used across recording, preprocessing, training,
reconstruction and visualization scripts are defined here.
Import what you need:

    from reconstruction_config import FPS, WHITE_THRESH, ...
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  Recording / frame rate
# ═══════════════════════════════════════════════════════════════════
FPS = 30
DELTA_T_US = int(1e6 / FPS)  # microseconds per frame

# Manual offset (signed integer, in depth frames) applied when aligning event
# frames to depth frames.  Positive = depth is ahead of events (shift event
# lookup forward); negative = depth is behind events (shift lookup backward).
# Set to 0 for no correction.  Example: set to -2 if depth is 2 frames behind.
DEPTH_EVENT_ALIGN_OFFSET_FRAMES: int = 0

# ═══════════════════════════════════════════════════════════════════
#  RealSense camera
# ═══════════════════════════════════════════════════════════════════
RS_WIDTH = 640
RS_HEIGHT = 480

# ═══════════════════════════════════════════════════════════════════
#  Event camera biases
# ═══════════════════════════════════════════════════════════════════
BIAS_DIFF_ON = 10
BIAS_DIFF_OFF = 80
BIAS_FO = 0
BIAS_HPF = 50
BIAS_REFR = 150

# ═══════════════════════════════════════════════════════════════════
#  Voxel grid
# ═══════════════════════════════════════════════════════════════════
NUM_BINS = 5  # number of temporal bins for voxel grid

# ═══════════════════════════════════════════════════════════════════
#  Depth parameters — E2Depth training (tabletop scene)
# ═══════════════════════════════════════════════════════════════════
D_MAX = 0.7    # maximum depth in metres (tabletop range)
ALPHA = 2.5    # log depth parameter: ln(D_MAX / D_MIN) ≈ ln(0.6/0.05)
DEPTH_MIN = 0.05  # minimum depth in metres (5 cm)

# ═══════════════════════════════════════════════════════════════════
#  Depth parameters — pose-to-plane encoding (full robot reach)
# ═══════════════════════════════════════════════════════════════════
POSE_D_MAX = 10.0  # maximum depth for pose-to-plane feature (metres)
POSE_ALPHA = 4.6   # ln(10.0 / 0.1) ≈ 4.6, covers 0.1 m to 10 m

# ═══════════════════════════════════════════════════════════════════
#  TSDF reconstruction
# ═══════════════════════════════════════════════════════════════════
TSDF_VOXEL_SIZE = 0.002        # metres
TSDF_SDF_TRUNC_FACTOR = 4.0
TSDF_DEPTH_MIN = 0.05          # metres (same as training DEPTH_MIN)
TSDF_DEPTH_MAX = 0.8           # metres (larger range for 3-D reconstruction)

# ═══════════════════════════════════════════════════════════════════
#  White-pixel masking
# ═══════════════════════════════════════════════════════════════════
WHITE_THRESH = 100  # RGB channel threshold for white detection

# ═══════════════════════════════════════════════════════════════════
#  Spatial masking (cube around target position)
# ═══════════════════════════════════════════════════════════════════
SPATIAL_CUBE_SIDE = 0.32    # metres (32 cm cube)
SPATIAL_CUBE_X_OFFSET = 0.05   # metres — shift of cube centre along X in robot base frame
SPATIAL_CUBE_Z_OFFSET = -0.03
SPATIAL_CUBE_CENTER_Z = SPATIAL_CUBE_Z_OFFSET + SPATIAL_CUBE_SIDE / 2  # metres — Z of cube centre in robot base frame
DEPTH_BLEED_RADIUS = 1      # px — half-width of bleed-correction kernel (3×3 default)
# Target point (cube centre) in robot base frame
SPATIAL_TARGET_X = 0.3 + SPATIAL_CUBE_X_OFFSET   # metres (robot workspace X + cube shift)
SPATIAL_TARGET_Y = 0.0                            # metres (centred on robot Y axis)
SPATIAL_TARGET_Z = SPATIAL_CUBE_CENTER_Z          # metres

# Vertical offset applied on top of the cube-bottom to place the table plane
# (positive = raise the plane above the cube bottom, negative = lower it)
TABLE_Z_OFFSET = 0.01   # metres

# ═══════════════════════════════════════════════════════════════════
#  Depth visualization (percentile normalization)
# ═══════════════════════════════════════════════════════════════════
DEPTH_VIZ_P_LOW = 2
DEPTH_VIZ_P_HIGH = 98
# Fixed colour range for depth images/videos (independent of training range)
DEPTH_VIZ_MIN = 0.05   # metres — maps to bottom of TURBO colourmap
DEPTH_VIZ_MAX = 1.0    # metres — maps to top of TURBO colourmap

# ═══════════════════════════════════════════════════════════════════
#  Training image resolution
# ═══════════════════════════════════════════════════════════════════
TRAIN_RESIZE_HW = (288, 384)  # (H, W) resize before crop during training
TRAIN_CROP_HW   = (240, 320)  # (H, W) center crop after resize during training
TRAIN_BATCH_SIZE = 10         # default batch size for E2Depth training
TRAIN_SEQ_LEN    = 10         # default sequence length for recurrent training

# ═══════════════════════════════════════════════════════════════════
#  Default paths
# ═══════════════════════════════════════════════════════════════════
CALIB_DIR = Path("camera_data")
DATA_ROOT = Path("data/real")
TEMPORAL_CHECK_ROOT = Path("data/temporal_check")
LIGHT_CHECK_ROOT = Path("data/light_check")
DEFAULT_OUT_DIR = Path("checkpoints_e2depth")

# ═══════════════════════════════════════════════════════════════════
#  ChArUco board (calibration)
# ═══════════════════════════════════════════════════════════════════
CHARUCO_SQUARES_H = 6
CHARUCO_SQUARES_V = 9
CHARUCO_SQUARE_LEN = 0.03    # metres
CHARUCO_MARKER_LEN = 0.015   # metres

# ═══════════════════════════════════════════════════════════════════
#  ZMQ addresses (recording synchronization)
# ═══════════════════════════════════════════════════════════════════
ZMQ_SYNC_ADDR = "tcp://localhost:6001"
ZMQ_POSE_ADDR = "tcp://localhost:6000"

# ═══════════════════════════════════════════════════════════════════
#  Pose timestamp correction
# ═══════════════════════════════════════════════════════════════════
# Manual offset added to pose timestamps before frame assignment.
# Positive values shift poses forward in time (use when events appear
# to lead the robot motion); negative values shift them backward.
# Applied on top of the automatically measured ZMQ transport delay.
POSE_TIME_OFFSET_MS: float = -20.0

# ═══════════════════════════════════════════════════════════════════
#  Visualization defaults
# ═══════════════════════════════════════════════════════════════════
POSE_VIZ_AXIS_LEN = 0.03    # metres, length of XYZ axes in 3-D plot
POSE_VIZ_ARROW_LEN = 0.06   # metres, length of facing arrow in 3-D plot
