"""
Centralized configuration for the 3D reconstruction pipeline.

All shared constants used across preprocessing, training, evaluation and
reconstruction scripts are defined here.
Import constants directly from this module, for example:

    from config import FPS, PREPROCESS_CROP_HW, PREPROCESS_RESIZE_HW
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  Default paths
# ═══════════════════════════════════════════════════════════════════
CALIB_DIR = Path("camera_data")
# Dataset root relative to the repository; download_dataset.py stores the
# published dataset here.
DATA_ROOT = Path("data/Event_and_Depth")

# ═══════════════════════════════════════════════════════════════════
#  Recording frame rate
# ═══════════════════════════════════════════════════════════════════
FPS = 30

# ═══════════════════════════════════════════════════════════════════
#  Model input preprocessing
# ═══════════════════════════════════════════════════════════════════
# Every spatial pipeline stage uses the same operation order: center-crop the
# native event-camera image, then resize to the model input resolution. Camera
# intrinsics must undergo this exact transform as well.
PREPROCESS_CROP_HW = (720, 960)   # (H, W) crop in native event-camera pixels
PREPROCESS_RESIZE_HW = (240, 320) # (H, W) final model input

# ═══════════════════════════════════════════════════════════════════
#  Voxel grid
# ═══════════════════════════════════════════════════════════════════
NUM_BINS = 5  # number of temporal bins for voxel grid

# ═══════════════════════════════════════════════════════════════════
#  Depth parameters for training
# ═══════════════════════════════════════════════════════════════════
D_MAX = 0.7    # maximum depth in metres (tabletop range)
DEPTH_MIN = 0.05  # minimum depth in metres (5 cm)

# ═══════════════════════════════════════════════════════════════════
#  TSDF reconstruction
# ═══════════════════════════════════════════════════════════════════
TSDF_VOXEL_SIZE = 0.002        # metres
TSDF_SDF_TRUNC_FACTOR = 4.0
TSDF_DEPTH_MAX = 0.8           # metres (larger range for 3-D reconstruction)


# ═══════════════════════════════════════════════════════════════════
#  Shared workspace cube for evaluation, reconstruction, and table prior
# ═══════════════════════════════════════════════════════════════════
SPATIAL_CUBE_SIDE = 0.32    # metres (32 cm cube)
SPATIAL_CUBE_X_OFFSET = 0.05   # metres — shift of cube centre along X in robot base frame
SPATIAL_CUBE_Z_OFFSET = -0.03
SPATIAL_CUBE_CENTER_Z = SPATIAL_CUBE_Z_OFFSET + SPATIAL_CUBE_SIDE / 2  # metres — Z of cube centre in robot base frame
# Target point (cube centre) in robot base frame
SPATIAL_TARGET_X = 0.3 + SPATIAL_CUBE_X_OFFSET   # metres (robot workspace X + cube shift)
SPATIAL_TARGET_Y = 0.0                            # metres (centred on robot Y axis)
SPATIAL_TARGET_Z = SPATIAL_CUBE_CENTER_Z          # metres

# Depth-projection bleed correction (3×3 kernel by default).
DEPTH_BLEED_RADIUS = 1

# Vertical offset applied on top of the cube-bottom to place the table plane
# (positive = raise the plane above the cube bottom, negative = lower it)
TABLE_Z_OFFSET = 0.01   # metres

# ═══════════════════════════════════════════════════════════════════
#  Depth visualization
# ═══════════════════════════════════════════════════════════════════
# Fixed colour range for depth images/videos (independent of training range)
DEPTH_VIZ_MIN = 0.05   # metres — maps to bottom of TURBO colourmap
DEPTH_VIZ_MAX = 0.7    # metres — maps to top of TURBO colourmap

