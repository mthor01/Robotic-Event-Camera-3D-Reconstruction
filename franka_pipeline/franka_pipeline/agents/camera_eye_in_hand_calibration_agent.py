# TODO: This is heavily AI-coded, future review is needed

"""Eye-in-hand camera calibration agent.

This module provides an agent that automates the eye-in-hand camera calibration
process by moving to predefined poses, capturing images, and computing the
camera-to-end-effector transformation.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pynput import keyboard
from scipy.spatial.transform import Rotation

from franka_pipeline.agents.agent import Agent
from franka_pipeline.agents.teleoperation_agent import TeleoperationAgent
from franka_pipeline.input_controllers.spacemouse import SpaceMouseController
from franka_pipeline.logging import get_logger
from franka_pipeline.robot_controllers.osc_pose_target_controller import (
    OscPoseTargetController,
)
from franka_pipeline.utils import (
    T_from_t_R,
    pos_rot_to_transformation_matrix,
    t_R_from_T,
)

logger = get_logger(__name__)


@dataclass
class CharucoProperties:
    """Properties for ChArUco board detection.

    Attributes:
        aruco_dict: OpenCV ArUco dictionary type.
        squares_horizontally: Number of squares in the horizontal direction.
        squares_vertically: Number of squares in the vertical direction.
        square_length: Size of each square in meters.
        marker_length: Size of each ArUco marker in meters.
    """

    aruco_dict: int = cv2.aruco.DICT_6X6_250
    squares_horizontally: int = 6
    squares_vertically: int = 9
    square_length: float = 0.03  # 3 cm
    marker_length: float = 0.015  # 1.5 cm


# TODO: this dataclass should maybe be removed. Values could be put in the agent directly.
# In any case, the tolerances should be checked again to put sensible values
@dataclass
class CalibrationConfig:
    """Configuration for camera calibration.

    Attributes:
        charuco: ChArUco board properties.
        output_dir: Directory to save calibration data.
        position_tolerance: Position tolerance in meters for pose reaching.
        rotation_tolerance: Rotation tolerance in radians for pose reaching.
        hand_eye_method: Method for hand-eye calibration (OpenCV method name).
        debug_visualization: Whether to show and save debug images.
    """

    charuco: CharucoProperties = field(default_factory=CharucoProperties)
    output_dir: Path = Path("calibration_data")
    # position_tolerance: float = 0.005  # 5 mm
    # rotation_tolerance: float = 0.02  # ~1 degree
    position_tolerance: float = 0.05  # TODO make this smaller
    rotation_tolerance: float = 0.06  # TODO make this smaller
    settling_time: float = 0.3  # Time to wait before capturing image (seconds)
    hand_eye_method: str = "Tsai-Lenz"
    debug_visualization: bool = True


# Available hand-eye calibration methods from OpenCV
HAND_EYE_METHODS = {
    "Tsai-Lenz": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def calibrate_intrinsics(
    images: Sequence[np.ndarray],
    charuco_config: CharucoProperties,
    debug_visualization: bool = False,
    output_dir: Optional[Path] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Perform intrinsic camera calibration using ChArUco board.

    Args:
        images: List of images (RGB or grayscale).
        charuco_config: ChArUco board configuration.
        debug_visualization: Whether to save debug images.
        output_dir: Directory to save debug images.

    Returns:
        Tuple of (camera_matrix, dist_coeffs, reprojection_error) if successful,
        None otherwise.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(charuco_config.aruco_dict)
    board = cv2.aruco.CharucoBoard(
        (charuco_config.squares_horizontally, charuco_config.squares_vertically),
        charuco_config.square_length,
        charuco_config.marker_length,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters()
    )

    all_corners = []
    all_ids = []

    debug_dir = None
    if debug_visualization and output_dir:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug images will be saved to {debug_dir}")

    for idx, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray
        )

        if charuco_corners is not None and len(charuco_corners) > 0:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

        if debug_visualization and debug_dir:
            _save_debug_image_intrinsics(
                image,
                idx,
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,
                debug_dir,
            )

    if len(all_corners) < 3:
        logger.error(f"Not enough valid corner detections: {len(all_corners)}")
        return None

    if debug_visualization and output_dir:
        cv2.destroyWindow("ChArUco Detection - Intrinsic Calibration")

    logger.info(f"Valid corner detections: {len(all_corners)}/{len(images)} images")

    # Calibrate
    image_size = images[0].shape[:2][::-1]  # (width, height)
    all_object_points = [board.getChessboardCorners()[ids.flatten()] for ids in all_ids]

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        all_object_points, all_corners, image_size, None, None
    )

    if ret:
        logger.info("Intrinsic calibration successful")
        logger.info(f"Reprojection error (RMS): {ret:.4f} pixels")
        return camera_matrix, dist_coeffs, ret

    logger.error("Intrinsic calibration failed")
    return None


def _save_debug_image_intrinsics(
    image: np.ndarray,
    idx: int,
    charuco_corners: Optional[np.ndarray],
    charuco_ids: Optional[np.ndarray],
    marker_corners: Optional[Sequence[np.ndarray]],
    marker_ids: Optional[np.ndarray],
    debug_dir: Path,
) -> None:
    """Save debug image for intrinsic calibration."""
    debug_img = (
        image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )

    if marker_corners is not None and len(marker_corners) > 0:
        cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)

    if charuco_corners is not None and len(charuco_corners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(
            debug_img, charuco_corners, charuco_ids, (0, 255, 0)
        )
        status = f"OK: {len(charuco_corners)} corners"
        color = (0, 255, 0)
    else:
        status = "FAILED: No corners detected"
        color = (0, 0, 255)

    cv2.putText(
        debug_img,
        f"Image {idx}: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    debug_path = debug_dir / f"intrinsic_detection_{idx:03d}.png"
    cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    # Show debug image
    cv2.imshow(
        "ChArUco Detection - Intrinsic Calibration",
        cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR),
    )
    cv2.waitKey(500)


def detect_charuco_poses(
    images: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    charuco_config: CharucoProperties,
    captured_poses: Optional[Sequence[np.ndarray]] = None,
    debug_visualization: bool = False,
    output_dir: Optional[Path] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """Detect ChArUco board poses in images.

    Args:
        images: List of images.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        charuco_config: ChArUco board configuration.
        captured_poses: Optional list of EE poses for debug visualization.
        debug_visualization: Whether to save debug images.
        output_dir: Directory to save debug images.

    Returns:
        Tuple of (T_cam2object_list, valid_indices).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(charuco_config.aruco_dict)
    board = cv2.aruco.CharucoBoard(
        (charuco_config.squares_horizontally, charuco_config.squares_vertically),
        charuco_config.square_length,
        charuco_config.marker_length,
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters()
    )

    T_cam2object_list = []
    valid_indices = []

    debug_dir = None
    if debug_visualization and output_dir:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Axis length for visualization (half the board size)
    axis_length = (
        charuco_config.square_length
        * min(charuco_config.squares_horizontally, charuco_config.squares_vertically)
        / 2
    )

    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray
        )

        rvec, tvec = None, None
        pose_success = False

        if (
            charuco_corners is not None
            and charuco_ids is not None
            and len(charuco_corners) >= 4
        ):
            # Use solvePnP instead of deprecated estimatePoseCharucoBoard
            obj_points, img_points = board.matchImagePoints(
                charuco_corners, charuco_ids
            )
            if obj_points is not None and len(obj_points) >= 4:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    img_points,
                    camera_matrix,
                    dist_coeffs,
                )

                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    T = T_from_t_R(tvec, R)
                    T_cam2object_list.append(T)
                    valid_indices.append(i)
                    pose_success = True

        if debug_visualization and debug_dir:
            _save_debug_image_poses(
                image,
                i,
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,
                rvec,
                tvec,
                pose_success,
                camera_matrix,
                dist_coeffs,
                axis_length,
                captured_poses,
                debug_dir,
            )

    if debug_visualization and output_dir:
        cv2.destroyWindow("Pose Detection")

    logger.info(f"Detected poses in {len(T_cam2object_list)}/{len(images)} images")
    return T_cam2object_list, valid_indices


def _save_debug_image_poses(
    image: np.ndarray,
    idx: int,
    charuco_corners: Optional[np.ndarray],
    charuco_ids: Optional[np.ndarray],
    marker_corners: Optional[Sequence[np.ndarray]],
    marker_ids: Optional[np.ndarray],
    rvec: Optional[np.ndarray],
    tvec: Optional[np.ndarray],
    pose_success: bool,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float,
    captured_poses: Optional[Sequence[np.ndarray]],
    debug_dir: Path,
) -> None:
    """Save debug image for pose detection."""
    debug_img = (
        image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )

    if marker_corners is not None and len(marker_corners) > 0:
        cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)

    if charuco_corners is not None and len(charuco_corners) > 0:
        cv2.aruco.drawDetectedCornersCharuco(
            debug_img, charuco_corners, charuco_ids, (0, 255, 0)
        )

    if pose_success and rvec is not None and tvec is not None:
        cv2.drawFrameAxes(
            debug_img,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            axis_length,
        )
        status = f"OK: dist={np.linalg.norm(tvec):.3f}m"
        color = (0, 255, 0)
    else:
        status = "FAILED: No pose"
        color = (0, 0, 255)

    cv2.putText(
        debug_img,
        f"Image {idx}: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    if captured_poses and idx < len(captured_poses):
        ee_pos = captured_poses[idx][:3, 3]
        cv2.putText(
            debug_img,
            f"EE pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    debug_path = debug_dir / f"pose_detection_{idx:03d}.png"
    cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    cv2.imshow("Pose Detection", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(500)


def calibrate_hand_eye(
    T_base2ee_list: Sequence[np.ndarray],
    T_cam2object_list: Sequence[np.ndarray],
    method_name: str = "Tsai-Lenz",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Perform hand-eye calibration.

    Args:
        T_base2ee_list: List of base-to-EE transformations.
        T_cam2object_list: List of camera-to-object transformations.
        method_name: Hand-eye calibration method name.

    Returns:
        Tuple of (T_ee2cam, R_cam2gripper, t_cam2gripper) if successful, None otherwise.
    """
    if method_name not in HAND_EYE_METHODS:
        logger.error(
            f"Unknown hand-eye method: {method_name}. "
            f"Available: {list(HAND_EYE_METHODS.keys())}"
        )
        return None

    # Extract rotation matrices and translation vectors
    R_gripper2base_list = []
    t_gripper2base_list = []
    for T in T_base2ee_list:
        t, R = t_R_from_T(T)
        R_gripper2base_list.append(R)
        t_gripper2base_list.append(t.reshape(3, 1))

    R_target2cam_list = []
    t_target2cam_list = []
    for T in T_cam2object_list:
        t, R = t_R_from_T(T)
        R_target2cam_list.append(R)
        t_target2cam_list.append(t.reshape(3, 1))

    # Run OpenCV hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base_list,
        t_gripper2base=t_gripper2base_list,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        method=HAND_EYE_METHODS[method_name],
    )

    T_ee2cam = T_from_t_R(t_cam2gripper, R_cam2gripper)
    t_cam = t_cam2gripper.flatten()
    R_euler = Rotation.from_matrix(R_cam2gripper).as_euler("xyz", degrees=True)

    logger.info(f"Hand-eye calibration successful using {method_name}")
    logger.info(
        f"Translation (EE to Camera): [{t_cam[0]:.4f}, {t_cam[1]:.4f}, {t_cam[2]:.4f}] m"
    )
    logger.info(
        f"Rotation (Euler XYZ deg): [{R_euler[0]:.2f}, {R_euler[1]:.2f}, {R_euler[2]:.2f}]"
    )

    return T_ee2cam, R_cam2gripper, t_cam2gripper


def run_calibration_standalone(
    data_dir: str | Path = "calibration_data",
    config: CalibrationConfig | None = None,
) -> dict[str, Any] | None:
    """Run calibration using data from disk.

    Args:
        data_dir: Directory containing 'images' and 'poses' subdirectories.
        config: Calibration configuration.

    Returns:
        Dictionary with calibration results if successful, None otherwise.
    """
    data_dir = Path(data_dir)
    config = config or CalibrationConfig()
    config.output_dir = data_dir

    images_dir = data_dir / "images"
    poses_dir = data_dir / "poses"

    if not images_dir.exists() or not poses_dir.exists():
        logger.error(f"Data directories not found in {data_dir}")
        return None

    # Load images
    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        logger.error(f"No images found in {images_dir}")
        return None

    images = []
    for f in image_files:
        if (img := cv2.imread(str(f))) is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            logger.warning(f"Failed to load image {f}")

    # Load poses
    pose_files = sorted(poses_dir.glob("*.npy"))
    if not pose_files:
        logger.error(f"No poses found in {poses_dir}")
        return None

    if len(images) != len(pose_files):
        logger.error(
            f"Mismatch between images ({len(images)}) and poses ({len(pose_files)})"
        )
        return None

    captured_poses = [np.load(str(f)) for f in pose_files]
    logger.info(f"Loaded {len(images)} samples from {data_dir}")

    # 1. Intrinsic Calibration
    result = calibrate_intrinsics(
        images,
        config.charuco,
        debug_visualization=config.debug_visualization,
        output_dir=config.output_dir,
    )
    if result is None:
        return None
    camera_matrix, dist_coeffs, reprojection_error = result

    # Save intrinsic calibration
    intrinsic_path = config.output_dir / "intrinsic_calibration.npz"
    np.savez(
        str(intrinsic_path),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=reprojection_error,
        image_size=images[0].shape[:2][::-1],
    )
    logger.info(f"Saved intrinsic calibration to {intrinsic_path}")

    # 2. Detect Poses
    T_cam2object_list, valid_indices = detect_charuco_poses(
        images,
        camera_matrix,
        dist_coeffs,
        config.charuco,
        captured_poses=captured_poses,
        debug_visualization=config.debug_visualization,
        output_dir=config.output_dir,
    )

    if len(T_cam2object_list) < 3:
        logger.error(f"Not enough valid pose detections: {len(T_cam2object_list)}")
        return None

    # 3. Hand-Eye Calibration
    T_base2ee_list = [captured_poses[i] for i in valid_indices]

    he_result = calibrate_hand_eye(
        T_base2ee_list,
        T_cam2object_list,
        method_name=config.hand_eye_method,
    )
    if he_result is None:
        return None
    T_ee2cam, _, _ = he_result

    # Save result
    output_path = config.output_dir / "calibration_result.npz"
    np.savez(
        str(output_path),
        T_ee2cam=T_ee2cam,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    logger.info(f"Saved calibration result to {output_path}")

    return {
        "T_ee2cam": T_ee2cam,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
    }


def save_calibration_result(
    T_ee2cam: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    output_dir: Path,
    filepath: str | Path | None = None,
) -> Path:
    """Save calibration results to disk.

    Args:
        T_ee2cam: End-effector to camera transformation.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        output_dir: Directory to save calibration data if filepath is None.
        filepath: Path to save calibration. If None, uses default location in output_dir.

    Returns:
        Path where calibration was saved.
    """
    if filepath is None:
        filepath = output_dir / "calibration_result.npz"
    else:
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(filepath),
        T_ee2cam=T_ee2cam,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    logger.info(f"Saved calibration to {filepath}")
    return filepath


def load_calibration_result(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load calibration results from disk.

    Args:
        filepath: Path to calibration file.

    Returns:
        Dictionary with T_ee2cam, camera_matrix, and dist_coeffs.
    """
    filepath = Path(filepath)
    data = np.load(str(filepath))
    result = {
        "T_ee2cam": data["T_ee2cam"],
        "camera_matrix": data["camera_matrix"],
        "dist_coeffs": data["dist_coeffs"],
    }
    logger.info(f"Loaded calibration from {filepath}")
    return result


def generate_charuco_board_pdf(
    charuco_config: CharucoProperties,
    output_dir: Path,
    filepath: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Generate a PDF of the ChArUco calibration board with printing instructions.

    Uses matplotlib to generate a properly-sized PDF that can be printed
    at the correct physical dimensions.

    Args:
        charuco_config: ChArUco board configuration.
        output_dir: Directory to save the PDF if filepath is None.
        filepath: Output path for the PDF. If None, saves to output_dir.
        dpi: Resolution for the board image (300 recommended for printing).

    Returns:
        Path to the generated PDF file.
    """
    # Calculate board dimensions in mm and inches
    board_width_mm = (
        charuco_config.squares_horizontally * charuco_config.square_length * 1000
    )
    board_height_mm = (
        charuco_config.squares_vertically * charuco_config.square_length * 1000
    )
    marker_length_mm = charuco_config.marker_length * 1000
    square_length_mm = charuco_config.square_length * 1000

    # Convert to inches for matplotlib (1 inch = 25.4 mm)
    board_width_inch = board_width_mm / 25.4
    board_height_inch = board_height_mm / 25.4

    # Generate ChArUco board image
    dictionary = cv2.aruco.getPredefinedDictionary(charuco_config.aruco_dict)
    board = cv2.aruco.CharucoBoard(
        (charuco_config.squares_horizontally, charuco_config.squares_vertically),
        charuco_config.square_length,
        charuco_config.marker_length,
        dictionary,
    )

    # Generate high-resolution board image
    pixels_per_meter = dpi / 0.0254
    img_width = int(
        charuco_config.squares_horizontally
        * charuco_config.square_length
        * pixels_per_meter
    )
    img_height = int(
        charuco_config.squares_vertically
        * charuco_config.square_length
        * pixels_per_meter
    )
    board_image = board.generateImage(
        (img_width, img_height), marginSize=0, borderBits=1
    )

    # Set output filepath
    if filepath is None:
        filepath = output_dir / "charuco_board.pdf"
    else:
        filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create PDF with matplotlib
    with PdfPages(str(filepath)) as pdf:
        # Page 1: Instructions
        fig_instructions = plt.figure(figsize=(8.5, 11))  # Letter size
        ax = fig_instructions.add_subplot(111)
        ax.axis("off")

        instructions_text = f"""
ChArUco Calibration Board - Printing Instructions
{'='*50}

BOARD SPECIFICATIONS:
  • Board size: {charuco_config.squares_horizontally} x {charuco_config.squares_vertically} squares
  • Square size: {square_length_mm:.1f} mm ({square_length_mm/25.4:.3f} inches)
  • Marker size: {marker_length_mm:.1f} mm
  • Total dimensions: {board_width_mm:.1f} x {board_height_mm:.1f} mm
  • ArUco dictionary: DICT_6X6_250

PRINTING INSTRUCTIONS (CRITICAL):
{'─'*50}

1. PRINT AT 100% SCALE
   • Do NOT use "Fit to Page" or "Shrink to Fit"
   • Select "Actual Size" or "100%" in print dialog
   • In Adobe Reader: Print → Page Sizing: "Actual Size"
   • In Chrome: Print → More Settings → Scale: 100%

2. VERIFY THE SIZE AFTER PRINTING
   • Measure any square with a ruler
   • It MUST be exactly {square_length_mm:.1f} x {square_length_mm:.1f} mm
   • If wrong, adjust printer settings and reprint

3. PAPER AND QUALITY
   • Use matte paper (avoid glossy - causes reflections)
   • Use highest quality print settings
   • Ensure sharp, crisp edges on markers

4. MOUNTING
   • Mount on a flat, rigid surface
   • Use foam board, MDF, or thick cardboard
   • Ensure board stays PERFECTLY FLAT during calibration
   • Any warping will reduce calibration accuracy

5. LIGHTING DURING CALIBRATION
   • Use diffuse lighting (avoid harsh shadows)
   • Avoid reflections on the board surface
   • Ensure consistent lighting across the board

VERIFICATION CHECKLIST:
☐ Printed at 100% scale (no scaling)
☐ Square measures exactly {square_length_mm:.1f} mm
☐ Mounted on flat, rigid surface
☐ No warping or bending
☐ Sharp, clear marker edges
"""
        ax.text(
            0.05,
            0.95,
            instructions_text,
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
        )
        pdf.savefig(fig_instructions, dpi=150)
        plt.close(fig_instructions)

        # Page 2: The actual board (sized correctly for printing)
        # Add margin for the title and verification text
        margin_inch = 0.5
        title_space_inch = 0.6
        footer_space_inch = 0.4

        fig_board = plt.figure(
            figsize=(
                board_width_inch + 2 * margin_inch,
                board_height_inch
                + title_space_inch
                + footer_space_inch
                + 2 * margin_inch,
            )
        )

        # Add title
        fig_board.text(
            0.5,
            0.98,
            f"ChArUco Board - {charuco_config.squares_horizontally}x{charuco_config.squares_vertically} - "
            f"Square: {square_length_mm:.1f}mm",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        # Add the board image
        ax_board = fig_board.add_axes(
            [
                margin_inch / fig_board.get_figwidth(),
                (margin_inch + footer_space_inch) / fig_board.get_figheight(),
                board_width_inch / fig_board.get_figwidth(),
                board_height_inch / fig_board.get_figheight(),
            ]
        )
        ax_board.imshow(board_image, cmap="gray", aspect="equal")
        ax_board.axis("off")

        # Add verification text at bottom
        fig_board.text(
            0.5,
            0.02,
            f"VERIFY: Each square should measure exactly {square_length_mm:.1f} x {square_length_mm:.1f} mm",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="red",
        )

        pdf.savefig(fig_board, dpi=dpi)
        plt.close(fig_board)

    # Also save just the board image as PNG for reference
    board_png_path = output_dir / "charuco_board.png"
    cv2.imwrite(str(board_png_path), board_image)

    logger.info(f"Generated ChArUco board PDF: {filepath}")
    logger.info(f"Also saved board image: {board_png_path}")
    logger.info(f"Board specifications:")
    logger.info(
        f"  - Size: {charuco_config.squares_horizontally}x{charuco_config.squares_vertically} squares"
    )
    logger.info(f"  - Square size: {square_length_mm:.1f} mm")
    logger.info(f"  - Marker size: {marker_length_mm:.1f} mm")
    logger.info(
        f"  - Total dimensions: {board_width_mm:.1f} x {board_height_mm:.1f} mm"
    )
    logger.info("")
    logger.info("PRINTING INSTRUCTIONS:")
    logger.info("  1. Print at 100% scale (Actual Size)")
    logger.info("  2. DO NOT use 'Fit to Page' or scaling")
    logger.info(
        f"  3. Verify by measuring a square: should be {square_length_mm:.1f} mm"
    )
    logger.info("  4. Mount on flat, rigid surface")

    return filepath


class CameraEyeInHandCalibrationAgent(Agent):
    """Agent for performing eye-in-hand camera calibration.

    This agent moves the robot to predefined calibration poses, captures images
    at each pose, and computes the transformation from the end-effector to the
    camera frame using hand-eye calibration.

    The calibration process:
    1. (Optional) Manually select calibration poses via teleoperation
    2. Move to each calibration pose in sequence
    3. Capture an image and record the end-effector pose
    4. Detect ChArUco board pose in each image
    5. Perform intrinsic camera calibration (if needed)
    6. Perform hand-eye calibration to find T_ee2cam

    Keyboard controls during manual pose selection:
        - 'c': Capture current pose as calibration pose
        - 's': End pose selection and start calibration
        - 'q': Quit calibration

    Attributes:
        config: Calibration configuration.
        calibration_poses: List of target poses [x, y, z, qx, qy, qz, qw].
        current_pose_index: Index of the current calibration pose.
        state: Current state of the calibration process.
    """

    class State:
        """Calibration state enumeration."""

        IDLE = "idle"
        MANUAL_POSE_SELECTION = "manual_pose_selection"
        MOVING_TO_POSE = "moving_to_pose"
        WAITING = "waiting"
        CAPTURING = "capturing"
        PROCESSING = "processing"
        COMPLETE = "complete"

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        calibration_poses: Optional[Sequence[np.ndarray]] = None,
        poses_file: Optional[Union[str, Path]] = None,
        generate_poses_new: bool = False,
        camera_name: str = "wrist_camera",
        input_controller: Optional[Any] = None,
    ) -> None:
        """Initialize the camera eye-in-hand calibration agent.

        Args:
            config: Calibration configuration. Uses defaults if None.
            calibration_poses: List of poses to visit for calibration.
                Each pose is [x, y, z, qx, qy, qz, qw].
            poses_file: Path to a file containing calibration poses. If provided
                and the file exists, poses are loaded from it. Otherwise, the
                agent starts in manual pose selection mode.
            camera_name: Name of the camera in observations dict.
            input_controller: Input controller for manual pose selection.
                If None, a SpaceMouseController is created.
        """
        super().__init__(action_type="OSC_POSE")
        self.config = config or CalibrationConfig()
        self.calibration_poses = list(calibration_poses) if calibration_poses else []
        self.camera_name = camera_name

        # State tracking
        self.current_pose_index = 0
        self.state = self.State.IDLE
        self.wait_start_time = 0.0
        self.gripper_action = 0.5  # Keep gripper open

        # Collected calibration data
        self.captured_images: List[np.ndarray] = []
        self.captured_poses: List[np.ndarray] = []  # T_base2ee matrices
        self.capture_pending = False
        self.osc_controller: Optional[OscPoseTargetController] = None

        # Calibration results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.T_ee2cam: Optional[np.ndarray] = None

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Keyboard control state for manual pose selection
        self._keyboard_control: Dict[str, bool] = {
            "capture_pose": False,
            "end_selection": False,
            "quit": False,
        }
        self._keyboard_listener: Optional[keyboard.Listener] = None

        # Teleoperation agent for manual control
        self._input_controller = input_controller
        self._teleop_agent: Optional[TeleoperationAgent] = None

        # Handle poses_file: load if exists, otherwise start manual selection
        self._poses_file = (
            Path(poses_file) if poses_file and not generate_poses_new else None
        )
        self._use_manual_selection = True

        if self._poses_file and self._poses_file.exists():
            self.load_poses_from_file(self._poses_file)
            self._use_manual_selection = False
            logger.info(
                f"Loaded {len(self.calibration_poses)} poses from {self._poses_file}"
            )
        elif self.calibration_poses:
            self._use_manual_selection = False
            logger.info(
                f"Using {len(self.calibration_poses)} provided calibration poses"
            )

    def _print_controls(self) -> None:
        """Print keyboard controls to console."""
        print("\n" + "=" * 60)
        print("Camera Eye-in-Hand Calibration - Manual Pose Selection")
        print("=" * 60)
        print("Use the SpaceMouse to move the robot to calibration poses.")
        print("Make sure the ChArUco board is visible in the camera view.")
        print("\nKeyboard Controls:")
        print("  'c' - Capture current pose as calibration pose")
        print("  's' - Start calibration (end pose selection)")
        print("  'q' - Quit calibration")
        print("=" * 60 + "\n")

    def _start_keyboard_listener(self) -> None:
        """Start the keyboard listener for manual pose selection."""
        if self._keyboard_listener is not None:
            return

        self._keyboard_control = {
            "capture_pose": False,
            "end_selection": False,
            "quit": False,
        }
        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self._keyboard_listener.start()

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _on_key_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle keyboard key press events."""
        try:
            if hasattr(key, "char") and key.char is not None:
                if key.char == "c":
                    self._keyboard_control["capture_pose"] = True
                elif key.char == "s":
                    self._keyboard_control["end_selection"] = True
                elif key.char == "q":
                    self._keyboard_control["quit"] = True
        except AttributeError:
            pass

    def _init_teleoperation(self) -> None:
        """Initialize the teleoperation agent for manual control."""
        if self._teleop_agent is not None:
            return

        if self._input_controller is None:
            self._input_controller = SpaceMouseController()

        self._teleop_agent = TeleoperationAgent(self._input_controller)
        logger.info("Initialized teleoperation agent for manual pose selection")

    def add_calibration_pose(self, pose: np.ndarray) -> None:
        """Add a calibration pose to the list.

        Args:
            pose: Target pose [x, y, z, qx, qy, qz, qw].
        """
        pose = np.asarray(pose).flatten()
        if pose.shape != (7,):
            raise ValueError(f"Pose must have 7 elements, got {pose.shape}")
        self.calibration_poses.append(pose.copy())
        logger.info(f"Added calibration pose {len(self.calibration_poses)}: {pose}")

    def add_current_pose(self, robot_state: dict[str, Any]) -> None:
        """Add the current robot pose as a calibration pose.

        Args:
            robot_state: Current robot state containing position and rotation.
        """
        pos = robot_state["osc_position"].flatten()
        rot = robot_state["osc_rotation_quaternion"].flatten()
        pose = np.concatenate([pos, rot])
        self.add_calibration_pose(pose)

    def clear_calibration_poses(self) -> None:
        """Clear all calibration poses."""
        self.calibration_poses.clear()
        logger.info("Cleared all calibration poses")

    def start(self) -> None:
        """Start the calibration agent.

        If poses are already loaded, starts the automatic calibration.
        Otherwise, enters manual pose selection mode.
        """
        if self._use_manual_selection:
            self._start_manual_pose_selection()
        else:
            self._start_automatic_calibration()

    def _start_manual_pose_selection(self) -> None:
        """Enter manual pose selection mode."""
        self._print_controls()
        self._init_teleoperation()
        self._start_keyboard_listener()
        self.state = self.State.MANUAL_POSE_SELECTION
        logger.info("Entered manual pose selection mode")

    def _start_automatic_calibration(self) -> None:
        """Start the automatic calibration process."""
        if len(self.calibration_poses) < 3:
            logger.error(
                f"Need at least 3 calibration poses, have {len(self.calibration_poses)}. "
                "Falling back to manual pose selection."
            )
            self._use_manual_selection = True
            self._start_manual_pose_selection()
            return

        self.current_pose_index = 0
        self.captured_images.clear()
        self.captured_poses.clear()
        self.state = self.State.MOVING_TO_POSE
        logger.info(
            f"Starting automatic calibration with {len(self.calibration_poses)} poses"
        )

    def start_calibration(self) -> None:
        """Start the calibration process (legacy method).

        Deprecated: Use start() instead.
        """
        self._start_automatic_calibration()
        logger.info(f"Starting calibration with {len(self.calibration_poses)} poses")

    def _handle_manual_pose_selection(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Handle the manual pose selection state.

        Args:
            robot_state: Current robot state.
            observation: Sensor observations.
            instruction: Instruction string.
            metadata: Metadata dict to update.

        Returns:
            Tuple of (action, metadata).
        """
        # Check for quit
        if self._keyboard_control["quit"]:
            self._keyboard_control["quit"] = False
            self._stop_keyboard_listener()
            self.state = self.State.IDLE
            metadata["quit"] = True
            logger.info("Calibration cancelled by user")
            action = np.zeros(7, dtype=np.float32)
            return action, metadata

        # Check for capture pose
        if self._keyboard_control["capture_pose"]:
            self._keyboard_control["capture_pose"] = False
            self.add_current_pose(robot_state)
            print(
                f"  >> Captured pose {len(self.calibration_poses)}. "
                f"Press 'c' for more, 's' to start calibration."
            )

        # Check for end selection
        if self._keyboard_control["end_selection"]:
            self._keyboard_control["end_selection"] = False

            if len(self.calibration_poses) < 3:
                print(
                    f"  >> Need at least 3 poses, have {len(self.calibration_poses)}. "
                    "Please capture more poses."
                )
            else:
                # Save poses and start calibration
                self._stop_keyboard_listener()
                poses_path = self.save_poses_to_file()
                print(f"  >> Saved {len(self.calibration_poses)} poses to {poses_path}")
                print("  >> Starting automatic calibration...")
                self._start_automatic_calibration()

                # Return the first action from the new state
                return self.act(robot_state, observation, instruction)

        # Use teleoperation agent for movement
        if self._teleop_agent is not None:
            action, teleop_metadata = self._teleop_agent.act(
                robot_state, observation, instruction
            )
            metadata.update(teleop_metadata)
        else:
            action = np.zeros(7, dtype=np.float32)

        metadata["num_captured_poses"] = len(self.calibration_poses)
        return action, metadata

    def act(
        self,
        robot_state: dict[str, Any],
        observation: dict[str, Any],
        instruction: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate action based on calibration state.

        Args:
            robot_state: Current robot state.
            observation: Sensor observations including camera images.
            instruction: Instruction string (unused).

        Returns:
            Tuple of (action, metadata).
        """
        metadata = {
            "action_type": self.action_type,
            "calibration_state": self.state,
            "current_pose_index": self.current_pose_index,
            "total_poses": len(self.calibration_poses),
        }

        if self.state == self.State.IDLE:
            # Return zero action when idle
            action = np.zeros(7, dtype=np.float32)
            return action, metadata

        if self.state == self.State.MANUAL_POSE_SELECTION:
            return self._handle_manual_pose_selection(
                robot_state, observation, instruction, metadata
            )

        if self.state == self.State.COMPLETE:
            # Calibration complete, return zero action
            action = np.zeros(7, dtype=np.float32)
            return action, metadata

        if self.state == self.State.MOVING_TO_POSE:
            # Move to current target pose
            target_pose = self.calibration_poses[self.current_pose_index]
            action, at_target = self._move_to_pose(robot_state, target_pose)

            if at_target:
                self.state = self.State.WAITING
                self.wait_start_time = time.time()
                logger.info(
                    f"Reached pose {self.current_pose_index + 1}/{len(self.calibration_poses)}. Waiting for stability..."
                )

            return action, metadata

        if self.state == self.State.WAITING:
            # Wait for robot to settle
            if time.time() - self.wait_start_time >= self.config.settling_time:
                self.state = self.State.CAPTURING
                self.capture_pending = True
                logger.info("Stability wait complete. Capturing...")

            # Maintain position
            target_pose = self.calibration_poses[self.current_pose_index]
            # action, _ = self._move_to_pose(robot_state, target_pose)
            action = np.zeros(7, dtype=np.float32)
            return action, metadata

        if self.state == self.State.CAPTURING:
            # Capture image and pose
            if self.capture_pending:
                self._capture_sample(robot_state, observation)
                self.capture_pending = False

                # Move to next pose or process
                self.current_pose_index += 1
                if self.current_pose_index >= len(self.calibration_poses):
                    self.state = self.State.PROCESSING
                    logger.info("All poses captured, starting calibration processing")
                else:
                    self.state = self.State.MOVING_TO_POSE

            action = np.zeros(7, dtype=np.float32)
            return action, metadata

        if self.state == self.State.PROCESSING:
            # Process calibration data
            success = self._run_calibration()
            if success:
                self.state = self.State.COMPLETE
                logger.info("Calibration complete!")
            else:
                logger.error("Calibration failed")
                self.state = self.State.IDLE

            action = np.zeros(7, dtype=np.float32)
            return action, metadata

        # Fallback
        action = np.zeros(7, dtype=np.float32)
        return action, metadata

    def _move_to_pose(
        self, robot_state: dict[str, Any], target_pose: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Move towards target pose using OSC_POSE control.

        Args:
            robot_state: Current robot state.
            target_pose: Target pose [x, y, z, qx, qy, qz, qw].

        Returns:
            Tuple of (action, at_target).
        """
        pos = robot_state["osc_position"].flatten()
        rot = robot_state["osc_rotation_quaternion"].flatten()
        current_pose = np.concatenate([pos, rot])

        # Compute command with gripper action
        command = np.concatenate([target_pose, [self.gripper_action]])

        if self.osc_controller is None or not np.array_equal(
            self.osc_controller.target_pose, command
        ):
            self.osc_controller = OscPoseTargetController(
                target_pose=command,
                threshold_reach=self.config.position_tolerance,
                threshold_rotation=self.config.rotation_tolerance,
            )

        action, at_target = self.osc_controller.calculate_action(current_pose)

        if not at_target:
            # We don't have easy access to errors here anymore, so we just log moving
            logger.debug(f"Moving to pose: {target_pose}")

        return action, at_target

    def _capture_sample(
        self, robot_state: Dict[str, Any], observation: Dict[str, Any]
    ) -> None:
        """Capture image and pose at current position.

        Args:
            robot_state: Current robot state.
            observation: Sensor observations.
        """

        if (image := observation[self.camera_name]) is None:
            logger.warning(
                f"Captured image is None for observation key '{self.camera_name}'"
            )
            return

        # Get current pose as transformation matrix
        T_base2ee = pos_rot_to_transformation_matrix(
            robot_state["osc_position"].flatten(),
            robot_state["osc_rotation_quaternion"].flatten(),
        )

        # Store data
        self.captured_images.append(image.copy())
        self.captured_poses.append(T_base2ee.copy())

        # Save to disk
        sample_num = len(self.captured_images)
        self._save_sample(image, T_base2ee, sample_num)

        logger.info(f"Captured sample {sample_num}")

    def _save_sample(
        self, image: np.ndarray, T_base2ee: np.ndarray, sample_num: int
    ) -> None:
        """Save captured sample to disk.

        Args:
            image: Captured RGB image.
            T_base2ee: End-effector pose as 4x4 transformation matrix.
            sample_num: Sample number for filename.
        """
        output_dir = self.config.output_dir
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "poses").mkdir(exist_ok=True)

        # Save image
        img_path = output_dir / "images" / f"image_{sample_num:03d}.png"
        # Convert RGB to BGR for OpenCV
        cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save pose
        pose_path = output_dir / "poses" / f"pose_{sample_num:03d}.npy"
        np.save(str(pose_path), T_base2ee)

    def _run_calibration(self) -> bool:
        """Run the full calibration pipeline.

        Returns:
            True if calibration succeeded, False otherwise.
        """
        if len(self.captured_images) < 3:
            logger.error(f"Need at least 3 samples, have {len(self.captured_images)}")
            return False

        # Run intrinsic calibration
        success = self._calibrate_intrinsics()
        if not success:
            return False

        # Detect charuco poses in all images
        T_cam2object_list, valid_indices = self._detect_charuco_poses()
        if len(T_cam2object_list) < 3:
            logger.error(f"Not enough valid pose detections: {len(T_cam2object_list)}")
            return False

        # Run hand-eye calibration
        success = self._calibrate_hand_eye(T_cam2object_list, valid_indices)
        return success

    def _calibrate_intrinsics(self) -> bool:
        """Perform intrinsic camera calibration using ChArUco board.

        Returns:
            True if calibration succeeded, False otherwise.
        """
        result = calibrate_intrinsics(
            self.captured_images,
            self.config.charuco,
            debug_visualization=self.config.debug_visualization,
            output_dir=self.config.output_dir,
        )

        if result:
            self.camera_matrix, self.dist_coeffs, reprojection_error = result

            # Log detailed calibration info
            image_size = self.captured_images[0].shape[:2][::-1]
            logger.info("Intrinsic calibration successful")
            logger.info(f"Reprojection error (RMS): {reprojection_error:.4f} pixels")
            logger.info(f"Image size: {image_size}")
            logger.info(f"Camera matrix:\n{self.camera_matrix}")
            logger.info(
                f"Focal lengths: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}"
            )
            logger.info(
                f"Principal point: cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}"
            )
            logger.info(f"Distortion coefficients: {self.dist_coeffs.flatten()}")

            # Save intrinsic calibration separately
            intrinsic_path = self.config.output_dir / "intrinsic_calibration.npz"
            np.savez(
                str(intrinsic_path),
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                reprojection_error=reprojection_error,
                image_size=image_size,
            )
            logger.info(f"Saved intrinsic calibration to {intrinsic_path}")

            return True
        else:
            return False

    def _detect_charuco_poses(self) -> tuple[list[np.ndarray], list[int]]:
        """Detect ChArUco board poses in all captured images.

        Returns:
            Tuple of (T_cam2object_list, valid_indices) where T_cam2object_list
            contains valid pose detections and valid_indices are the corresponding
            image indices.
        """
        return detect_charuco_poses(
            self.captured_images,
            self.camera_matrix,
            self.dist_coeffs,
            self.config.charuco,
            captured_poses=self.captured_poses,
            debug_visualization=self.config.debug_visualization,
            output_dir=self.config.output_dir,
        )

    def _calibrate_hand_eye(
        self, T_cam2object_list: list[np.ndarray], valid_indices: list[int]
    ) -> bool:
        """Perform hand-eye calibration.

        Args:
            T_cam2object_list: List of camera-to-object transformations.
            valid_indices: Indices of valid samples in captured_poses.

        Returns:
            True if calibration succeeded, False otherwise.
        """
        T_base2ee_list = [self.captured_poses[i] for i in valid_indices]

        result = calibrate_hand_eye(
            T_base2ee_list,
            T_cam2object_list,
            method_name=self.config.hand_eye_method,
        )

        if result:
            self.T_ee2cam, _, _ = result

            # Validate result - check if transformation is reasonable
            t_cam = self.T_ee2cam[:3, 3]
            t_norm = np.linalg.norm(t_cam)
            if t_norm > 0.5:  # Camera more than 50cm from EE seems suspicious
                logger.warning(
                    f"Camera-to-EE distance ({t_norm:.3f}m) seems large. Verify calibration."
                )
            elif t_norm < 0.01:  # Camera less than 1cm from EE seems suspicious
                logger.warning(
                    f"Camera-to-EE distance ({t_norm:.3f}m) seems very small. Verify calibration."
                )
            else:
                logger.info(f"Camera-to-EE distance: {t_norm:.4f}m")

            return True

        return False

    def save_calibration(self, filepath: str | Path | None = None) -> Path:
        """Save calibration results to disk.

        Args:
            filepath: Path to save calibration. If None, uses default location.

        Returns:
            Path where calibration was saved.

        Raises:
            ValueError: If calibration has not been performed.
        """
        if self.T_ee2cam is None:
            raise ValueError("No calibration data to save. Run calibration first.")

        return save_calibration_result(
            self.T_ee2cam,
            self.camera_matrix,
            self.dist_coeffs,
            self.config.output_dir,
            filepath,
        )

    def load_calibration(self, filepath: str | Path) -> None:
        """Load calibration results from disk.

        Args:
            filepath: Path to calibration file.
        """
        result = load_calibration_result(filepath)
        self.T_ee2cam = result["T_ee2cam"]
        self.camera_matrix = result["camera_matrix"]
        self.dist_coeffs = result["dist_coeffs"]

    def get_calibration_result(self) -> dict[str, np.ndarray | None]:
        """Get calibration results.

        Returns:
            Dictionary with T_ee2cam, camera_matrix, and dist_coeffs.
        """
        return {
            "T_ee2cam": self.T_ee2cam,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
        }

    def generate_board_pdf(
        self,
        filepath: str | Path | None = None,
        dpi: int = 300,
    ) -> Path:
        """Generate a PDF of the ChArUco calibration board with printing instructions.

        Uses matplotlib to generate a properly-sized PDF that can be printed
        at the correct physical dimensions.

        Args:
            filepath: Output path for the PDF. If None, saves to output_dir.
            dpi: Resolution for the board image (300 recommended for printing).

        Returns:
            Path to the generated PDF file.

        Note:
            IMPORTANT PRINTING INSTRUCTIONS:
            1. Print at 100% scale (no scaling/fit-to-page)
            2. Disable any "fit to page" or "shrink to fit" options
            3. Select "Actual Size" in print dialog
            4. After printing, measure a square with a ruler to verify:
               - Square size should match config.charuco.square_length
               - For default config: squares should be exactly 30mm x 30mm
            5. If measurements don't match, check printer settings and reprint
            6. Mount the board on a flat, rigid surface (foam board, wood, etc.)
        """
        return generate_charuco_board_pdf(
            self.config.charuco,
            self.config.output_dir,
            filepath,
            dpi,
        )

    @staticmethod
    def generate_board_pdf_standalone(
        output_path: str | Path = "charuco_board.pdf",
        squares_horizontally: int = 6,
        squares_vertically: int = 9,
        square_length_m: float = 0.03,
        marker_length_m: float = 0.015,
        dpi: int = 300,
    ) -> Path:
        """Generate a ChArUco board PDF without instantiating the full agent.

        This is a convenience method for generating a calibration board without
        needing to set up the full calibration agent.

        Args:
            output_path: Output path for the PDF file.
            squares_horizontally: Number of squares horizontally.
            squares_vertically: Number of squares vertically.
            square_length_m: Square size in meters.
            marker_length_m: ArUco marker size in meters.
            dpi: Resolution for printing (300 recommended).

        Returns:
            Path to the generated PDF file.

        Example:
            >>> from franka_pipeline.agents import CameraEyeInHandCalibrationAgent
            >>> CameraEyeInHandCalibrationAgent.generate_board_pdf_standalone(
            ...     "my_board.pdf",
            ...     squares_horizontally=6,
            ...     squares_vertically=9,
            ...     square_length_m=0.03,  # 30mm squares
            ... )
        """
        charuco_config = CharucoProperties(
            squares_horizontally=squares_horizontally,
            squares_vertically=squares_vertically,
            square_length=square_length_m,
            marker_length=marker_length_m,
        )

        output_dir = (
            Path(output_path).parent
            if Path(output_path).parent != Path(".")
            else Path("calibration_data")
        )

        return generate_charuco_board_pdf(
            charuco_config,
            output_dir,
            filepath=output_path,
            dpi=dpi,
        )

    def load_poses_from_file(self, filepath: str | Path) -> None:
        """Load calibration poses from a file.

        Args:
            filepath: Path to numpy file containing poses.
                Expected shape: (N, 7) where each row is [x, y, z, qx, qy, qz, qw].
        """
        filepath = Path(filepath)
        poses = np.load(str(filepath))
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        for pose in poses:
            self.add_calibration_pose(pose)
        logger.info(f"Loaded {len(poses)} calibration poses from {filepath}")

    def save_poses_to_file(self, filepath: str | Path | None = None) -> Path:
        """Save current calibration poses to a file.

        Args:
            filepath: Path to save poses. If None, uses default location.

        Returns:
            Path where poses were saved.
        """
        if filepath is None:
            filepath = self.config.output_dir / "calibration_poses.npy"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        poses = np.array(self.calibration_poses)
        np.save(str(filepath), poses)
        logger.info(f"Saved {len(self.calibration_poses)} poses to {filepath}")
        return filepath

    def reset(self) -> None:
        """Reset the agent state."""
        self.current_pose_index = 0
        self.state = self.State.IDLE
        self.captured_images.clear()
        self.captured_poses.clear()
        self.capture_pending = False

        # Reset keyboard controls
        self._keyboard_control = {
            "capture_pose": False,
            "end_selection": False,
            "quit": False,
        }

    def stop(self) -> None:
        """Stop the calibration agent and clean up resources.

        Call this method when the agent is no longer needed to clean up
        the keyboard listener and teleoperation agent.
        """
        self._stop_keyboard_listener()
        if self._teleop_agent is not None:
            self._teleop_agent.reset()
            self._teleop_agent = None
        logger.info("Calibration agent stopped")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()
