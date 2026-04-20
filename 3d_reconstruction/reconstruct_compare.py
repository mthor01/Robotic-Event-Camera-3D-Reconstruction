"""
Side-by-side 3D reconstruction: ground-truth depth vs model prediction.

Builds two TSDF volumes from the same recording — one using the projected
GT depth and one using E2DepthNet predictions on precomputed voxels.
Both are masked by the spatial cube mask.  The two meshes are placed
next to each other in one OBJ file for visual comparison.

Reads:
    <object_dir>/hdf5/depth_in_event_frame.h5   – GT depth in event cam (N, H, W) float32 m
    <object_dir>/hdf5/spatial_mask.h5            – cube mask (N, H, W) uint8
    <object_dir>/hdf5/poses.h5                   – ee_T (N, 4, 4)
    <object_dir>/events/voxels_cam0/             – voxel grids (or voxels_pose_cam0/)
    camera_data/event_intrinsics.npz             – event cam K
    camera_data/T_event_from_rgb.npz + T_rgb_from_ee.npz
    checkpoints_e2depth/best.pt                  – trained model

Outputs:
    <object_dir>/mesh_compare.obj  – GT mesh (left) and predicted mesh (right)

Usage:
    python3 reconstruct_compare.py data/real/block/1

    # Custom checkpoint:
    python3 reconstruct_compare.py data/real/block/1 --checkpoint checkpoints_e2depth/epoch_010.pt

    # Adjust TSDF parameters:
    python3 reconstruct_compare.py data/real/block/1 --voxel_size 0.004 --skip 2
"""

import argparse
from pathlib import Path

import numpy as np
import h5py
import torch
import open3d as o3d

from reconstruction_config import (
    D_MAX, DEPTH_MIN, SPATIAL_CUBE_SIDE,
    TSDF_VOXEL_SIZE, TSDF_SDF_TRUNC_FACTOR, TSDF_DEPTH_MIN, TSDF_DEPTH_MAX,
    CALIB_DIR as _CALIB_DIR_REL,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CAM_DATA = SCRIPT_DIR / _CALIB_DIR_REL
DEFAULT_CKPT = SCRIPT_DIR / "checkpoints_e2depth" / "best.pt"


# ─── calibration ─────────────────────────────────────────────────────
def load_event_calibration():
    """Load transforms and intrinsics for event-camera-frame reconstruction."""
    ev = np.load(CAM_DATA / "event_intrinsics.npz")
    K_event = ev["camera_matrix"]            # (3,3)
    ev_size = ev["image_size"]               # [W, H]

    T_rgb_from_ee = np.load(CAM_DATA / "T_rgb_from_ee.npz")["T"]
    T_event_from_rgb = np.load(CAM_DATA / "T_event_from_rgb.npz")["T"]
    T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee
    T_ee_from_event = np.linalg.inv(T_event_from_ee)

    return {
        "K_event": K_event,
        "ev_w": int(ev_size[0]),
        "ev_h": int(ev_size[1]),
        "T_ee_from_event": T_ee_from_event,
    }


def make_o3d_intrinsic(K, width, height):
    return o3d.camera.PinholeCameraIntrinsic(
        width, height,
        float(K[0, 0]), float(K[1, 1]),
        float(K[0, 2]), float(K[1, 2]),
    )


# ─── model loading ───────────────────────────────────────────────────
def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained E2DepthNet from checkpoint."""
    from real_train import E2DepthNet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = E2DepthNet(
        in_channels=cfg.get("in_channels", 5),
        base=cfg.get("base", 32),
        num_encoders=cfg.get("num_encoders", 3),
        num_residuals=cfg.get("num_residuals", 2),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    depth_min = cfg.get("depth_min", DEPTH_MIN)
    depth_max = cfg.get("depth_max", D_MAX)
    use_pose = cfg.get("use_pose", False)
    return model, depth_min, depth_max, use_pose


# ─── TSDF integration ────────────────────────────────────────────────
def integrate_depth_map(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    depth_m: np.ndarray,
    mask: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    depth_max: float,
):
    """Integrate a single masked depth frame into the TSDF volume."""
    depth_masked = depth_m.copy()
    depth_masked[mask == 0] = 0.0

    depth_o3d = o3d.geometry.Image(depth_masked.astype(np.float32))
    H, W = depth_masked.shape
    # Dummy grey colour — we only care about geometry
    color = np.full((H, W, 3), 128, dtype=np.uint8)
    color_o3d = o3d.geometry.Image(color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False,
    )
    volume.integrate(rgbd, intrinsic, extrinsic)


def extract_mesh(volume):
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


# ─── main reconstruction ─────────────────────────────────────────────
def reconstruct_compare(
    object_dir: Path,
    checkpoint: Path = DEFAULT_CKPT,
    voxel_size: float = TSDF_VOXEL_SIZE,
    sdf_trunc_factor: float = TSDF_SDF_TRUNC_FACTOR,
    depth_min_tsdf: float = TSDF_DEPTH_MIN,
    depth_max_tsdf: float = TSDF_DEPTH_MAX,
    skip: int = 1,
    output_format: str = "obj",
    device_name: str = "cuda",
):
    object_dir = Path(object_dir)
    hdf5_dir = object_dir / "hdf5"

    # ── calibration ──────────────────────────────────────────────────
    calib = load_event_calibration()
    K_event = calib["K_event"]
    ev_w, ev_h = calib["ev_w"], calib["ev_h"]
    T_ee_from_event = calib["T_ee_from_event"]
    intrinsic = make_o3d_intrinsic(K_event, ev_w, ev_h)

    # ── load poses ───────────────────────────────────────────────────
    with h5py.File(hdf5_dir / "poses.h5", "r") as pf:
        ee_T = pf["ee_T"][:]  # (N, 4, 4)
    N = ee_T.shape[0]

    # Pre-compute extrinsics (Open3D wants T_camera_from_world = inv(T_world_from_camera))
    frame_indices = list(range(0, N, skip))
    extrinsics = []
    for i in frame_indices:
        T_base_from_event = ee_T[i] @ T_ee_from_event
        extrinsics.append(np.linalg.inv(T_base_from_event))

    print(f"[Compare] {N} total frames, using every {skip}-th → {len(frame_indices)} frames")

    # ── load GT depth + spatial mask ─────────────────────────────────
    gt_depth_path = hdf5_dir / "depth_in_event_frame.h5"
    mask_path = hdf5_dir / "spatial_mask.h5"
    if not gt_depth_path.exists():
        raise FileNotFoundError(f"GT depth not found: {gt_depth_path}\n"
                                "Run: python3 project_realsense_to_event.py")
    if not mask_path.exists():
        raise FileNotFoundError(f"Spatial mask not found: {mask_path}\n"
                                "Run: python3 precompute_spatial_mask.py")

    # ── load model ───────────────────────────────────────────────────
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, model_depth_min, model_depth_max, use_pose = load_model(checkpoint, device)
    print(f"[Compare] Model loaded from {checkpoint}")
    print(f"[Compare] Model depth range: [{model_depth_min:.2f}, {model_depth_max:.2f}] m, use_pose={use_pose}")

    # ── locate voxels ────────────────────────────────────────────────
    if use_pose:
        voxels_dir = object_dir / "events" / "voxels_pose_cam0"
    else:
        voxels_dir = object_dir / "events" / "voxels_cam0"
        if not voxels_dir.exists():
            voxels_dir = object_dir / "events" / "voxels"
    if not voxels_dir.exists():
        raise FileNotFoundError(f"Voxels not found: {voxels_dir}")
    print(f"[Compare] Voxels from {voxels_dir}")

    # ── create two TSDF volumes ──────────────────────────────────────
    sdf_trunc = voxel_size * sdf_trunc_factor
    def new_volume():
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    vol_gt = new_volume()
    vol_pred = new_volume()

    # ── integrate ────────────────────────────────────────────────────
    states = None
    with h5py.File(gt_depth_path, "r") as df, h5py.File(mask_path, "r") as mf:
        depth_ds = df["depth"]
        mask_ds = mf["mask"]

        for frame_num, idx in enumerate(frame_indices):
            gt_depth = depth_ds[idx]   # (H, W) float32 metres
            mask = mask_ds[idx]        # (H, W) uint8

            # Mask + depth range for GT
            gt_depth_clean = gt_depth.copy()
            gt_depth_clean[(gt_depth_clean < depth_min_tsdf) | (gt_depth_clean > depth_max_tsdf)] = 0.0

            integrate_depth_map(vol_gt, gt_depth_clean, mask, extrinsics[frame_num], intrinsic, depth_max_tsdf)

            # ── model prediction ─────────────────────────────────────
            voxel_path = voxels_dir / f"voxel_{idx:06d}.npy"
            if not voxel_path.exists():
                continue
            voxel = np.load(voxel_path)  # (C, H, W)

            voxel_t = torch.from_numpy(voxel).float().unsqueeze(0).to(device)  # (1, C, H, W)
            with torch.no_grad():
                pred_norm, states = model(voxel_t, states)
            # pred_norm: (1, 1, H, W) in [0, 1]
            pred_depth = pred_norm[0, 0].cpu().numpy() * (model_depth_max - model_depth_min) + model_depth_min
            pred_depth[(pred_depth < depth_min_tsdf) | (pred_depth > depth_max_tsdf)] = 0.0

            integrate_depth_map(vol_pred, pred_depth, mask, extrinsics[frame_num], intrinsic, depth_max_tsdf)

            if (frame_num + 1) % 50 == 0 or frame_num == 0:
                print(f"  Integrated {frame_num + 1}/{len(frame_indices)} frames")

    print(f"  Integrated {len(frame_indices)}/{len(frame_indices)} frames — extracting meshes …")

    # ── extract meshes ───────────────────────────────────────────────
    mesh_gt = extract_mesh(vol_gt)
    mesh_pred = extract_mesh(vol_pred)

    print(f"[Compare] GT mesh:   {len(mesh_gt.vertices)} verts, {len(mesh_gt.triangles)} tris")
    print(f"[Compare] Pred mesh: {len(mesh_pred.vertices)} verts, {len(mesh_pred.triangles)} tris")

    # ── colour the two meshes differently ────────────────────────────
    # GT = blue-ish, Pred = orange-ish
    gt_color = np.full((len(mesh_gt.vertices), 3), [0.3, 0.5, 0.9])
    pred_color = np.full((len(mesh_pred.vertices), 3), [0.9, 0.5, 0.2])
    mesh_gt.vertex_colors = o3d.utility.Vector3dVector(gt_color)
    mesh_pred.vertex_colors = o3d.utility.Vector3dVector(pred_color)

    # ── place side by side ───────────────────────────────────────────
    # Compute bounding box of GT mesh to determine offset
    if len(mesh_gt.vertices) > 0 and len(mesh_pred.vertices) > 0:
        gt_bbox = mesh_gt.get_axis_aligned_bounding_box()
        gt_extent = gt_bbox.get_extent()
        # Shift prediction mesh along Y axis (lateral) by 1.5× the GT width
        offset = np.array([0.0, gt_extent[1] * 1.5 + 0.05, 0.0])
        mesh_pred.translate(offset)
    elif len(mesh_gt.vertices) == 0:
        print("[Compare] WARNING: GT mesh is empty!")
    elif len(mesh_pred.vertices) == 0:
        print("[Compare] WARNING: Pred mesh is empty!")

    # Merge into one mesh
    combined = mesh_gt + mesh_pred

    # ── export ───────────────────────────────────────────────────────
    fmt = output_format.lower().lstrip(".")
    out_path = object_dir / f"mesh_compare.{fmt}"
    o3d.io.write_triangle_mesh(str(out_path), combined)
    print(f"[Compare] Saved → {out_path}")
    print(f"          GT (blue) on left, Prediction (orange) on right")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side 3D reconstruction: GT depth vs model prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("object_dir", type=str,
                        help="Path to recorded object directory (e.g. data/real/block/1)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT),
                        help="Path to model checkpoint")
    parser.add_argument("--voxel_size", type=float, default=TSDF_VOXEL_SIZE,
                        help="TSDF voxel size in metres")
    parser.add_argument("--sdf_trunc_factor", type=float, default=TSDF_SDF_TRUNC_FACTOR,
                        help="SDF truncation = voxel_size × factor")
    parser.add_argument("--depth_min", type=float, default=TSDF_DEPTH_MIN,
                        help="Min depth for TSDF integration (metres)")
    parser.add_argument("--depth_max", type=float, default=TSDF_DEPTH_MAX,
                        help="Max depth for TSDF integration (metres)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Use every N-th frame")
    parser.add_argument("--format", type=str, default="obj",
                        choices=["obj", "stl", "ply"],
                        help="Output mesh format")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device")
    args = parser.parse_args()

    reconstruct_compare(
        object_dir=args.object_dir,
        checkpoint=Path(args.checkpoint),
        voxel_size=args.voxel_size,
        sdf_trunc_factor=args.sdf_trunc_factor,
        depth_min_tsdf=args.depth_min,
        depth_max_tsdf=args.depth_max,
        skip=args.skip,
        output_format=args.format,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
