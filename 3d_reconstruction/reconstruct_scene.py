"""
TSDF-based 3D scene reconstruction from depth frames + robot poses.

Reads:
    <object_dir>/hdf5/realsense.h5   – depth (uint16 mm) and rgb frames
    <object_dir>/hdf5/poses.h5       – per-frame ee_T (4×4 SE(3) in robot base frame)
    camera_data/T_rgb_from_ee.npz    – hand-eye calibration
    camera_data/T_color_from_depth.npz – RealSense depth→RGB extrinsic
    camera_data/rs_depth_intrinsics.npz – depth camera K matrix
    camera_data/depth_scale.npz      – depth tick → metres factor

Outputs:
    <object_dir>/mesh.obj  (or .stl / .ply via --format)

Transform chain (depth camera → robot base):
    T_base_from_depth[i] = ee_T[i] @ inv(T_rgb_from_ee) @ T_color_from_depth

Usage:
    python3 reconstruct_scene.py data/real/1

    # Custom parameters:
    python3 reconstruct_scene.py data/real/1 \\
        --voxel_size 0.002 --depth_max 1.5 --skip 2 --format stl

    # Use every 5th frame, larger voxels for speed:
    python3 reconstruct_scene.py data/real/1 --skip 5 --voxel_size 0.005
"""

import argparse
from pathlib import Path

import numpy as np
import h5py
import open3d as o3d


# ── paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CAM_DATA   = SCRIPT_DIR / "camera_data"


def load_calibration():
    """Load all static calibration matrices."""
    T_rgb_from_ee      = np.load(CAM_DATA / "T_rgb_from_ee.npz")["T"]        # (4,4)
    T_color_from_depth = np.load(CAM_DATA / "T_color_from_depth.npz")["T"]   # (4,4)
    depth_scale        = float(np.load(CAM_DATA / "depth_scale.npz")["scale"])

    intr_data   = np.load(CAM_DATA / "rs_depth_intrinsics.npz")
    K           = intr_data["camera_matrix"]       # (3,3)
    image_size  = intr_data["image_size"]           # [W, H]

    # Static transform: depth camera → EE
    T_ee_from_rgb   = np.linalg.inv(T_rgb_from_ee)
    T_ee_from_depth = T_ee_from_rgb @ T_color_from_depth

    return T_ee_from_depth, K, int(image_size[0]), int(image_size[1]), depth_scale


def make_o3d_intrinsic(K, width, height):
    """Create an Open3D PinholeCameraIntrinsic from a 3×3 K matrix."""
    return o3d.camera.PinholeCameraIntrinsic(
        width, height,
        float(K[0, 0]), float(K[1, 1]),
        float(K[0, 2]), float(K[1, 2]),
    )


def reconstruct(
    object_dir: Path,
    voxel_size: float = 0.003,
    sdf_trunc_factor: float = 4.0,
    depth_min: float = 0.05,
    depth_max: float = 2.0,
    skip: int = 1,
    output_format: str = "obj",
):
    """Run TSDF integration and export mesh."""
    object_dir = Path(object_dir)
    hdf5_dir   = object_dir / "hdf5"

    # ── calibration ──────────────────────────────────────────────────
    T_ee_from_depth, K, W, H, depth_scale = load_calibration()
    intrinsic = make_o3d_intrinsic(K, W, H)

    # ── load poses ───────────────────────────────────────────────────
    with h5py.File(hdf5_dir / "poses.h5", "r") as pf:
        ee_T = pf["ee_T"][:]                     # (N, 4, 4)

    N = ee_T.shape[0]
    print(f"[Reconstruct] {N} frames, using every {skip}-th → {len(range(0, N, skip))} frames")
    print(f"[Reconstruct] voxel_size={voxel_size:.4f}  depth=[{depth_min:.2f}, {depth_max:.2f}] m")

    # Pre-compute per-frame camera extrinsic (world → camera, as Open3D expects)
    # T_base_from_depth = ee_T @ T_ee_from_depth
    # Open3D wants the *extrinsic* = T_camera_from_world = inv(T_world_from_camera)
    # Our T_base_from_depth IS T_world_from_camera, so invert it.
    extrinsics = []
    for i in range(0, N, skip):
        T_base_from_depth = ee_T[i] @ T_ee_from_depth
        extrinsics.append(np.linalg.inv(T_base_from_depth))

    # ── TSDF volume ──────────────────────────────────────────────────
    sdf_trunc = voxel_size * sdf_trunc_factor
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # ── integrate frames ─────────────────────────────────────────────
    with h5py.File(hdf5_dir / "realsense.h5", "r") as rf:
        depth_ds = rf["depth"]
        rgb_ds   = rf["rgb"]

        for frame_num, idx in enumerate(range(0, N, skip)):
            depth_raw = depth_ds[idx]                          # uint16
            rgb_raw   = rgb_ds[idx]                            # uint8 BGR

            # Convert depth to float32 metres
            depth_m = depth_raw.astype(np.float32) * depth_scale

            # Mask out-of-range
            depth_m[(depth_m < depth_min) | (depth_m > depth_max)] = 0.0

            # Open3D images
            depth_o3d = o3d.geometry.Image(depth_m)
            color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_raw[:, :, ::-1]))  # BGR→RGB

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,          # already in metres
                depth_trunc=depth_max,
                convert_rgb_to_intensity=False,
            )

            volume.integrate(
                rgbd,
                intrinsic,
                extrinsics[frame_num],
            )

            if (frame_num + 1) % 50 == 0 or frame_num == 0:
                print(f"  Integrated {frame_num + 1}/{len(extrinsics)} frames")

    print(f"  Integrated {len(extrinsics)}/{len(extrinsics)} frames — extracting mesh …")

    # ── extract mesh ─────────────────────────────────────────────────
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    n_triangles = len(mesh.triangles)
    n_vertices  = len(mesh.vertices)
    print(f"[Reconstruct] Mesh: {n_vertices} vertices, {n_triangles} triangles")

    # ── export ───────────────────────────────────────────────────────
    fmt = output_format.lower().lstrip(".")
    out_path = object_dir / f"mesh.{fmt}"

    if fmt == "stl":
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
    elif fmt == "ply":
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
    elif fmt == "obj":
        o3d.io.write_triangle_mesh(str(out_path), mesh)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}  (use obj, stl, or ply)")

    print(f"[Reconstruct] Saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="TSDF 3D reconstruction from depth + robot poses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("object_dir", type=str,
                        help="Path to recorded object directory (e.g. data/real/1)")
    parser.add_argument("--voxel_size", type=float, default=0.003,
                        help="TSDF voxel size in metres")
    parser.add_argument("--sdf_trunc_factor", type=float, default=4.0,
                        help="SDF truncation = voxel_size × this factor")
    parser.add_argument("--depth_min", type=float, default=0.05,
                        help="Minimum depth in metres")
    parser.add_argument("--depth_max", type=float, default=2.0,
                        help="Maximum depth in metres")
    parser.add_argument("--skip", type=int, default=1,
                        help="Use every N-th frame (1 = all frames)")
    parser.add_argument("--format", type=str, default="obj",
                        choices=["obj", "stl", "ply"],
                        help="Output mesh format")
    args = parser.parse_args()

    reconstruct(
        object_dir=args.object_dir,
        voxel_size=args.voxel_size,
        sdf_trunc_factor=args.sdf_trunc_factor,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        skip=args.skip,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
