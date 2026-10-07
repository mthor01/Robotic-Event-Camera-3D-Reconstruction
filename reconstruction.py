#!/usr/bin/env python3
"""
TSDF reconstruction from a trained MVS model.

Supports checkpoints produced by ``training/train_mvs.py``.
Uses the precomputed table-plane channel stored in hdf5/table_plane.h5
(produced by data_precomputation/precompute_table_plane.py) instead of
computing it on-the-fly.

Predicted and GT depth maps of evenly spaced frames are fused into TSDF meshes
that are cropped to the workspace cube, and the predicted surface is compared
with the GT surface. By default predictions are fused weighted by the learned
per-pixel confidence; uniform fusion, or both for a direct comparison, and
largest-connected-surface copies of the meshes are only written on request.

Outputs, by default below reconstruction_results/<checkpoint_name>/ next to this script:
  <sequence>/<sequence>_gt_mesh.obj         — TSDF-fused GT mesh
  <sequence>/<sequence>_<variant>_mesh.obj  — TSDF-fused predicted mesh, where
                                              <variant> is uncertainty_weighted
                                              and/or uniform
  <sequence>/*_largest_component.obj        — only with --save_largest_connected_surface
  <sequence>/reconstruction_metrics.json/.txt
  reconstruction_summary.json/.txt          — means over all sequences
  chamfer_by_object.png

Usage:
    python3 reconstruction.py \\
        --checkpoint training/checkpoints/mvs/best_l1_myrun.pth \\
        --data_dir   data/eval/20

    python3 reconstruction.py \\
        --checkpoint training/checkpoints/mvs/best_l1_myrun.pth \\
        --data_dir   data/eval \\
        --compare_uncertainty_tsdf --save_largest_connected_surface
"""

import sys
import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "training"))

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    D_MAX, DEPTH_MIN,
    CALIB_DIR as _CALIB_DIR,
    PREPROCESS_CROP_HW, PREPROCESS_RESIZE_HW,
    TSDF_VOXEL_SIZE, TSDF_SDF_TRUNC_FACTOR,
    SPATIAL_CUBE_SIDE, SPATIAL_TARGET_X, SPATIAL_TARGET_Y, SPATIAL_TARGET_Z,
)
from helpers import (
    INTRINSICS_TRANSFORM,
    camera_centers_world,
    find_precomputed_sequences,
    points_in_cube,
    select_pose_views,
    transform_intrinsics,
)
from train_mvs import _inverse_depth_candidates, load_checkpoint

CALIB_DIR = _HERE / _CALIB_DIR
OUTPUT_ROOT = _HERE / "reconstruction_results"
# Surface metrics written to the per-sequence and summary files.
SURFACE_METRICS = (
    "accuracy_mean_m", "accuracy_median_m", "completeness_mean_m",
    "completeness_median_m", "chamfer_mean_m", "chamfer_median_m",
    "normal_consistency_symmetric", "precision_1cm", "recall_1cm",
    "fscore_1cm", "precision_2cm", "recall_2cm", "fscore_2cm",
    "precision_5cm", "recall_5cm", "fscore_5cm",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_K(
    calib_dir: Path,
    resize_hw: Optional[Tuple[int, int]] = None,
    crop_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, int, int]:
    """Return camera intrinsics scaled to the model's input resolution."""
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K = ev["camera_matrix"].copy().astype(np.float64)
    native_W = int(ev["image_size"][0])
    native_H = int(ev["image_size"][1])
    if resize_hw is None or crop_hw is None:
        return K, native_H, native_W
    K = transform_intrinsics(K, (native_H, native_W), resize_hw, crop_hw)
    return K, resize_hw[0], resize_hw[1]


def load_T_ee_from_event(calib_dir: Path) -> np.ndarray:
    """Return T_ee_from_event (4×4) composed from T_event_from_rgb and T_rgb_from_ee."""
    T_rgb_from_ee    = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float64)
    T_event_from_ee  = T_event_from_rgb @ T_rgb_from_ee
    return np.linalg.inv(T_event_from_ee)


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_resize(
    arr: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
    mode: str = "bilinear",
) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    final_hw = resize_hw
    if final_hw is not None and tuple(t.shape[-2:]) == tuple(final_hw):
        return arr
    kw = {} if mode == "nearest" else {"align_corners": False}
    if crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    if resize_hw is not None:
        t = F.interpolate(t, size=resize_hw, mode=mode, **kw)
    return t[0, 0].numpy()


def _preprocess_voxels(
    vox_raw: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
) -> np.ndarray:
    """Center-crop and resize a voxel grid to the target resolution."""
    final_h = resize_hw[0] if resize_hw is not None else vox_raw.shape[1]
    final_w = resize_hw[1] if resize_hw is not None else vox_raw.shape[2]
    if vox_raw.shape[1] == final_h and vox_raw.shape[2] == final_w:
        return vox_raw
    t = torch.from_numpy(vox_raw).unsqueeze(0)  # (1, C, H, W)
    if crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    if resize_hw is not None:
        t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def _multiview_view_ids(
    frame_idx: int,
    ckpt: dict,
    centers_world: np.ndarray,
) -> List[int]:
    """Select target/source frames by camera motion, as in train_mvs.py.

    Missing source slots near sequence boundaries are marked with -1.
    """
    view_ids = select_pose_views(
        centers_world,
        frame_idx,
        int(ckpt["num_views"]),
        float(ckpt["pose_move_threshold"]),
        bool(
            ckpt.get(
                "allow_fewer_pose_views",
                ckpt.get("allow_unbalanced_pose_views", False),
            )
        ),
    )
    if view_ids is None:
        raise RuntimeError(f"Frame {frame_idx} has no valid source-view layout")
    return view_ids


def _multiview_input(
    vox_ds,
    tbl_ds,
    idx: int,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw: Optional[Tuple[int, int]],
) -> torch.Tensor:
    vox_np = _preprocess_voxels(vox_ds[idx].astype(np.float32), resize_hw, crop_hw)
    h, w = vox_np.shape[1], vox_np.shape[2]
    tbl_t = torch.from_numpy(tbl_ds[idx].astype(np.float32)).unsqueeze(0)
    if tbl_t.shape[-2] != h or tbl_t.shape[-1] != w:
        tbl_t = F.interpolate(
            tbl_t.unsqueeze(0), (h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
    return torch.cat([torch.from_numpy(vox_np), tbl_t], dim=0)


@torch.no_grad()
def _infer_multiview(
    model: torch.nn.Module,
    frame_idx: int,
    vox_ds,
    tbl_ds,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
    ckpt: dict,
    device: torch.device,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw: Optional[Tuple[int, int]],
    centers_world: np.ndarray,
    return_uncertainty: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    view_ids = _multiview_view_ids(frame_idx, ckpt, centers_world)
    target_input = _multiview_input(
        vox_ds, tbl_ds, frame_idx, resize_hw, crop_hw
    )
    imgs = torch.stack(
        [
            target_input if i == frame_idx else
            _multiview_input(vox_ds, tbl_ds, i, resize_hw, crop_hw) if i >= 0 else
            torch.zeros_like(target_input)
            for i in view_ids
        ],
        dim=0,
    ).unsqueeze(0).to(device)
    cam_mats = torch.from_numpy(
        np.stack([
            T_cam_from_world[i] if i >= 0 else T_cam_from_world[frame_idx]
            for i in view_ids
        ])
    ).unsqueeze(0).to(device)
    view_valid_mask = torch.tensor(
        [[i >= 0 for i in view_ids]], dtype=torch.bool, device=device
    )
    K_t = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device)
    depth_values = torch.from_numpy(
        _inverse_depth_candidates(
            int(ckpt.get("coarse_depths", 32)),
            float(ckpt.get("depth_min", DEPTH_MIN)),
            float(ckpt.get("depth_max", D_MAX)),
        )
    ).to(device)
    out = model(
        imgs,
        cam_mats,
        K_t,
        depth_values,
        return_uncertainty=return_uncertainty,
        view_valid_mask=view_valid_mask,
    )
    if return_uncertainty:
        pred, confidence = out
        uncertainty = 1.0 - confidence
        return pred[0, 0].cpu().numpy(), uncertainty[0, 0].cpu().numpy()
    return out[0, 0].cpu().numpy(), None


# ─────────────────────────────────────────────────────────────────────────────
#  TSDF fusion and surface evaluation
# ─────────────────────────────────────────────────────────────────────────────

def tsdf_fuse(
    frames:           list,
    K:                np.ndarray,
    out_path:         Path,
    voxel_length:     float = 0.004,
    sdf_trunc_factor: float = 5.0,
    depth_max:        float = 0.6,
    cube_center:      Optional[np.ndarray] = None,
    cube_half_side:   float = 0.0,
    confidence_levels: int = 0,
    min_confidence:   float = 0.05,
) -> bool:
    """TSDF fusion, optionally weighted by a per-pixel confidence map.

    Open3D's legacy TSDF integrator has no observation-weight image.  We
    approximate continuous confidence weights with nested masks: a pixel is
    integrated once for every confidence level it exceeds.  This is equivalent
    to a uniformly quantized per-observation TSDF weight.  Pixels below the
    first level are set to invalid depth, so they neither update the surface
    nor carve free space.

    ``frames`` entries are ``(depth, T_world_from_cam)`` or
    ``(depth, T_world_from_cam, confidence)`` with confidence in [0, 1].
    """
    try:
        import open3d as o3d
    except ImportError:
        print("  [mesh] open3d not available — skipping. Install: pip install open3d")
        return False

    if not frames:
        print("  [mesh] No frames to fuse — skipping")
        return False

    H, W = frames[0][0].shape
    confidence_levels = max(0, int(confidence_levels))
    if not 0.0 <= min_confidence < 1.0:
        raise ValueError("min_confidence must be in [0, 1)")
    sdf_trunc = voxel_length * sdf_trunc_factor
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H,
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    dummy_color = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))

    weighting_msg = (
        f", confidence-weighted={confidence_levels} levels, min={min_confidence:.2f}"
        if confidence_levels > 0 else ""
    )
    print(f"  TSDF fusing {len(frames)} depth frames "
          f"(voxel={voxel_length * 1000:.1f} mm, trunc={sdf_trunc * 1000:.1f} mm"
          f"{weighting_msg}) …")

    # Open3D expects the extrinsic as world-to-camera; depth beyond depth_max
    # is ignored.
    def integrate_depth(depth_m: np.ndarray, T_cam_from_world: np.ndarray) -> None:
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_m, dtype=np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy_color, depth_o3d, depth_scale=1.0, depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, T_cam_from_world)

    for frame in frames:
        depth_m, T_world_from_cam = frame[:2]
        confidence = frame[2] if len(frame) > 2 else None
        T_cam_from_world = np.linalg.inv(T_world_from_cam)
        if confidence_levels <= 0 or confidence is None:
            integrate_depth(depth_m, T_cam_from_world)
            continue
        if confidence.shape != depth_m.shape:
            raise ValueError(
                f"Confidence shape {confidence.shape} does not match depth {depth_m.shape}"
            )
        confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        thresholds = min_confidence + (
            np.arange(confidence_levels, dtype=np.float32) + 0.5
        ) * ((1.0 - min_confidence) / confidence_levels)
        valid_depth = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < depth_max)
        for threshold in thresholds:
            weighted_depth = np.where(
                valid_depth & (confidence >= threshold), depth_m, np.float32(0.0)
            ).astype(np.float32, copy=False)
            if np.any(weighted_depth > 0.0):
                integrate_depth(weighted_depth, T_cam_from_world)

    full_mesh = volume.extract_triangle_mesh()
    full_mesh.compute_vertex_normals()
    mesh = full_mesh
    if cube_center is not None and cube_half_side > 0.0:
        verts = np.asarray(full_mesh.vertices)
        inside = np.flatnonzero(points_in_cube(verts, cube_center, cube_half_side))
        mesh = full_mesh.select_by_index(inside)
        mesh.compute_vertex_normals()
    print(f"  Cube crop (side={cube_half_side*2*100:.0f} cm) → "
          f"{len(np.asarray(mesh.vertices)):,} verts, "
          f"{len(np.asarray(mesh.triangles)):,} triangles")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_normals=True)
    print(f"  Mesh → {out_path}")
    return len(np.asarray(mesh.vertices)) > 0 and len(np.asarray(mesh.triangles)) > 0


def save_largest_connected_surface(source_path: Path) -> Path:
    """Write a copy of a triangle mesh containing only its largest component."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "--save_largest_connected_surface requires Open3D"
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(source_path))
    triangle_count = len(np.asarray(mesh.triangles))
    if triangle_count == 0:
        raise RuntimeError(
            f"Cannot clean {source_path}: mesh has no triangles"
        )

    triangle_clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters, dtype=np.int64)
    cluster_sizes = np.asarray(cluster_sizes, dtype=np.int64)
    if cluster_sizes.size == 0:
        raise RuntimeError(
            f"Cannot clean {source_path}: no connected surface was found"
        )

    largest_cluster = int(np.argmax(cluster_sizes))
    keep = triangle_clusters == largest_cluster
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    cleaned_path = source_path.with_name(
        f"{source_path.stem}_largest_component{source_path.suffix}"
    )
    write_normals = source_path.suffix.lower() != ".ply"
    if not o3d.io.write_triangle_mesh(
        str(cleaned_path), mesh, write_vertex_normals=write_normals
    ):
        raise RuntimeError(f"Failed to write cleaned mesh: {cleaned_path}")

    kept_triangles = len(np.asarray(mesh.triangles))
    kept_vertices = len(np.asarray(mesh.vertices))
    print(
        f"  Largest surface → {cleaned_path} "
        f"({kept_vertices:,} verts, {kept_triangles:,}/{triangle_count:,} triangles)"
    )
    return cleaned_path


def _nearest_distances_and_indices(
    src: np.ndarray,
    dst: np.ndarray,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour distances and indices from src to dst."""
    try:
        from scipy.spatial import cKDTree
        distances, indices = cKDTree(dst).query(src, k=1)
        return distances.astype(np.float64), indices.astype(np.int64)
    except ImportError:
        if len(src) * len(dst) > 25_000_000:
            raise RuntimeError(
                "scipy is required for this many mesh samples. Install scipy "
                "or lower --surface_samples."
            )
        distances = np.empty(len(src), dtype=np.float64)
        indices = np.empty(len(src), dtype=np.int64)
        for start in range(0, len(src), chunk_size):
            chunk = src[start:start + chunk_size]
            d2 = np.sum((chunk[:, None, :] - dst[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(d2, axis=1)
            indices[start:start + len(chunk)] = nearest
            distances[start:start + len(chunk)] = np.sqrt(
                d2[np.arange(len(chunk)), nearest]
            )
        return distances, indices


def evaluate_tsdf_meshes(
    pred_mesh_path: Path,
    gt_mesh_path: Path,
    surface_samples: int,
) -> Optional[dict]:
    """Evaluate uniformly sampled predicted and GT TSDF mesh surfaces."""
    try:
        import open3d as o3d
    except ImportError:
        print("  [metrics] open3d unavailable — skipping TSDF mesh metrics")
        return None

    pred_mesh = o3d.io.read_triangle_mesh(str(pred_mesh_path))
    gt_mesh = o3d.io.read_triangle_mesh(str(gt_mesh_path))
    if (
        len(pred_mesh.vertices) == 0
        or len(pred_mesh.triangles) == 0
        or len(gt_mesh.vertices) == 0
        or len(gt_mesh.triangles) == 0
    ):
        print("  [metrics] Predicted or GT TSDF mesh is empty")
        return None

    pred_mesh.compute_vertex_normals()
    gt_mesh.compute_vertex_normals()
    sample_count = max(1000, int(surface_samples))
    # Use identical deterministic surface samples across reconstruction
    # variants so comparison differences come from fusion, not sampling noise.
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(0)
    pred_cloud = pred_mesh.sample_points_uniformly(
        number_of_points=sample_count,
        use_triangle_normal=True,
    )
    if hasattr(o3d.utility, "random"):
        o3d.utility.random.seed(1)
    gt_cloud = gt_mesh.sample_points_uniformly(
        number_of_points=sample_count,
        use_triangle_normal=True,
    )
    pred_points = np.asarray(pred_cloud.points, dtype=np.float64)
    gt_points = np.asarray(gt_cloud.points, dtype=np.float64)
    pred_normals = np.asarray(pred_cloud.normals, dtype=np.float64)
    gt_normals = np.asarray(gt_cloud.normals, dtype=np.float64)

    pred_to_gt, pred_match = _nearest_distances_and_indices(pred_points, gt_points)
    gt_to_pred, gt_match = _nearest_distances_and_indices(gt_points, pred_points)

    pred_normal_dot = np.abs(
        np.sum(pred_normals * gt_normals[pred_match], axis=1)
    )
    gt_normal_dot = np.abs(
        np.sum(gt_normals * pred_normals[gt_match], axis=1)
    )
    pred_normal_dot = np.clip(pred_normal_dot, 0.0, 1.0)
    gt_normal_dot = np.clip(gt_normal_dot, 0.0, 1.0)

    metrics = {
        "surface_samples_pred": int(len(pred_points)),
        "surface_samples_gt": int(len(gt_points)),
        "accuracy_mean_m": float(pred_to_gt.mean()),
        "accuracy_median_m": float(np.median(pred_to_gt)),
        "completeness_mean_m": float(gt_to_pred.mean()),
        "completeness_median_m": float(np.median(gt_to_pred)),
        "chamfer_mean_m": float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
        "chamfer_median_m": float(
            0.5 * (np.median(pred_to_gt) + np.median(gt_to_pred))
        ),
        "normal_consistency_pred_to_gt": float(pred_normal_dot.mean()),
        "normal_consistency_gt_to_pred": float(gt_normal_dot.mean()),
        "normal_consistency_symmetric": float(
            0.5 * (pred_normal_dot.mean() + gt_normal_dot.mean())
        ),
    }
    for tolerance_m in (0.01, 0.02, 0.05):
        label = f"{int(round(tolerance_m * 100))}cm"
        precision = float(np.mean(pred_to_gt < tolerance_m))
        recall = float(np.mean(gt_to_pred < tolerance_m))
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        metrics[f"precision_{label}"] = precision
        metrics[f"recall_{label}"] = recall
        metrics[f"fscore_{label}"] = fscore
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────


def reconstruct_sequence(
    sequence_dir: Path,
    out_dir: Path,
    model: torch.nn.Module,
    ckpt: dict,
    variants: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Optional[dict]]:
    """Fuse GT and predicted meshes for one sequence and return surface metrics."""
    resize_hw, crop_hw = PREPROCESS_RESIZE_HW, PREPROCESS_CROP_HW
    depth_h5_path = sequence_dir / "hdf5" / "depth_in_event_frame.h5"
    voxels_h5_path = sequence_dir / "events" / "voxels_cam0.h5"
    poses_h5_path = sequence_dir / "hdf5" / "poses.h5"
    table_plane_h5_path = sequence_dir / "hdf5" / "table_plane.h5"
    with h5py.File(table_plane_h5_path, "r") as table_file:
        transform = table_file.attrs.get("intrinsics_transform", "")
        if isinstance(transform, bytes):
            transform = transform.decode("utf-8", errors="replace")
    if transform != INTRINSICS_TRANSFORM:
        raise RuntimeError(
            f"{table_plane_h5_path} uses obsolete direct-scaling geometry. "
            "Regenerate it with data_precomputation/precompute_table_plane.py "
            "--overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    K, out_H, out_W = load_K(CALIB_DIR, resize_hw, crop_hw)
    depth_min = ckpt.get("depth_min", DEPTH_MIN)
    depth_max = ckpt.get("depth_max", D_MAX)
    cube_center = np.array([SPATIAL_TARGET_X, SPATIAL_TARGET_Y, SPATIAL_TARGET_Z], dtype=np.float64)
    cube_half_side = SPATIAL_CUBE_SIDE / 2.0

    with h5py.File(poses_h5_path, "r") as f:
        ee_T_all = f["ee_T"][:].astype(np.float64)   # (N, 4, 4)
    T_ee_from_event = load_T_ee_from_event(CALIB_DIR)
    T_event_from_ee = np.linalg.inv(T_ee_from_event).astype(np.float32)
    T_ee_inv = np.linalg.inv(ee_T_all.astype(np.float32))
    T_cam_from_world = np.einsum("ij,njk->nik", T_event_from_ee, T_ee_inv).astype(np.float32)
    cam_centers_world = camera_centers_world(T_cam_from_world)

    with h5py.File(depth_h5_path, "r") as f:
        n_total = f["depth"].shape[0]
    with h5py.File(voxels_h5_path, "r") as f:
        n_total = min(n_total, f["voxels"].shape[0])
    with h5py.File(table_plane_h5_path, "r") as f:
        n_total = min(n_total, f["table_plane"].shape[0], len(T_cam_from_world))

    eligible_frames = list(range(n_total))
    allow_fewer_pose_views = bool(
        ckpt.get(
            "allow_fewer_pose_views",
            ckpt.get("allow_unbalanced_pose_views", False),
        )
    )
    if not allow_fewer_pose_views:
        eligible_frames = [
            frame_idx
            for frame_idx in eligible_frames
            if select_pose_views(
                cam_centers_world,
                frame_idx,
                int(ckpt["num_views"]),
                float(ckpt["pose_move_threshold"]),
                False,
            ) is not None
        ]
        if not eligible_frames:
            raise RuntimeError(
                "No target frame has the strictly balanced pose-view layout "
                "required by the active reconstruction policy."
            )
    n_mesh = min(len(eligible_frames), max(1, args.mesh_frame_count))
    positions = np.round(np.linspace(0, len(eligible_frames) - 1, n_mesh)).astype(int)
    mesh_frames = sorted({eligible_frames[position] for position in positions})
    print(f"  Frames         : {len(mesh_frames)} of {n_total} fused", flush=True)

    use_confidence = "uncertainty_weighted" in variants
    pred_frames = []
    gt_frames = []
    with h5py.File(depth_h5_path, "r") as depth_f, \
            h5py.File(voxels_h5_path, "r") as voxels_f, \
            h5py.File(table_plane_h5_path, "r") as table_f:
        for frame_idx in mesh_frames:
            pred_norm, uncertainty_map = _infer_multiview(
                model,
                frame_idx,
                voxels_f["voxels"],
                table_f["table_plane"],
                T_cam_from_world,
                K,
                ckpt,
                device,
                resize_hw,
                crop_hw,
                cam_centers_world,
                return_uncertainty=use_confidence,
            )
            pred_m = depth_min + pred_norm * (depth_max - depth_min)
            gt = depth_f["depth"][frame_idx].astype(np.float32)
            if gt.shape[0] != out_H or gt.shape[1] != out_W:
                gt = crop_resize(gt, resize_hw, crop_hw, mode="bilinear")
            gt_valid = np.isfinite(gt) & (gt > depth_min) & (gt < depth_max)
            pred_valid = np.isfinite(pred_m) & (pred_m > depth_min) & (pred_m < depth_max)
            T_base_from_event = ee_T_all[frame_idx] @ T_ee_from_event
            confidence = (
                np.clip(1.0 - uncertainty_map, 0.0, 1.0).astype(np.float32, copy=False)
                if uncertainty_map is not None
                else None
            )
            pred_frames.append((
                np.where(pred_valid, pred_m, np.float32(0.0)).astype(np.float32),
                T_base_from_event,
                confidence,
            ))
            gt_frames.append((
                np.where(gt_valid, gt, np.float32(0.0)).astype(np.float32),
                T_base_from_event,
            ))

    tsdf_kw = dict(
        voxel_length=args.voxel_size,
        sdf_trunc_factor=args.sdf_trunc_factor,
        depth_max=depth_max,
        cube_center=cube_center,
        cube_half_side=cube_half_side,
    )
    name = sequence_dir.name
    gt_mesh_path = out_dir / f"{name}_gt_mesh.obj"
    gt_mesh_created = tsdf_fuse(gt_frames, K, gt_mesh_path, **tsdf_kw)
    if gt_mesh_created and args.save_largest_connected_surface:
        gt_mesh_path = save_largest_connected_surface(gt_mesh_path)

    metrics: dict[str, Optional[dict]] = {}
    for variant in variants:
        weighted = variant == "uncertainty_weighted"
        pred_mesh_path = out_dir / f"{name}_{variant}_mesh.obj"
        pred_mesh_created = tsdf_fuse(
            pred_frames if weighted else [frame[:2] for frame in pred_frames],
            K,
            pred_mesh_path,
            confidence_levels=args.tsdf_confidence_levels if weighted else 0,
            min_confidence=args.tsdf_min_confidence,
            **tsdf_kw,
        )
        metrics[variant] = None
        if pred_mesh_created and gt_mesh_created:
            if args.save_largest_connected_surface:
                pred_mesh_path = save_largest_connected_surface(pred_mesh_path)
            metrics[variant] = evaluate_tsdf_meshes(
                pred_mesh_path, gt_mesh_path, surface_samples=args.surface_samples
            )
        surface = metrics[variant]
        print(
            f"  {variant:15s}: "
            + (
                f"Chamfer={surface['chamfer_mean_m'] * 100:.2f} cm, "
                f"normal={surface['normal_consistency_symmetric']:.3f}, "
                f"F@1/2/5cm={surface['fscore_1cm']:.1%}/"
                f"{surface['fscore_2cm']:.1%}/{surface['fscore_5cm']:.1%}"
                if surface is not None
                else "no surface metrics (empty predicted or GT mesh)"
            ),
            flush=True,
        )

    payload = {
        "surface_selection": _surface_selection(args),
        "cube_center_world_m": cube_center.tolist(),
        "cube_side_m": SPATIAL_CUBE_SIDE,
        "fused_frames": mesh_frames,
        "surface_metrics": metrics,
    }
    (out_dir / "reconstruction_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [f"Reconstruction metrics for {name} ({payload['surface_selection']})"]
    for variant, surface in metrics.items():
        lines.append(f"  {variant}")
        for metric in SURFACE_METRICS:
            if surface is not None:
                lines.append(f"    {metric}: {surface[metric]:.6f}")
    (out_dir / "reconstruction_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def _surface_selection(args: argparse.Namespace) -> str:
    return (
        "largest_connected_surface"
        if args.save_largest_connected_surface
        else "all_surfaces"
    )


def write_reconstruction_summary(
    output_root: Path,
    results: dict[str, dict[str, Optional[dict]]],
    variants: list[str],
    args: argparse.Namespace,
) -> None:
    """Write unweighted means over sequences and, for two variants, their difference."""
    means: dict[str, dict[str, float]] = {}
    for variant in variants:
        means[variant] = {}
        for metric in SURFACE_METRICS:
            values = [
                float(sequence_metrics[variant][metric])
                for sequence_metrics in results.values()
                if sequence_metrics[variant] is not None
                and np.isfinite(sequence_metrics[variant][metric])
            ]
            if values:
                means[variant][metric] = float(np.mean(values))
    lines = [
        "Reconstruction summary (unweighted mean across objects)",
        f"Surface selection: {_surface_selection(args)}",
        f"Objects: {len(results)} ({', '.join(results)})",
        "",
    ]
    for variant in variants:
        lines.append(variant)
        lines.extend(f"  {metric}: {value:.6f}" for metric, value in means[variant].items())
        lines.append("")
    comparison = {}
    if len(variants) == 2:
        lines.append(
            "uncertainty_weighted vs uniform (difference = weighted - uniform, "
            "factor = weighted / uniform)"
        )
        for metric, uniform_value in means["uniform"].items():
            weighted_value = means["uncertainty_weighted"].get(metric)
            if weighted_value is None:
                continue
            difference = weighted_value - uniform_value
            factor = weighted_value / uniform_value if uniform_value != 0.0 else None
            comparison[metric] = {
                "uniform": uniform_value,
                "uncertainty_weighted": weighted_value,
                "difference": difference,
                "weighted_over_uniform_factor": factor,
            }
            factor_text = f"{factor:.6f}x" if factor is not None else "undefined"
            lines.append(
                f"  {metric}: difference={difference:+.6f}, factor={factor_text}"
            )
    (output_root / "reconstruction_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "reconstruction_summary.json").write_text(
        json.dumps(
            {
                "object_count": len(results),
                "objects": list(results),
                "surface_selection": _surface_selection(args),
                "mean_surface_metrics": means,
                "comparison": comparison,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    names = list(results)
    x = np.arange(len(names))
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(max(8, 1.35 * len(names)), 5))
    for offset, variant in enumerate(variants):
        values = [
            100.0 * sequence_metrics[variant]["chamfer_mean_m"]
            if sequence_metrics[variant] is not None
            else np.nan
            for sequence_metrics in results.values()
        ]
        ax.bar(x + (offset - (len(variants) - 1) / 2) * width, values, width, label=variant)
    ax.set_xticks(x, names)
    ax.set_ylabel("Post-TSDF surface Chamfer [cm]")
    ax.set_title("Reconstruction Chamfer by object")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "chamfer_by_object.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSDF reconstruction from a trained MVS model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MVS checkpoint (.pth) saved by training/train_mvs.py",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help=(
            "One sequence directory, or a parent whose immediate child "
            "directories are sequences"
        ),
    )
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (default: reconstruction_results/<checkpoint_name>)")
    parser.add_argument("--mesh_frame_count", type=int, default=80,
                        help="Number of evenly spaced frames fused into each mesh")
    pose_layout_group = parser.add_mutually_exclusive_group()
    pose_layout_group.add_argument(
        "--allow_fewer_pose_views",
        "--allow-fewer-pose-views",
        "--allow_unbalanced_pose_views",
        "--allow-unbalanced-pose-views",
        dest="pose_layout_override",
        action="store_const",
        const=True,
        help="Override the checkpoint and allow fewer masked source views at boundaries",
    )
    pose_layout_group.add_argument(
        "--strict_balanced_pose_views",
        "--strict-balanced-pose-views",
        dest="pose_layout_override",
        action="store_const",
        const=False,
        help="Override the checkpoint and require balanced pose-view layouts",
    )
    parser.set_defaults(pose_layout_override=None)
    parser.add_argument("--voxel_size",        type=float, default=TSDF_VOXEL_SIZE)
    parser.add_argument("--sdf_trunc_factor",  type=float, default=TSDF_SDF_TRUNC_FACTOR)
    fusion_group = parser.add_mutually_exclusive_group()
    fusion_group.add_argument(
        "--uniform_tsdf",
        action="store_true",
        help=(
            "Fuse the predicted mesh uniformly instead of weighting it with the "
            "checkpoint's learned per-pixel confidence (needed for checkpoints "
            "trained without --uncertainty)"
        ),
    )
    fusion_group.add_argument(
        "--compare_uncertainty_tsdf",
        "--compare-uncertainty-tsdf",
        action="store_true",
        help=(
            "Fuse both uniform and uncertainty-weighted predicted meshes with "
            "identical settings and compare their metrics in the summary"
        ),
    )
    parser.add_argument(
        "--tsdf_confidence_levels",
        type=int,
        default=8,
        help="Number of quantized confidence-weight levels (default: 8)",
    )
    parser.add_argument(
        "--tsdf_min_confidence",
        type=float,
        default=0.05,
        help="Confidence floor below which a depth pixel does not update TSDF",
    )
    parser.add_argument(
        "--save_largest_connected_surface",
        "--save-largest-connected-surface",
        "--save_largest_connected_component",
        action="store_true",
        help=(
            "Write largest-connected-surface copies of the predicted and GT "
            "TSDF meshes and use those copies for the surface metrics"
        ),
    )
    parser.add_argument("--surface_samples", type=int, default=100000,
                        help="Uniform samples per TSDF mesh for surface metrics")
    args = parser.parse_args()
    if args.mesh_frame_count <= 0:
        parser.error("--mesh_frame_count must be > 0")
    if args.tsdf_confidence_levels <= 0:
        parser.error("--tsdf_confidence_levels must be > 0")
    if not 0.0 <= args.tsdf_min_confidence < 1.0:
        parser.error("--tsdf_min_confidence must be in [0, 1)")
    if args.surface_samples <= 0:
        parser.error("--surface_samples must be > 0")
    if not args.checkpoint.is_file():
        parser.error(
            f"Checkpoint does not exist: {args.checkpoint}\n"
            "  Download the pretrained model (README, section 'Pretrained model') "
            "or train one with training/train_mvs.sh."
        )

    sequence_dirs = find_precomputed_sequences(args.data_dir)
    if not sequence_dirs:
        parser.error(
            f"{args.data_dir} is neither a precomputed sequence nor a directory "
            "containing them; run data_precomputation/precompute_all.sh first"
        )
    if args.compare_uncertainty_tsdf:
        variants = ["uniform", "uncertainty_weighted"]
    elif args.uniform_tsdf:
        variants = ["uniform"]
    else:
        variants = ["uncertainty_weighted"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    if "uncertainty_weighted" in variants and not ckpt.get("uncertainty", False):
        parser.error(
            "Learned uncertainty requires a train_mvs.py checkpoint trained with "
            "--uncertainty; use --uniform_tsdf for other checkpoints"
        )
    if args.pose_layout_override is not None:
        ckpt["allow_unbalanced_pose_views"] = args.pose_layout_override
        ckpt["allow_fewer_pose_views"] = args.pose_layout_override
    output_root = args.out_dir or OUTPUT_ROOT / args.checkpoint.stem
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint : {args.checkpoint.name} (device {device})")
    print(f"Variants   : {', '.join(variants)}")
    print(f"Output     : {output_root}")

    results = {}
    for sequence_number, sequence_dir in enumerate(sequence_dirs, start=1):
        print(
            f"\n[{sequence_number}/{len(sequence_dirs)}] Reconstructing {sequence_dir.name}",
            flush=True,
        )
        results[sequence_dir.name] = reconstruct_sequence(
            sequence_dir,
            output_root / sequence_dir.name,
            model,
            ckpt,
            variants,
            args,
            device,
        )
    write_reconstruction_summary(output_root, results, variants, args)
    print(f"\nSummary: {output_root / 'reconstruction_summary.txt'}")


if __name__ == "__main__":
    main()
