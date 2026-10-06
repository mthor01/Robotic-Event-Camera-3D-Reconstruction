#!/usr/bin/env python3
"""
Depth-map, point-cloud, and TSDF reconstruction from a trained MVS model.

Supports checkpoints produced by ``training/train_mvs.py``.
Uses the precomputed table-plane channel stored in hdf5/table_plane.h5
(produced by data_precomputation/precompute_table_plane.py) instead of
computing it on-the-fly.

Outputs are written by default into each sequence directory, under
<sequence>/reconstruction_output/<checkpoint_name>/ (or to --out_dir).
When --data_dir is a parent folder, the cross-sequence summary is written to
<data_dir>/reconstruction_output/<checkpoint_name>/. Each output folder contains:
  frame_NNNNN/depth_pred.png   — colourised predicted depth
  frame_NNNNN/depth_gt.png     — colourised GT depth
  frame_NNNNN/pointcloud.ply   — predicted depth backprojected to 3-D
  overview.png                 — side-by-side grid of all viz frames
  <sequence>_mesh.obj          — TSDF-fused predicted mesh
                                  (.ply with vertex RGB when --color_mesh is used)
  <sequence>_gt_mesh.obj       — TSDF-fused GT mesh
  tsdf_metrics.txt/.json/.png  — surface and held-out rendered-depth metrics

Usage:
    python3 reconstruction.py \\
        --checkpoint training/checkpoints/mvs/best_l1_myrun.pth \\
        --data_dir   data/new/eval/1

    python3 reconstruction.py \\
        --checkpoint training/checkpoints/mvs/best_l1_myrun.pth \\
        --data_dir   data/new/eval/1 \\
        --save_frame_visualizations \\
        --indices 0 50 100 200 400 \\
        --out_dir results/lego_1

    # A parent directory reconstructs every sequence and writes a summary to
    # data/new/eval/reconstruction_output/<checkpoint_name>/.
    python3 reconstruction.py \\
        --checkpoint training/checkpoints/mvs/best_l1_myrun.pth \\
        --data_dir   data/new/eval
"""

import sys
import argparse
import json
import subprocess
import time
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
import matplotlib.cm as cm

from config import (
    D_MAX, DEPTH_MIN, NUM_BINS,
    CALIB_DIR as _CALIB_DIR,
    PREPROCESS_CROP_HW, PREPROCESS_RESIZE_HW,
    TSDF_VOXEL_SIZE, TSDF_SDF_TRUNC_FACTOR, TSDF_DEPTH_MAX,
    SPATIAL_CUBE_SIDE, SPATIAL_TARGET_X, SPATIAL_TARGET_Y, SPATIAL_TARGET_Z,
)
from helpers import (
    INTRINSICS_TRANSFORM,
    camera_centers_world,
    depth_cube_mask,
    fixed_source_offsets,
    points_in_cube,
    select_pose_views,
    transform_intrinsics,
)
from train_mvs import (
    ModernMVSNet,
    _inverse_depth_candidates,
    _linear_depth_candidates,
)

CALIB_DIR = _HERE / _CALIB_DIR
RAISED_OBJECT_BOTTOM_Z_M = 0.015
OUTPUT_DIRNAME = "reconstruction_output"


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


def load_original_depth_calibration(calib_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return native RealSense depth intrinsics, T_ee_from_depth, and depth scale."""
    depth_intr = np.load(calib_dir / "rs_depth_intrinsics.npz")
    K_depth = depth_intr["camera_matrix"].copy().astype(np.float64)
    T_rgb_from_ee = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_color_from_depth = np.load(calib_dir / "T_color_from_depth.npz")["T"].astype(np.float64)
    T_ee_from_depth = np.linalg.inv(T_rgb_from_ee) @ T_color_from_depth
    depth_scale_path = calib_dir / "depth_scale.npz"
    depth_scale = (
        float(np.load(depth_scale_path)["scale"])
        if depth_scale_path.exists()
        else 0.001
    )
    return K_depth, T_ee_from_depth, depth_scale


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
    """Load a train_mvs.py checkpoint and return (model, ckpt_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_arch = str(ckpt.get("model_arch", "ModernMVSNet"))
    if model_arch != "ModernMVSNet" or "num_views" not in ckpt or "coarse_depths" not in ckpt:
        raise ValueError(
            f"Checkpoint does not appear to be a train_mvs.py checkpoint: {ckpt_path}\n"
            "Expected ModernMVSNet weights with 'num_views'/'coarse_depths' metadata."
        )
    fine_window = float(ckpt.get("fine_window", 0.08))
    fine_offset_radius = float(ckpt.get("fine_offset_radius", 2.0))
    model = ModernMVSNet(
        in_ch=ckpt.get("in_ch", NUM_BINS + 1),
        base=ckpt.get("base", ckpt.get("base_channels_arg", 32)),
        feature_ch=ckpt.get("feature_channels", None),
        cost_base=ckpt.get("cost_channels", None),
        fine_depths=ckpt.get("fine_depths", 5),
        fine_window=fine_window,
        fine_offset_radius=fine_offset_radius,
        fine_window_min=float(
            ckpt.get(
                "fine_window_min",
                0.25 * fine_window * fine_offset_radius,
            )
        ),
        fine_window_max=float(
            ckpt.get(
                "fine_window_max",
                fine_window * fine_offset_radius,
            )
        ),
        learned_fine_window=ckpt.get("learned_fine_window", False),
        reference_channels=ckpt.get("reference_channels", 0),
        coarse_cost_channels=ckpt.get("coarse_cost_channels", 0),
        fine_cost_channels=ckpt.get("fine_cost_channels", 0),
        refiner_channels=ckpt.get("refiner_channels", 0),
        refiner_max_residual_m=ckpt.get("refiner_max_residual_m"),
        refiner_reference_input=not ckpt.get(
            "no_refiner_reference_input", False
        ),
        no_2d_refinement=ckpt.get("no_2d_refinement", False),
        coarse_hourglass_levels=ckpt.get("coarse_hourglass_levels", 2),
        fine_hourglass_levels=ckpt.get("fine_hourglass_levels", 2),
        fpn_dropout=ckpt.get("fpn_dropout", 0.0),
        reference_dropout=ckpt.get("reference_dropout", 0.0),
        hourglass_dropout=ckpt.get("hourglass_dropout", 0.0),
        drop_path_rate=ckpt.get("drop_path_rate", 0.0),
        middle_depths=ckpt.get("middle_depths", 8),
        middle_window=ckpt.get("middle_window", 0.12),
        middle_cost_channels=ckpt.get("middle_cost_channels", 0),
        middle_hourglass_levels=ckpt.get("middle_hourglass_levels", 2),
        middle_feature_channels=ckpt.get("middle_feature_channels", 0),
        fine_feature_channels=ckpt.get("fine_feature_channels", 0),
        fpn_lateral_convolutions=not ckpt.get(
            "no_fpn_lateral_convolutions", False
        ),
    ).to(device)
    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_confidence = [
        k for k in incompatible.missing_keys
        if not k.startswith("confidence_head.")
    ]
    if missing_non_confidence or unexpected:
        details = []
        if missing_non_confidence:
            details.append(f"missing keys: {missing_non_confidence}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model_arch}: "
            + "; ".join(details)
        )
    if incompatible.missing_keys:
        print(
            "  note           : checkpoint has no confidence_head weights; "
            "ignoring learned confidence for reconstruction"
        )
    model.eval()
    return model, ckpt


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


def crop_resize_rgb(
    rgb: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
) -> np.ndarray:
    """Center-crop and resize RGB to the reconstruction resolution."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
    final_h = resize_hw[0] if resize_hw is not None else rgb.shape[0]
    final_w = resize_hw[1] if resize_hw is not None else rgb.shape[1]
    if rgb.shape[0] == final_h and rgb.shape[1] == final_w:
        return rgb.astype(np.uint8, copy=False)
    t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    if crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    if resize_hw is not None:
        t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
    out = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).numpy()
    return out.astype(np.uint8)


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


def _multiview_view_ids(
    frame_idx: int,
    n_frames: int,
    ckpt: dict,
    centers_world: Optional[np.ndarray],
) -> List[int]:
    """Select target/source frames using the same conventions as train_mvs.py.

    Edge frames can lack the exact training-time source layout. In that case we
    keep reconstruction running by filling the missing source slots with nearest
    temporal neighbours, then the target frame if the sequence is too short.
    """
    num_views = int(ckpt.get("num_views", 5))
    if ckpt.get("pose_view_selection", False) and centers_world is not None:
        threshold = float(ckpt.get("pose_move_threshold", 0.01))
        selected = select_pose_views(
            centers_world,
            frame_idx,
            num_views,
            threshold,
            bool(
                ckpt.get(
                    "allow_fewer_pose_views",
                    ckpt.get("allow_unbalanced_pose_views", False),
                )
            ),
        )
        view_ids = selected or [frame_idx]
    else:
        interval = int(ckpt.get("view_interval", 5))
        view_ids = [frame_idx] + [
            min(max(frame_idx + o, 0), n_frames - 1)
            for o in fixed_source_offsets(num_views, interval)
        ]

    if len(view_ids) < num_views:
        used = set(view_ids)
        candidates = sorted(
            (i for i in range(n_frames) if i not in used),
            key=lambda i: abs(i - frame_idx),
        )
        view_ids.extend(candidates[:num_views - len(view_ids)])
    while len(view_ids) < num_views:
        view_ids.append(frame_idx)
    return view_ids[:num_views]


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
    centers_world: Optional[np.ndarray],
    return_uncertainty: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    view_ids = _multiview_view_ids(frame_idx, len(T_cam_from_world), ckpt, centers_world)
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
    candidate_fn = (
        _linear_depth_candidates
        if ckpt.get("linear_depth_candidates", False)
        else _inverse_depth_candidates
    )
    depth_values = torch.from_numpy(
        candidate_fn(
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
#  Point-cloud helpers
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_pointcloud(depth_m: np.ndarray, mask: np.ndarray,
                        K: np.ndarray) -> np.ndarray:
    """Backproject masked depth (H, W) → (N, 3) XYZ point cloud in camera frame."""
    H, W = depth_m.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    valid = (mask > 0.5) & (depth_m > 0)
    z = depth_m[valid]
    x = (xx[valid] - K[0, 2]) * z / K[0, 0]
    y = (yy[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def save_ply(pts: np.ndarray, path: Path) -> None:
    """Write (N, 3) float32 XYZ as binary PLY."""
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(pts.tobytes())


def depth_error_stats(pred_m: np.ndarray, gt_m: np.ndarray, mask: np.ndarray) -> dict:
    """Return depth error sums for masked pixels."""
    valid = mask > 0.5
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0}
    err = (pred_m[valid] - gt_m[valid]).astype(np.float64)
    abs_err = np.abs(err)
    return {
        "n": n,
        "abs_sum": float(abs_err.sum()),
        "sq_sum": float(np.square(err).sum()),
        "max_abs": float(abs_err.max()),
    }


def merge_depth_error_stats(total: dict, frame_stats: dict) -> None:
    total["n"] += frame_stats["n"]
    total["abs_sum"] += frame_stats["abs_sum"]
    total["sq_sum"] += frame_stats["sq_sum"]
    total["max_abs"] = max(total["max_abs"], frame_stats["max_abs"])


def summarize_depth_error(total: dict) -> Optional[dict]:
    if total["n"] <= 0:
        return None
    return {
        "pixels": int(total["n"]),
        "mae_m": total["abs_sum"] / total["n"],
        "rmse_m": float(np.sqrt(total["sq_sum"] / total["n"])),
        "max_m": total["max_abs"],
    }


def _subsample_points(pts: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or len(pts) <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=max_points, replace=False)
    return pts[np.sort(idx)]


def _nearest_distances(src: np.ndarray, dst: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    """Nearest-neighbour distances from src to dst, using scipy when available."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(dst).query(src, k=1)[0].astype(np.float64)
    except ImportError:
        if len(src) * len(dst) > 25_000_000:
            raise RuntimeError(
                "scipy is not installed and the point clouds are too large for "
                "the fallback nearest-neighbour computation. Increase "
                "--error_stride or lower --error_max_points."
            )
        out = np.empty(len(src), dtype=np.float64)
        for start in range(0, len(src), chunk_size):
            chunk = src[start:start + chunk_size]
            d2 = np.sum((chunk[:, None, :] - dst[None, :, :]) ** 2, axis=2)
            out[start:start + len(chunk)] = np.sqrt(d2.min(axis=1))
        return out


def reconstruction_error_stats(
    pred_frames: list,
    gt_frames: list,
    K: np.ndarray,
    stride: int,
    max_points: int,
    cube_center: Optional[np.ndarray] = None,
    cube_half_side: float = 0.0,
) -> Optional[dict]:
    """Compare predicted and GT reconstructions as fused world-frame point sets."""
    if not pred_frames or not gt_frames:
        return None

    pred_pts_all = []
    gt_pts_all = []
    stride = max(1, int(stride))
    for (pred_depth, T_world_from_cam), (gt_depth, T_gt_world_from_cam) in zip(pred_frames, gt_frames):
        if not np.allclose(T_world_from_cam, T_gt_world_from_cam):
            raise ValueError("Predicted and GT reconstruction frames are not pose-aligned.")

        pred_s = pred_depth[::stride, ::stride]
        gt_s = gt_depth[::stride, ::stride]
        K_s = K.copy()
        K_s[0, :] /= stride
        K_s[1, :] /= stride

        pred_cam = depth_to_pointcloud(pred_s, pred_s > 0.0, K_s)
        gt_cam = depth_to_pointcloud(gt_s, gt_s > 0.0, K_s)
        if len(pred_cam):
            pred_h = np.concatenate([pred_cam, np.ones((len(pred_cam), 1), dtype=np.float32)], axis=1)
            pred_world = (T_world_from_cam @ pred_h.T).T[:, :3].astype(np.float32)
            if cube_center is not None:
                pred_world = pred_world[points_in_cube(pred_world, cube_center, cube_half_side)]
            if len(pred_world):
                pred_pts_all.append(pred_world)
        if len(gt_cam):
            gt_h = np.concatenate([gt_cam, np.ones((len(gt_cam), 1), dtype=np.float32)], axis=1)
            gt_world = (T_world_from_cam @ gt_h.T).T[:, :3].astype(np.float32)
            if cube_center is not None:
                gt_world = gt_world[points_in_cube(gt_world, cube_center, cube_half_side)]
            if len(gt_world):
                gt_pts_all.append(gt_world)

    if not pred_pts_all or not gt_pts_all:
        return None

    pred_pts = _subsample_points(np.concatenate(pred_pts_all, axis=0), max_points, seed=11)
    gt_pts = _subsample_points(np.concatenate(gt_pts_all, axis=0), max_points, seed=17)
    pred_to_gt = _nearest_distances(pred_pts, gt_pts)
    gt_to_pred = _nearest_distances(gt_pts, pred_pts)

    return {
        "pred_points": int(len(pred_pts)),
        "gt_points": int(len(gt_pts)),
        "accuracy_m": float(pred_to_gt.mean()),
        "completeness_m": float(gt_to_pred.mean()),
        "chamfer_l1_m": float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
        "accuracy_p95_m": float(np.percentile(pred_to_gt, 95)),
        "completeness_p95_m": float(np.percentile(gt_to_pred, 95)),
    }


def write_error_report(out_path: Path, depth_stats: Optional[dict], recon_stats: Optional[dict]) -> None:
    lines = []
    if depth_stats is not None:
        lines.extend([
            "Depth error over inferred frames",
            f"  valid_pixels : {depth_stats['pixels']}",
            f"  MAE          : {depth_stats['mae_m']:.6f} m ({depth_stats['mae_m'] * 100:.2f} cm)",
            f"  RMSE         : {depth_stats['rmse_m']:.6f} m ({depth_stats['rmse_m'] * 100:.2f} cm)",
            f"  Max abs      : {depth_stats['max_m']:.6f} m ({depth_stats['max_m'] * 100:.2f} cm)",
            "",
        ])
    if recon_stats is not None:
        lines.extend([
            "Reconstruction point-cloud error over TSDF frames",
            f"  pred_points  : {recon_stats['pred_points']}",
            f"  gt_points    : {recon_stats['gt_points']}",
            f"  Chamfer-L1   : {recon_stats['chamfer_l1_m']:.6f} m ({recon_stats['chamfer_l1_m'] * 100:.2f} cm)",
            f"  Pred -> GT   : {recon_stats['accuracy_m']:.6f} m ({recon_stats['accuracy_m'] * 100:.2f} cm)",
            f"  GT -> Pred   : {recon_stats['completeness_m']:.6f} m ({recon_stats['completeness_m'] * 100:.2f} cm)",
            f"  P95 P->GT    : {recon_stats['accuracy_p95_m']:.6f} m ({recon_stats['accuracy_p95_m'] * 100:.2f} cm)",
            f"  P95 GT->P    : {recon_stats['completeness_p95_m']:.6f} m ({recon_stats['completeness_p95_m'] * 100:.2f} cm)",
            "",
        ])
    if not lines:
        lines.append("No valid error statistics were available.\n")
    out_path.write_text("\n".join(lines))











def _select_global_mae_frames(
    candidates: list,
    keep_count: int,
    use_filtered_depth: bool,
) -> list:
    finite = [cand for cand in candidates if np.isfinite(cand["mae_m"])]
    selected = sorted(finite, key=lambda cand: (cand["mae_m"], cand["frame_idx"]))
    return [
        {
            "mae_m": cand["mae_m"],
            "frame_idx": cand["frame_idx"],
            "pred_depth": cand["pred_filtered"] if use_filtered_depth else cand["pred_masked"],
            "gt_depth": cand["gt_masked"],
            "T": cand["T"],
        }
        for cand in selected[:keep_count]
    ]






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
    use_color:        bool = False,
    additional_crops: Optional[list[tuple[Path, np.ndarray]]] = None,
    full_mesh_path:   Optional[Path] = None,
) -> bool:
    """TSDF fusion, optionally weighted by a per-pixel confidence map.

    Open3D's legacy TSDF integrator has no observation-weight image.  We
    approximate continuous confidence weights with nested masks: a pixel is
    integrated once for every confidence level it exceeds.  This is equivalent
    to a uniformly quantized per-observation TSDF weight.  Pixels below the
    first level are set to invalid depth, so they neither update the surface
    nor carve free space.

    ``frames`` entries are ``(depth, T_world_from_cam)``,
    ``(depth, T_world_from_cam, confidence)``, or
    ``(depth, T_world_from_cam, confidence, rgb)``.  Confidence is in [0, 1];
    rgb is uint8 in the same image plane as depth.
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
        color_type=(
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if use_color
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        ),
    )
    dummy_color = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))

    weighting_msg = (
        f", confidence-weighted={confidence_levels} levels, min={min_confidence:.2f}"
        if confidence_levels > 0 else ""
    )
    color_msg = ", RGB color" if use_color else ""
    print(f"  TSDF fusing {len(frames)} depth frames "
          f"(voxel={voxel_length * 1000:.1f} mm, trunc={sdf_trunc * 1000:.1f} mm"
          f"{weighting_msg}{color_msg}) …")

    def integrate_depth(
        depth_m: np.ndarray,
        T_cam_from_world: np.ndarray,
        color_np: Optional[np.ndarray] = None,
    ) -> None:
        depth_for_color = depth_m
        if use_color:
            if color_np is None:
                raise ValueError("use_color=True requires RGB frames")
            if color_np.shape[:2] != depth_m.shape or color_np.shape[-1] != 3:
                raise ValueError(
                    f"RGB shape {color_np.shape} does not match depth {depth_m.shape}"
                )
            # rgb_in_event_frame.h5 stores projected RealSense RGB in the event
            # image plane.  Pixels without a projected RGB sample are black.
            # Open3D has no separate color-validity mask, so if we pass those
            # pixels through it treats "no RGB sample" as an actual black color
            # observation and averages it into the TSDF colors.  For colored
            # meshes, suppress those depth observations as well; geometry-only
            # fusion is unchanged because this branch only runs with use_color.
            color_valid = np.any(color_np > 0, axis=2)
            depth_for_color = np.where(color_valid, depth_m, np.float32(0.0))
            color_o3d = o3d.geometry.Image(
                np.ascontiguousarray(color_np, dtype=np.uint8)
            )
        else:
            color_o3d = dummy_color
        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_for_color, dtype=np.float32)
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, T_cam_from_world)

    for frame in frames:
        depth_m, T_world_from_cam = frame[:2]
        confidence = frame[2] if len(frame) > 2 else None
        color_np = frame[3] if len(frame) > 3 else None
        T_cam_from_world = np.linalg.inv(T_world_from_cam)
        if confidence_levels <= 0 or confidence is None:
            integrate_depth(depth_m, T_cam_from_world, color_np)
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
                integrate_depth(weighted_depth, T_cam_from_world, color_np)

    full_mesh = volume.extract_triangle_mesh()
    full_mesh.compute_vertex_normals()

    def cropped_mesh(center: Optional[np.ndarray]):
        if center is None or cube_half_side <= 0.0:
            return full_mesh
        verts = np.asarray(full_mesh.vertices)
        inside = np.flatnonzero(points_in_cube(verts, center, cube_half_side))
        result = full_mesh.select_by_index(inside)
        result.compute_vertex_normals()
        return result

    mesh = cropped_mesh(cube_center)
    print(f"  Cube crop (side={cube_half_side*2*100:.0f} cm) → "
          f"{len(np.asarray(mesh.vertices)):,} verts, "
          f"{len(np.asarray(mesh.triangles)):,} triangles")

    def write_mesh(path: Path, mesh_to_write) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(
            str(path), mesh_to_write, write_vertex_normals=not use_color,
        )
        print(f"  Mesh → {path}  ({len(np.asarray(mesh_to_write.vertices)):,} verts, "
              f"{len(np.asarray(mesh_to_write.triangles)):,} triangles)")

    # For colored PLYs, prefer a simple xyz+rgb vertex layout.  Some viewers
    # pick the normal/scalar array for false-color rendering when Open3D writes
    # normals before red/green/blue, which makes the mesh look psychedelic even
    # though the stored RGB values are sane.  Viewers can recompute normals.
    if full_mesh_path is not None:
        write_mesh(full_mesh_path, full_mesh)
    write_mesh(out_path, mesh)
    for extra_path, extra_center in additional_crops or []:
        write_mesh(extra_path, cropped_mesh(extra_center))
    n_v = len(np.asarray(mesh.vertices))
    n_t = len(np.asarray(mesh.triangles))
    return n_v > 0 and n_t > 0


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


def _render_mesh_depth(
    mesh_path: Path,
    K: np.ndarray,
    T_world_from_cam: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Render metric ray depth from a mesh using an Open3D raycasting scene."""
    import open3d as o3d

    legacy_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K.astype(np.float64)),
        extrinsic_matrix=o3d.core.Tensor(
            np.linalg.inv(T_world_from_cam).astype(np.float64)
        ),
        width_px=width,
        height_px=height,
    )
    depth = scene.cast_rays(rays)["t_hit"].numpy().astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def evaluate_rendered_depth(
    mesh_path: Path,
    frame_indices: List[int],
    depth_h5_path: Path,
    ee_T_all: np.ndarray,
    T_ee_from_event: np.ndarray,
    K: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw: Optional[Tuple[int, int]],
    depth_min: float,
    depth_max: float,
) -> Optional[dict]:
    """Render the predicted mesh from held-out poses and compare with GT depth."""
    if not frame_indices:
        return None
    try:
        import open3d  # noqa: F401
    except ImportError:
        print("  [metrics] open3d unavailable — skipping rendered-depth metrics")
        return None

    abs_sum = sq_sum = abs_rel_sum = 0.0
    valid_gt_pixels = rendered_valid_pixels = compared_pixels = 0
    frame_rows = []
    with h5py.File(depth_h5_path, "r") as depth_file:
        for frame_idx in frame_indices:
            gt = depth_file["depth"][frame_idx].astype(np.float32)
            if resize_hw is not None or crop_hw is not None:
                gt = crop_resize(gt, resize_hw, crop_hw, mode="bilinear")
            T_world_from_cam = ee_T_all[frame_idx] @ T_ee_from_event
            rendered = _render_mesh_depth(
                mesh_path, K, T_world_from_cam, gt.shape[0], gt.shape[1]
            )
            gt_valid = np.isfinite(gt) & (gt > depth_min) & (gt < depth_max)
            render_valid = (
                np.isfinite(rendered)
                & (rendered > depth_min)
                & (rendered < depth_max)
            )
            compared = gt_valid & render_valid
            valid_gt_pixels += int(gt_valid.sum())
            rendered_valid_pixels += int((gt_valid & render_valid).sum())
            compared_pixels += int(compared.sum())
            if not compared.any():
                continue
            error = rendered[compared].astype(np.float64) - gt[compared].astype(np.float64)
            abs_error = np.abs(error)
            frame_rows.append({
                "frame_idx": int(frame_idx),
                "pixels": int(compared.sum()),
                "mae_m": float(abs_error.mean()),
                "rmse_m": float(np.sqrt(np.square(error).mean())),
                "abs_rel": float((abs_error / gt[compared]).mean()),
                "valid_render_fraction": float(
                    (gt_valid & render_valid).sum() / max(int(gt_valid.sum()), 1)
                ),
            })
            abs_sum += float(abs_error.sum())
            sq_sum += float(np.square(error).sum())
            abs_rel_sum += float((abs_error / gt[compared]).sum())

    if compared_pixels == 0:
        return None
    return {
        "held_out_frames": int(len(frame_indices)),
        "frames_with_overlap": int(len(frame_rows)),
        "compared_pixels": int(compared_pixels),
        "mae_m": abs_sum / compared_pixels,
        "rmse_m": float(np.sqrt(sq_sum / compared_pixels)),
        "abs_rel": abs_rel_sum / compared_pixels,
        "valid_render_percentage": (
            rendered_valid_pixels / valid_gt_pixels if valid_gt_pixels > 0 else 0.0
        ),
        "per_frame": frame_rows,
    }


def write_tsdf_metric_outputs(
    out_dir: Path,
    surface_metrics: Optional[dict],
    rendered_metrics: Optional[dict],
    file_stem: str = "tsdf_metrics",
) -> None:
    payload = {
        "surface_metrics": surface_metrics,
        "rendered_depth_metrics": rendered_metrics,
    }
    (out_dir / f"{file_stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = ["TSDF reconstruction metrics", ""]
    if surface_metrics is None:
        lines.append("Surface metrics: unavailable")
    else:
        lines.extend([
            "Uniform mesh-surface metrics",
            f"  Accuracy mean      : {surface_metrics['accuracy_mean_m']:.6f} m",
            f"  Accuracy median    : {surface_metrics['accuracy_median_m']:.6f} m",
            f"  Completeness mean  : {surface_metrics['completeness_mean_m']:.6f} m",
            f"  Completeness median: {surface_metrics['completeness_median_m']:.6f} m",
            f"  Chamfer mean       : {surface_metrics['chamfer_mean_m']:.6f} m",
            f"  Chamfer median     : {surface_metrics['chamfer_median_m']:.6f} m",
            f"  Normal consistency : {surface_metrics['normal_consistency_symmetric']:.6f}",
        ])
        for label in ("1cm", "2cm", "5cm"):
            lines.append(
                f"  {label}: precision={surface_metrics[f'precision_{label}']:.2%}, "
                f"recall={surface_metrics[f'recall_{label}']:.2%}, "
                f"F-score={surface_metrics[f'fscore_{label}']:.2%}"
            )
    lines.append("")
    if rendered_metrics is None:
        lines.append("Rendered-depth metrics: unavailable")
    else:
        lines.extend([
            "Held-out rendered-depth metrics",
            f"  Frames              : {rendered_metrics['held_out_frames']}",
            f"  Valid render        : {rendered_metrics['valid_render_percentage']:.2%}",
            f"  MAE                 : {rendered_metrics['mae_m']:.6f} m",
            f"  RMSE                : {rendered_metrics['rmse_m']:.6f} m",
            f"  AbsRel              : {rendered_metrics['abs_rel']:.6f}",
        ])
    (out_dir / f"{file_stem}.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    if surface_metrics is None and rendered_metrics is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if surface_metrics is not None:
        axes[0].bar(
            ["Accuracy", "Completeness", "Chamfer"],
            [
                surface_metrics["accuracy_mean_m"] * 100,
                surface_metrics["completeness_mean_m"] * 100,
                surface_metrics["chamfer_mean_m"] * 100,
            ],
            color=["#4472c4", "#ed7d31", "#70ad47"],
        )
        axes[0].set_ylabel("Mean distance [cm]")
        axes[0].set_title("Surface distance")
        labels = ["1 cm", "2 cm", "5 cm"]
        x = np.arange(3)
        width = 0.25
        axes[1].bar(
            x - width,
            [100 * surface_metrics[f"precision_{k}"] for k in ("1cm", "2cm", "5cm")],
            width,
            label="Precision",
        )
        axes[1].bar(
            x,
            [100 * surface_metrics[f"recall_{k}"] for k in ("1cm", "2cm", "5cm")],
            width,
            label="Recall",
        )
        axes[1].bar(
            x + width,
            [100 * surface_metrics[f"fscore_{k}"] for k in ("1cm", "2cm", "5cm")],
            width,
            label="F-score",
        )
        axes[1].set_xticks(x, labels)
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("Score [%]")
        axes[1].set_title("Surface thresholds")
        axes[1].legend()
    else:
        axes[0].axis("off")
        axes[1].axis("off")
    if rendered_metrics is not None:
        axes[2].bar(
            ["MAE", "RMSE"],
            [rendered_metrics["mae_m"] * 100, rendered_metrics["rmse_m"] * 100],
            color=["#5b9bd5", "#c55a11"],
        )
        axes[2].set_ylabel("Error [cm]")
        axes[2].set_title(
            f"Rendered depth\nvalid={rendered_metrics['valid_render_percentage']:.1%}"
        )
    else:
        axes[2].axis("off")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{file_stem}.png", dpi=180)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def colorize(arr: np.ndarray, vmin: float, vmax: float,
             cmap: str = "turbo") -> np.ndarray:
    normed = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba   = getattr(cm, cmap)(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_depth_png(depth_m: np.ndarray, mask: np.ndarray,
                   path: Path, vmin: float, vmax: float) -> None:
    img = colorize(np.where(mask > 0.5, depth_m, np.nan), vmin, vmax)
    img[~(mask > 0.5)] = 40
    plt.imsave(str(path), img)




def save_overview(samples: list, out_path: Path,
                  depth_min: float, depth_max: float) -> None:
    """Side-by-side: events | GT depth | pred depth | abs error."""
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5), squeeze=False)
    col_titles = ["Events (summed)", "GT depth", "Pred depth", "Abs error"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for row, s in enumerate(samples):
        ax_ev, ax_gt, ax_pr, ax_er = axes[row]
        frame_idx = s["frame_idx"]

        ev_sum = s["voxel"].sum(axis=0)
        ev_vis = np.clip((ev_sum - ev_sum.min()) / (ev_sum.max() - ev_sum.min() + 1e-6), 0, 1)
        ax_ev.imshow(ev_vis, cmap="gray")
        ax_ev.set_ylabel(f"frame {frame_idx}", fontsize=8)

        gt_mask = s["gt_mask"]
        pred_mask = s["pred_mask"]
        gt_valid = gt_mask > 0.5
        pred_valid = pred_mask > 0.5

        gt_vis = colorize(np.where(gt_valid, s["gt"], 0), depth_min, depth_max)
        pr_vis = colorize(np.where(pred_valid, s["pred"], 0), depth_min, depth_max)
        gt_vis[~gt_valid] = 40
        pr_vis[~pred_valid] = 40
        ax_gt.imshow(gt_vis)
        ax_pr.imshow(pr_vis)

        err = np.abs(s["pred"] - s["gt"]) * gt_valid
        err_vis = colorize(err, 0, 0.1)
        err_vis[~gt_valid] = 40
        ax_er.imshow(err_vis)
        ax_er.set_xlabel(
            f"MAE {err[gt_valid].mean() * 100:.1f} cm" if gt_valid.any() else "",
            fontsize=8,
        )

        for ax in (ax_ev, ax_gt, ax_pr, ax_er):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview → {out_path}")








# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def _is_reconstruction_sequence(path: Path) -> bool:
    """Return whether path contains the inputs required for reconstruction."""
    return (
        (path / "hdf5" / "depth_in_event_frame.h5").is_file()
        and (path / "events" / "voxels_cam0.h5").is_file()
        and (path / "hdf5" / "table_plane.h5").is_file()
    )


def _default_output_dir(data_dir: str | Path, out_subdir: str | Path) -> Path:
    """Return the default output folder stored inside a data directory."""
    return Path(data_dir) / OUTPUT_DIRNAME / out_subdir


def _set_cli_option(argv: list[str], option: str, value: str) -> list[str]:
    """Replace one single-valued CLI option, accepting --option=value too."""
    updated: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == option:
            index += 2
            continue
        if token.startswith(option + "="):
            index += 1
            continue
        updated.append(token)
        index += 1
    updated.extend([option, value])
    return updated


def _remove_cli_flag(argv: list[str], option: str) -> list[str]:
    """Remove a boolean CLI flag, including its ``--flag=value`` form."""
    return [
        token
        for token in argv
        if token != option and not token.startswith(option + "=")
    ]


def _load_reconstruction_run_summary(output_dir: Path) -> dict:
    """Load an aggregate summary or normalize a single-sequence result."""
    aggregate_path = output_dir / "reconstruction_summary.json"
    if aggregate_path.is_file():
        return json.loads(aggregate_path.read_text(encoding="utf-8"))

    sequence_path = output_dir / "reconstruction_metrics.json"
    if not sequence_path.is_file():
        raise FileNotFoundError(
            f"No reconstruction summary found below {output_dir}"
        )
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    return {
        "object_count": 1,
        "objects": [output_dir.name],
        "surface_selection": payload.get("surface_selection", "all_surfaces"),
        "cases": payload.get("cases", {}),
    }


def _write_uncertainty_comparison_summary(
    output_root: Path,
    uniform_dir: Path,
    weighted_dir: Path,
) -> None:
    """Compare aggregate metrics from uniform and confidence-weighted TSDF.

    Each comparison includes the signed difference and weighted/uniform ratio.
    """
    uniform = _load_reconstruction_run_summary(uniform_dir)
    weighted = _load_reconstruction_run_summary(weighted_dir)
    uniform_surface_selection = uniform.get("surface_selection", "all_surfaces")
    weighted_surface_selection = weighted.get("surface_selection", "all_surfaces")
    if uniform_surface_selection != weighted_surface_selection:
        raise ValueError(
            "Cannot compare uniform and uncertainty-weighted runs with "
            "different surface selection policies: "
            f"{uniform_surface_selection!r} versus "
            f"{weighted_surface_selection!r}"
        )
    variant_payloads = {
        "uniform_tsdf": uniform,
        "uncertainty_weighted_tsdf": weighted,
    }
    comparison: dict[str, dict] = {}
    lines = [
        "Reconstruction comparison: uniform vs uncertainty-weighted TSDF",
        f"Uniform objects: {uniform.get('object_count', 'N/A')}",
        f"Weighted objects: {weighted.get('object_count', 'N/A')}",
        f"Surface selection: {uniform_surface_selection}",
        "Difference is uncertainty-weighted minus uniform.",
        (
            "Factor is uncertainty-weighted divided by uniform; for error "
            "metrics, values below 1 indicate a reduction and values above 1 "
            "an increase."
        ),
        "",
    ]

    case_names = list(
        dict.fromkeys(
            list(uniform.get("cases", {})) + list(weighted.get("cases", {}))
        )
    )
    for case_name in case_names:
        lines.append(case_name)
        comparison[case_name] = {}
        uniform_case = uniform.get("cases", {}).get(case_name, {}) or {}
        weighted_case = weighted.get("cases", {}).get(case_name, {}) or {}
        section_names = list(
            dict.fromkeys(list(uniform_case) + list(weighted_case))
        )
        for section_name in section_names:
            uniform_section = uniform_case.get(section_name) or {}
            weighted_section = weighted_case.get(section_name) or {}
            if not isinstance(uniform_section, dict) or not isinstance(
                weighted_section, dict
            ):
                continue
            metric_names = list(
                dict.fromkeys(list(uniform_section) + list(weighted_section))
            )
            section_comparison = {}
            section_lines = []
            for metric_name in metric_names:
                uniform_value = uniform_section.get(metric_name)
                weighted_value = weighted_section.get(metric_name)
                if not isinstance(uniform_value, (int, float)) or not isinstance(
                    weighted_value, (int, float)
                ):
                    continue
                uniform_value = float(uniform_value)
                weighted_value = float(weighted_value)
                if not np.isfinite(uniform_value) or not np.isfinite(weighted_value):
                    continue
                difference = weighted_value - uniform_value
                factor = (
                    weighted_value / uniform_value
                    if uniform_value != 0.0
                    else None
                )
                section_comparison[metric_name] = {
                    "uniform": uniform_value,
                    "uncertainty_weighted": weighted_value,
                    "difference": difference,
                    "weighted_over_uniform_factor": factor,
                }
                factor_text = (
                    f"{factor:.6f}x"
                    if factor is not None
                    else "undefined (uniform=0)"
                )
                section_lines.append(
                    f"    {metric_name}: uniform={uniform_value:.6f}, "
                    f"weighted={weighted_value:.6f}, difference={difference:+.6f}, "
                    f"factor={factor_text}"
                )
            if section_comparison:
                comparison[case_name][section_name] = section_comparison
                lines.append(f"  {section_name}")
                lines.extend(section_lines)
        lines.append("")

    summary_json = {
        "description": (
            "uncertainty-weighted versus uniform TSDF comparison; includes "
            "weighted-minus-uniform differences and weighted/uniform factors"
        ),
        "surface_selection": uniform_surface_selection,
        "variants": variant_payloads,
        "comparison": comparison,
    }
    (output_root / "reconstruction_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (output_root / "reconstruction_summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
    )


def _run_uncertainty_tsdf_comparison(args: argparse.Namespace) -> bool:
    """Run uniform and uncertainty-weighted reconstruction variants."""
    if not args.compare_uncertainty_tsdf:
        return False

    out_subdir = args.out_subdir or Path(args.checkpoint).stem
    output_root = (
        Path(args.out_dir)
        if args.out_dir is not None
        else _default_output_dir(args.data_dir, out_subdir)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base_args = _remove_cli_flag(
        sys.argv[1:], "--compare_uncertainty_tsdf"
    )
    base_args = _remove_cli_flag(base_args, "--compare-uncertainty-tsdf")
    base_args = _remove_cli_flag(base_args, "--uncertainty_weighted_tsdf")
    variants = (
        ("uniform_tsdf", False),
        ("uncertainty_weighted_tsdf", True),
    )
    for variant_name, weighted in variants:
        if args.out_dir is not None:
            child_args = _set_cli_option(
                base_args, "--out_dir", str(output_root / variant_name)
            )
        else:
            child_args = _set_cli_option(
                base_args, "--out_subdir", str(Path(out_subdir) / variant_name)
            )
        if weighted:
            child_args.append("--uncertainty_weighted_tsdf")
        command = [sys.executable, str(Path(__file__).resolve()), *child_args]
        print(f"\nRunning reconstruction variant: {variant_name}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Reconstruction variant {variant_name} failed with exit "
                f"code {result.returncode}"
            )

    _write_uncertainty_comparison_summary(
        output_root,
        output_root / "uniform_tsdf",
        output_root / "uncertainty_weighted_tsdf",
    )
    print(
        f"\nTSDF comparison summary: "
        f"{output_root / 'reconstruction_summary.txt'}",
        flush=True,
    )
    return True


def _write_reconstruction_summary(
    output_root: Path, sequence_outputs: list[tuple[str, Path]]
) -> None:
    """Aggregate per-sequence two-mask metrics and write plots/mean values."""
    runs = []
    for sequence_name, sequence_out in sequence_outputs:
        path = sequence_out / "reconstruction_metrics.json"
        if path.is_file():
            runs.append((sequence_name, json.loads(path.read_text(encoding="utf-8"))))
    if not runs:
        return

    surface_selections = {
        payload.get("surface_selection", "all_surfaces")
        for _, payload in runs
    }
    if len(surface_selections) != 1:
        raise ValueError(
            "Cannot aggregate reconstruction runs with different surface "
            f"selection policies: {sorted(surface_selections)}"
        )
    surface_selection = next(iter(surface_selections))

    def values(case: str, section: str, metric: str) -> list[float]:
        result = []
        for _, payload in runs:
            value = ((payload.get("cases", {}).get(case, {}).get(section) or {}).get(metric))
            if isinstance(value, (int, float)) and np.isfinite(value):
                result.append(float(value))
        return result

    cases = (
        ("full_scale", "Full scale"),
        ("current_mask", "Current mask"),
        ("raised_object_cube", "Raised object cube (+1.5 cm)"),
    )
    sections = {
        "depth_metrics": ("mae_m", "rmse_m", "max_m"),
        "prefusion_pointcloud_metrics": (
            "chamfer_l1_m", "accuracy_m", "completeness_m",
            "accuracy_p95_m", "completeness_p95_m",
        ),
        "surface_metrics": (
            "accuracy_mean_m", "accuracy_median_m", "completeness_mean_m",
            "completeness_median_m", "chamfer_mean_m", "chamfer_median_m",
            "normal_consistency_symmetric", "precision_1cm", "recall_1cm",
            "fscore_1cm", "precision_2cm", "recall_2cm", "fscore_2cm",
            "precision_5cm", "recall_5cm", "fscore_5cm",
        ),
        "original_depth_surface_metrics": (
            "accuracy_mean_m", "accuracy_median_m", "completeness_mean_m",
            "completeness_median_m", "chamfer_mean_m", "chamfer_median_m",
            "normal_consistency_symmetric", "precision_1cm", "recall_1cm",
            "fscore_1cm", "precision_2cm", "recall_2cm", "fscore_2cm",
            "precision_5cm", "recall_5cm", "fscore_5cm",
        ),
        "rendered_depth_metrics": ("mae_m", "rmse_m", "abs_rel", "valid_render_percentage"),
    }
    lines = [
        "Reconstruction summary (unweighted mean across objects)",
        f"Surface selection: {surface_selection}",
        f"Objects: {len(runs)} ({', '.join(name for name, _ in runs)})",
        "",
    ]
    summary_json = {
        "object_count": len(runs),
        "objects": [name for name, _ in runs],
        "surface_selection": surface_selection,
        "cases": {},
    }
    for case, label in cases:
        lines.append(label)
        summary_json["cases"][case] = {}
        for section, metrics in sections.items():
            lines.append(f"  {section}")
            summary_json["cases"][case][section] = {}
            for metric in metrics:
                vals = values(case, section, metric)
                if not vals:
                    continue
                mean = float(np.mean(vals))
                summary_json["cases"][case][section][metric] = mean
                unit = " m" if metric.endswith("_m") else ""
                lines.append(f"    {metric}: {mean:.6f}{unit}  (n={len(vals)})")
        lines.append("")
    (output_root / "reconstruction_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "reconstruction_summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n", encoding="utf-8"
    )

    names = [name for name, _ in runs]
    x = np.arange(len(names))
    width = 0.38
    plotted_cases = cases[1:]
    fig, ax = plt.subplots(figsize=(max(8, 1.35 * len(names)), 5))
    for offset, (case, label) in zip((-width / 2, width / 2), plotted_cases):
        vals = [
            100.0 * float(((payload["cases"].get(case, {}).get("surface_metrics") or {}).get("chamfer_mean_m", np.nan)))
            for _, payload in runs
        ]
        ax.bar(x + offset, vals, width, label=label)
    ax.set_xticks(x, names)
    ax.set_ylabel("Post-TSDF surface Chamfer [cm]")
    ax.set_title("Reconstruction Chamfer by object and spatial mask")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "chamfer_by_object_and_mask.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = (
        ("depth_metrics", "mae_m", "Depth MAE [cm]"),
        ("surface_metrics", "chamfer_mean_m", "Surface Chamfer [cm]"),
        ("rendered_depth_metrics", "mae_m", "Rendered-depth MAE [cm]"),
    )
    for ax, (section, metric, title) in zip(axes, panels):
        for offset, (case, label) in zip(
            (-width / 2, width / 2), plotted_cases
        ):
            vals = [100.0 * float(((payload["cases"].get(case, {}).get(section) or {}).get(metric, np.nan))) for _, payload in runs]
            ax.bar(x + offset, vals, width, label=label)
        ax.set_xticks(x, names, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_root / "reconstruction_error_overview.png", dpi=180)
    plt.close(fig)


def _run_sequence_directory(args: argparse.Namespace) -> bool:
    """Run one child process per sequence when --data_dir is a parent folder."""
    data_root = Path(args.data_dir)
    if _is_reconstruction_sequence(data_root):
        return False
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")

    sequence_dirs = sorted(
        child
        for child in data_root.iterdir()
        if child.is_dir() and _is_reconstruction_sequence(child)
    )
    if not sequence_dirs:
        raise FileNotFoundError(
            f"{data_root} is neither a valid sequence nor a directory containing "
            "valid sequence subdirectories"
        )

    out_subdir = args.out_subdir or Path(args.checkpoint).stem
    output_root = (
        Path(args.out_dir)
        if args.out_dir is not None
        else _default_output_dir(data_root, out_subdir)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sequence_outputs = [
        (
            sequence_dir.name,
            output_root / sequence_dir.name
            if args.out_dir is not None
            else _default_output_dir(sequence_dir, out_subdir),
        )
        for sequence_dir in sequence_dirs
    ]
    print(
        f"Detected sequence parent: {data_root}\n"
        f"Reconstructing {len(sequence_dirs)} sequences; summary: {output_root}",
        flush=True,
    )

    failures: list[tuple[Path, int]] = []
    for sequence_number, (sequence_dir, (_, sequence_out)) in enumerate(
        zip(sequence_dirs, sequence_outputs), start=1
    ):
        child_args = _set_cli_option(sys.argv[1:], "--data_dir", str(sequence_dir))
        child_args = _set_cli_option(child_args, "--out_dir", str(sequence_out))
        command = [sys.executable, str(Path(__file__).resolve()), *child_args]
        print(
            f"\n[{sequence_number}/{len(sequence_dirs)}] "
            f"Reconstructing {sequence_dir.name}",
            flush=True,
        )
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append((sequence_dir, result.returncode))

    if failures:
        failure_text = ", ".join(
            f"{path.name} (exit {returncode})"
            for path, returncode in failures
        )
        raise RuntimeError(
            f"Reconstruction failed for {len(failures)}/{len(sequence_dirs)} "
            f"sequences: {failure_text}"
        )

    _write_reconstruction_summary(output_root, sequence_outputs)

    print(
        f"\nCompleted reconstruction for all {len(sequence_dirs)} sequences. "
        f"Output root: {output_root}",
        flush=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth-map, point-cloud, and TSDF reconstruction from a trained MVS model.",
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
        type=str,
        required=True,
        help=(
            "One sequence directory, or a parent whose immediate child "
            "directories are sequences"
        ),
    )
    parser.add_argument("--save_frame_visualizations", "--save-frame-visualizations",
                        action="store_true",
                        help="Write per-frame depth/point-cloud folders and overview.png")
    parser.add_argument("--n_frames", type=int, default=5,
                        help="Number of frames to visualise in the overview")
    parser.add_argument("--frame_step", type=int, default=None,
                        help="Use every Nth frame for the mesh (overrides --mesh_frame_count)")
    parser.add_argument("--mesh_frame_count", type=int, default=80,
                        help="Total number of evenly spaced frames to predict when --frame_step is not set")
    parser.add_argument("--best_mae_frames", type=int, default=None, metavar="N",
                        help="Fuse the N predicted frames with the lowest whole-image MAE")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Explicit visualisation frame indices (overrides --n_frames)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: <data_dir>/reconstruction_output/<checkpoint_name>)")
    # Internal: path below reconstruction_output/ that parent runs pass to
    # child processes when --out_dir is not set.
    parser.add_argument("--out_subdir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--depth_min", type=float, default=DEPTH_MIN)
    parser.add_argument("--depth_max", type=float, default=D_MAX)
    parser.add_argument("--resize_h", type=int, default=PREPROCESS_RESIZE_HW[0])
    parser.add_argument("--resize_w", type=int, default=PREPROCESS_RESIZE_HW[1])
    parser.add_argument("--crop_h",   type=int, default=PREPROCESS_CROP_HW[0])
    parser.add_argument("--crop_w",   type=int, default=PREPROCESS_CROP_HW[1])
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
    parser.add_argument(
        "--uncertainty_weighted_tsdf",
        action="store_true",
        help=(
            "Weight predicted TSDF updates with the checkpoint's learned "
            "per-pixel confidence; does not affect GT TSDF fusion"
        ),
    )
    parser.add_argument(
        "--compare_uncertainty_tsdf",
        "--compare-uncertainty-tsdf",
        action="store_true",
        help=(
            "Run both uniform and uncertainty-weighted TSDF reconstruction "
            "with identical settings, then compare their metrics in the root "
            "reconstruction summary"
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
    parser.add_argument("--skip_gt_mesh", action="store_true",
                        help="Do not fuse gt_mesh.obj; useful for faster iteration")
    parser.add_argument(
        "--save_largest_connected_surface",
        "--save-largest-connected-surface",
        "--save_largest_connected_component",
        action="store_true",
        help=(
            "Write largest-connected-surface copies of the predicted and GT "
            "TSDF meshes and use those copies for all primary surface and "
            "rendered-depth metrics, including the uncertainty comparison"
        ),
    )
    parser.add_argument(
        "--evaluate_full_scale",
        "--evaluate-full-scale",
        action="store_true",
        help=(
            "Save uncropped predicted/GT TSDF meshes and report their surface "
            "and held-out rendered-depth errors"
        ),
    )
    parser.add_argument(
        "--color_mesh",
        "--rgb_mesh",
        action="store_true",
        help=(
            "Fuse recorded RGB projected into the event frame into the predicted "
            "TSDF mesh. Writes the predicted mesh as PLY so vertex colors are preserved."
        ),
    )
    parser.add_argument(
        "--save_original_depth_mesh",
        action="store_true",
        help=(
            "Also TSDF-fuse hdf5/realsense.h5 native RealSense depth frames "
            "without projecting them into the event-camera plane"
        ),
    )
    parser.add_argument(
        "--evaluate_original_depth_mesh",
        action="store_true",
        help=(
            "Create the native-resolution RealSense depth TSDF mesh and use "
            "it as an additional reference for surface evaluation of the "
            "predicted workspace mesh"
        ),
    )
    parser.add_argument("--error_stride", type=int, default=4,
                        help="Pixel stride for reconstruction-level point-cloud error")
    parser.add_argument("--error_max_points", type=int, default=5000,
                        help="Maximum predicted/GT points used for reconstruction-level error")
    parser.add_argument("--surface_samples", type=int, default=100000,
                        help="Uniform samples per TSDF mesh for surface metrics")
    parser.add_argument("--render_eval_frames", type=int, default=20,
                        help="Held-out camera poses used for rendered-depth evaluation")
    parser.add_argument("--cube_side", type=float, default=SPATIAL_CUBE_SIDE)
    parser.add_argument("--target_x", type=float, default=SPATIAL_TARGET_X)
    parser.add_argument("--target_y", type=float, default=SPATIAL_TARGET_Y)
    parser.add_argument("--target_z", type=float, default=SPATIAL_TARGET_Z)
    args = parser.parse_args()
    use_original_depth_mesh = bool(
        args.save_original_depth_mesh or args.evaluate_original_depth_mesh
    )
    if args.mesh_frame_count <= 0:
        parser.error("--mesh_frame_count must be > 0")
    if args.tsdf_confidence_levels <= 0:
        parser.error("--tsdf_confidence_levels must be > 0")
    if not 0.0 <= args.tsdf_min_confidence < 1.0:
        parser.error("--tsdf_min_confidence must be in [0, 1)")
    if args.best_mae_frames is not None and args.best_mae_frames <= 0:
        parser.error("--best_mae_frames must be > 0")
    if args.surface_samples <= 0:
        parser.error("--surface_samples must be > 0")
    if args.render_eval_frames < 0:
        parser.error("--render_eval_frames must be >= 0")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    if _run_uncertainty_tsdf_comparison(args):
        return

    if _run_sequence_directory(args):
        return

    ckpt_path      = Path(args.checkpoint)
    data_dir       = Path(args.data_dir)
    cube_center    = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)
    cube_half_side = args.cube_side / 2.0
    raised_object_center = np.array(
        [args.target_x, args.target_y, RAISED_OBJECT_BOTTOM_Z_M + cube_half_side],
        dtype=np.float64,
    )

    resize_hw = (args.resize_h, args.resize_w)
    crop_hw = (args.crop_h, args.crop_w)
    if min(*resize_hw, *crop_hw) <= 0:
        parser.error("crop and resize dimensions must be positive")

    if args.out_dir is None:
        out_dir = _default_output_dir(data_dir, args.out_subdir or ckpt_path.stem)
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data paths ────────────────────────────────────────────────
    depth_h5_path      = data_dir / "hdf5" / "depth_in_event_frame.h5"
    realsense_h5_path  = data_dir / "hdf5" / "realsense.h5"
    rgb_h5_path        = data_dir / "hdf5" / "rgb_in_event_frame.h5"
    voxels_h5_path     = data_dir / "events" / "voxels_cam0.h5"
    poses_h5_path      = data_dir / "hdf5" / "poses.h5"
    table_plane_h5_path = data_dir / "hdf5" / "table_plane.h5"

    for p, hint in [
        (depth_h5_path,       "Run data_precomputation/project_realsense_to_event.py first."),
        (voxels_h5_path,      "Run data_precomputation/precompute_voxels.py first."),
        (table_plane_h5_path, "Run data_precomputation/precompute_table_plane.py first."),
        (poses_h5_path,       "The MVS model requires camera poses for source-view warping."),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}\n{hint}")
    with h5py.File(table_plane_h5_path, "r") as table_file:
        transform = table_file.attrs.get("intrinsics_transform", "")
        if isinstance(transform, bytes):
            transform = transform.decode("utf-8", errors="replace")
        corrected_table_transform = transform == INTRINSICS_TRANSFORM
    if not corrected_table_transform:
        raise RuntimeError(
            f"{table_plane_h5_path} uses obsolete direct-scaling geometry. "
            "Regenerate it with data_precomputation/precompute_table_plane.py "
            "--overwrite."
        )
    if use_original_depth_mesh and not realsense_h5_path.exists():
        raise FileNotFoundError(
            f"Missing: {realsense_h5_path}\n"
            "Native-depth mesh creation requires the raw RealSense depth file."
        )
    if args.color_mesh and not rgb_h5_path.exists():
        raise FileNotFoundError(
            f"Missing: {rgb_h5_path}\n"
            "--color_mesh requires RGB projected into the event frame. "
            "Run data_precomputation/project_realsense_to_event.py with RGB enabled first."
        )

    # ── Camera intrinsics ─────────────────────────────────────────
    K, out_H, out_W = load_K(CALIB_DIR, resize_hw, crop_hw)
    print(f"Input resolution : {out_W}×{out_H}")
    if use_original_depth_mesh:
        K_depth, T_ee_from_depth, original_depth_scale = load_original_depth_calibration(CALIB_DIR)
        print("Original depth   : native RealSense depth enabled")
    else:
        K_depth = T_ee_from_depth = None
        original_depth_scale = 0.001

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")

    # ── Model ─────────────────────────────────────────────────────
    model, ckpt = load_model(ckpt_path, device)
    if args.pose_layout_override is not None:
        ckpt["allow_unbalanced_pose_views"] = args.pose_layout_override
        ckpt["allow_fewer_pose_views"] = args.pose_layout_override
    expected_transform = INTRINSICS_TRANSFORM
    checkpoint_transform = ckpt.get("intrinsics_transform", "")
    if checkpoint_transform != expected_transform:
        raise RuntimeError(
            f"Checkpoint uses {checkpoint_transform!r}, but reconstruction requested "
            f"{expected_transform!r}. Use a crop-then-resize checkpoint."
        )
    use_learned_uncertainty = args.uncertainty_weighted_tsdf
    if use_learned_uncertainty:
        if not bool(ckpt.get("uncertainty", False)):
            parser.error(
                "Learned uncertainty requires a train_mvs.py checkpoint trained with --uncertainty"
            )
        if not any(key.startswith("confidence_head.") for key in ckpt.get("model", {})):
            parser.error(
                "Learned uncertainty was requested, but the checkpoint has no confidence-head weights"
            )
    depth_min = ckpt.get("depth_min", args.depth_min)
    depth_max = ckpt.get("depth_max", args.depth_max)

    print(f"Checkpoint       : {ckpt_path.name}")
    print(f"  in_ch          : {ckpt.get('in_ch', NUM_BINS + 1)}")
    print(f"  architecture   : {ckpt.get('model_arch', 'ModernMVSNet')}")
    print(f"  num_views      : {ckpt.get('num_views', 5)}")
    print(f"  coarse_depths  : {ckpt.get('coarse_depths', 32)}")
    print(f"  view_interval  : {ckpt.get('view_interval', 5)}")
    print(f"  pose selection : {ckpt.get('pose_view_selection', False)}")
    if ckpt.get("pose_view_selection", False):
        print(
            "  pose layouts   : "
            + (
                "fewer masked boundary views allowed"
                if ckpt.get(
                    "allow_fewer_pose_views",
                    ckpt.get("allow_unbalanced_pose_views", False),
                )
                else "strictly balanced targets only"
            )
        )
    print("  feature encoder: deep_fpn")
    print(f"  uncertainty    : {bool(ckpt.get('uncertainty', False))}")
    print(f"  table_z        : from {table_plane_h5_path.name}")
    print(f"  depth range    : {depth_min} – {depth_max} m")

    # ── Poses ─────────────────────────────────────────────────────
    with h5py.File(poses_h5_path, "r") as f:
        ee_T_all = f["ee_T"][:].astype(np.float64)   # (N, 4, 4)
    T_ee_from_event = load_T_ee_from_event(CALIB_DIR)
    print(f"Poses loaded     : {len(ee_T_all)} frames")

    T_event_from_ee = np.linalg.inv(T_ee_from_event).astype(np.float32)
    T_ee_inv = np.linalg.inv(ee_T_all.astype(np.float32))
    T_cam_from_world = np.einsum("ij,njk->nik", T_event_from_ee, T_ee_inv).astype(np.float32)
    cam_centers_world = camera_centers_world(T_cam_from_world)

    # ── Frame counts ──────────────────────────────────────────────
    with h5py.File(depth_h5_path, "r") as f:
        n_total = f["depth"].shape[0]
    with h5py.File(voxels_h5_path, "r") as f:
        n_vox = f["voxels"].shape[0]
    with h5py.File(table_plane_h5_path, "r") as f:
        n_tbl = f["table_plane"].shape[0]
    if args.color_mesh:
        with h5py.File(rgb_h5_path, "r") as f:
            if "rgb" not in f:
                raise KeyError(f"{rgb_h5_path} does not contain dataset 'rgb'")
            n_rgb = f["rgb"].shape[0]
    else:
        n_rgb = n_total
    if use_original_depth_mesh:
        with h5py.File(realsense_h5_path, "r") as f:
            n_rs = f["depth"].shape[0]
        n_total = min(n_total, n_rs)

    n_total = min(n_total, n_vox, n_tbl, n_rgb, len(T_cam_from_world))   # don't go past what's precomputed
    frame_start = 0
    frame_end = n_total - 1
    region_count = frame_end - frame_start + 1

    eligible_frames = list(range(frame_start, frame_end + 1))
    allow_fewer_pose_views = bool(
        ckpt.get(
            "allow_fewer_pose_views",
            ckpt.get("allow_unbalanced_pose_views", False),
        )
    )
    if ckpt.get("pose_view_selection", False) and not allow_fewer_pose_views:
        num_views = int(ckpt.get("num_views", 1))
        threshold = float(ckpt.get("pose_move_threshold", 0.01))
        eligible_frames = [
            frame_idx
            for frame_idx in eligible_frames
            if select_pose_views(
                cam_centers_world,
                frame_idx,
                num_views,
                threshold,
                False,
            ) is not None
        ]
        if not eligible_frames:
            raise RuntimeError(
                "No target frame has the strictly balanced pose-view layout "
                "required by the active reconstruction policy."
            )

    # ── Frame selection ────────────────────────────────────────────
    if args.frame_step is not None:
        step = max(1, args.frame_step)
        mesh_frames = [frame for frame in eligible_frames if frame % step == 0]
        if not mesh_frames:
            mesh_frames = [eligible_frames[0]]
    else:
        n_mesh = min(len(eligible_frames), max(1, args.mesh_frame_count))
        positions = np.round(
            np.linspace(0, len(eligible_frames) - 1, n_mesh)
        ).astype(int)
        mesh_frames = sorted({eligible_frames[position] for position in positions})
        step = None

    if not args.save_frame_visualizations:
        viz_frames = []
    elif args.indices is not None:
        eligible_set = set(eligible_frames)
        viz_frames = sorted([
            int(i) for i in args.indices
            if int(i) in eligible_set
        ])
    else:
        if len(mesh_frames) <= args.n_frames:
            viz_frames = list(mesh_frames)
        else:
            picks      = np.round(np.linspace(0, len(mesh_frames) - 1, args.n_frames)).astype(int)
            viz_frames = sorted({mesh_frames[i] for i in picks})

    run_frames = sorted(set(mesh_frames) | set(viz_frames))

    print(f"Total frames     : {n_total}")
    print(f"Index range      : {frame_start}..{frame_end} ({region_count} frames)")
    if len(eligible_frames) != region_count:
        print(
            f"Eligible targets : {len(eligible_frames)}/{region_count} "
            "under strict balanced pose-view selection"
        )
    if step is None:
        print(
            f"Prediction frames: {len(mesh_frames)} evenly spaced "
            f"(target={args.mesh_frame_count})"
        )
    else:
        print(f"Frame step       : {step}  ({len(mesh_frames)} prediction frames)")
    print(f"Viz frames ({len(viz_frames):3d}) : {viz_frames}")
    if args.best_mae_frames is not None:
        print(
            f"MAE frame select : keep {args.best_mae_frames} "
            "globally lowest-MAE frames"
        )
    if args.uncertainty_weighted_tsdf:
        print(
            "TSDF confidence  : per-pixel learned weighting "
            f"({args.tsdf_confidence_levels} levels, "
            f"minimum={args.tsdf_min_confidence:.2f})"
        )

    # ── Inference ─────────────────────────────────────────────────
    depth_f  = h5py.File(depth_h5_path,       "r")
    voxels_f = h5py.File(voxels_h5_path,      "r")
    table_f  = h5py.File(table_plane_h5_path, "r")
    rgb_f = h5py.File(rgb_h5_path, "r") if args.color_mesh else None
    if rgb_f is not None and "rgb" not in rgb_f:
        raise KeyError(f"{rgb_h5_path} does not contain dataset 'rgb'")

    samples      = []
    mesh_data    = []
    gt_mesh_data = []
    final_mesh_frame_ids = []
    mesh_candidates = []
    confidence_by_frame = {}
    rgb_by_frame = {}
    viz_set      = set(viz_frames)
    mesh_set     = set(mesh_frames)
    t_infer_total = 0.0
    depth_error_total = {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0}
    depth_error_by_case = {
        "current_mask": {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0},
        "raised_object_cube": {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0},
    }

    try:
        for frame_idx in run_frames:
            t_frame0 = time.perf_counter()
            # Voxels
            vox_raw  = voxels_f["voxels"][frame_idx].astype(np.float32)
            vox_np   = _preprocess_voxels(vox_raw, resize_hw, crop_hw)

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
                return_uncertainty=use_learned_uncertainty,
            )
            pred_m    = depth_min + pred_norm * (depth_max - depth_min)
            if uncertainty_map is not None:
                confidence_by_frame[frame_idx] = np.clip(
                    1.0 - uncertainty_map, 0.0, 1.0
                ).astype(np.float32, copy=False)
            t_infer_total += time.perf_counter() - t_frame0

            # GT depth
            gt = depth_f["depth"][frame_idx].astype(np.float32)
            if gt.shape[0] != out_H or gt.shape[1] != out_W:
                gt = crop_resize(gt, resize_hw, crop_hw, mode="bilinear")
            gt_valid = np.isfinite(gt) & (gt > depth_min) & (gt < depth_max)
            pred_valid = np.isfinite(pred_m) & (pred_m > depth_min) & (pred_m < depth_max)
            gt_mask = gt_valid.astype(np.float32)
            pred_mask = pred_valid.astype(np.float32)
            frame_err = depth_error_stats(pred_m, gt, gt_mask)
            merge_depth_error_stats(depth_error_total, frame_err)

            if frame_idx < len(ee_T_all):
                T_base_from_event = ee_T_all[frame_idx] @ T_ee_from_event
                T_event_from_base = np.linalg.inv(T_base_from_event)
                for case_name, case_center in (
                    ("current_mask", cube_center),
                    ("raised_object_cube", raised_object_center),
                ):
                    spatial = depth_cube_mask(
                        gt, T_event_from_base, K, case_center, cube_half_side
                    )
                    case_err = depth_error_stats(pred_m, gt, gt_mask * spatial)
                    merge_depth_error_stats(depth_error_by_case[case_name], case_err)

            # Per-frame point cloud
            pts = depth_to_pointcloud(pred_m, pred_mask, K)

            # Accumulate for TSDF
            if frame_idx < len(ee_T_all) and frame_idx in mesh_set:
                T_base_from_event = ee_T_all[frame_idx] @ T_ee_from_event
                pred_masked = np.where(pred_valid, pred_m, np.float32(0.0)).astype(np.float32)
                gt_masked   = np.where(gt_valid,   gt,     np.float32(0.0)).astype(np.float32)
                if rgb_f is not None:
                    rgb = rgb_f["rgb"][frame_idx]
                    rgb_by_frame[frame_idx] = crop_resize_rgb(rgb, resize_hw, crop_hw)
                mesh_candidates.append({
                    "frame_idx": frame_idx,
                    "pred_masked": pred_masked,
                    "gt_masked": gt_masked,
                    "valid": pred_valid,
                    "mae_m": frame_err["abs_sum"] / frame_err["n"] if frame_err["n"] > 0 else float("inf"),
                    "T": T_base_from_event,
                })

            if frame_idx not in viz_set:
                continue

            frame_dir = out_dir / f"frame_{frame_idx:05d}"
            frame_dir.mkdir(exist_ok=True)

            save_depth_png(pred_m, pred_mask, frame_dir / "depth_pred.png", depth_min, depth_max)
            save_depth_png(gt,     gt_mask,   frame_dir / "depth_gt.png",   depth_min, depth_max)
            save_ply(pts, frame_dir / "pointcloud.ply")

            mae = frame_err["abs_sum"] / frame_err["n"] if frame_err["n"] > 0 else float("nan")
            print(f"  frame {frame_idx:5d} | {len(pts):6d} pts | MAE {mae * 100:.2f} cm | → {frame_dir.name}/")

            samples.append({
                "frame_idx": frame_idx,
                "gt":        gt,
                "pred":      pred_m,
                "gt_mask":   gt_mask,
                "pred_mask": pred_mask,
                "voxel":     vox_np,
            })
    finally:
        depth_f.close()
        voxels_f.close()
        table_f.close()
        if rgb_f is not None:
            rgb_f.close()

    # ── Select the lowest-MAE frames for TSDF fusion ─────────────
    mesh_data = [(c["pred_masked"], c["T"]) for c in mesh_candidates]
    gt_mesh_data = [(c["gt_masked"], c["T"]) for c in mesh_candidates]
    final_mesh_frame_ids = [c["frame_idx"] for c in mesh_candidates]

    if args.best_mae_frames is not None and mesh_candidates:
        keep_count = min(args.best_mae_frames, len(mesh_candidates))
        selected = _select_global_mae_frames(
            mesh_candidates,
            keep_count,
            use_filtered_depth=False,
        )
        mesh_data = [(cand["pred_depth"], cand["T"]) for cand in selected]
        gt_mesh_data = [(cand["gt_depth"], cand["T"]) for cand in selected]
        final_mesh_frame_ids = [cand["frame_idx"] for cand in selected]
        if selected:
            maes = [cand["mae_m"] for cand in selected]
            print(
                f"MAE frame select : kept {len(selected)}/{len(mesh_candidates)} candidates "
                f"(best={min(maes) * 100:.2f} cm, "
                f"worst kept={max(maes) * 100:.2f} cm)"
            )
            print(
                "  selected frames: "
                + ", ".join(str(cand["frame_idx"]) for cand in selected[:20])
                + (" ..." if len(selected) > 20 else "")
            )
        else:
            print("MAE frame select : no finite-MAE candidates available")

    print(f"Inference/prep   : {len(run_frames)} frames in {t_infer_total:.1f}s")

    depth_summary = summarize_depth_error(depth_error_total)
    if depth_summary is not None:
        print(
            f"Depth error      : MAE={depth_summary['mae_m'] * 100:.2f} cm, "
            f"RMSE={depth_summary['rmse_m'] * 100:.2f} cm, "
            f"max={depth_summary['max_m'] * 100:.2f} cm "
            f"over {depth_summary['pixels']:,} valid pixels"
        )
    else:
        print("Depth error      : no valid GT pixels")

    recon_summary = None
    recon_by_case = {}
    depth_by_case = {
        name: summarize_depth_error(stats)
        for name, stats in depth_error_by_case.items()
    }
    if mesh_data and gt_mesh_data:
        t_err0 = time.perf_counter()
        try:
            recon_summary = reconstruction_error_stats(
                mesh_data,
                gt_mesh_data,
                K,
                stride=args.error_stride,
                max_points=args.error_max_points,
            )
            for case_name, case_center in (
                ("current_mask", cube_center),
                ("raised_object_cube", raised_object_center),
            ):
                recon_by_case[case_name] = reconstruction_error_stats(
                    mesh_data, gt_mesh_data, K,
                    stride=args.error_stride,
                    max_points=args.error_max_points,
                    cube_center=case_center,
                    cube_half_side=cube_half_side,
                )
        except RuntimeError as exc:
            print(f"Recon error      : skipped ({exc})")
        if recon_summary is not None:
            print(
                f"Recon error      : Chamfer-L1={recon_summary['chamfer_l1_m'] * 100:.2f} cm "
                f"(pred→GT={recon_summary['accuracy_m'] * 100:.2f} cm, "
                f"GT→pred={recon_summary['completeness_m'] * 100:.2f} cm, "
                f"p95={recon_summary['accuracy_p95_m'] * 100:.2f}/"
                f"{recon_summary['completeness_p95_m'] * 100:.2f} cm, "
                f"points={recon_summary['pred_points']:,}/{recon_summary['gt_points']:,}, "
                f"time={time.perf_counter() - t_err0:.1f}s)"
            )
        else:
            print("Recon error      : no valid predicted/GT reconstruction points")
    else:
        print("Recon error      : no pose-aligned TSDF frames available")

    write_error_report(out_dir / "error_report.txt", depth_summary, recon_summary)
    for case_name in ("current_mask", "raised_object_cube"):
        write_error_report(
            out_dir / f"error_report_{case_name}.txt",
            depth_by_case.get(case_name),
            recon_by_case.get(case_name),
        )
    print(f"  Error report   : {out_dir / 'error_report.txt'}")

    # ── TSDF mesh ─────────────────────────────────────────────────
    pred_mesh_created = False
    gt_mesh_created = False
    original_depth_mesh_created = False
    if args.best_mae_frames is not None:
        frame_selection_name = "best_mae"
    else:
        frame_selection_name = "all_frames"

    sequence_name = data_dir.name
    pred_mesh_ext = ".ply" if args.color_mesh else ".obj"
    pred_mesh_path = (
        out_dir / f"{sequence_name}_{frame_selection_name}_mesh{pred_mesh_ext}"
    )
    gt_mesh_path = (
        out_dir / f"{sequence_name}_{frame_selection_name}_gt_mesh.obj"
    )
    full_pred_mesh_path = out_dir / (
        f"{sequence_name}_{frame_selection_name}_full_mesh{pred_mesh_ext}"
    )
    full_gt_mesh_path = out_dir / (
        f"{sequence_name}_{frame_selection_name}_full_gt_mesh.obj"
    )
    raised_object_pred_mesh_path = out_dir / (
        f"{sequence_name}_{frame_selection_name}_raised_object_mesh{pred_mesh_ext}"
    )
    raised_object_gt_mesh_path = out_dir / (
        f"{sequence_name}_{frame_selection_name}_raised_object_gt_mesh.obj"
    )
    original_depth_mesh_path = (
        out_dir / f"{sequence_name}_{frame_selection_name}_original_depth_mesh.obj"
    )
    evaluation_pred_mesh_path = pred_mesh_path
    evaluation_gt_mesh_path = gt_mesh_path
    evaluation_full_pred_mesh_path = full_pred_mesh_path
    evaluation_full_gt_mesh_path = full_gt_mesh_path
    evaluation_raised_pred_mesh_path = raised_object_pred_mesh_path
    evaluation_raised_gt_mesh_path = raised_object_gt_mesh_path
    original_depth_mesh_data = []
    if use_original_depth_mesh:
        if not final_mesh_frame_ids:
            print("  Original depth : no final TSDF frames available")
        else:
            with h5py.File(realsense_h5_path, "r") as rs_f:
                rs_depth_ds = rs_f["depth"]
                for frame_idx in final_mesh_frame_ids:
                    if frame_idx >= rs_depth_ds.shape[0]:
                        continue
                    depth_raw = rs_depth_ds[frame_idx].astype(np.float32)
                    depth_native = depth_raw * np.float32(original_depth_scale)
                    depth_valid = (
                        np.isfinite(depth_native)
                        & (depth_native > depth_min)
                        & (depth_native < TSDF_DEPTH_MAX)
                    )
                    depth_native = np.where(
                        depth_valid,
                        depth_native,
                        np.float32(0.0),
                    ).astype(np.float32)
                    T_base_from_depth = ee_T_all[frame_idx] @ T_ee_from_depth
                    original_depth_mesh_data.append((depth_native, T_base_from_depth))
            print(
                f"  Original depth : prepared {len(original_depth_mesh_data)}/"
                f"{len(final_mesh_frame_ids)} native RealSense frames"
            )
    if mesh_data:
        t_tsdf0 = time.perf_counter()
        pred_tsdf_data = mesh_data
        confidence_levels = 0
        if args.uncertainty_weighted_tsdf:
            if len(final_mesh_frame_ids) != len(mesh_data):
                raise RuntimeError(
                    "TSDF frame IDs are not aligned with predicted depth frames"
                )
            pred_tsdf_data = [
                (depth, T, confidence_by_frame.get(frame_idx))
                for frame_idx, (depth, T) in zip(final_mesh_frame_ids, mesh_data)
            ]
            missing_confidence = sum(frame[2] is None for frame in pred_tsdf_data)
            if missing_confidence:
                raise RuntimeError(
                    f"Missing confidence maps for {missing_confidence} selected TSDF frames"
                )
            confidence_levels = args.tsdf_confidence_levels
        if args.color_mesh:
            if len(final_mesh_frame_ids) != len(mesh_data):
                raise RuntimeError(
                    "TSDF frame IDs are not aligned with predicted depth frames"
                )
            colored_tsdf_data = []
            for frame_idx, frame in zip(final_mesh_frame_ids, pred_tsdf_data):
                rgb = rgb_by_frame.get(frame_idx)
                if rgb is None:
                    raise RuntimeError(f"Missing RGB for selected TSDF frame {frame_idx}")
                if len(frame) > 2:
                    colored_tsdf_data.append((frame[0], frame[1], frame[2], rgb))
                else:
                    colored_tsdf_data.append((frame[0], frame[1], None, rgb))
            pred_tsdf_data = colored_tsdf_data
        pred_mesh_created = tsdf_fuse(
            pred_tsdf_data,
            K,
            pred_mesh_path,
            voxel_length=args.voxel_size,
            sdf_trunc_factor=args.sdf_trunc_factor,
            depth_max=depth_max,
            cube_center=cube_center,
            cube_half_side=cube_half_side,
            confidence_levels=confidence_levels,
            min_confidence=args.tsdf_min_confidence,
            use_color=args.color_mesh,
            additional_crops=[(raised_object_pred_mesh_path, raised_object_center)],
            full_mesh_path=(full_pred_mesh_path if args.evaluate_full_scale else None),
        )
        print(f"  Pred TSDF time : {time.perf_counter() - t_tsdf0:.1f}s")
    if gt_mesh_data and not args.skip_gt_mesh:
        t_tsdf0 = time.perf_counter()
        gt_mesh_created = tsdf_fuse(
            gt_mesh_data,
            K,
            gt_mesh_path,
            voxel_length=args.voxel_size,
            sdf_trunc_factor=args.sdf_trunc_factor,
            depth_max=depth_max,
            cube_center=cube_center,
            cube_half_side=cube_half_side,
            additional_crops=[(raised_object_gt_mesh_path, raised_object_center)],
            full_mesh_path=(full_gt_mesh_path if args.evaluate_full_scale else None),
        )
        print(f"  GT TSDF time   : {time.perf_counter() - t_tsdf0:.1f}s")
    elif gt_mesh_data:
        print("  GT TSDF        : skipped (--skip_gt_mesh)")
    if args.save_largest_connected_surface:
        if pred_mesh_created:
            evaluation_pred_mesh_path = save_largest_connected_surface(
                pred_mesh_path
            )
            evaluation_raised_pred_mesh_path = save_largest_connected_surface(
                raised_object_pred_mesh_path
            )
            if args.evaluate_full_scale:
                evaluation_full_pred_mesh_path = save_largest_connected_surface(
                    full_pred_mesh_path
                )
        if gt_mesh_created:
            evaluation_gt_mesh_path = save_largest_connected_surface(gt_mesh_path)
            evaluation_raised_gt_mesh_path = save_largest_connected_surface(
                raised_object_gt_mesh_path
            )
            if args.evaluate_full_scale:
                evaluation_full_gt_mesh_path = save_largest_connected_surface(
                    full_gt_mesh_path
                )
    if original_depth_mesh_data:
        t_tsdf0 = time.perf_counter()
        original_depth_mesh_created = tsdf_fuse(
            original_depth_mesh_data,
            K_depth,
            original_depth_mesh_path,
            voxel_length=args.voxel_size,
            sdf_trunc_factor=args.sdf_trunc_factor,
            depth_max=TSDF_DEPTH_MAX,
            cube_center=cube_center,
            cube_half_side=cube_half_side,
        )
        print(f"  Original depth TSDF time: {time.perf_counter() - t_tsdf0:.1f}s")
        if not original_depth_mesh_created:
            print("  Original depth TSDF     : empty mesh")

    # ── Post-TSDF reconstruction evaluation ──────────────────────
    full_surface_metrics = None
    surface_metrics = None
    raised_object_surface_metrics = None
    original_depth_surface_metrics = None
    rendered_metrics = None
    raised_object_rendered_metrics = None
    if pred_mesh_created and gt_mesh_created:
        t_metric0 = time.perf_counter()
        try:
            if args.evaluate_full_scale:
                full_surface_metrics = evaluate_tsdf_meshes(
                    evaluation_full_pred_mesh_path,
                    evaluation_full_gt_mesh_path,
                    surface_samples=args.surface_samples,
                )
            surface_metrics = evaluate_tsdf_meshes(
                evaluation_pred_mesh_path,
                evaluation_gt_mesh_path,
                surface_samples=args.surface_samples,
            )
            raised_object_surface_metrics = evaluate_tsdf_meshes(
                evaluation_raised_pred_mesh_path,
                evaluation_raised_gt_mesh_path,
                surface_samples=args.surface_samples,
            )
        except RuntimeError as exc:
            print(f"  Surface metrics: skipped ({exc})")
        if surface_metrics is not None:
            print(
                f"  Surface metrics: F@1/2/5cm="
                f"{surface_metrics['fscore_1cm']:.1%}/"
                f"{surface_metrics['fscore_2cm']:.1%}/"
                f"{surface_metrics['fscore_5cm']:.1%}, "
                f"Chamfer={surface_metrics['chamfer_mean_m'] * 100:.2f} cm, "
                f"normal={surface_metrics['normal_consistency_symmetric']:.3f}"
            )
            print(f"  Surface metric time: {time.perf_counter() - t_metric0:.1f}s")
        if full_surface_metrics is not None:
            print(
                f"  Surface full   : F@1cm={full_surface_metrics['fscore_1cm']:.1%}, "
                f"Chamfer={full_surface_metrics['chamfer_mean_m'] * 100:.2f} cm, "
                f"normal={full_surface_metrics['normal_consistency_symmetric']:.3f}"
            )
        if raised_object_surface_metrics is not None:
            print(
                f"  Surface raised object: F@1cm={raised_object_surface_metrics['fscore_1cm']:.1%}, "
                f"Chamfer={raised_object_surface_metrics['chamfer_mean_m'] * 100:.2f} cm, "
                f"normal={raised_object_surface_metrics['normal_consistency_symmetric']:.3f}"
            )
    elif args.skip_gt_mesh:
        print("  Surface metrics: unavailable because --skip_gt_mesh was used")

    if (
        args.evaluate_original_depth_mesh
        and pred_mesh_created
        and original_depth_mesh_created
    ):
        try:
            original_depth_surface_metrics = evaluate_tsdf_meshes(
                evaluation_pred_mesh_path,
                original_depth_mesh_path,
                surface_samples=args.surface_samples,
            )
        except RuntimeError as exc:
            print(f"  Native-depth surface metrics: skipped ({exc})")
        if original_depth_surface_metrics is not None:
            print(
                "  Surface vs native depth: "
                f"F@1/2/5cm="
                f"{original_depth_surface_metrics['fscore_1cm']:.1%}/"
                f"{original_depth_surface_metrics['fscore_2cm']:.1%}/"
                f"{original_depth_surface_metrics['fscore_5cm']:.1%}, "
                f"Chamfer="
                f"{original_depth_surface_metrics['chamfer_mean_m'] * 100:.2f} cm, "
                f"normal="
                f"{original_depth_surface_metrics['normal_consistency_symmetric']:.3f}"
            )
    elif args.evaluate_original_depth_mesh:
        print(
            "  Native-depth surface metrics: unavailable because the predicted "
            "or native-depth mesh was not created"
        )

    full_rendered_metrics = None
    if pred_mesh_created and args.render_eval_frames > 0:
        fusion_ids = set(final_mesh_frame_ids)
        held_out_candidates = [
            idx for idx in range(min(n_total, len(ee_T_all)))
            if idx not in fusion_ids
        ]
        if held_out_candidates:
            count = min(args.render_eval_frames, len(held_out_candidates))
            positions = np.round(
                np.linspace(0, len(held_out_candidates) - 1, count)
            ).astype(int)
            held_out_frames = [held_out_candidates[pos] for pos in sorted(set(positions))]
            t_render0 = time.perf_counter()
            if args.evaluate_full_scale:
                full_rendered_metrics = evaluate_rendered_depth(
                    evaluation_full_pred_mesh_path,
                    held_out_frames,
                    depth_h5_path,
                    ee_T_all,
                    T_ee_from_event,
                    K,
                    resize_hw,
                    crop_hw,
                    depth_min,
                    depth_max,
                )
            rendered_metrics = evaluate_rendered_depth(
                evaluation_pred_mesh_path,
                held_out_frames,
                depth_h5_path,
                ee_T_all,
                T_ee_from_event,
                K,
                resize_hw,
                crop_hw,
                depth_min,
                depth_max,
            )
            raised_object_rendered_metrics = evaluate_rendered_depth(
                evaluation_raised_pred_mesh_path,
                held_out_frames,
                depth_h5_path,
                ee_T_all,
                T_ee_from_event,
                K,
                resize_hw,
                crop_hw,
                depth_min,
                depth_max,
            )
            if rendered_metrics is not None:
                print(
                    f"  Rendered depth : MAE={rendered_metrics['mae_m'] * 100:.2f} cm, "
                    f"RMSE={rendered_metrics['rmse_m'] * 100:.2f} cm, "
                    f"AbsRel={rendered_metrics['abs_rel']:.4f}, "
                    f"valid={rendered_metrics['valid_render_percentage']:.1%}, "
                    f"frames={rendered_metrics['held_out_frames']}, "
                    f"time={time.perf_counter() - t_render0:.1f}s"
                )
        else:
            print("  Rendered depth : no camera frames held out from TSDF fusion")

    write_tsdf_metric_outputs(out_dir, surface_metrics, rendered_metrics)
    reconstruction_metrics = {
        "surface_selection": (
            "largest_connected_surface"
            if args.save_largest_connected_surface
            else "all_surfaces"
        ),
        "mask_definition": {
            "current_mask_center_world_m": cube_center.tolist(),
            "raised_object_bottom_z_m": RAISED_OBJECT_BOTTOM_Z_M,
            "raised_object_center_world_m": raised_object_center.tolist(),
            "cube_side_m": args.cube_side,
        },
        "cases": {
            "full_scale": {
                "depth_metrics": None,
                "prefusion_pointcloud_metrics": None,
                "surface_metrics": full_surface_metrics,
                "rendered_depth_metrics": full_rendered_metrics,
            },
            "current_mask": {
                "depth_metrics": depth_by_case.get("current_mask"),
                "prefusion_pointcloud_metrics": recon_by_case.get("current_mask"),
                "surface_metrics": surface_metrics,
                "original_depth_surface_metrics": original_depth_surface_metrics,
                "rendered_depth_metrics": rendered_metrics,
            },
            "raised_object_cube": {
                "depth_metrics": depth_by_case.get("raised_object_cube"),
                "prefusion_pointcloud_metrics": recon_by_case.get("raised_object_cube"),
                "surface_metrics": raised_object_surface_metrics,
                "rendered_depth_metrics": raised_object_rendered_metrics,
            },
        },
    }
    (out_dir / "reconstruction_metrics.json").write_text(
        json.dumps(reconstruction_metrics, indent=2) + "\n", encoding="utf-8"
    )
    case_lines = [
        "Reconstruction metrics for full-scale and spatially cropped meshes",
        (
            "Surface selection: largest connected surface"
            if args.save_largest_connected_surface
            else "Surface selection: all extracted surfaces"
        ),
        f"Current cube center: {cube_center.tolist()} m",
        f"Raised object cube bottom: z={RAISED_OBJECT_BOTTOM_Z_M:.3f} m; center={raised_object_center.tolist()} m",
        "",
    ]
    for case_name, label in (
        ("full_scale", "Full scale (uncropped)"),
        ("current_mask", "Current mask"),
        ("raised_object_cube", "Raised object cube (+1.5 cm)"),
    ):
        case = reconstruction_metrics["cases"][case_name]
        case_lines.append(label)
        depth_case = case["depth_metrics"]
        point_case = case["prefusion_pointcloud_metrics"]
        surface_case = case["surface_metrics"]
        original_surface_case = case.get("original_depth_surface_metrics")
        render_case = case["rendered_depth_metrics"]
        if depth_case:
            case_lines.append(f"  Depth MAE/RMSE: {depth_case['mae_m']:.6f}/{depth_case['rmse_m']:.6f} m")
        if point_case:
            case_lines.append(f"  Pre-fusion Chamfer-L1: {point_case['chamfer_l1_m']:.6f} m")
        if surface_case:
            case_lines.append(f"  Post-TSDF Chamfer mean/median: {surface_case['chamfer_mean_m']:.6f}/{surface_case['chamfer_median_m']:.6f} m")
            case_lines.append(f"  Normal consistency: {surface_case['normal_consistency_symmetric']:.6f}")
            case_lines.append(f"  F-score @1/2/5 cm: {surface_case['fscore_1cm']:.2%}/{surface_case['fscore_2cm']:.2%}/{surface_case['fscore_5cm']:.2%}")
        if original_surface_case:
            case_lines.append(
                "  Versus native-depth mesh Chamfer mean/median: "
                f"{original_surface_case['chamfer_mean_m']:.6f}/"
                f"{original_surface_case['chamfer_median_m']:.6f} m"
            )
            case_lines.append(
                "  Versus native-depth mesh normal consistency: "
                f"{original_surface_case['normal_consistency_symmetric']:.6f}"
            )
            case_lines.append(
                "  Versus native-depth mesh F-score @1/2/5 cm: "
                f"{original_surface_case['fscore_1cm']:.2%}/"
                f"{original_surface_case['fscore_2cm']:.2%}/"
                f"{original_surface_case['fscore_5cm']:.2%}"
            )
        if render_case:
            case_lines.append(f"  Rendered MAE/RMSE: {render_case['mae_m']:.6f}/{render_case['rmse_m']:.6f} m")
            case_lines.append(f"  Render coverage: {render_case['valid_render_percentage']:.2%}")
        case_lines.append("")
    (out_dir / "reconstruction_metrics.txt").write_text("\n".join(case_lines), encoding="utf-8")
    print(f"  TSDF metrics   : {out_dir / 'tsdf_metrics.txt'}")

    # ── Overview PNG ──────────────────────────────────────────────
    if samples:
        save_overview(samples, out_dir / "overview.png", depth_min, depth_max)

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
