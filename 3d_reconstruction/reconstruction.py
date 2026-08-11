#!/usr/bin/env python3
"""
Depth-map, point-cloud, and TSDF reconstruction from trained depth models.

Supports single- and multi-view early-fusion U-Net checkpoints, recurrent
U-Net checkpoints, and geometric multiview checkpoints produced by the two
current training entry points: ``training/train_unet.py`` and
``training/multiview.py``.
Uses the precomputed table-plane channel stored in hdf5/table_plane.h5
(produced by data_precomputation/precompute_table_plane.py) instead of
computing it on-the-fly.

Outputs are written by default to
training/results/<checkpoint_name>/ (or to --out_dir):
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
        --checkpoint training/checkpoints/unet_table/best_myrun.pth \\
        --data_dir   data/new/eval/1

    python3 reconstruction.py \\
        --checkpoint training/checkpoints/unet_table/best_myrun.pth \\
        --data_dir   data/new/eval/1 \\
        --indices 0 50 100 200 400 \\
        --out_dir results/lego_1_table

    python3 reconstruction.py \\
        --checkpoint training/checkpoints/unet_table/best_myrun.pth \\
        --data_dir   data/new/eval
"""

import sys
import argparse
import json
import math
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
    TRAIN_RESIZE_HW, TRAIN_CROP_HW,
    CROP_THEN_RESIZE_CROP_HW, CROP_THEN_RESIZE_HW,
    TSDF_VOXEL_SIZE, TSDF_SDF_TRUNC_FACTOR, TSDF_DEPTH_MAX,
    SPATIAL_CUBE_SIDE, SPATIAL_TARGET_X, SPATIAL_TARGET_Y, SPATIAL_TARGET_Z,
)
from preprocessing_geometry import transform_intrinsics, transform_name
from spatial_mask import points_in_cube
from train_unet import RecurrentUNet, UNet
from multiview import (
    ModernMVSNet,
    _inverse_depth_candidates,
    _linear_depth_candidates,
)

CALIB_DIR = _HERE / _CALIB_DIR
CROP_THEN_RESIZE_MODE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_K(
    calib_dir: Path,
    resize_hw: Optional[Tuple[int, int]] = None,
    crop_hw: Optional[Tuple[int, int]] = None,
    crop_then_resize: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """Return camera intrinsics scaled to the model's input resolution."""
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K = ev["camera_matrix"].copy().astype(np.float64)
    native_W = int(ev["image_size"][0])
    native_H = int(ev["image_size"][1])
    if resize_hw is None or crop_hw is None:
        return K, native_H, native_W
    K = transform_intrinsics(
        K, (native_H, native_W), resize_hw, crop_hw, crop_then_resize
    )
    final_hw = resize_hw if crop_then_resize else crop_hw
    return K, final_hw[0], final_hw[1]


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

def _detect_table_ckpt_type(ckpt: dict) -> str:
    """Infer which table-prior training script produced a checkpoint."""
    if (
        "num_views" in ckpt
        and "coarse_depths" in ckpt
        and ("feature_channels" in ckpt or any(k.startswith("feature.") for k in ckpt.get("model", {}).keys()))
    ):
        return "multiview"
    if "embed_dim" in ckpt or "window_size" in ckpt or "seq_len" in ckpt:
        return "ereformer"
    if "model" in ckpt and any(k.startswith("patch_embed.") for k in ckpt["model"].keys()):
        return "ereformer"
    if (
        "table_z" in ckpt
        or ("model" in ckpt and "config" not in ckpt and ckpt.get("in_ch", NUM_BINS) > NUM_BINS)
    ):
        return "unet_table"
    return "unknown"


def _infer_multiview_model_arch(ckpt: dict) -> str:
    """Return the sole supported multiview architecture."""
    return str(ckpt.get("model_arch", "ModernMVSNet"))


def load_model(ckpt_path: Path, device: torch.device):
    """Load a table-prior checkpoint and return (model, ckpt_dict, ckpt_type)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_type = _detect_table_ckpt_type(ckpt)

    if ckpt_type == "unet_table":
        in_ch = ckpt.get("in_ch", NUM_BINS + 1)
        base  = ckpt.get("base",  32)
        if bool(ckpt.get("recurrent", False)) or ckpt.get("model_arch") == "RecurrentUNet":
            model = RecurrentUNet(in_ch=in_ch, base=base).to(device)
        else:
            model = UNet(in_ch=in_ch, base=base).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt, ckpt_type

    if ckpt_type == "ereformer":
        in_ch       = ckpt.get("in_ch", NUM_BINS + 1)
        embed_dim   = ckpt.get("embed_dim", 96)
        window_size = ckpt.get("window_size", 8)
        model = EReFormer(
            in_ch=in_ch,
            embed_dim=embed_dim,
            window_size=window_size,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt, ckpt_type

    if ckpt_type == "multiview":
        model_arch = _infer_multiview_model_arch(ckpt)
        if model_arch != "ModernMVSNet":
            raise ValueError(f"Unsupported multiview checkpoint architecture: {model_arch}")
        model = ModernMVSNet(
            in_ch=ckpt.get("in_ch", NUM_BINS + 1),
            base=ckpt.get("base", ckpt.get("base_channels_arg", 32)),
            feature_ch=ckpt.get("feature_channels", None),
            cost_base=ckpt.get("cost_channels", None),
            fine_depths=ckpt.get("fine_depths", 5),
            fine_window=ckpt.get("fine_window", 0.08),
            fine_offset_radius=ckpt.get("fine_offset_radius", 2.0),
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
        return model, ckpt, ckpt_type

    if ckpt_type == "unknown":
        raise ValueError(
            f"Checkpoint does not appear to be a supported table-prior checkpoint: {ckpt_path}\n"
            "Expected a table-prior U-Net checkpoint ('table_z' or 'in_ch' > NUM_BINS) "
            "an EReFormer checkpoint ('embed_dim'/'window_size' metadata or patch_embed weights), "
            "or a multiview.py checkpoint "
            "('num_views'/'coarse_depths' metadata)."
        )
    raise AssertionError(f"Unhandled checkpoint type: {ckpt_type}")


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def resize_crop(
    arr: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
    mode: str = "bilinear",
) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    final_hw = resize_hw if CROP_THEN_RESIZE_MODE else crop_hw
    if final_hw is not None and tuple(t.shape[-2:]) == tuple(final_hw):
        return arr
    kw = {} if mode == "nearest" else {"align_corners": False}
    if CROP_THEN_RESIZE_MODE:
        if crop_hw is not None:
            ch, cw = crop_hw
            y0 = (t.shape[2] - ch) // 2
            x0 = (t.shape[3] - cw) // 2
            t = t[:, :, y0:y0 + ch, x0:x0 + cw]
        if resize_hw is not None:
            t = F.interpolate(t, size=resize_hw, mode=mode, **kw)
    else:
        if resize_hw is not None:
            t = F.interpolate(t, size=resize_hw, mode=mode, **kw)
        if crop_hw is not None:
            ch, cw = crop_hw
            y0 = (t.shape[2] - ch) // 2
            x0 = (t.shape[3] - cw) // 2
            t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    return t[0, 0].numpy()


def resize_crop_rgb(
    rgb: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
) -> np.ndarray:
    """Resize/crop an RGB image to the target reconstruction resolution."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
    final_h = (resize_hw[0] if CROP_THEN_RESIZE_MODE and resize_hw is not None else
               crop_hw[0]   if crop_hw   is not None else
               resize_hw[0] if resize_hw is not None else rgb.shape[0])
    final_w = (resize_hw[1] if CROP_THEN_RESIZE_MODE and resize_hw is not None else
               crop_hw[1]   if crop_hw   is not None else
               resize_hw[1] if resize_hw is not None else rgb.shape[1])
    if rgb.shape[0] == final_h and rgb.shape[1] == final_w:
        return rgb.astype(np.uint8, copy=False)
    t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    if CROP_THEN_RESIZE_MODE and crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    if resize_hw is not None:
        t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
    if not CROP_THEN_RESIZE_MODE and crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    out = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).numpy()
    return out.astype(np.uint8)


def _preprocess_voxels(
    vox_raw: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
) -> np.ndarray:
    """Resize/crop a (C, H, W) voxel grid to the target resolution."""
    final_h = (resize_hw[0] if CROP_THEN_RESIZE_MODE and resize_hw is not None else
               crop_hw[0]   if crop_hw   is not None else
               resize_hw[0] if resize_hw is not None else vox_raw.shape[1])
    final_w = (resize_hw[1] if CROP_THEN_RESIZE_MODE and resize_hw is not None else
               crop_hw[1]   if crop_hw   is not None else
               resize_hw[1] if resize_hw is not None else vox_raw.shape[2])
    if vox_raw.shape[1] == final_h and vox_raw.shape[2] == final_w:
        return vox_raw
    t = torch.from_numpy(vox_raw).unsqueeze(0)  # (1, C, H, W)
    if CROP_THEN_RESIZE_MODE and crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    if resize_hw is not None:
        t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
    if not CROP_THEN_RESIZE_MODE and crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    return t.squeeze(0).numpy()


def _pose_channels_from_base_event(T_base_from_event: np.ndarray) -> np.ndarray:
    """Return six pose values: event-camera position and optical axis in base frame."""
    position = T_base_from_event[:3, 3].astype(np.float32)
    optical_axis = T_base_from_event[:3, 2].astype(np.float32)
    norm = float(np.linalg.norm(optical_axis))
    if norm > 1e-6:
        optical_axis = optical_axis / norm
    return np.concatenate([position, optical_axis]).astype(np.float32)


def _unet_input_np(
    vox_np: np.ndarray,
    tbl_np: np.ndarray,
    pose_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build one early-fusion U-Net input frame."""
    vox_H, vox_W = vox_np.shape[1], vox_np.shape[2]
    tbl_t = torch.from_numpy(tbl_np.astype(np.float32)).unsqueeze(0)
    if tbl_t.shape[-2] != vox_H or tbl_t.shape[-1] != vox_W:
        tbl_t = F.interpolate(
            tbl_t.unsqueeze(0), (vox_H, vox_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    input_channels = [vox_np, tbl_t.numpy()]
    if pose_values is not None:
        pose_np = np.broadcast_to(
            pose_values.astype(np.float32)[:, None, None],
            (6, vox_H, vox_W),
        )
        input_channels.append(pose_np)
    return np.concatenate(input_channels, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer(
    model:    torch.nn.Module,
    vox_np:   np.ndarray,   # (C, H, W)  preprocessed voxels
    tbl_np:   np.ndarray,   # (H_tbl, W_tbl) precomputed table-plane channel [0, 1]
    device:   torch.device,
    states:   Optional[list] = None,
    pose_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[list]]:
    """Run a table-prior forward pass. Returns ((H, W) normalised depth, states)."""
    inp_np = _unet_input_np(vox_np, tbl_np, pose_values=pose_values)
    inp    = torch.from_numpy(inp_np).unsqueeze(0).to(device)   # (1, C+1, H, W)
    if isinstance(model, EReFormer):
        out = model(inp, states)
    else:
        out = model(inp)

    if isinstance(out, tuple):
        pred, states = out
    else:
        pred, states = out, None
    return pred[0, 0].cpu().numpy(), states


@torch.no_grad()
def _infer_early_fusion_unet(
    model: torch.nn.Module,
    frame_idx: int,
    vox_ds,
    tbl_ds,
    T_cam_from_world: Optional[np.ndarray],
    ckpt: dict,
    device: torch.device,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw: Optional[Tuple[int, int]],
    centers_world: Optional[np.ndarray],
) -> np.ndarray:
    """Run a checkpoint-configured early-fusion U-Net forward pass.

    The target/source ordering, temporal offsets, pose-based view selection,
    table-prior channels, and optional pose channels match TablePriorDataset.
    """
    num_views = int(ckpt.get("num_views", 1))
    n_frames = min(len(vox_ds), len(tbl_ds))
    if T_cam_from_world is not None:
        n_frames = min(n_frames, len(T_cam_from_world))
    view_ids = _multiview_view_ids(
        frame_idx,
        n_frames,
        ckpt,
        centers_world,
    )

    use_pose_channels = bool(ckpt.get("pose_channels", False))
    if use_pose_channels and T_cam_from_world is None:
        raise RuntimeError(
            "This U-Net checkpoint uses pose channels, but camera poses are unavailable."
        )

    view_inputs = []
    for view_idx in view_ids:
        vox_np = _preprocess_voxels(
            vox_ds[view_idx].astype(np.float32), resize_hw, crop_hw
        )
        tbl_np = tbl_ds[view_idx].astype(np.float32)
        pose_values = None
        if use_pose_channels:
            T_base_from_event = np.linalg.inv(T_cam_from_world[view_idx]).astype(
                np.float32
            )
            pose_values = _pose_channels_from_base_event(T_base_from_event)
        view_inputs.append(
            _unet_input_np(vox_np, tbl_np, pose_values=pose_values)
        )

    inp_np = np.concatenate(view_inputs, axis=0)
    expected_channels = int(ckpt.get("in_ch", inp_np.shape[0]))
    if inp_np.shape[0] != expected_channels:
        raise RuntimeError(
            "Early-fusion U-Net input-channel mismatch: constructed "
            f"{inp_np.shape[0]} channels from {num_views} view(s), but the "
            f"checkpoint expects {expected_channels}."
        )
    inp = torch.from_numpy(inp_np).unsqueeze(0).to(device)
    out = model(inp)
    pred = out[0] if isinstance(out, tuple) else out
    return pred[0, 0].cpu().numpy()


def _infer_recurrent_unet(
    model: torch.nn.Module,
    frame_idx: int,
    vox_ds,
    tbl_ds,
    T_cam_from_world: np.ndarray,
    ckpt: dict,
    device: torch.device,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    enrollment = int(ckpt.get("recurrent_enrollment_range", 0))
    frame_ids = [max(0, frame_idx - offset) for offset in range(enrollment, -1, -1)]
    inputs = []
    for idx in frame_ids:
        vox_np = _preprocess_voxels(vox_ds[idx].astype(np.float32), resize_hw, crop_hw)
        tbl_np = tbl_ds[idx].astype(np.float32)
        pose_values = None
        if bool(ckpt.get("pose_channels", False)):
            pose_values = _pose_channels_from_base_event(
                np.linalg.inv(T_cam_from_world[idx]).astype(np.float32)
            )
        inputs.append(_unet_input_np(vox_np, tbl_np, pose_values=pose_values))
    inp = torch.from_numpy(np.stack(inputs, axis=0)).unsqueeze(0).to(device)
    out = model(inp)
    pred = out[0] if isinstance(out, tuple) else out
    return pred[0, 0].cpu().numpy()


def _multiview_source_offsets(num_views: int, view_interval: int) -> List[int]:
    offsets: List[int] = []
    k = 1
    while len(offsets) < num_views - 1:
        offsets.append(-k * view_interval)
        if len(offsets) < num_views - 1:
            offsets.append(k * view_interval)
        k += 1
    return offsets


def _camera_centers_world(T_cam_from_world: np.ndarray) -> np.ndarray:
    R = T_cam_from_world[:, :3, :3]
    t = T_cam_from_world[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t).astype(np.float32)


def _pose_neighbours(
    centers: np.ndarray,
    idx: int,
    direction: int,
    per_direction: int,
    move_threshold: float,
) -> List[int]:
    neighbours: List[int] = []
    anchor = idx
    cursor = idx + direction
    n_frames = len(centers)
    while 0 <= cursor < n_frames and len(neighbours) < per_direction:
        moved = np.linalg.norm(centers[cursor] - centers[anchor])
        if moved >= move_threshold:
            neighbours.append(cursor)
            anchor = cursor
        cursor += direction
    return neighbours


def _multiview_view_ids(
    frame_idx: int,
    n_frames: int,
    ckpt: dict,
    centers_world: Optional[np.ndarray],
) -> List[int]:
    """Select target/source frames using the same conventions as multiview.py.

    Edge frames can lack the exact training-time source layout. In that case we
    keep reconstruction running by filling the missing source slots with nearest
    temporal neighbours, then the target frame if the sequence is too short.
    """
    num_views = int(ckpt.get("num_views", 5))
    if ckpt.get("pose_view_selection", False) and centers_world is not None:
        per_direction = (num_views - 1) // 2
        threshold = float(ckpt.get("pose_move_threshold", 0.01))
        before = _pose_neighbours(centers_world, frame_idx, -1, per_direction, threshold)
        after = _pose_neighbours(centers_world, frame_idx, 1, per_direction, threshold)
        view_ids = [frame_idx] + before + after
    else:
        interval = int(ckpt.get("view_interval", 5))
        view_ids = [frame_idx] + [
            min(max(frame_idx + o, 0), n_frames - 1)
            for o in _multiview_source_offsets(num_views, interval)
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
    imgs = torch.stack([
        _multiview_input(vox_ds, tbl_ds, i, resize_hw, crop_hw)
        for i in view_ids
    ], dim=0).unsqueeze(0).to(device)
    cam_mats = torch.from_numpy(np.stack([T_cam_from_world[i] for i in view_ids])).unsqueeze(0).to(device)
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
            pred_pts_all.append((T_world_from_cam @ pred_h.T).T[:, :3].astype(np.float32))
        if len(gt_cam):
            gt_h = np.concatenate([gt_cam, np.ones((len(gt_cam), 1), dtype=np.float32)], axis=1)
            gt_pts_all.append((T_world_from_cam @ gt_h.T).T[:, :3].astype(np.float32))

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

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    if cube_center is not None and cube_half_side > 0.0:
        verts  = np.asarray(mesh.vertices)
        inside = np.flatnonzero(points_in_cube(verts, cube_center, cube_half_side))
        mesh   = mesh.select_by_index(inside)
        mesh.compute_vertex_normals()
        print(f"  Cube crop (side={cube_half_side*2*100:.0f} cm) → "
              f"{len(np.asarray(mesh.vertices)):,} verts, "
              f"{len(np.asarray(mesh.triangles)):,} triangles")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # For colored PLYs, prefer a simple xyz+rgb vertex layout.  Some viewers
    # pick the normal/scalar array for false-color rendering when Open3D writes
    # normals before red/green/blue, which makes the mesh look psychedelic even
    # though the stored RGB values are sane.  Viewers can recompute normals.
    o3d.io.write_triangle_mesh(
        str(out_path),
        mesh,
        write_vertex_normals=not use_color,
    )
    n_v = len(np.asarray(mesh.vertices))
    n_t = len(np.asarray(mesh.triangles))
    print(f"  Mesh → {out_path}  ({n_v:,} verts, {n_t:,} triangles)")
    return n_v > 0 and n_t > 0


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
    pred_cloud = pred_mesh.sample_points_uniformly(
        number_of_points=sample_count,
        use_triangle_normal=True,
    )
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
                gt = resize_crop(gt, resize_hw, crop_hw, mode="bilinear")
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
) -> None:
    payload = {
        "surface_metrics": surface_metrics,
        "rendered_depth_metrics": rendered_metrics,
    }
    (out_dir / "tsdf_metrics.json").write_text(
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
    (out_dir / "tsdf_metrics.txt").write_text(
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
    fig.savefig(out_dir / "tsdf_metrics.png", dpi=180)
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

    checkpoint_stem = Path(args.checkpoint).stem
    output_root = (
        Path(args.out_dir)
        if args.out_dir is not None
        else _HERE / "training" / "results" / checkpoint_stem
    )
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Detected sequence parent: {data_root}\n"
        f"Reconstructing {len(sequence_dirs)} sequences into: {output_root}",
        flush=True,
    )

    failures: list[tuple[Path, int]] = []
    for sequence_number, sequence_dir in enumerate(sequence_dirs, start=1):
        sequence_out = output_root / sequence_dir.name
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

    print(
        f"\nCompleted reconstruction for all {len(sequence_dirs)} sequences. "
        f"Output root: {output_root}",
        flush=True,
    )
    return True


def main() -> None:
    global CROP_THEN_RESIZE_MODE
    parser = argparse.ArgumentParser(
        description="Depth-map, point-cloud, and TSDF reconstruction from trained depth models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a table-prior checkpoint (.pth)")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "One sequence directory, or a parent whose immediate child "
            "directories are sequences"
        ),
    )
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
                        help="Output directory (default: training/results/<checkpoint_name>)")
    parser.add_argument("--depth_min", type=float, default=DEPTH_MIN)
    parser.add_argument("--depth_max", type=float, default=D_MAX)
    parser.add_argument("--resize_h", type=int, default=TRAIN_RESIZE_HW[0])
    parser.add_argument("--resize_w", type=int, default=TRAIN_RESIZE_HW[1])
    parser.add_argument("--crop_h",   type=int, default=TRAIN_CROP_HW[0])
    parser.add_argument("--crop_w",   type=int, default=TRAIN_CROP_HW[1])
    parser.add_argument("--crop_then_resize", "--crop-then-resize", action="store_true",
                        help="Use native center-crop followed by resize, with "
                             "CROP_THEN_RESIZE_* settings from config.py")
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

    if _run_sequence_directory(args):
        return

    ckpt_path      = Path(args.checkpoint)
    data_dir       = Path(args.data_dir)
    cube_center    = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)
    cube_half_side = args.cube_side / 2.0

    CROP_THEN_RESIZE_MODE = args.crop_then_resize
    if args.crop_then_resize:
        resize_hw = CROP_THEN_RESIZE_HW
        crop_hw = CROP_THEN_RESIZE_CROP_HW
    else:
        resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else TRAIN_RESIZE_HW
        crop_hw = (args.crop_h, args.crop_w) if args.crop_h > 0 and args.crop_w > 0 else TRAIN_CROP_HW

    if args.out_dir is None:
        out_dir = _HERE / "training" / "results" / ckpt_path.stem
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
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}\n{hint}")
    with h5py.File(table_plane_h5_path, "r") as table_file:
        transform = table_file.attrs.get("intrinsics_transform", "")
        if isinstance(transform, bytes):
            transform = transform.decode("utf-8", errors="replace")
        corrected_table_transform = transform == transform_name(args.crop_then_resize)
    if not corrected_table_transform:
        raise RuntimeError(
            f"{table_plane_h5_path} uses obsolete direct-scaling geometry. "
            "Regenerate it with data_precomputation/precompute_table_plane.py "
            "--overwrite."
        )
    if args.save_original_depth_mesh and not realsense_h5_path.exists():
        raise FileNotFoundError(
            f"Missing: {realsense_h5_path}\n"
            "--save_original_depth_mesh requires the raw RealSense depth file."
        )
    if args.color_mesh and not rgb_h5_path.exists():
        raise FileNotFoundError(
            f"Missing: {rgb_h5_path}\n"
            "--color_mesh requires RGB projected into the event frame. "
            "Run data_precomputation/project_realsense_to_event.py with RGB enabled first."
        )

    # ── Camera intrinsics ─────────────────────────────────────────
    K, out_H, out_W = load_K(
        CALIB_DIR, resize_hw, crop_hw, args.crop_then_resize
    )
    print(f"Input resolution : {out_W}×{out_H}")
    if args.save_original_depth_mesh:
        K_depth, T_ee_from_depth, original_depth_scale = load_original_depth_calibration(CALIB_DIR)
        print("Original depth   : native RealSense depth enabled")
    else:
        K_depth = T_ee_from_depth = None
        original_depth_scale = 0.001

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")

    # ── Model ─────────────────────────────────────────────────────
    model, ckpt, ckpt_type = load_model(ckpt_path, device)
    expected_transform = transform_name(args.crop_then_resize)
    checkpoint_transform = ckpt.get("intrinsics_transform", "resize_center_crop")
    if checkpoint_transform != expected_transform:
        raise RuntimeError(
            f"Checkpoint uses {checkpoint_transform!r}, but reconstruction requested "
            f"{expected_transform!r}. Use the matching preprocessing flag."
        )
    use_learned_uncertainty = args.uncertainty_weighted_tsdf
    if use_learned_uncertainty:
        if ckpt_type != "multiview":
            parser.error("Learned uncertainty is only supported for multiview.py checkpoints")
        if not bool(ckpt.get("uncertainty", False)):
            parser.error(
                "Learned uncertainty requires a multiview.py checkpoint trained with --uncertainty"
            )
        if not any(key.startswith("confidence_head.") for key in ckpt.get("model", {})):
            parser.error(
                "Learned uncertainty was requested, but the checkpoint has no confidence-head weights"
            )
    depth_min = ckpt.get("depth_min", args.depth_min)
    depth_max = ckpt.get("depth_max", args.depth_max)
    table_z   = ckpt.get("table_z",   None)

    print(f"Checkpoint       : {ckpt_path.name}")
    print(f"  model          : {ckpt_type}")
    print(f"  in_ch          : {ckpt.get('in_ch', NUM_BINS + 1)}")
    if ckpt_type == "ereformer":
        print(f"  embed_dim      : {ckpt.get('embed_dim', 96)}")
        print(f"  window_size    : {ckpt.get('window_size', 8)}")
        print(f"  seq_len        : {ckpt.get('seq_len', 'N/A')}")
    if ckpt_type == "multiview":
        print(f"  architecture   : {_infer_multiview_model_arch(ckpt)}")
        print(f"  num_views      : {ckpt.get('num_views', 5)}")
        print(f"  coarse_depths  : {ckpt.get('coarse_depths', 32)}")
        print(f"  view_interval  : {ckpt.get('view_interval', 5)}")
        print(f"  pose selection : {ckpt.get('pose_view_selection', False)}")
        print("  feature encoder: deep_fpn")
        print(f"  uncertainty    : {bool(ckpt.get('uncertainty', False))}")
    if ckpt_type == "unet_table":
        print(f"  num_views      : {ckpt.get('num_views', 1)}")
        print(f"  view_interval  : {ckpt.get('view_interval', 5)}")
        print(f"  pose selection : {bool(ckpt.get('pose_view_selection', False))}")
        print(f"  pose channels  : {bool(ckpt.get('pose_channels', False))}")
    table_z_msg = (
        f"{table_z} m"
        if table_z is not None
        else f"from {table_plane_h5_path.name}"
    )
    print(f"  table_z        : {table_z_msg}")
    print(f"  depth range    : {depth_min} – {depth_max} m")

    # Read table_z from the precomputed h5 as a cross-check
    with h5py.File(table_plane_h5_path, "r") as _tf:
        precomp_table_z = float(_tf.attrs.get("table_z_m", 0.0))
    if table_z is not None and abs(table_z - precomp_table_z) > 1e-4:
        print(f"  WARNING: checkpoint table_z ({table_z:.4f} m) differs from "
              f"precomputed table_z ({precomp_table_z:.4f} m). "
              "Using precomputed value for channel loading; model was trained with "
              "the checkpoint value.")

    # ── Poses ─────────────────────────────────────────────────────
    if poses_h5_path.exists():
        with h5py.File(poses_h5_path, "r") as f:
            ee_T_all = f["ee_T"][:].astype(np.float64)   # (N, 4, 4)
        T_ee_from_event = load_T_ee_from_event(CALIB_DIR)
        use_poses = True
        print(f"Poses loaded     : {len(ee_T_all)} frames")
    else:
        ee_T_all        = None
        T_ee_from_event = None
        use_poses       = False
        print("WARNING: poses.h5 not found — TSDF mesh will be in camera frame only")
    needs_model_poses = (
        ckpt_type == "multiview"
        or bool(ckpt.get("pose_channels", False))
        or bool(ckpt.get("pose_view_selection", False))
    )
    if needs_model_poses and not use_poses:
        raise FileNotFoundError(
            f"Missing: {poses_h5_path}\n"
            "This checkpoint requires camera poses for view selection, pose "
            "channels, or geometric source-view warping."
        )
    if args.save_original_depth_mesh and not use_poses:
        raise FileNotFoundError(
            f"Missing: {poses_h5_path}\n"
            "--save_original_depth_mesh requires poses for native depth-camera TSDF fusion."
        )

    T_cam_from_world = None
    cam_centers_world = None
    if use_poses:
        T_event_from_ee = np.linalg.inv(T_ee_from_event).astype(np.float32)
        T_ee_inv = np.linalg.inv(ee_T_all.astype(np.float32))
        T_cam_from_world = np.einsum("ij,njk->nik", T_event_from_ee, T_ee_inv).astype(np.float32)
        cam_centers_world = _camera_centers_world(T_cam_from_world)

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
    if args.save_original_depth_mesh:
        with h5py.File(realsense_h5_path, "r") as f:
            n_rs = f["depth"].shape[0]
        n_total = min(n_total, n_rs)

    n_total = min(n_total, n_vox, n_tbl, n_rgb)   # don't go past what's precomputed
    if T_cam_from_world is not None:
        n_total = min(n_total, len(T_cam_from_world))
    frame_start = 0
    frame_end = n_total - 1
    region_count = frame_end - frame_start + 1

    # ── Frame selection ────────────────────────────────────────────
    if args.frame_step is not None:
        step = max(1, args.frame_step)
        mesh_frames = list(range(frame_start, frame_end + 1, step))
    else:
        n_mesh = min(region_count, max(1, args.mesh_frame_count))
        mesh_frames = np.round(np.linspace(frame_start, frame_end, n_mesh)).astype(int).tolist()
        mesh_frames = sorted(set(mesh_frames))
        step = None

    if args.indices is not None:
        viz_frames = sorted([
            int(i) for i in args.indices
            if frame_start <= int(i) <= frame_end
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
    recurrent_states = None
    t_infer_total = 0.0
    depth_error_total = {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "max_abs": 0.0}

    try:
        for frame_idx in run_frames:
            t_frame0 = time.perf_counter()
            # Voxels
            vox_raw  = voxels_f["voxels"][frame_idx].astype(np.float32)
            vox_np   = _preprocess_voxels(vox_raw, resize_hw, crop_hw)

            # Table-plane channel (precomputed, native camera resolution)
            tbl_np   = table_f["table_plane"][frame_idx].astype(np.float32)
            tbl_eval = tbl_np
            if tbl_eval.shape[0] != out_H or tbl_eval.shape[1] != out_W:
                tbl_eval = resize_crop(tbl_eval, resize_hw, crop_hw, mode="bilinear")

            if ckpt_type == "multiview":
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
                recurrent_states = None
            elif (
                ckpt_type == "unet_table"
                and (
                    bool(ckpt.get("recurrent", False))
                    or ckpt.get("model_arch") == "RecurrentUNet"
                )
            ):
                pred_norm = _infer_recurrent_unet(
                    model,
                    frame_idx,
                    voxels_f["voxels"],
                    table_f["table_plane"],
                    T_cam_from_world,
                    ckpt,
                    device,
                    resize_hw,
                    crop_hw,
                )
                recurrent_states = None
                uncertainty_map = None
            elif ckpt_type == "unet_table":
                pred_norm = _infer_early_fusion_unet(
                    model,
                    frame_idx,
                    voxels_f["voxels"],
                    table_f["table_plane"],
                    T_cam_from_world,
                    ckpt,
                    device,
                    resize_hw,
                    crop_hw,
                    cam_centers_world,
                )
                recurrent_states = None
                uncertainty_map = None
            else:
                pred_norm, recurrent_states = _infer(
                    model, vox_np, tbl_np, device, recurrent_states
                )
                uncertainty_map = None
            pred_m    = depth_min + pred_norm * (depth_max - depth_min)
            if uncertainty_map is not None:
                confidence_by_frame[frame_idx] = np.clip(
                    1.0 - uncertainty_map, 0.0, 1.0
                ).astype(np.float32, copy=False)
            t_infer_total += time.perf_counter() - t_frame0

            # GT depth
            gt = depth_f["depth"][frame_idx].astype(np.float32)
            if gt.shape[0] != out_H or gt.shape[1] != out_W:
                gt = resize_crop(gt, resize_hw, crop_hw, mode="bilinear")
            gt_valid = np.isfinite(gt) & (gt > depth_min) & (gt < depth_max)
            pred_valid = np.isfinite(pred_m) & (pred_m > depth_min) & (pred_m < depth_max)
            gt_mask = gt_valid.astype(np.float32)
            pred_mask = pred_valid.astype(np.float32)
            frame_err = depth_error_stats(pred_m, gt, gt_mask)
            merge_depth_error_stats(depth_error_total, frame_err)

            # Per-frame point cloud
            pts = depth_to_pointcloud(pred_m, pred_mask, K)

            # Accumulate for TSDF
            if use_poses and frame_idx < len(ee_T_all) and frame_idx in mesh_set:
                T_base_from_event = ee_T_all[frame_idx] @ T_ee_from_event
                pred_masked = np.where(pred_valid, pred_m, np.float32(0.0)).astype(np.float32)
                gt_masked   = np.where(gt_valid,   gt,     np.float32(0.0)).astype(np.float32)
                if rgb_f is not None:
                    rgb = rgb_f["rgb"][frame_idx]
                    rgb_by_frame[frame_idx] = resize_crop_rgb(rgb, resize_hw, crop_hw)
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
    print(f"  Error report   : {out_dir / 'error_report.txt'}")

    # ── TSDF mesh ─────────────────────────────────────────────────
    pred_mesh_created = False
    gt_mesh_created = False
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
    original_depth_mesh_path = (
        out_dir / f"{sequence_name}_{frame_selection_name}_original_depth_mesh.obj"
    )
    original_depth_mesh_data = []
    if args.save_original_depth_mesh:
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
        )
        print(f"  GT TSDF time   : {time.perf_counter() - t_tsdf0:.1f}s")
    elif gt_mesh_data:
        print("  GT TSDF        : skipped (--skip_gt_mesh)")
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
    surface_metrics = None
    rendered_metrics = None
    if pred_mesh_created and gt_mesh_created:
        t_metric0 = time.perf_counter()
        try:
            surface_metrics = evaluate_tsdf_meshes(
                pred_mesh_path,
                gt_mesh_path,
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
    elif args.skip_gt_mesh:
        print("  Surface metrics: unavailable because --skip_gt_mesh was used")

    if pred_mesh_created and use_poses and args.render_eval_frames > 0:
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
            rendered_metrics = evaluate_rendered_depth(
                pred_mesh_path,
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
    print(f"  TSDF metrics   : {out_dir / 'tsdf_metrics.txt'}")

    # ── Overview PNG ──────────────────────────────────────────────
    if samples:
        save_overview(samples, out_dir / "overview.png", depth_min, depth_max)

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
