#!/usr/bin/env python3
"""Evaluate U-Net, basic, legacy, and modern MVS checkpoints.

Example:
    python3 evaluation.py \
        --checkpoint checkpoints/multiview/best_model.pth \
        --data_dir ../data/real/eval

The data directory may be either a folder containing multiple sequence
folders or one sequence folder. Results are written to
results/<checkpoint-stem>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multiview import (
    DEPTH_MIN,
    D_MAX,
    NUM_BINS,
    MultiViewAugConfig,
    MultiViewDepthNet,
    MultiViewTableDataset,
    _find_sequences,
    _load_event_calibration,
)
from config import (
    SPATIAL_CUBE_SIDE,
    SPATIAL_TARGET_X,
    SPATIAL_TARGET_Y,
    SPATIAL_TARGET_Z,
)


DEPTH_METRIC_NAMES = (
    "abs_rel",
    "sq_rel_m",
    "mae_m",
    "rmse_m",
    "rmse_log",
    "delta_1",
    "delta_2",
    "delta_3",
)
MVC_THRESHOLDS_M = (0.01, 0.02, 0.05)


@dataclass
class ErrorAccumulator:
    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0

    def add(self, error: np.ndarray) -> None:
        values = np.asarray(error, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        self.count += int(values.size)
        self.abs_sum += float(np.abs(values).sum())
        self.sq_sum += float(np.square(values).sum())

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "mae_m": math.nan, "rmse_m": math.nan}
        return {
            "count": self.count,
            "mae_m": self.abs_sum / self.count,
            "rmse_m": math.sqrt(self.sq_sum / self.count),
        }


@dataclass
class DepthAccumulator:
    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    abs_rel_sum: float = 0.0
    sq_rel_sum: float = 0.0
    log_sq_sum: float = 0.0
    delta_1_count: int = 0
    delta_2_count: int = 0
    delta_3_count: int = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred64 = np.asarray(pred, dtype=np.float64).reshape(-1)
        gt64 = np.asarray(gt, dtype=np.float64).reshape(-1)
        valid = (
            np.isfinite(pred64)
            & np.isfinite(gt64)
            & (pred64 > 0.0)
            & (gt64 > 0.0)
        )
        pred64 = pred64[valid]
        gt64 = gt64[valid]
        if pred64.size == 0:
            return

        error = pred64 - gt64
        abs_error = np.abs(error)
        ratio = np.maximum(pred64 / gt64, gt64 / pred64)
        log_error = np.log(pred64) - np.log(gt64)

        self.count += int(pred64.size)
        self.abs_sum += float(abs_error.sum())
        self.sq_sum += float(np.square(error).sum())
        self.abs_rel_sum += float((abs_error / gt64).sum())
        self.sq_rel_sum += float((np.square(error) / gt64).sum())
        self.log_sq_sum += float(np.square(log_error).sum())
        self.delta_1_count += int((ratio < 1.25).sum())
        self.delta_2_count += int((ratio < 1.25**2).sum())
        self.delta_3_count += int((ratio < 1.25**3).sum())

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"valid_pixels": 0, **{name: math.nan for name in DEPTH_METRIC_NAMES}}
        n = float(self.count)
        return {
            "valid_pixels": self.count,
            "abs_rel": self.abs_rel_sum / n,
            "sq_rel_m": self.sq_rel_sum / n,
            "mae_m": self.abs_sum / n,
            "rmse_m": math.sqrt(self.sq_sum / n),
            "rmse_log": math.sqrt(self.log_sq_sum / n),
            "delta_1": self.delta_1_count / n,
            "delta_2": self.delta_2_count / n,
            "delta_3": self.delta_3_count / n,
        }


@dataclass
class OffsetMetricAccumulator:
    frames: int = 0
    pixels: int = 0
    l1_sum: float = 0.0
    p95_sum: float = 0.0
    worst10_sum: float = 0.0

    def add(self, errors: np.ndarray) -> None:
        values = np.asarray(errors, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        values = np.abs(values)
        worst_count = max(1, int(math.ceil(0.10 * values.size)))
        worst_start = values.size - worst_count
        worst_values = np.partition(values, worst_start)[worst_start:]
        self.frames += 1
        self.pixels += int(values.size)
        self.l1_sum += float(values.mean())
        self.p95_sum += float(np.percentile(values, 95))
        self.worst10_sum += float(worst_values.mean())

    def metrics(self, z_offset_m: float, center_z_m: float) -> dict[str, Any]:
        denominator = max(self.frames, 1)
        return {
            "z_offset_m": z_offset_m,
            "cube_center_z_m": center_z_m,
            "frames": self.frames,
            "pixels": self.pixels,
            "l1_m": self.l1_sum / denominator if self.frames else math.nan,
            "p95_m": self.p95_sum / denominator if self.frames else math.nan,
            "l1_worst10_m": (
                self.worst10_sum / denominator if self.frames else math.nan
            ),
        }


@dataclass
class EvalTimer:
    """Small wall-clock timer bucket collector for evaluation profiling."""

    seconds: dict[str, float]
    counts: dict[str, int]

    def __init__(self) -> None:
        self.seconds = {}
        self.counts = {}

    def add(self, name: str, elapsed_s: float, count: int = 1) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + float(elapsed_s)
        self.counts[name] = self.counts.get(name, 0) + int(count)

    def rows(self, frame_count: int) -> list[dict[str, Any]]:
        total = sum(self.seconds.values())
        rows = []
        for name, seconds in sorted(
            self.seconds.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            rows.append(
                {
                    "stage": name,
                    "seconds": seconds,
                    "percent_of_timed_total": (
                        100.0 * seconds / total if total > 0.0 else math.nan
                    ),
                    "ms_per_eval_frame": (
                        1000.0 * seconds / frame_count
                        if frame_count > 0 else math.nan
                    ),
                    "count": self.counts.get(name, 0),
                    "ms_per_count": (
                        1000.0 * seconds / self.counts[name]
                        if self.counts.get(name, 0) > 0 else math.nan
                    ),
                }
            )
        return rows


@dataclass
class MVCAccumulator:
    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    below_1cm: int = 0
    below_2cm: int = 0
    below_5cm: int = 0
    pairs: int = 0

    def add(self, errors: np.ndarray) -> None:
        values = np.asarray(errors, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        values = np.abs(values)
        self.count += int(values.size)
        self.abs_sum += float(values.sum())
        self.sq_sum += float(np.square(values).sum())
        self.below_1cm += int((values < MVC_THRESHOLDS_M[0]).sum())
        self.below_2cm += int((values < MVC_THRESHOLDS_M[1]).sum())
        self.below_5cm += int((values < MVC_THRESHOLDS_M[2]).sum())

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "frame_pairs": self.pairs,
                "correspondences": 0,
                "mae_m": math.nan,
                "rmse_m": math.nan,
                "within_1cm": math.nan,
                "within_2cm": math.nan,
                "within_5cm": math.nan,
            }
        n = float(self.count)
        return {
            "frame_pairs": self.pairs,
            "correspondences": self.count,
            "mae_m": self.abs_sum / n,
            "rmse_m": math.sqrt(self.sq_sum / n),
            "within_1cm": self.below_1cm / n,
            "within_2cm": self.below_2cm / n,
            "within_5cm": self.below_5cm / n,
        }


@dataclass
class FramePrediction:
    frame_idx: int
    pred: np.ndarray
    T_cam_from_world: np.ndarray
    K: np.ndarray


def _cube_depth_mask(
    depth_m: np.ndarray,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
    cube_center: np.ndarray,
    cube_half_side: float,
) -> np.ndarray:
    """Return pixels whose measured 3-D point lies inside the world-frame cube."""
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.zeros((height, width), dtype=bool)

    z = depth[ys, xs]
    x = (xs.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (ys.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
    T_world_from_cam = np.linalg.inv(T_cam_from_world.astype(np.float64))
    points_world = (T_world_from_cam @ points_cam)[:3].T
    inside = np.all(
        np.abs(points_world - cube_center[None, :]) <= cube_half_side,
        axis=1,
    )

    mask = np.zeros((height, width), dtype=bool)
    mask[ys[inside], xs[inside]] = True
    return mask


def _cube_depth_masks_for_z_offsets(
    depth_m: np.ndarray,
    T_cam_from_world: np.ndarray,
    K: np.ndarray,
    target_x: float,
    target_y: float,
    cube_half_side: float,
    z_offsets_m: np.ndarray,
) -> list[np.ndarray]:
    """Recompute hard spatial masks with each offset as the cube-bottom Z."""
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape
    measured = np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(measured)
    masks = [np.zeros((height, width), dtype=bool) for _ in z_offsets_m]
    if xs.size == 0:
        return masks

    z = depth[ys, xs]
    x = (xs.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (ys.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    points_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
    points_world = (
        np.linalg.inv(T_cam_from_world.astype(np.float64)) @ points_cam
    )[:3].T
    inside_xy = (
        (np.abs(points_world[:, 0] - target_x) <= cube_half_side)
        & (np.abs(points_world[:, 1] - target_y) <= cube_half_side)
    )
    for mask, z_offset_m in zip(masks, z_offsets_m):
        center_z = float(z_offset_m + cube_half_side)
        inside = inside_xy & (
            np.abs(points_world[:, 2] - center_z) <= cube_half_side
        )
        mask[ys[inside], xs[inside]] = True
    return masks


def _activity_error_summary(
    rows: list[dict[str, Any]],
    n_bins: int = 8,
) -> dict[str, Any]:
    """Summarize cube-masked event activity against unchanged frame MAE."""
    valid_rows = [
        row for row in rows
        if math.isfinite(float(row["cube_event_activity"]))
        and math.isfinite(float(row["mae_m"]))
    ]
    if not valid_rows:
        return {
            "frames": 0,
            "correlation_activity_vs_mae": math.nan,
            "mean_cube_event_activity": math.nan,
            "mean_mae_m": math.nan,
            "mean_cube_depth_pixels": math.nan,
            "activity_bins": [],
        }

    activity = np.asarray(
        [row["cube_event_activity"] for row in valid_rows], dtype=np.float64
    )
    mae = np.asarray([row["mae_m"] for row in valid_rows], dtype=np.float64)
    cube_pixels = np.asarray(
        [row["cube_depth_pixels"] for row in valid_rows], dtype=np.float64
    )
    if (
        activity.size < 2
        or float(np.std(activity)) < 1e-12
        or float(np.std(mae)) < 1e-12
    ):
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(activity, mae)[0, 1])

    bins = []
    for bin_index, indices in enumerate(np.array_split(np.argsort(activity), n_bins)):
        if indices.size == 0:
            continue
        bins.append(
            {
                "bin": bin_index,
                "frames": int(indices.size),
                "activity_min": float(activity[indices].min()),
                "activity_max": float(activity[indices].max()),
                "mean_activity": float(activity[indices].mean()),
                "mean_mae_m": float(mae[indices].mean()),
            }
        )
    return {
        "frames": len(valid_rows),
        "correlation_activity_vs_mae": correlation,
        "mean_cube_event_activity": float(activity.mean()),
        "mean_mae_m": float(mae.mean()),
        "mean_cube_depth_pixels": float(cube_pixels.mean()),
        "activity_bins": bins,
    }


def _rotation_matrix_to_xyz_euler_deg(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to intrinsic XYZ Euler angles in degrees."""
    matrices = np.asarray(rotation, dtype=np.float64)
    sy = np.sqrt(
        np.square(matrices[:, 0, 0]) + np.square(matrices[:, 1, 0])
    )
    singular = sy < 1e-8

    x = np.arctan2(matrices[:, 2, 1], matrices[:, 2, 2])
    y = np.arctan2(-matrices[:, 2, 0], sy)
    z = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
    if np.any(singular):
        x[singular] = np.arctan2(
            -matrices[singular, 1, 2],
            matrices[singular, 1, 1],
        )
        z[singular] = 0.0

    # Unwrap each trajectory so crossings at +/-180 degrees do not create
    # artificial discontinuities in the plots.
    angles = np.stack([x, y, z], axis=1)
    return np.rad2deg(np.unwrap(angles, axis=0))


def _load_pose_motion(sequence_dir: Path) -> dict[str, np.ndarray]:
    """Load per-frame end-effector pose and derive translational speed."""
    import h5py

    with h5py.File(sequence_dir / "hdf5" / "poses.h5", "r") as handle:
        ee_T = handle["ee_T"][:].astype(np.float64)

    timestamps_s: np.ndarray | None = None
    realsense_path = sequence_dir / "hdf5" / "realsense.h5"
    if realsense_path.exists():
        with h5py.File(realsense_path, "r") as handle:
            if "t_global_ms" in handle:
                timestamps_s = handle["t_global_ms"][:].astype(np.float64) / 1e3
            elif "t_hw_as_sys_ns" in handle:
                timestamps_s = handle["t_hw_as_sys_ns"][:].astype(np.float64) / 1e9
            elif "t_sys_ns" in handle:
                timestamps_s = handle["t_sys_ns"][:].astype(np.float64) / 1e9

    n_frames = len(ee_T)
    if timestamps_s is None:
        metadata_path = sequence_dir / "hdf5" / "metadata.h5"
        fps = 30.0
        if metadata_path.exists():
            with h5py.File(metadata_path, "r") as handle:
                fps = float(handle.attrs.get("fps", fps))
        timestamps_s = np.arange(n_frames, dtype=np.float64) / max(fps, 1e-6)

    n_frames = min(n_frames, len(timestamps_s))
    ee_T = ee_T[:n_frames]
    timestamps_s = timestamps_s[:n_frames]
    positions = ee_T[:, :3, 3]
    rotations_deg = _rotation_matrix_to_xyz_euler_deg(ee_T[:, :3, :3])

    speed_m_s = np.full(n_frames, np.nan, dtype=np.float64)
    if n_frames >= 2:
        left = np.maximum(np.arange(n_frames) - 1, 0)
        right = np.minimum(np.arange(n_frames) + 1, n_frames - 1)
        dt = timestamps_s[right] - timestamps_s[left]
        displacement = positions[right] - positions[left]
        valid_dt = np.isfinite(dt) & (dt > 1e-9)
        speed_m_s[valid_dt] = (
            np.linalg.norm(displacement[valid_dt], axis=1) / dt[valid_dt]
        )

    return {
        "position_m": positions,
        "rotation_xyz_deg": rotations_deg,
        "speed_m_s": speed_m_s,
    }


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(val) for val in value]
    return value


def _checkpoint_state(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    if "model" in checkpoint:
        state = checkpoint["model"]
        metadata = checkpoint
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        metadata = checkpoint
    elif checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state = checkpoint
        metadata = {}
    else:
        raise KeyError("Checkpoint has no 'model' or 'state_dict' model weights.")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    return state, metadata


def _required_metadata(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in metadata:
        return metadata[key]
    if default is not None:
        return default
    raise KeyError(
        f"Checkpoint is missing '{key}'. Use a checkpoint saved by the current multiview.py."
    )


def _build_model(
    metadata: dict[str, Any],
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    base = int(_required_metadata(metadata, "base"))
    model_arch = str(metadata.get("model_arch", "MultiViewDepthNet"))
    model_cls = MultiViewDepthNet
    if model_arch == "ModernMVSNet":
        from modern_multiview import ModernMVSNet, upgrade_legacy_modern_state_dict

        model_cls = ModernMVSNet
        state = upgrade_legacy_modern_state_dict(state)
    elif model_arch == "BasicMVSNet":
        from basic_multiview import BasicMVSNet

        model_cls = BasicMVSNet
    elif model_arch in ("UNet", "UNet+uncertainty") or "predict_uncertainty" in metadata:
        from train_unet import UNet
        from train_unet_table import UncertaintyUNet

        predicts_uncertainty = bool(metadata.get("predict_uncertainty", False))
        unet_cls = UncertaintyUNet if predicts_uncertainty else UNet
        unet = unet_cls(
            in_ch=int(_required_metadata(metadata, "in_ch", NUM_BINS + 1)),
            base=base,
        )
        incompatible = unet.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint architecture does not match {model_arch}: {incompatible}"
            )

        class EarlyFusionUNetEvaluationAdapter(torch.nn.Module):
            def __init__(self, depth_model: torch.nn.Module) -> None:
                super().__init__()
                self.depth_model = depth_model

            def forward(
                self,
                images: torch.Tensor,
                camera_matrices: torch.Tensor,
                intrinsics: torch.Tensor,
                depth_values: torch.Tensor,
            ) -> torch.Tensor:
                del camera_matrices, intrinsics, depth_values
                batch, views, channels, height, width = images.shape
                fused = images.reshape(batch, views * channels, height, width)
                output = self.depth_model(fused)
                return output[0] if isinstance(output, tuple) else output

        return EarlyFusionUNetEvaluationAdapter(unet).to(device).eval()
    elif model_arch not in ("MultiViewDepthNet", "legacy_multiview"):
        raise ValueError(f"Unsupported multiview checkpoint architecture: {model_arch}")

    model = model_cls(
        in_ch=int(_required_metadata(metadata, "in_ch", NUM_BINS + 1)),
        base=base,
        feature_ch=int(_required_metadata(metadata, "feature_channels", base * 4)),
        cost_base=int(_required_metadata(metadata, "cost_channels", max(base // 2, 8))),
        fine_depths=int(_required_metadata(metadata, "fine_depths", 5)),
        fine_window=float(_required_metadata(metadata, "fine_window", 0.08)),
        fine_offset_radius=float(_required_metadata(metadata, "fine_offset_radius", 2.0)),
        learned_fine_window=bool(_required_metadata(metadata, "learned_fine_window", False)),
        masked_warp_aggregation=bool(
            _required_metadata(metadata, "masked_warp_aggregation", False)
        ),
        cost_volume_ref_features=bool(
            _required_metadata(metadata, "cost_volume_ref_features", False)
        ),
        single_view_fallback=bool(
            _required_metadata(metadata, "single_view_fallback", False)
        ),
        feature_encoder=str(_required_metadata(metadata, "feature_encoder", "cnn")),
        decoder_type=str(metadata.get("decoder_type", "auto")),
        correlation_groups=int(metadata.get("correlation_groups", 0)),
        reference_channels=int(metadata.get("reference_channels", 0)),
        coarse_cost_channels=int(metadata.get("coarse_cost_channels", 0)),
        fine_cost_channels=int(metadata.get("fine_cost_channels", 0)),
        refiner_channels=int(metadata.get("refiner_channels", 0)),
        hourglass_levels=int(metadata.get("hourglass_levels", 2)),
        coarse_hourglass_levels=int(metadata.get("coarse_hourglass_levels", 0)),
        fine_hourglass_levels=int(metadata.get("fine_hourglass_levels", 0)),
        learned_view_weighting=bool(metadata.get("learned_view_weighting", False)),
        two_mode_fine_candidates=bool(metadata.get("two_mode_fine_candidates", False)),
        fine_supervision=bool(metadata.get("fine_supervision", False)),
        fine_loss_weight=float(metadata.get("fine_loss_weight", 0.3)),
        variance_channels=int(metadata.get("variance_channels", 0)),
        convex_upsampling=bool(metadata.get("convex_upsampling", False)),
        fullres_geometry=bool(metadata.get("fullres_geometry", False)),
        fullres_depths=int(metadata.get("fullres_depths", 3)),
        fullres_window=float(metadata.get("fullres_window", 0.01)),
        fpn_dropout=float(metadata.get("fpn_dropout", 0.0)),
        reference_dropout=float(metadata.get("reference_dropout", 0.0)),
        hourglass_dropout=float(metadata.get("hourglass_dropout", 0.0)),
        drop_path_rate=float(metadata.get("drop_path_rate", 0.0)),
        cost_volume_type=str(metadata.get("cost_volume_type", "correlation")),
        middle_depths=int(metadata.get("middle_depths", 8)),
        middle_window=float(metadata.get("middle_window", 0.12)),
        middle_cost_channels=int(metadata.get("middle_cost_channels", 0)),
        middle_hourglass_levels=int(metadata.get("middle_hourglass_levels", 0)),
        middle_feature_channels=int(metadata.get("middle_feature_channels", 0)),
        fine_feature_channels=int(metadata.get("fine_feature_channels", 0)),
        middle_supervision=bool(metadata.get("middle_supervision", False)),
        middle_loss_weight=float(metadata.get("middle_loss_weight", 0.3)),
        fullres_fine_volume=bool(metadata.get("fullres_fine_volume", False)),
        h4_coarse_volume=bool(metadata.get("h4_coarse_volume", False)),
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing_non_confidence = [
        key for key in incompatible.missing_keys
        if not key.startswith("confidence_head.")
    ]
    if missing_non_confidence or incompatible.unexpected_keys:
        details = []
        if missing_non_confidence:
            details.append(f"missing keys: {missing_non_confidence}")
        if incompatible.unexpected_keys:
            details.append(f"unexpected keys: {incompatible.unexpected_keys}")
        raise RuntimeError(
            f"Checkpoint architecture does not match {model_arch} ("
            + "; ".join(details)
            + ")."
        )
    if incompatible.missing_keys:
        print(
            "Checkpoint has no learned confidence head; loading depth model "
            "weights only.",
            flush=True,
        )
    return model.to(device).eval()


def _frame_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    accumulator = DepthAccumulator()
    accumulator.add(pred[valid], gt[valid])
    return accumulator.metrics()


def _boundary_masks(
    gt: torch.Tensor,
    valid: torch.Tensor,
    threshold_m: float,
    dilation_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return valid boundary and non-boundary masks for a single depth map."""
    depth = gt[None, None]
    mask = valid[None, None].bool()
    edge = torch.zeros_like(mask)

    horizontal_valid = mask[:, :, :, 1:] & mask[:, :, :, :-1]
    horizontal_edge = (
        torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1]) >= threshold_m
    ) & horizontal_valid
    edge[:, :, :, 1:] |= horizontal_edge
    edge[:, :, :, :-1] |= horizontal_edge

    vertical_valid = mask[:, :, 1:, :] & mask[:, :, :-1, :]
    vertical_edge = (
        torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :]) >= threshold_m
    ) & vertical_valid
    edge[:, :, 1:, :] |= vertical_edge
    edge[:, :, :-1, :] |= vertical_edge

    if dilation_px > 0:
        kernel = 2 * dilation_px + 1
        edge = F.max_pool2d(edge.float(), kernel, stride=1, padding=dilation_px) > 0
    boundary = edge & mask
    non_boundary = (~edge) & mask
    return boundary[0, 0], non_boundary[0, 0]


def _reproject_consistency_errors(
    source: FramePrediction,
    target: FramePrediction,
    pixel_stride: int,
    occlusion_tolerance_m: float,
) -> np.ndarray:
    """Warp source prediction into target and compare z-buffer-visible points."""
    source_depth = source.pred
    target_depth = target.pred
    height, width = source_depth.shape
    if target_depth.shape != (height, width):
        raise ValueError("Multi-view consistency requires equal prediction resolutions.")

    ys, xs = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    z = source_depth[::pixel_stride, ::pixel_stride]
    valid = np.isfinite(z) & (z > 0.0)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    if z.size == 0:
        return np.empty(0, dtype=np.float64)

    K_source = source.K.astype(np.float64)
    x = (xs - K_source[0, 2]) * z / K_source[0, 0]
    y = (ys - K_source[1, 2]) * z / K_source[1, 1]
    points_source = np.stack([x, y, z, np.ones_like(z)], axis=0)

    T_target_from_source = (
        target.T_cam_from_world.astype(np.float64)
        @ np.linalg.inv(source.T_cam_from_world.astype(np.float64))
    )
    points_target = T_target_from_source @ points_source
    z_target = points_target[2]
    in_front = np.isfinite(z_target) & (z_target > 1e-6)
    if not np.any(in_front):
        return np.empty(0, dtype=np.float64)

    points_target = points_target[:, in_front]
    z_target = z_target[in_front]
    K_target = target.K.astype(np.float64)
    u = K_target[0, 0] * points_target[0] / z_target + K_target[0, 2]
    v = K_target[1, 1] * points_target[1] / z_target + K_target[1, 2]
    u_int = np.rint(u).astype(np.int64)
    v_int = np.rint(v).astype(np.int64)
    inside = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u_int >= 0)
        & (u_int < width)
        & (v_int >= 0)
        & (v_int < height)
    )
    if not np.any(inside):
        return np.empty(0, dtype=np.float64)

    u_int = u_int[inside]
    v_int = v_int[inside]
    z_target = z_target[inside]
    flat_index = v_int * width + u_int

    # Keep only the closest projected source point per target pixel.
    order = np.lexsort((z_target, flat_index))
    sorted_index = flat_index[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = sorted_index[1:] != sorted_index[:-1]
    visible = order[first]

    target_values = target_depth[v_int[visible], u_int[visible]].astype(np.float64)
    transformed_values = z_target[visible]
    valid_target = (
        np.isfinite(target_values)
        & (target_values > 0.0)
        & np.isfinite(transformed_values)
        & (transformed_values <= target_values + occlusion_tolerance_m)
    )
    return np.abs(transformed_values[valid_target] - target_values[valid_target])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(name: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if name.startswith("delta_") or name.startswith("within_"):
        return f"{100.0 * float(value):.2f}%"
    if name.endswith("_m"):
        return f"{float(value):.6f} m"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    depth = summary["depth_metrics"]
    boundary = summary["boundary_metrics"]
    mvc = summary["multiview_consistency"]
    activity = summary["cube_event_activity_vs_error"]
    offset_metrics = summary["spatial_mask_z_offset_metrics"]
    lines = [
        f"Checkpoint: {summary['checkpoint']}",
        f"Evaluation root: {summary['evaluation_root']}",
        f"Sequences: {summary['sequence_count']}",
        f"Frames: {summary['frame_count']}",
        f"Evaluation frame step: {summary['evaluation_configuration']['frame_step']}",
        "",
        "Performance",
        f"  inference_batch_size: {summary['performance']['inference_batch_size']}",
        f"  timed_inference_frames: {summary['performance']['timed_inference_frames']}",
        f"  model_inference_ms_per_frame: "
        f"{summary['performance']['model_inference_ms_per_frame']:.3f}",
        f"  model_inference_fps: {summary['performance']['model_inference_fps']:.3f}",
        f"  evaluation_loop_ms_per_frame: "
        f"{summary['performance']['evaluation_loop_ms_per_frame']:.3f}",
        "",
        "Depth metrics",
    ]
    for name in DEPTH_METRIC_NAMES:
        lines.append(f"  {name}: {_format_metric(name, depth[name])}")
    lines.extend(
        [
            f"  valid_pixels: {depth['valid_pixels']}",
            "",
            "Depth-boundary metrics",
            f"  boundary_pixels: {boundary['boundary']['count']}",
            f"  boundary_mae_m: {_format_metric('mae_m', boundary['boundary']['mae_m'])}",
            f"  boundary_rmse_m: {_format_metric('rmse_m', boundary['boundary']['rmse_m'])}",
            f"  non_boundary_pixels: {boundary['non_boundary']['count']}",
            f"  non_boundary_mae_m: {_format_metric('mae_m', boundary['non_boundary']['mae_m'])}",
            f"  non_boundary_rmse_m: {_format_metric('rmse_m', boundary['non_boundary']['rmse_m'])}",
            "",
            "Multi-view consistency",
            f"  frame_pairs: {mvc['frame_pairs']}",
            f"  correspondences: {mvc['correspondences']}",
            f"  mae_m: {_format_metric('mae_m', mvc['mae_m'])}",
            f"  rmse_m: {_format_metric('rmse_m', mvc['rmse_m'])}",
            f"  within_1cm: {_format_metric('within_1cm', mvc['within_1cm'])}",
            f"  within_2cm: {_format_metric('within_2cm', mvc['within_2cm'])}",
            f"  within_5cm: {_format_metric('within_5cm', mvc['within_5cm'])}",
            "",
            "Cube-masked event activity vs whole-frame error",
            f"  frames: {activity['frames']}",
            f"  correlation_activity_vs_mae: "
            f"{_format_metric('correlation', activity['correlation_activity_vs_mae'])}",
            f"  mean_cube_event_activity: "
            f"{_format_metric('activity', activity['mean_cube_event_activity'])}",
            f"  mean_cube_depth_pixels: "
            f"{_format_metric('pixels', activity['mean_cube_depth_pixels'])}",
            f"  mean_whole_frame_mae_m: "
            f"{_format_metric('mae_m', activity['mean_mae_m'])}",
        ]
    )
    lines.extend(["", "Spatial-mask Z-offset metrics"])
    for row in offset_metrics:
        lines.append(
            f"  bottom_z={1000.0 * row['z_offset_m']:.1f} mm: "
            f"L1={1000.0 * row['l1_m']:.3f} mm, "
            f"p95={1000.0 * row['p95_m']:.3f} mm, "
            f"worst10={1000.0 * row['l1_worst10_m']:.3f} mm "
            f"({row['pixels']} pixels)"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup_seaborn_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette="deep",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.25,
        },
    )
    return plt, sns


def _write_eval_times(
    output_dir: Path,
    rows: list[dict[str, Any]],
    frame_count: int,
) -> None:
    if not rows:
        return
    _write_csv(output_dir / "eval_times.csv", rows)
    with (output_dir / "eval_times.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite_or_none(rows), handle, indent=2)
        handle.write("\n")

    total_seconds = sum(float(row["seconds"]) for row in rows)
    lines = [
        "Evaluation timing breakdown",
        f"Frames: {frame_count}",
        f"Timed total: {total_seconds:.3f} s",
        "",
        (
            f"{'stage':34s} {'seconds':>10s} {'share':>8s} "
            f"{'ms/frame':>10s} {'count':>8s} {'ms/count':>10s}"
        ),
        "-" * 86,
    ]
    for row in rows:
        lines.append(
            f"{row['stage']:34s} "
            f"{float(row['seconds']):10.3f} "
            f"{float(row['percent_of_timed_total']):7.1f}% "
            f"{float(row['ms_per_eval_frame']):10.3f} "
            f"{int(row['count']):8d} "
            f"{float(row['ms_per_count']):10.3f}"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "  dataloader_wait includes waiting for worker/HDF5 loading and collation.",
            "  device_transfer includes host-to-device tensor copies.",
            "  model_forward is CUDA-synchronized forward time.",
            "  cpu_transfer includes prediction and metadata conversion back to CPU.",
            "  online_metrics includes depth metrics, mask sweeps, boundary metrics,",
            "  event-activity summaries, pose bookkeeping, and MVC reprojection.",
            "  output_writing and plotting are after the evaluation loop.",
        ]
    )
    (output_dir / "eval_times").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_results(
    output_dir: Path,
    summary: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
) -> None:
    plt, sns = _setup_seaborn_plotting()

    depth = summary["depth_metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    error_names = ["abs_rel", "sq_rel_m", "mae_m", "rmse_m", "rmse_log"]
    sns.barplot(
        x=error_names,
        y=[depth[name] for name in error_names],
        ax=axes[0],
        hue=error_names,
        palette="Blues_d",
        legend=False,
    )
    axes[0].set_title("Depth errors")
    axes[0].tick_params(axis="x", rotation=30)
    delta_names = ["delta_1", "delta_2", "delta_3"]
    delta_labels = [r"$\delta<1.25$", r"$\delta<1.25^2$", r"$\delta<1.25^3$"]
    sns.barplot(
        x=delta_labels,
        y=[100.0 * depth[name] for name in delta_names],
        ax=axes[1],
        hue=delta_labels,
        palette="Greens_d",
        legend=False,
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].set_title("Threshold accuracy")
    figure.tight_layout()
    figure.savefig(output_dir / "depth_metrics.png", dpi=180)
    plt.close(figure)

    boundary = summary["boundary_metrics"]
    mvc = summary["multiview_consistency"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    boundary_labels = ["Boundary MAE", "Non-boundary MAE"]
    sns.barplot(
        x=boundary_labels,
        y=[boundary["boundary"]["mae_m"], boundary["non_boundary"]["mae_m"]],
        ax=axes[0],
        hue=boundary_labels,
        palette="flare",
        legend=False,
    )
    axes[0].set_ylabel("Error [m]")
    axes[0].set_title("Depth-boundary error")
    mvc_labels = ["<1 cm", "<2 cm", "<5 cm"]
    sns.barplot(
        x=mvc_labels,
        y=[
            100.0 * mvc["within_1cm"],
            100.0 * mvc["within_2cm"],
            100.0 * mvc["within_5cm"],
        ],
        ax=axes[1],
        hue=mvc_labels,
        palette="Purples_d",
        legend=False,
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Consistent correspondences [%]")
    axes[1].set_title("Multi-view consistency thresholds")
    figure.tight_layout()
    figure.savefig(output_dir / "boundary_and_consistency.png", dpi=180)
    plt.close(figure)

    if sequence_rows:
        sequence_names = [str(row["sequence"]) for row in sequence_rows]
        sequence_mae_cm = [
            100.0 * float(row["mae_m"]) for row in sequence_rows
        ]
        sequence_rmse_cm = [
            100.0 * float(row["rmse_m"]) for row in sequence_rows
        ]
        plot_rows = [
            {"sequence": seq, "metric": "MAE", "error_cm": mae}
            for seq, mae in zip(sequence_names, sequence_mae_cm)
        ] + [
            {"sequence": seq, "metric": "RMSE", "error_cm": rmse}
            for seq, rmse in zip(sequence_names, sequence_rmse_cm)
        ]
        figure_width = max(8.0, 0.75 * len(sequence_rows))
        figure, axis = plt.subplots(figsize=(figure_width, 5.5))
        sns.barplot(
            data=plot_rows,
            x="sequence",
            y="error_cm",
            hue="metric",
            ax=axis,
            palette="deep",
        )
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("Depth error [cm]")
        axis.set_title("Depth error per sequence")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "error_per_sequence.png", dpi=180)
        plt.close(figure)

    if frame_rows:
        mae_cm = [100.0 * float(row["mae_m"]) for row in frame_rows]
        rmse_cm = [100.0 * float(row["rmse_m"]) for row in frame_rows]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.histplot(mae_cm, bins=40, ax=axes[0], color=sns.color_palette()[0])
        axes[0].set_xlabel("Per-frame MAE [cm]")
        axes[0].set_ylabel("Frames")
        sns.histplot(rmse_cm, bins=40, ax=axes[1], color=sns.color_palette()[1])
        axes[1].set_xlabel("Per-frame RMSE [cm]")
        axes[1].set_ylabel("Frames")
        figure.tight_layout()
        figure.savefig(output_dir / "per_frame_error_histograms.png", dpi=180)
        plt.close(figure)

        activity = np.asarray(
            [row["cube_event_activity"] for row in frame_rows], dtype=np.float64
        )
        mae_m = np.asarray([row["mae_m"] for row in frame_rows], dtype=np.float64)
        finite = np.isfinite(activity) & np.isfinite(mae_m)
        if finite.any():
            activity = activity[finite]
            mae_m = mae_m[finite]
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            sns.scatterplot(
                x=activity,
                y=mae_m,
                ax=axes[0],
                s=18,
                alpha=0.35,
                linewidth=0,
            )
            axes[0].set_xlabel("Event activity at GT-depth pixels inside cube")
            axes[0].set_ylabel("Whole-frame MAE [m]")
            axes[0].set_title("Relevant-region activity vs prediction error")

            bins = summary["cube_event_activity_vs_error"]["activity_bins"]
            sns.lineplot(
                x=[row["mean_activity"] for row in bins],
                y=[row["mean_mae_m"] for row in bins],
                marker="o",
                ax=axes[1],
                color=sns.color_palette("flare", 3)[1],
            )
            axes[1].set_xlabel("Mean cube-masked event activity")
            axes[1].set_ylabel("Mean whole-frame MAE [m]")
            axes[1].set_title("Equal-frame activity bins")
            figure.tight_layout()
            figure.savefig(
                output_dir / "cube_event_activity_vs_error.png", dpi=180
            )
            plt.close(figure)

        outside_activity = np.asarray(
            [row["outside_cube_event_activity"] for row in frame_rows],
            dtype=np.float64,
        )
        mae_m = np.asarray([row["mae_m"] for row in frame_rows], dtype=np.float64)
        finite = np.isfinite(outside_activity) & np.isfinite(mae_m)
        if finite.any():
            outside_activity = outside_activity[finite]
            mae_m = mae_m[finite]
            sorted_bins = [
                indices
                for indices in np.array_split(
                    np.argsort(outside_activity), min(8, outside_activity.size)
                )
                if indices.size
            ]
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            sns.scatterplot(
                x=outside_activity,
                y=mae_m,
                ax=axes[0],
                s=18,
                alpha=0.35,
                linewidth=0,
            )
            axes[0].set_xlabel(
                "Event activity at valid GT-depth pixels outside cube"
            )
            axes[0].set_ylabel("Whole-frame MAE [m]")
            axes[0].set_title("Outside-cube activity vs prediction error")

            sns.lineplot(
                x=[
                    float(outside_activity[indices].mean())
                    for indices in sorted_bins
                ],
                y=[float(mae_m[indices].mean()) for indices in sorted_bins],
                marker="o",
                ax=axes[1],
                color=sns.color_palette("flare", 3)[1],
            )
            axes[1].set_xlabel("Mean outside-cube event activity")
            axes[1].set_ylabel("Mean whole-frame MAE [m]")
            axes[1].set_title("Equal-frame activity bins")
            figure.tight_layout()
            figure.savefig(
                output_dir / "outside_cube_event_activity_vs_error.png",
                dpi=180,
            )
            plt.close(figure)

        pose_fields = (
            ("pose_x_m", "Position X [m]"),
            ("pose_y_m", "Position Y [m]"),
            ("pose_z_m", "Position Z [m]"),
            ("rotation_x_deg", "Rotation X [deg]"),
            ("rotation_y_deg", "Rotation Y [deg]"),
            ("rotation_z_deg", "Rotation Z [deg]"),
        )
        if all(field in frame_rows[0] for field, _ in pose_fields):
            figure, axes = plt.subplots(2, 3, figsize=(15, 9))
            for axis, (field, label) in zip(axes.flat, pose_fields):
                x = np.asarray([row[field] for row in frame_rows], dtype=np.float64)
                error_cm = np.asarray(
                    [100.0 * row["mae_m"] for row in frame_rows],
                    dtype=np.float64,
                )
                finite = np.isfinite(x) & np.isfinite(error_cm)
                x = x[finite]
                error_cm = error_cm[finite]
                sns.scatterplot(
                    x=x,
                    y=error_cm,
                    ax=axis,
                    s=14,
                    alpha=0.22,
                    linewidth=0,
                )
                if x.size:
                    sorted_bins = [
                        indices
                        for indices in np.array_split(
                            np.argsort(x), min(12, x.size)
                        )
                        if indices.size
                    ]
                    sns.lineplot(
                        x=[float(x[indices].mean()) for indices in sorted_bins],
                        y=[
                            float(error_cm[indices].mean())
                            for indices in sorted_bins
                        ],
                        marker="o",
                        ax=axis,
                        color=sns.color_palette("flare", 3)[1],
                    )
                axis.set_xlabel(label)
                axis.set_ylabel("Per-frame MAE [cm]")
            figure.suptitle(
                "Depth error over end-effector pose\n"
                "Dots: frames; orange: equal-count bin means",
                fontsize=13,
            )
            figure.tight_layout()
            figure.savefig(output_dir / "pose_vs_error.png", dpi=180)
            plt.close(figure)

        if "arm_speed_m_s" in frame_rows[0]:
            speed = np.asarray(
                [row["arm_speed_m_s"] for row in frame_rows], dtype=np.float64
            )
            error_cm = np.asarray(
                [100.0 * row["mae_m"] for row in frame_rows], dtype=np.float64
            )
            finite = np.isfinite(speed) & np.isfinite(error_cm)
            speed = speed[finite]
            error_cm = error_cm[finite]
            if speed.size:
                figure, axis = plt.subplots(figsize=(7.5, 5.5))
                sns.scatterplot(
                    x=speed,
                    y=error_cm,
                    ax=axis,
                    s=18,
                    alpha=0.25,
                    linewidth=0,
                )
                sorted_bins = [
                    indices
                    for indices in np.array_split(
                        np.argsort(speed), min(12, speed.size)
                    )
                    if indices.size
                ]
                sns.lineplot(
                    x=[float(speed[indices].mean()) for indices in sorted_bins],
                    y=[float(error_cm[indices].mean()) for indices in sorted_bins],
                    marker="o",
                    ax=axis,
                    color=sns.color_palette("flare", 3)[1],
                    label="Equal-count bin mean",
                )
                axis.set_xlabel("End-effector translational speed [m/s]")
                axis.set_ylabel("Per-frame MAE [cm]")
                axis.set_title("Arm speed vs depth error")
                axis.legend()
                figure.tight_layout()
                figure.savefig(output_dir / "arm_speed_vs_error.png", dpi=180)
                plt.close(figure)

        if "target_distance_m" in frame_rows[0]:
            distance = np.asarray(
                [row["target_distance_m"] for row in frame_rows],
                dtype=np.float64,
            )
            error_cm = np.asarray(
                [100.0 * row["mae_m"] for row in frame_rows],
                dtype=np.float64,
            )
            finite = np.isfinite(distance) & np.isfinite(error_cm)
            distance = distance[finite]
            error_cm = error_cm[finite]
            if distance.size:
                figure, axis = plt.subplots(figsize=(7.5, 5.5))
                sns.scatterplot(
                    x=distance,
                    y=error_cm,
                    ax=axis,
                    s=18,
                    alpha=0.25,
                    linewidth=0,
                )
                sorted_bins = [
                    indices
                    for indices in np.array_split(
                        np.argsort(distance), min(12, distance.size)
                    )
                    if indices.size
                ]
                sns.lineplot(
                    x=[float(distance[indices].mean()) for indices in sorted_bins],
                    y=[float(error_cm[indices].mean()) for indices in sorted_bins],
                    marker="o",
                    ax=axis,
                    color=sns.color_palette("flare", 3)[1],
                    label="Equal-count bin mean",
                )
                axis.set_xlabel("Camera distance to recording target [m]")
                axis.set_ylabel("Per-frame MAE [cm]")
                axis.set_title("Target distance vs depth error")
                axis.legend()
                figure.tight_layout()
                figure.savefig(output_dir / "target_distance_vs_error.png", dpi=180)
                plt.close(figure)


def _plot_worst_frames(
    output_dir: Path,
    samples: list[dict[str, Any]],
) -> None:
    """Save the highest-MAE frame from each evaluated object/sequence."""
    if not samples:
        return

    plt, _sns = _setup_seaborn_plotting()

    samples = sorted(samples, key=lambda sample: sample["sequence"])
    figure, axes = plt.subplots(
        len(samples),
        4,
        figsize=(16, 3.5 * len(samples)),
        squeeze=False,
    )
    for column, title in enumerate(
        ("Event activity (shared scale)", "GT depth", "Pred depth", "Abs error")
    ):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")

    depth_cmap = plt.get_cmap("turbo")
    error_cmap = plt.get_cmap("turbo")
    invalid_color = np.array([40, 40, 40], dtype=np.uint8)

    def colorize(values: np.ndarray, vmin: float, vmax: float, cmap: Any) -> np.ndarray:
        normalized = np.clip((values - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        return (cmap(normalized)[..., :3] * 255).astype(np.uint8)

    # Use one visualization scale for every sequence. Previously each frame
    # was min-max normalized independently, which could make weak and strong
    # event frames look equally active. This affects plots only; model inputs
    # are the unmodified tensors returned by MultiViewTableDataset.
    event_activity_images = [
        np.abs(sample["events"]).sum(axis=0) for sample in samples
    ]
    nonzero_activity = np.concatenate(
        [activity[activity > 0.0] for activity in event_activity_images]
    ) if any(np.any(activity > 0.0) for activity in event_activity_images) else np.array([])
    event_vmax = (
        max(float(np.percentile(nonzero_activity, 99.0)), 1e-6)
        if nonzero_activity.size
        else 1.0
    )

    for row, (sample, event_activity) in enumerate(
        zip(samples, event_activity_images)
    ):
        ax_events, ax_gt, ax_pred, ax_error = axes[row]
        valid = sample["valid"]
        pred_valid = (
            np.isfinite(sample["pred"])
            & (sample["pred"] >= DEPTH_MIN)
            & (sample["pred"] <= D_MAX)
        )

        ax_events.imshow(
            event_activity,
            cmap="gray",
            vmin=0.0,
            vmax=event_vmax,
        )
        ax_events.set_ylabel(
            f"{sample['sequence']}\nframe {sample['frame_idx']}",
            fontsize=8,
        )

        gt_vis = colorize(sample["gt"], DEPTH_MIN, D_MAX, depth_cmap)
        gt_vis[~valid] = invalid_color
        ax_gt.imshow(gt_vis)

        pred_vis = colorize(sample["pred"], DEPTH_MIN, D_MAX, depth_cmap)
        pred_vis[~pred_valid] = invalid_color
        ax_pred.imshow(pred_vis)

        error = np.abs(sample["pred"] - sample["gt"])
        error_vis = colorize(error, 0.0, 0.1, error_cmap)
        error_vis[~valid] = invalid_color
        ax_error.imshow(error_vis)
        ax_error.set_xlabel(f"MAE {sample['mae_m'] * 100.0:.2f} cm", fontsize=8)

        for axis in (ax_events, ax_gt, ax_pred, ax_error):
            axis.set_xticks([])
            axis.set_yticks([])

    figure.tight_layout()
    figure.savefig(output_dir / "worst_error_overview.png", dpi=180)
    plt.close(figure)


def _plot_spatial_mask_offset_metrics(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    plt, sns = _setup_seaborn_plotting()
    labels = [f"{1000.0 * float(row['z_offset_m']):.1f}" for row in rows]
    plot_rows = []
    for label, row in zip(labels, rows):
        plot_rows.extend(
            [
                {"offset_mm": label, "metric": "L1", "error_mm": 1000.0 * float(row["l1_m"])},
                {"offset_mm": label, "metric": "p95", "error_mm": 1000.0 * float(row["p95_m"])},
                {
                    "offset_mm": label,
                    "metric": "Worst-10% L1",
                    "error_mm": 1000.0 * float(row["l1_worst10_m"]),
                },
            ]
        )
    figure, axis = plt.subplots(figsize=(12, 5.5))
    sns.barplot(
        data=plot_rows,
        x="offset_mm",
        y="error_mm",
        hue="metric",
        ax=axis,
        palette="deep",
    )
    axis.set_xlabel("Spatial-mask cube-bottom Z offset [mm]")
    axis.set_ylabel("Depth error [mm]")
    axis.set_title("Depth error under recomputed spatial masks")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "spatial_mask_z_offset_metrics.png", dpi=180)
    plt.close(figure)


def _plot_spatial_mask_offset_overview(
    output_dir: Path,
    samples: list[dict[str, Any]],
    z_offsets_m: np.ndarray,
) -> None:
    """Save one mask-sweep row per evaluated sequence/object."""
    if not samples or len(z_offsets_m) == 0:
        return

    plt, _sns = _setup_seaborn_plotting()

    samples = sorted(samples, key=lambda sample: sample["sequence"])
    n_rows = len(samples)
    n_cols = len(z_offsets_m)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(2.4 * n_cols, 8.0), max(2.2 * n_rows, 3.0)),
        squeeze=False,
    )

    depth_cmap = plt.get_cmap("turbo")
    invalid_color = np.array([35, 35, 35], dtype=np.uint8)
    mask_color = np.array([255, 255, 255], dtype=np.uint8)
    outline_color = np.array([0, 255, 255], dtype=np.uint8)

    def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
        normalized = np.clip(
            (depth_m - DEPTH_MIN) / max(D_MAX - DEPTH_MIN, 1e-6),
            0.0,
            1.0,
        )
        image = (depth_cmap(normalized)[..., :3] * 255).astype(np.uint8)
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        image[~valid] = invalid_color
        return image

    def mask_outline(mask: np.ndarray) -> np.ndarray:
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
        eroded = (
            -F.max_pool2d(-mask_t, kernel_size=3, stride=1, padding=1)
        )[0, 0].numpy() > 0.5
        return mask & ~eroded

    for row, sample in enumerate(samples):
        gt_vis = colorize_depth(sample["gt"])
        for col, (z_offset_m, mask) in enumerate(
            zip(z_offsets_m, sample["masks"])
        ):
            image = gt_vis.copy()
            mask_bool = mask.astype(bool, copy=False)
            image[mask_bool] = (
                0.35 * image[mask_bool].astype(np.float32)
                + 0.65 * mask_color.astype(np.float32)
            ).astype(np.uint8)
            image[mask_outline(mask_bool)] = outline_color

            axis = axes[row, col]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(
                    f"{1000.0 * float(z_offset_m):.1f} mm",
                    fontsize=9,
                )
            if col == 0:
                axis.set_ylabel(
                    f"{sample['sequence']}\nframe {sample['frame_idx']}",
                    fontsize=9,
                )

    figure.suptitle(
        "Spatial mask Z-offset sweep: GT depth with recomputed mask overlay",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "spatial_mask_z_offset_overview.png", dpi=180)
    plt.close(figure)


def _evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    print(f"\nLoading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, metadata = _checkpoint_state(checkpoint)
    model = _build_model(metadata, state, device)

    model_arch = str(metadata.get("model_arch", "MultiViewDepthNet"))
    unet_model = (
        model_arch in ("UNet", "UNet+uncertainty")
        or "predict_uncertainty" in metadata
    )
    num_views = int(metadata.get("num_views", 1)) if unet_model else int(
        _required_metadata(metadata, "num_views", 5)
    )
    view_interval = int(_required_metadata(metadata, "view_interval", 5))
    pose_view_selection = bool(_required_metadata(metadata, "pose_view_selection", False))
    pose_move_threshold = float(
        _required_metadata(metadata, "pose_move_threshold", 0.01)
    )
    num_depths = int(_required_metadata(metadata, "num_depths", 32))

    evaluation_root = args.data_dir
    sequence_dirs = _find_sequences(evaluation_root)
    if not sequence_dirs:
        raise RuntimeError(
            f"No valid sequences found at or directly under {evaluation_root}"
        )

    output_dir = args.results_folder / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = _load_event_calibration()
    depth_total = DepthAccumulator()
    boundary_total = ErrorAccumulator()
    non_boundary_total = ErrorAccumulator()
    mvc_total = MVCAccumulator()
    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    worst_frame_by_sequence: dict[str, dict[str, Any]] = {}
    spatial_mask_offset_samples: dict[str, dict[str, Any]] = {}
    frame_count = 0
    spatial_mask_z_offsets_m = np.linspace(
        args.spatial_mask_offset_min,
        args.spatial_mask_offset_max,
        args.spatial_mask_offset_steps,
        dtype=np.float64,
    )
    spatial_mask_offset_accumulators = [
        OffsetMetricAccumulator() for _ in spatial_mask_z_offsets_m
    ]
    inference_seconds = 0.0
    inference_frames = 0
    inference_batches = 0
    warmup_complete = False
    timers = EvalTimer()
    evaluation_loop_start = time.perf_counter()
    cube_center = np.asarray(
        [args.target_x, args.target_y, args.target_z], dtype=np.float64
    )
    cube_half_side = args.cube_side / 2.0

    for sequence_number, sequence_dir in enumerate(sequence_dirs, start=1):
        t_sequence_setup = time.perf_counter()
        dataset = MultiViewTableDataset(
            sequence_dir,
            calib=calib,
            num_views=num_views,
            view_interval=view_interval,
            pose_view_selection=pose_view_selection,
            pose_move_threshold=pose_move_threshold,
            num_depths=num_depths,
            use_mask=not args.no_mask,
            fill_invalid=args.fill_invalid,
            aug=MultiViewAugConfig(enabled=False),
        )
        available_frames = len(dataset)
        if args.fast_mode > 1:
            dataset.valid_indices = dataset.valid_indices[
                dataset.valid_indices % args.fast_mode == 0
            ]
            if len(dataset) == 0:
                print(
                    f"[{sequence_number}/{len(sequence_dirs)}] {sequence_dir.name}: "
                    f"skipping because --fast_mode {args.fast_mode} selected no valid targets",
                    flush=True,
                )
                continue
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        sequence_depth = DepthAccumulator()
        sequence_boundary = ErrorAccumulator()
        sequence_non_boundary = ErrorAccumulator()
        sequence_mvc = MVCAccumulator()
        history: deque[FramePrediction] = deque(maxlen=max(args.mvc_frame_offset, 1))
        timers.add(
            "sequence_dataset_and_loader_setup",
            time.perf_counter() - t_sequence_setup,
        )
        t_pose = time.perf_counter()
        pose_motion = _load_pose_motion(sequence_dir)
        timers.add("pose_motion_loading", time.perf_counter() - t_pose)

        print(
            f"[{sequence_number}/{len(sequence_dirs)}] {sequence_dir.name}: "
            f"{len(dataset)}/{available_frames} target frames"
            + (
                f" (every {args.fast_mode}th original frame)"
                if args.fast_mode > 1
                else ""
            ),
            flush=True,
        )
        with torch.inference_mode():
            loader_iter = iter(loader)
            batch_number = 0
            while True:
                t_batch_wait = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    timers.add("dataloader_wait", time.perf_counter() - t_batch_wait)
                    break
                batch_number += 1
                timers.add("dataloader_wait", time.perf_counter() - t_batch_wait)

                t_transfer = time.perf_counter()
                imgs = batch["imgs"].to(device, non_blocking=True)
                cam_mats = batch["cam_mats"].to(device, non_blocking=True)
                K = batch["K"].to(device, non_blocking=True)
                depth_values = batch["depth_values"].to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timers.add("device_transfer", time.perf_counter() - t_transfer)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_start = time.perf_counter()
                prediction_norm = model(imgs, cam_mats, K, depth_values)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_elapsed = time.perf_counter() - inference_start
                if warmup_complete:
                    inference_seconds += inference_elapsed
                    inference_frames += int(imgs.shape[0])
                    inference_batches += 1
                else:
                    # Exclude CUDA context/kernel warm-up from steady-state timing.
                    warmup_complete = True
                timers.add("model_forward", inference_elapsed, count=int(imgs.shape[0]))

                t_cpu_transfer = time.perf_counter()
                prediction = (
                    prediction_norm * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                ).detach().cpu()

                gt_batch = batch["dep_t"].float()
                mask_batch = batch["mask_t"] > 0.5
                ref_indices = batch["ref_idx"].tolist()
                cam_batch = batch["cam_mats"][:, 0].float().numpy()
                K_batch = batch["K"].float().numpy()
                timers.add(
                    "cpu_transfer",
                    time.perf_counter() - t_cpu_transfer,
                    count=int(prediction.shape[0]),
                )

                t_metrics = time.perf_counter()
                for index_in_batch in range(prediction.shape[0]):
                    pred_t = prediction[index_in_batch, 0]
                    gt_t = gt_batch[index_in_batch, 0]
                    valid_t = (
                        mask_batch[index_in_batch, 0]
                        & torch.isfinite(gt_t)
                        & torch.isfinite(pred_t)
                        & (gt_t > 0.0)
                        & (pred_t > 0.0)
                    )
                    pred_np = pred_t.numpy()
                    gt_np = gt_t.numpy()
                    valid_np = valid_t.numpy()

                    offset_masks = _cube_depth_masks_for_z_offsets(
                        gt_np,
                        cam_batch[index_in_batch],
                        K_batch[index_in_batch],
                        args.target_x,
                        args.target_y,
                        cube_half_side,
                        spatial_mask_z_offsets_m,
                    )
                    if sequence_dir.name not in spatial_mask_offset_samples:
                        spatial_mask_offset_samples[sequence_dir.name] = {
                            "sequence": sequence_dir.name,
                            "frame_idx": int(ref_indices[index_in_batch]),
                            "gt": gt_np.copy(),
                            "masks": [mask.copy() for mask in offset_masks],
                        }
                    measured_prediction_valid = (
                        np.isfinite(gt_np)
                        & (gt_np > 0.0)
                        & np.isfinite(pred_np)
                        & (pred_np > 0.0)
                    )
                    absolute_error = np.abs(pred_np - gt_np)
                    for offset_accumulator, offset_mask in zip(
                        spatial_mask_offset_accumulators, offset_masks
                    ):
                        offset_accumulator.add(
                            absolute_error[measured_prediction_valid & offset_mask]
                        )

                    depth_total.add(pred_np[valid_np], gt_np[valid_np])
                    sequence_depth.add(pred_np[valid_np], gt_np[valid_np])
                    frame_metrics = _frame_depth_metrics(pred_np, gt_np, valid_np)
                    frame_mae = float(frame_metrics["mae_m"])
                    cube_mask = _cube_depth_mask(
                        gt_np,
                        cam_batch[index_in_batch],
                        K_batch[index_in_batch],
                        cube_center,
                        cube_half_side,
                    )
                    target_events = (
                        batch["imgs"][index_in_batch, 0, :NUM_BINS]
                        .float()
                        .numpy()
                    )
                    event_activity_image = np.abs(target_events).sum(axis=0)
                    cube_event_activity = float(event_activity_image[cube_mask].sum())
                    # Do not use valid_np here: when spatial masking is enabled,
                    # valid_np already contains the precomputed inside-cube mask.
                    # Intersecting it with ~cube_mask therefore measures only
                    # disagreement around the cube boundary, not outside-cube
                    # activity. Use every pixel with measured GT depth instead.
                    gt_depth_valid = np.isfinite(gt_np) & (gt_np > 0.0)
                    outside_cube_mask = gt_depth_valid & ~cube_mask
                    outside_cube_event_activity = float(
                        event_activity_image[outside_cube_mask].sum()
                    )

                    previous_worst = worst_frame_by_sequence.get(sequence_dir.name)
                    if (
                        math.isfinite(frame_mae)
                        and (
                            previous_worst is None
                            or frame_mae > previous_worst["mae_m"]
                        )
                    ):
                        worst_frame_by_sequence[sequence_dir.name] = {
                            "sequence": sequence_dir.name,
                            "frame_idx": int(ref_indices[index_in_batch]),
                            "mae_m": frame_mae,
                            "events": (
                                batch["imgs"][index_in_batch, 0, :NUM_BINS]
                                .float()
                                .numpy()
                                .copy()
                            ),
                            "gt": gt_np.copy(),
                            "pred": pred_np.copy(),
                            "valid": valid_np.copy(),
                        }

                    boundary_mask, non_boundary_mask = _boundary_masks(
                        gt_t,
                        valid_t,
                        threshold_m=args.boundary_threshold,
                        dilation_px=args.boundary_dilation,
                    )
                    error_np = np.abs(pred_np - gt_np)
                    boundary_np = boundary_mask.numpy()
                    non_boundary_np = non_boundary_mask.numpy()
                    boundary_total.add(error_np[boundary_np])
                    non_boundary_total.add(error_np[non_boundary_np])
                    sequence_boundary.add(error_np[boundary_np])
                    sequence_non_boundary.add(error_np[non_boundary_np])

                    current = FramePrediction(
                        frame_idx=int(ref_indices[index_in_batch]),
                        pred=pred_np,
                        T_cam_from_world=cam_batch[index_in_batch],
                        K=K_batch[index_in_batch],
                    )
                    if len(history) >= args.mvc_frame_offset:
                        previous = history[-args.mvc_frame_offset]
                        forward_errors = _reproject_consistency_errors(
                            previous,
                            current,
                            args.mvc_pixel_stride,
                            args.mvc_occlusion_tolerance,
                        )
                        backward_errors = _reproject_consistency_errors(
                            current,
                            previous,
                            args.mvc_pixel_stride,
                            args.mvc_occlusion_tolerance,
                        )
                        mvc_total.pairs += 1
                        sequence_mvc.pairs += 1
                        mvc_total.add(forward_errors)
                        mvc_total.add(backward_errors)
                        sequence_mvc.add(forward_errors)
                        sequence_mvc.add(backward_errors)
                    history.append(current)

                    pose_idx = current.frame_idx
                    if pose_idx < len(pose_motion["position_m"]):
                        position = pose_motion["position_m"][pose_idx]
                        rotation = pose_motion["rotation_xyz_deg"][pose_idx]
                        arm_speed = float(pose_motion["speed_m_s"][pose_idx])
                    else:
                        position = np.full(3, np.nan, dtype=np.float64)
                        rotation = np.full(3, np.nan, dtype=np.float64)
                        arm_speed = math.nan
                    T_world_from_cam = np.linalg.inv(
                        current.T_cam_from_world.astype(np.float64)
                    )
                    camera_position = T_world_from_cam[:3, 3]
                    target_distance = float(
                        np.linalg.norm(camera_position - cube_center)
                    )

                    frame_rows.append(
                        {
                            "sequence": sequence_dir.name,
                            "frame_idx": current.frame_idx,
                            **frame_metrics,
                            "boundary_pixels": int(boundary_np.sum()),
                            "boundary_mae_m": (
                                float(error_np[boundary_np].mean())
                                if boundary_np.any()
                                else math.nan
                            ),
                            "non_boundary_pixels": int(non_boundary_np.sum()),
                            "non_boundary_mae_m": (
                                float(error_np[non_boundary_np].mean())
                                if non_boundary_np.any()
                                else math.nan
                            ),
                            "cube_event_activity": cube_event_activity,
                            "cube_depth_pixels": int(cube_mask.sum()),
                            "outside_cube_event_activity": (
                                outside_cube_event_activity
                            ),
                            "outside_cube_depth_pixels": int(
                                outside_cube_mask.sum()
                            ),
                            "pose_x_m": float(position[0]),
                            "pose_y_m": float(position[1]),
                            "pose_z_m": float(position[2]),
                            "rotation_x_deg": float(rotation[0]),
                            "rotation_y_deg": float(rotation[1]),
                            "rotation_z_deg": float(rotation[2]),
                            "arm_speed_m_s": arm_speed,
                            "target_distance_m": target_distance,
                        }
                    )
                    frame_count += 1

                timers.add(
                    "online_metrics",
                    time.perf_counter() - t_metrics,
                    count=int(prediction.shape[0]),
                )

                if args.progress_every > 0 and batch_number % args.progress_every == 0:
                    print(
                        f"  batch {batch_number}/{len(loader)}",
                        flush=True,
                    )

        sequence_metrics = sequence_depth.metrics()
        sequence_frame_rows = [
            row for row in frame_rows if row["sequence"] == sequence_dir.name
        ]
        sequence_activity = _activity_error_summary(sequence_frame_rows)
        sequence_target_distances = [
            float(row["target_distance_m"])
            for row in sequence_frame_rows
            if math.isfinite(float(row["target_distance_m"]))
        ]
        sequence_rows.append(
            {
                "sequence": sequence_dir.name,
                "frames": len(dataset),
                **sequence_metrics,
                "boundary_mae_m": sequence_boundary.metrics()["mae_m"],
                "boundary_rmse_m": sequence_boundary.metrics()["rmse_m"],
                "non_boundary_mae_m": sequence_non_boundary.metrics()["mae_m"],
                "non_boundary_rmse_m": sequence_non_boundary.metrics()["rmse_m"],
                "mvc_mae_m": sequence_mvc.metrics()["mae_m"],
                "mvc_rmse_m": sequence_mvc.metrics()["rmse_m"],
                "mvc_within_1cm": sequence_mvc.metrics()["within_1cm"],
                "mvc_within_2cm": sequence_mvc.metrics()["within_2cm"],
                "mvc_within_5cm": sequence_mvc.metrics()["within_5cm"],
                "cube_activity_vs_mae_correlation": sequence_activity[
                    "correlation_activity_vs_mae"
                ],
                "mean_cube_event_activity": sequence_activity[
                    "mean_cube_event_activity"
                ],
                "mean_cube_depth_pixels": sequence_activity[
                    "mean_cube_depth_pixels"
                ],
                "mean_target_distance_m": (
                    float(np.mean(sequence_target_distances))
                    if sequence_target_distances
                    else math.nan
                ),
            }
        )

    evaluation_loop_seconds = time.perf_counter() - evaluation_loop_start
    activity_summary = _activity_error_summary(frame_rows)
    spatial_mask_offset_rows = [
        accumulator.metrics(
            float(z_offset_m),
            float(z_offset_m + cube_half_side),
        )
        for accumulator, z_offset_m in zip(
            spatial_mask_offset_accumulators, spatial_mask_z_offsets_m
        )
    ]
    model_ms_per_frame = (
        1000.0 * inference_seconds / inference_frames
        if inference_frames > 0 else math.nan
    )
    model_fps = (
        inference_frames / inference_seconds
        if inference_seconds > 0 else math.nan
    )
    evaluation_loop_ms_per_frame = (
        1000.0 * evaluation_loop_seconds / frame_count
        if frame_count > 0 else math.nan
    )
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_val_l1_m": metadata.get("val_l1"),
        "checkpoint_val_p95_m": metadata.get("val_p95"),
        "checkpoint_val_l1_worst10_m": metadata.get("val_l1_worst10"),
        "evaluation_root": str(evaluation_root.resolve()),
        "sequence_count": len(sequence_dirs),
        "frame_count": frame_count,
        "performance": {
            "inference_batch_size": args.batch_size,
            "timed_inference_batches": inference_batches,
            "timed_inference_frames": inference_frames,
            "model_inference_seconds": inference_seconds,
            "model_inference_ms_per_frame": model_ms_per_frame,
            "model_inference_fps": model_fps,
            "evaluation_loop_seconds": evaluation_loop_seconds,
            "evaluation_loop_ms_per_frame": evaluation_loop_ms_per_frame,
            "model_timing_scope": (
                "synchronized forward pass only; excludes data loading, host/device "
                "transfer, metrics, temporal consistency, CSV output, and plotting"
            ),
            "evaluation_loop_scope": (
                "data iteration, host/device transfer, forward pass, and online "
                "metrics; excludes final CSV/JSON writing and plotting"
            ),
        },
        "model_configuration": {
            "model_arch": model_arch,
            "num_views": num_views,
            "view_interval": view_interval,
            "pose_view_selection": pose_view_selection,
            "pose_move_threshold_m": pose_move_threshold,
            "num_depths": num_depths,
            "fine_depths": metadata.get("fine_depths"),
            "feature_encoder": metadata.get("feature_encoder"),
            "decoder_type": metadata.get("decoder_type", "auto"),
        },
        "evaluation_configuration": {
            "use_spatial_mask": not args.no_mask,
            "fill_invalid": args.fill_invalid,
            "frame_step": args.fast_mode,
            "boundary_threshold_m": args.boundary_threshold,
            "boundary_dilation_px": args.boundary_dilation,
            "mvc_frame_offset": args.mvc_frame_offset,
            "mvc_pixel_stride": args.mvc_pixel_stride,
            "mvc_occlusion_tolerance_m": args.mvc_occlusion_tolerance,
            "mvc_bidirectional": True,
            "mvc_visibility": (
                "nearest projected source point per target pixel (z-buffer), "
                "excluding points behind target depth beyond the occlusion tolerance"
            ),
            "event_activity_region": (
                "sum(abs(target event voxels)) at pixels whose GT-depth point "
                "backprojects inside the configured world-frame cube"
            ),
            "event_activity_error_region": (
                "unchanged evaluation/training valid mask; cube mask is not "
                "applied to depth error"
            ),
            "cube_center_world_m": cube_center.tolist(),
            "cube_side_m": args.cube_side,
            "spatial_mask_offset_min_m": args.spatial_mask_offset_min,
            "spatial_mask_offset_max_m": args.spatial_mask_offset_max,
            "spatial_mask_offset_steps": args.spatial_mask_offset_steps,
            "spatial_mask_offset_definition": (
                "offset is the world-frame cube-bottom Z; cube center Z equals "
                "offset + cube_side/2; recomputed mask replaces the stored mask"
            ),
        },
        "depth_metrics": depth_total.metrics(),
        "boundary_metrics": {
            "boundary": boundary_total.metrics(),
            "non_boundary": non_boundary_total.metrics(),
        },
        "multiview_consistency": mvc_total.metrics(),
        "cube_event_activity_vs_error": activity_summary,
        "spatial_mask_z_offset_metrics": spatial_mask_offset_rows,
    }

    t_output_writing = time.perf_counter()
    _write_csv(output_dir / "per_frame_metrics.csv", frame_rows)
    _write_csv(output_dir / "per_sequence_metrics.csv", sequence_rows)
    _write_csv(
        output_dir / "spatial_mask_z_offset_metrics.csv",
        spatial_mask_offset_rows,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite_or_none(summary), handle, indent=2)
        handle.write("\n")
    _write_summary_text(output_dir / "summary.txt", summary)
    timers.add("output_writing", time.perf_counter() - t_output_writing)

    t_plotting = time.perf_counter()
    _plot_results(output_dir, summary, frame_rows, sequence_rows)
    _plot_spatial_mask_offset_metrics(output_dir, spatial_mask_offset_rows)
    _plot_spatial_mask_offset_overview(
        output_dir,
        list(spatial_mask_offset_samples.values()),
        spatial_mask_z_offsets_m,
    )
    worst_frame_samples = list(worst_frame_by_sequence.values())
    _plot_worst_frames(output_dir, worst_frame_samples)
    timers.add("plotting", time.perf_counter() - t_plotting)
    eval_time_rows = timers.rows(frame_count)
    _write_eval_times(output_dir, eval_time_rows, frame_count)

    print((output_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    if worst_frame_samples:
        print(
            f"Worst-frame overview: "
            f"{(output_dir / 'worst_error_overview.png').resolve()}",
            flush=True,
        )
    print(f"Evaluation timing: {(output_dir / 'eval_times').resolve()}", flush=True)
    print(f"Results written to: {output_dir.resolve()}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net-table, basic, legacy, or modern MVS checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more supported depth-model checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("../data/real/eval"),
        help=(
            "Final evaluation-data folder: either one sequence or a folder "
            "containing multiple sequence folders."
        ),
    )
    parser.add_argument(
        "--results_folder",
        type=Path,
        default=Path("results"),
        help="Root output directory. Each checkpoint gets its own subdirectory.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--fast_mode",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Evaluate only target frames whose original frame index is divisible "
            "by N; source views required by each selected target are still loaded."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Ignore spatial_mask.h5; pixels with GT depth <= 0 remain invalid.",
    )
    parser.add_argument(
        "--fill_invalid",
        action="store_true",
        help="Fill invalid GT pixels using the table-plane prior before evaluation.",
    )
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=0.02,
        help="Minimum neighbouring GT depth jump in metres defining a boundary.",
    )
    parser.add_argument(
        "--boundary_dilation",
        type=int,
        default=2,
        help="Boundary-region dilation radius in pixels.",
    )
    parser.add_argument(
        "--mvc_frame_offset",
        type=int,
        default=1,
        help="Compare target frames this many evaluable samples apart.",
    )
    parser.add_argument(
        "--mvc_pixel_stride",
        type=int,
        default=1,
        help="Subsample source pixels for multi-view consistency; 1 uses every pixel.",
    )
    parser.add_argument(
        "--mvc_occlusion_tolerance",
        type=float,
        default=0.02,
        help="Discard reprojected points more than this many metres behind target depth.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Print progress every N batches; 0 disables batch progress.",
    )
    parser.add_argument(
        "--cube_side",
        type=float,
        default=SPATIAL_CUBE_SIDE,
        help="World-frame cube side length used only for event-activity filtering.",
    )
    parser.add_argument("--target_x", type=float, default=SPATIAL_TARGET_X)
    parser.add_argument("--target_y", type=float, default=SPATIAL_TARGET_Y)
    parser.add_argument("--target_z", type=float, default=SPATIAL_TARGET_Z)
    parser.add_argument(
        "--spatial_mask_offset_min",
        type=float,
        default=-0.02,
        help="Minimum cube-bottom Z offset in metres for mask-sweep metrics.",
    )
    parser.add_argument(
        "--spatial_mask_offset_max",
        type=float,
        default=0.02,
        help="Maximum cube-bottom Z offset in metres for mask-sweep metrics.",
    )
    parser.add_argument(
        "--spatial_mask_offset_steps",
        type=int,
        default=11,
        help="Number of evenly spaced mask offsets from minimum to maximum.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be > 0")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.fast_mode <= 0:
        parser.error("--fast_mode must be > 0")
    if args.boundary_threshold <= 0:
        parser.error("--boundary_threshold must be > 0")
    if args.boundary_dilation < 0:
        parser.error("--boundary_dilation must be >= 0")
    if args.mvc_frame_offset <= 0:
        parser.error("--mvc_frame_offset must be > 0")
    if args.mvc_pixel_stride <= 0:
        parser.error("--mvc_pixel_stride must be > 0")
    if args.mvc_occlusion_tolerance < 0:
        parser.error("--mvc_occlusion_tolerance must be >= 0")
    if args.cube_side <= 0:
        parser.error("--cube_side must be > 0")
    if args.spatial_mask_offset_min >= args.spatial_mask_offset_max:
        parser.error("--spatial_mask_offset_min must be < --spatial_mask_offset_max")
    if args.spatial_mask_offset_steps < 2:
        parser.error("--spatial_mask_offset_steps must be >= 2")
    for checkpoint in args.checkpoint:
        if not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
    return args


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("[ERROR] --device cuda requested, but CUDA is unavailable.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.results_folder.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    for checkpoint_path in args.checkpoint:
        _evaluate_checkpoint(checkpoint_path, args, device)


if __name__ == "__main__":
    main()
