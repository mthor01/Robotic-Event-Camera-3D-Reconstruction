#!/usr/bin/env python3
"""Evaluate U-Net and multiview checkpoints.

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

_CAMERA_DATA_DIR = Path(__file__).resolve().parent.parent / "camera_data"

from multiview import (
    MultiViewAugConfig,
    ModernMVSNet,
    MultiViewTableDataset,
)
from helpers import (
    INTRINSICS_TRANSFORM,
    depth_cube_mask,
    depth_cube_masks_for_bottom_offsets,
    find_precomputed_sequences,
    load_event_calibration,
)
from config import (
    DEPTH_MIN,
    D_MAX,
    NUM_BINS,
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
DEPTH_REGION_LABELS = {
    "whole_image": "Whole Frame",
    "lower_cube": "Workspace Cube",
    "upper_cube": "Raised Cube",
}
PER_SEQUENCE_REGION_SPECS = (
    ("whole_frame", "Whole Frame", "mae_m", "rmse_m"),
    (
        "workspace_cube",
        "Workspace Cube",
        "lower_cube_mae_m",
        "lower_cube_rmse_m",
    ),
    (
        "raised_cube",
        "Raised Cube",
        "upper_cube_mae_m",
        "upper_cube_rmse_m",
    ),
)
DOMAIN_SUMMARY_REGION_KEYS = {
    "whole_frame": "whole_image",
    "workspace_cube": "lower_cube",
    "raised_cube": "upper_cube",
}
MVC_THRESHOLDS_M = (0.01, 0.02, 0.05)
SPATIAL_MASK_VERTICAL_SHIFT_M = 0.005
BOUNDARY_SPATIAL_MASK_OFFSET_M = 0.01 + SPATIAL_MASK_VERTICAL_SHIFT_M
BOUNDARY_VISUALIZATION_SAMPLE_COUNT = 5
QUALITATIVE_RANDOM_SEED = 20260813
LEGEND_FONT_SIZE = 12
LEGEND_TITLE_FONT_SIZE = 13
DIAGNOSTIC_LEGEND_FONT_SIZE = 15
DIAGNOSTIC_LEGEND_TITLE_FONT_SIZE = 16
BOUNDARY_VISUALIZATION_RANDOM_SEED = 0


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
    """Load pose, timestamps, normalized recording time, and motion speed."""
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

    normalized_time = np.zeros(n_frames, dtype=np.float64)
    if n_frames >= 2:
        duration_s = timestamps_s[-1] - timestamps_s[0]
        if np.isfinite(duration_s) and duration_s > 1e-9:
            normalized_time = np.clip(
                (timestamps_s - timestamps_s[0]) / duration_s,
                0.0,
                1.0,
            )
        else:
            normalized_time = np.linspace(0.0, 1.0, n_frames)

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
        "normalized_time": normalized_time,
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
    model_arch = str(metadata.get("model_arch", "ModernMVSNet"))
    model_cls = ModernMVSNet
    if model_arch == "ModernMVSNet":
        pass
    elif model_arch in ("UNet", "UNet+uncertainty", "RecurrentUNet") or "predict_uncertainty" in metadata:
        from unet_models import RecurrentUNet, UNet, UncertaintyUNet

        predicts_uncertainty = bool(metadata.get("predict_uncertainty", False))
        recurrent = bool(metadata.get("recurrent", False)) or model_arch == "RecurrentUNet"
        if recurrent:
            unet_cls = RecurrentUNet
        else:
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
                view_valid_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                del camera_matrices, intrinsics, depth_values, view_valid_mask
                if recurrent:
                    output = self.depth_model(images)
                else:
                    batch, views, channels, height, width = images.shape
                    fused = images.reshape(batch, views * channels, height, width)
                    output = self.depth_model(fused)
                return output[0] if isinstance(output, tuple) else output

        return EarlyFusionUNetEvaluationAdapter(unet).to(device).eval()
    else:
        raise ValueError(f"Unsupported multiview checkpoint architecture: {model_arch}")

    fine_window = float(_required_metadata(metadata, "fine_window", 0.08))
    fine_offset_radius = float(
        _required_metadata(metadata, "fine_offset_radius", 2.0)
    )
    model = model_cls(
        in_ch=int(_required_metadata(metadata, "in_ch", NUM_BINS + 1)),
        base=base,
        feature_ch=int(_required_metadata(metadata, "feature_channels", base * 4)),
        cost_base=int(_required_metadata(metadata, "cost_channels", max(base // 2, 8))),
        fine_depths=int(_required_metadata(metadata, "fine_depths", 5)),
        fine_window=fine_window,
        fine_offset_radius=fine_offset_radius,
        fine_window_min=float(
            metadata.get("fine_window_min", 0.25 * fine_window * fine_offset_radius)
        ),
        fine_window_max=float(
            metadata.get("fine_window_max", fine_window * fine_offset_radius)
        ),
        learned_fine_window=bool(_required_metadata(metadata, "learned_fine_window", False)),
        reference_channels=int(metadata.get("reference_channels", 0)),
        coarse_cost_channels=int(metadata.get("coarse_cost_channels", 0)),
        fine_cost_channels=int(metadata.get("fine_cost_channels", 0)),
        refiner_channels=int(metadata.get("refiner_channels", 0)),
        refiner_max_residual_m=metadata.get("refiner_max_residual_m"),
        refiner_reference_input=not bool(
            metadata.get("no_refiner_reference_input", False)
        ),
        no_2d_refinement=bool(metadata.get("no_2d_refinement", False)),
        coarse_hourglass_levels=int(metadata.get("coarse_hourglass_levels", 2)),
        fine_hourglass_levels=int(metadata.get("fine_hourglass_levels", 2)),
        fpn_dropout=float(metadata.get("fpn_dropout", 0.0)),
        reference_dropout=float(metadata.get("reference_dropout", 0.0)),
        hourglass_dropout=float(metadata.get("hourglass_dropout", 0.0)),
        drop_path_rate=float(metadata.get("drop_path_rate", 0.0)),
        middle_depths=int(metadata.get("middle_depths", 8)),
        middle_window=float(metadata.get("middle_window", 0.12)),
        middle_cost_channels=int(metadata.get("middle_cost_channels", 0)),
        middle_hourglass_levels=int(metadata.get("middle_hourglass_levels", 2)),
        middle_feature_channels=int(metadata.get("middle_feature_channels", 0)),
        fine_feature_channels=int(metadata.get("fine_feature_channels", 0)),
        fpn_lateral_convolutions=not bool(
            metadata.get("no_fpn_lateral_convolutions", False)
        ),
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


def _worst_fraction_l1(errors: np.ndarray, fraction: float = 0.10) -> float:
    """Return the mean absolute error among the largest error fraction."""
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return math.nan
    count = max(1, int(math.ceil(fraction * values.size)))
    start = values.size - count
    return float(np.partition(values, start)[start:].mean())


def _boundary_masks(
    domain: torch.Tensor,
    evaluation_valid: torch.Tensor,
    dilation_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split evaluated pixels by proximity to the edge of a valid domain.

    A boundary starts on the inside of ``domain`` wherever a four-connected
    neighbour is outside it.  For spatially masked evaluation, ``domain`` is
    defined GT intersected with the spatial mask; consequently undefined GT
    and pixels outside the spatial mask both produce a boundary.
    """
    domain = domain[None, None].bool()
    evaluation_mask = evaluation_valid[None, None].bool()
    edge = torch.zeros_like(domain)

    horizontal_transition = domain[:, :, :, 1:] != domain[:, :, :, :-1]
    edge[:, :, :, 1:] |= horizontal_transition & domain[:, :, :, 1:]
    edge[:, :, :, :-1] |= horizontal_transition & domain[:, :, :, :-1]

    vertical_transition = domain[:, :, 1:, :] != domain[:, :, :-1, :]
    edge[:, :, 1:, :] |= vertical_transition & domain[:, :, 1:, :]
    edge[:, :, :-1, :] |= vertical_transition & domain[:, :, :-1, :]

    if dilation_px > 0:
        kernel = 2 * dilation_px + 1
        edge = F.max_pool2d(edge.float(), kernel, stride=1, padding=dilation_px) > 0
    boundary = edge & evaluation_mask
    non_boundary = (~edge) & evaluation_mask
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


def _depth_region_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one machine-readable row for each spatial evaluation region."""
    metrics_by_region = summary["depth_metrics_by_region"]
    return [
        {
            "region": region,
            "label": DEPTH_REGION_LABELS[region],
            **metrics_by_region[region],
        }
        for region in DEPTH_REGION_LABELS
    ]


def _pose_layout_directory_indicator(allow_unbalanced: bool) -> str:
    """Return the filename-safe prefix describing the evaluation policy."""
    return (
        "unbalanced_pose_views_allowed"
        if allow_unbalanced
        else "unbalanced_pose_views_disallowed"
    )


def _model_display_name(is_unet: bool, num_views: int) -> str:
    """Return the architecture name used in plots and comparison tables."""
    if not is_unet:
        return "MVS"
    return "Single-View U-Net" if num_views == 1 else "Multi-View U-Net"


def _write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    boundary = summary["boundary_metrics"]
    masked_boundary = summary["boundary_metrics_spatial_mask_plus_1cm"]
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
        "Depth metrics by spatial region",
    ]
    for row in _depth_region_rows(summary):
        lines.append(f"  {row['label']}")
        for name in DEPTH_METRIC_NAMES:
            lines.append(f"    {name}: {_format_metric(name, row[name])}")
        lines.append(f"    valid_pixels: {row['valid_pixels']}")
    lines.extend(
        [
            "",
            "Depth-boundary metrics",
            f"  boundary_pixels: {boundary['boundary']['count']}",
            f"  boundary_mae_m: {_format_metric('mae_m', boundary['boundary']['mae_m'])}",
            f"  boundary_rmse_m: {_format_metric('rmse_m', boundary['boundary']['rmse_m'])}",
            f"  non_boundary_pixels: {boundary['non_boundary']['count']}",
            f"  non_boundary_mae_m: {_format_metric('mae_m', boundary['non_boundary']['mae_m'])}",
            f"  non_boundary_rmse_m: {_format_metric('rmse_m', boundary['non_boundary']['rmse_m'])}",
            "",
            "Depth-boundary metrics inside +1.5 cm spatial mask",
            f"  spatial_mask_cube_bottom_z_offset_m: "
            f"{masked_boundary['z_offset_m']:.6f}",
            f"  boundary_pixels: {masked_boundary['boundary']['count']}",
            f"  boundary_mae_m: "
            f"{_format_metric('mae_m', masked_boundary['boundary']['mae_m'])}",
            f"  boundary_rmse_m: "
            f"{_format_metric('rmse_m', masked_boundary['boundary']['rmse_m'])}",
            f"  non_boundary_pixels: {masked_boundary['non_boundary']['count']}",
            f"  non_boundary_mae_m: "
            f"{_format_metric('mae_m', masked_boundary['non_boundary']['mae_m'])}",
            f"  non_boundary_rmse_m: "
            f"{_format_metric('rmse_m', masked_boundary['non_boundary']['rmse_m'])}",
            "",
            "Multi-view consistency",
            "  enabled: "
            f"{summary['evaluation_configuration'].get('mvc_enabled', True)}",
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
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
        },
    )
    return plt, sns


def _sequence_display_map(sequence_names: list[str]) -> dict[str, str]:
    """Map raw sequence directory names to presentation labels 1..N."""
    def sort_key(name: str) -> tuple[int, Any]:
        return (0, int(name)) if name.isdigit() else (1, name)

    ordered = sorted({str(name) for name in sequence_names}, key=sort_key)
    return {name: f"Sequence {index}" for index, name in enumerate(ordered, 1)}


def _plot_uncertainty_vs_prediction_error(
    output_dir: Path,
    uncertainty: list[float],
    absolute_error_m: list[float],
    domain_slug: str,
    domain_label: str,
) -> None:
    """Plot predicted confidence against absolute depth error."""
    if not uncertainty:
        return
    x = 1.0 - np.asarray(uncertainty, dtype=np.float64)
    y_cm = 100.0 * np.asarray(absolute_error_m, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y_cm)
    x, y_cm = x[finite], y_cm[finite]
    if x.size < 2:
        return

    plt, sns = _setup_seaborn_plotting()
    figure, axis = plt.subplots(figsize=(8.2, 5.8))
    sns.scatterplot(
        x=x,
        y=y_cm,
        s=24,
        alpha=0.38,
        linewidth=0,
        color=sns.color_palette("deep")[0],
        rasterized=True,
        ax=axis,
    )
    axis.set(
        xlabel="Predicted Confidence",
        ylabel="Absolute Depth Error [cm]",
        title=f"Predicted Confidence vs. Depth Error: {domain_label}",
        xlim=(-0.05, 1.05),
    )
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    figure.tight_layout()
    figure.savefig(
        output_dir / f"uncertainty_vs_prediction_error_{domain_slug}.png",
        dpi=220,
    )
    figure.savefig(
        output_dir / f"uncertainty_vs_prediction_error_{domain_slug}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _add_uncertainty_reservoir_samples(
    state: dict[str, Any],
    uncertainty: np.ndarray,
    absolute_error_m: np.ndarray,
    rng: np.random.Generator,
    capacity: int = 20000,
    per_frame_limit: int = 4096,
) -> None:
    """Add paired confidence/error samples to a bounded reservoir."""
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    absolute_error_m = np.asarray(absolute_error_m, dtype=np.float64).reshape(-1)
    finite = np.isfinite(uncertainty) & np.isfinite(absolute_error_m)
    uncertainty = uncertainty[finite]
    absolute_error_m = absolute_error_m[finite]
    if uncertainty.size > per_frame_limit:
        selected = rng.choice(
            uncertainty.size, size=per_frame_limit, replace=False
        )
        uncertainty = uncertainty[selected]
        absolute_error_m = absolute_error_m[selected]
    for uncertainty_value, error_value in zip(uncertainty, absolute_error_m):
        state["seen"] += 1
        if len(state["uncertainty"]) < capacity:
            state["uncertainty"].append(float(uncertainty_value))
            state["error"].append(float(error_value))
            continue
        replacement = int(rng.integers(0, state["seen"]))
        if replacement < capacity:
            state["uncertainty"][replacement] = float(uncertainty_value)
            state["error"][replacement] = float(error_value)


def _set_inverse_percentage_axis(axis: Any, values: list[float]) -> None:
    """Use a readable zero-based scale for inverse near-perfect percentages."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        axis.set_ylim(0.0, 1.0)
        return
    maximum = max(0.0, float(finite.max()))
    upper = max(0.1, 1.15 * maximum)
    axis.set_ylim(0.0, 100.0 if upper >= 100.0 else upper)


def _equal_count_mean_curve(
    rows: list[dict[str, Any]],
    x_field: str,
    y_field: str = "mae_m",
    bins: int = 12,
    y_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted equal-count-bin means without exposing frame scatter."""
    if not rows or x_field not in rows[0] or y_field not in rows[0]:
        return np.empty(0), np.empty(0)
    x = np.asarray([row.get(x_field, math.nan) for row in rows], dtype=np.float64)
    y = np.asarray([row.get(y_field, math.nan) for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite] * y_scale
    if x.size == 0:
        return np.empty(0), np.empty(0)
    groups = [
        indices
        for indices in np.array_split(np.argsort(x), min(bins, x.size))
        if indices.size
    ]
    mean_x = np.asarray([float(x[indices].mean()) for indices in groups])
    mean_y = np.asarray([float(y[indices].mean()) for indices in groups])
    order = np.argsort(mean_x)
    return mean_x[order], mean_y[order]


def _fixed_width_mean_curve(
    x: np.ndarray,
    y: np.ndarray,
    interval: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean y in fixed-width x bins anchored at zero."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.empty(0), np.empty(0)
    bin_indices = np.floor(x / interval).astype(np.int64)
    populated_bins = np.unique(bin_indices)
    centers = (populated_bins.astype(np.float64) + 0.5) * interval
    means = np.asarray(
        [float(y[bin_indices == index].mean()) for index in populated_bins]
    )
    return centers, means


def _fixed_width_mean_curve_from_rows(
    rows: list[dict[str, Any]],
    x_field: str,
    interval: float,
    y_field: str = "mae_m",
    y_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed-width mean curve from per-frame metric rows."""
    if not rows or x_field not in rows[0] or y_field not in rows[0]:
        return np.empty(0), np.empty(0)
    x = np.asarray([row.get(x_field, math.nan) for row in rows], dtype=np.float64)
    y = np.asarray([row.get(y_field, math.nan) for row in rows], dtype=np.float64)
    return _fixed_width_mean_curve(x, y * y_scale, interval)


def _column_mapping(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row dictionaries to Seaborn's version-compatible wide mapping."""
    if not rows:
        return {}
    fields = list(rows[0])
    return {field: [row.get(field) for row in rows] for field in fields}


def _normalized_time_error_curves(
    frame_rows: list[dict[str, Any]],
    bins: int = 25,
    error_field: str = "mae_m",
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Bin frame MAE over recording progress and weight recordings equally."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    centers = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    sequence_curves: dict[str, np.ndarray] = {}
    sequence_names = sorted({str(row["sequence"]) for row in frame_rows})
    for sequence in sequence_names:
        sequence_rows = [
            row for row in frame_rows if str(row["sequence"]) == sequence
        ]
        progress = np.asarray(
            [row.get("normalized_time", math.nan) for row in sequence_rows],
            dtype=np.float64,
        )
        error_cm = np.asarray(
            [
                100.0 * float(row.get(error_field, math.nan))
                for row in sequence_rows
            ],
            dtype=np.float64,
        )
        finite = np.isfinite(progress) & np.isfinite(error_cm)
        progress = np.clip(progress[finite], 0.0, 1.0)
        error_cm = error_cm[finite]
        curve = np.full(bins, np.nan, dtype=np.float64)
        if progress.size:
            bin_indices = np.minimum((progress * bins).astype(np.int64), bins - 1)
            for bin_index in np.unique(bin_indices):
                curve[bin_index] = float(error_cm[bin_indices == bin_index].mean())
        sequence_curves[sequence] = curve

    if not sequence_curves:
        empty = np.full(bins, np.nan, dtype=np.float64)
        return centers, {}, empty, empty.copy(), np.zeros(bins, dtype=np.int64)

    stacked = np.stack(list(sequence_curves.values()), axis=0)
    finite = np.isfinite(stacked)
    recording_counts = finite.sum(axis=0)
    mean = np.full(bins, np.nan, dtype=np.float64)
    std = np.full(bins, np.nan, dtype=np.float64)
    populated = recording_counts > 0
    mean[populated] = np.nansum(stacked[:, populated], axis=0) / recording_counts[
        populated
    ]
    for bin_index in np.flatnonzero(populated):
        values = stacked[finite[:, bin_index], bin_index]
        std[bin_index] = float(values.std())
    return centers, sequence_curves, mean, std, recording_counts


def _plot_normalized_time_error(
    output_dir: Path,
    frame_rows: list[dict[str, Any]],
    domain_slug: str,
    domain_label: str,
    error_field: str,
    bins: int = 25,
) -> None:
    """Plot per-recording and equally weighted mean MAE over normalized time."""
    centers, sequence_curves, mean, std, recording_counts = (
        _normalized_time_error_curves(
            frame_rows, bins=bins, error_field=error_field
        )
    )
    finite_mean = np.isfinite(mean)
    if not sequence_curves or not finite_mean.any():
        return

    curve_rows = [
        {
            "normalized_time": float(centers[index]),
            "mean_mae_cm": float(mean[index]),
            "std_mae_cm": float(std[index]),
            "recordings": int(recording_counts[index]),
        }
        for index in np.flatnonzero(finite_mean)
    ]
    _write_csv(
        output_dir / f"error_over_normalized_time_{domain_slug}.csv",
        curve_rows,
    )

    plt, sns = _setup_seaborn_plotting()
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sequence_labels = _sequence_display_map(list(sequence_curves))
    colors = sns.color_palette("husl", n_colors=max(1, len(sequence_curves)))
    for color, (sequence, curve) in zip(colors, sequence_curves.items()):
        finite = np.isfinite(curve)
        if finite.any():
            axis.plot(
                centers[finite],
                curve[finite],
                color=color,
                linewidth=1.0,
                alpha=0.32,
                label=sequence_labels[sequence],
            )
    axis.plot(
        centers[finite_mean],
        mean[finite_mean],
        color="black",
        linewidth=2.8,
        label="Equal-Recording Mean",
        zorder=10,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Normalized Recording Time")
    axis.set_ylabel("Per-Frame MAE [cm]")
    axis.set_title(f"Depth Error Over Normalized Recording Time: {domain_label}")
    axis.legend(
        title="Recording",
        fontsize=DIAGNOSTIC_LEGEND_FONT_SIZE,
        title_fontsize=DIAGNOSTIC_LEGEND_TITLE_FONT_SIZE,
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"error_over_normalized_time_{domain_slug}.png",
        dpi=180,
    )
    plt.close(figure)


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
            "  event-activity summaries, pose bookkeeping, and MVC reprojection",
            "  when MVC is enabled.",
            "  output_writing and plotting are after the evaluation loop.",
        ]
    )
    (output_dir / "eval_times.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot_results(
    output_dir: Path,
    summary: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    sequence_rows: list[dict[str, Any]],
) -> None:
    plt, sns = _setup_seaborn_plotting()

    error_names = ["abs_rel", "sq_rel_m", "mae_m", "rmse_m", "rmse_log"]
    delta_names = ["delta_1", "delta_2", "delta_3"]
    delta_labels = [r"$\delta<1.25$", r"$\delta<1.25^2$", r"$\delta<1.25^3$"]
    for domain_slug, domain_label, _, _ in PER_SEQUENCE_REGION_SPECS:
        region_key = DOMAIN_SUMMARY_REGION_KEYS[domain_slug]
        depth = summary["depth_metrics_by_region"][region_key]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.barplot(
            x=error_names,
            y=[depth[name] for name in error_names],
            ax=axes[0],
            hue=error_names,
            palette="Blues_d",
            legend=False,
        )
        axes[0].set_title(f"Depth Errors: {domain_label}")
        axes[0].tick_params(axis="x", rotation=30)
        delta_failure_percentages = [
            100.0 * (1.0 - float(depth[name])) for name in delta_names
        ]
        sns.barplot(
            x=delta_labels,
            y=delta_failure_percentages,
            ax=axes[1],
            hue=delta_labels,
            palette="Greens_d",
            legend=False,
        )
        _set_inverse_percentage_axis(axes[1], delta_failure_percentages)
        axes[1].set_ylabel("Pixels Outside Threshold [%]")
        axes[1].set_title(f"Threshold Failure Rate: {domain_label}")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"depth_metrics_{domain_slug}.png", dpi=180
        )
        plt.close(figure)

    region_error_rows = [
        {
            "region": row["label"],
            "metric": metric,
            "error_cm": 100.0 * float(row[field]),
        }
        for row in _depth_region_rows(summary)
        for metric, field in (("MAE", "mae_m"), ("RMSE", "rmse_m"))
    ]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(
        data=_column_mapping(region_error_rows),
        x="region",
        y="error_cm",
        hue="metric",
        ax=axis,
        palette="deep",
    )
    axis.set_xlabel("")
    axis.set_ylabel("Depth Error [cm]")
    axis.set_title("MAE and RMSE by Evaluation Region")
    axis.tick_params(axis="x", rotation=12)
    axis.legend(title="Metric")
    figure.tight_layout()
    figure.savefig(output_dir / "mae_rmse_by_region.png", dpi=180)
    plt.close(figure)

    mae_region_rows = [
        row for row in region_error_rows if row["metric"] == "MAE"
    ]
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    sns.barplot(
        data=_column_mapping(mae_region_rows),
        x="region",
        y="error_cm",
        hue="region",
        ax=axis,
        palette="deep",
        legend=False,
    )
    axis.set_xlabel("")
    axis.set_ylabel("MAE [cm]")
    axis.set_title("MAE by Evaluation Region")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    figure.savefig(output_dir / "mae_by_region.png", dpi=180)
    plt.close(figure)

    boundary = summary["boundary_metrics"]
    mvc = summary["multiview_consistency"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    boundary_labels = ["Boundary MAE", "Non-Boundary MAE"]
    sns.barplot(
        x=boundary_labels,
        y=[boundary["boundary"]["mae_m"], boundary["non_boundary"]["mae_m"]],
        ax=axes[0],
        hue=boundary_labels,
        palette="flare",
        legend=False,
    )
    axes[0].set_ylabel("Error [m]")
    axes[0].set_title("Depth-Boundary Error")
    mvc_labels = ["<1 cm", "<2 cm", "<5 cm"]
    mvc_enabled = summary["evaluation_configuration"].get("mvc_enabled", True)
    if mvc_enabled:
        sns.barplot(
            x=mvc_labels,
            y=(mvc_inconsistency_percentages := [
                100.0 * (1.0 - float(mvc["within_1cm"])),
                100.0 * (1.0 - float(mvc["within_2cm"])),
                100.0 * (1.0 - float(mvc["within_5cm"])),
            ]),
            ax=axes[1],
            hue=mvc_labels,
            palette="Purples_d",
            legend=False,
        )
        _set_inverse_percentage_axis(axes[1], mvc_inconsistency_percentages)
        axes[1].set_ylabel("Inconsistent Correspondences [%]")
        axes[1].set_title("Multi-View Inconsistency Thresholds")
    else:
        axes[1].set_axis_off()
        axes[1].text(
            0.5,
            0.5,
            "Multi-view consistency disabled\n(--no_mvc)",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(output_dir / "boundary_and_consistency.png", dpi=180)
    plt.close(figure)

    boundary_metrics_by_region = summary["boundary_metrics_by_region"]
    for domain_slug, _, _, _ in PER_SEQUENCE_REGION_SPECS:
        domain_boundary = boundary_metrics_by_region[domain_slug]
        boundary_labels = ["Boundary", "Non-Boundary"]
        figure, axis = plt.subplots(figsize=(7.5, 5.0))
        sns.barplot(
            x=boundary_labels,
            y=[
                100.0 * float(domain_boundary["boundary"]["mae_m"]),
                100.0 * float(domain_boundary["non_boundary"]["mae_m"]),
            ],
            ax=axis,
            hue=boundary_labels,
            palette="flare",
            legend=False,
        )
        axis.set_xlabel("Region")
        axis.set_ylabel("MAE [cm]")
        figure.tight_layout()
        figure.savefig(output_dir / f"boundary_error_{domain_slug}.png", dpi=180)
        plt.close(figure)

    if sequence_rows:
        sequence_labels = _sequence_display_map(
            [str(row["sequence"]) for row in sequence_rows]
        )
        figure_width = max(8.0, 0.75 * len(sequence_rows))
        for filename_region, region_label, mae_field, rmse_field in (
            PER_SEQUENCE_REGION_SPECS
        ):
            plot_rows = []
            for row in sequence_rows:
                sequence_name = sequence_labels[str(row["sequence"])]
                plot_rows.extend(
                    (
                        {
                            "sequence": sequence_name,
                            "metric": "MAE",
                            "error_cm": 100.0 * float(row[mae_field]),
                        },
                        {
                            "sequence": sequence_name,
                            "metric": "RMSE",
                            "error_cm": 100.0 * float(row[rmse_field]),
                        },
                    )
                )
            figure, axis = plt.subplots(figsize=(figure_width, 5.5))
            sns.barplot(
                data=_column_mapping(plot_rows),
                x="sequence",
                y="error_cm",
                hue="metric",
                ax=axis,
                palette="deep",
            )
            axis.tick_params(axis="x", rotation=45)
            axis.set_xlabel("")
            axis.set_ylabel("Depth Error [cm]")
            axis.set_title(f"Depth Error per Sequence: {region_label}")
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                output_dir / f"error_per_sequence_{filename_region}.png",
                dpi=180,
            )
            if filename_region == "whole_frame":
                figure.savefig(output_dir / "error_per_sequence.png", dpi=180)
            plt.close(figure)

        all_domain_mae_rows = [
            {
                "sequence": sequence_labels[str(row["sequence"])].removeprefix(
                    "Sequence "
                ),
                "domain": region_label,
                "mae_cm": 100.0 * float(row[mae_field]),
            }
            for row in sequence_rows
            for _, region_label, mae_field, _ in PER_SEQUENCE_REGION_SPECS
        ]
        figure, axis = plt.subplots(figsize=(figure_width, 5.5))
        sns.barplot(
            data=_column_mapping(all_domain_mae_rows),
            x="sequence",
            y="mae_cm",
            hue="domain",
            hue_order=[spec[1] for spec in PER_SEQUENCE_REGION_SPECS],
            ax=axis,
            palette="deep",
        )
        axis.tick_params(axis="x", rotation=0)
        axis.set_xlabel("Sequence")
        axis.set_ylabel("MAE [cm]")
        axis.set_title("MAE per Sequence Across Evaluation Domains")
        axis.legend(title="Domain")
        figure.tight_layout()
        figure.savefig(
            output_dir / "mae_per_sequence_all_domains.png",
            dpi=180,
        )
        plt.close(figure)

    if frame_rows:
        for domain_slug, domain_label, mae_field, rmse_field in (
            PER_SEQUENCE_REGION_SPECS
        ):
            mae_cm = [
                value
                for row in frame_rows
                if math.isfinite(
                    value := 100.0 * float(row[mae_field])
                )
            ]
            rmse_cm = [
                value
                for row in frame_rows
                if math.isfinite(
                    value := 100.0 * float(row[rmse_field])
                )
            ]
            if not mae_cm or not rmse_cm:
                continue
            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            sns.histplot(
                mae_cm, bins=40, ax=axes[0], color=sns.color_palette()[0]
            )
            axes[0].set_xlabel("Per-Frame MAE [cm]")
            axes[0].set_ylabel("Frames")
            sns.histplot(
                rmse_cm, bins=40, ax=axes[1], color=sns.color_palette()[1]
            )
            axes[1].set_xlabel("Per-Frame RMSE [cm]")
            axes[1].set_ylabel("Frames")
            figure.suptitle(f"Per-Frame Depth Errors: {domain_label}")
            figure.tight_layout()
            figure.savefig(
                output_dir / f"per_frame_error_histograms_{domain_slug}.png",
                dpi=180,
            )
            plt.close(figure)

        def plot_activity_error(
            activity_field: str,
            activity_xlabel: str,
            title_prefix: str,
            filename_prefix: str,
        ) -> None:
            activity = np.asarray(
                [row[activity_field] for row in frame_rows], dtype=np.float64
            )
            for domain_slug, domain_label, mae_field, _ in (
                PER_SEQUENCE_REGION_SPECS
            ):
                mae_m = np.asarray(
                    [row[mae_field] for row in frame_rows], dtype=np.float64
                )
                finite = np.isfinite(activity) & np.isfinite(mae_m)
                if not finite.any():
                    continue
                x = activity[finite]
                y = 100.0 * mae_m[finite]
                mean_x, mean_y = _fixed_width_mean_curve(x, y, 500.0)
                figure, axes = plt.subplots(1, 2, figsize=(16, 6.5))
                sns.scatterplot(
                    x=x,
                    y=y,
                    ax=axes[0],
                    s=18,
                    alpha=0.35,
                    linewidth=0,
                )
                axes[0].set_xlabel(activity_xlabel)
                axes[0].set_ylabel("MAE [cm]")
                axes[0].set_title(f"{title_prefix}: {domain_label}")
                sns.lineplot(
                    x=mean_x,
                    y=mean_y,
                    marker="o",
                    ax=axes[1],
                    color=sns.color_palette("flare", 3)[1],
                )
                axes[1].set_xlabel(activity_xlabel)
                axes[1].set_ylabel("MAE [cm]")
                axes[1].set_title("Fixed 500-Activity-Unit Bins")
                figure.tight_layout()
                figure.savefig(
                    output_dir / f"{filename_prefix}_{domain_slug}.png",
                    dpi=180,
                )
                plt.close(figure)

        plot_activity_error(
            "cube_event_activity",
            "Event Activity at GT-Depth Pixels Inside Workspace Cube",
            "Workspace-Cube Activity vs. Prediction Error",
            "cube_event_activity_vs_error",
        )
        plot_activity_error(
            "outside_cube_event_activity",
            "Event Activity at Valid GT-Depth Pixels Outside Workspace Cube",
            "Outside-Cube Activity vs. Prediction Error",
            "outside_cube_event_activity_vs_error",
        )

        def plot_scalar_diagnostic(
            x_field: str,
            xlabel: str,
            title_prefix: str,
            filename_prefix: str,
            legend_loc: str,
        ) -> None:
            if x_field not in frame_rows[0]:
                return
            x_all = np.asarray(
                [row[x_field] for row in frame_rows], dtype=np.float64
            )
            for domain_slug, domain_label, mae_field, _ in (
                PER_SEQUENCE_REGION_SPECS
            ):
                error_all = np.asarray(
                    [100.0 * row[mae_field] for row in frame_rows],
                    dtype=np.float64,
                )
                finite = np.isfinite(x_all) & np.isfinite(error_all)
                x = x_all[finite]
                error_cm = error_all[finite]
                if not x.size:
                    continue
                if x_field == "arm_speed_m_s":
                    mean_x, mean_error_cm = _fixed_width_mean_curve(
                        x, error_cm, 0.01
                    )
                    line_label = "Fixed 0.01 m/s-Bin Mean"
                else:
                    groups = [
                        indices
                        for indices in np.array_split(
                            np.argsort(x), min(12, x.size)
                        )
                        if indices.size
                    ]
                    mean_x = np.asarray(
                        [float(x[indices].mean()) for indices in groups]
                    )
                    mean_error_cm = np.asarray(
                        [float(error_cm[indices].mean()) for indices in groups]
                    )
                    line_label = "Equal-Count-Bin Mean"
                figure, axis = plt.subplots(figsize=(11, 7))
                sns.scatterplot(
                    x=x,
                    y=error_cm,
                    ax=axis,
                    s=18,
                    alpha=0.25,
                    linewidth=0,
                )
                sns.lineplot(
                    x=mean_x,
                    y=mean_error_cm,
                    marker="o",
                    ax=axis,
                    color=sns.color_palette("flare", 3)[1],
                    label=line_label,
                )
                axis.set_xlabel(xlabel)
                axis.set_ylabel(
                    "MAE [cm]"
                    if x_field == "arm_speed_m_s"
                    else f"{domain_label} Per-Frame MAE [cm]"
                )
                axis.set_title(f"{title_prefix}: {domain_label}")
                axis.legend(
                    loc=legend_loc,
                    fontsize=DIAGNOSTIC_LEGEND_FONT_SIZE,
                    title_fontsize=DIAGNOSTIC_LEGEND_TITLE_FONT_SIZE,
                )
                figure.tight_layout()
                figure.savefig(
                    output_dir / f"{filename_prefix}_{domain_slug}.png",
                    dpi=180,
                )
                plt.close(figure)

        plot_scalar_diagnostic(
            "arm_speed_m_s",
            "End-Effector Translational Velocity [m/s]",
            "Velocity vs. Depth Error",
            "arm_speed_vs_error",
            "upper right",
        )
        plot_scalar_diagnostic(
            "target_distance_m",
            "Camera Distance to Recording Target [m]",
            "Target Distance vs. Depth Error",
            "target_distance_vs_error",
            "upper left",
        )


def _write_comparison_results(
    output_dir: Path,
    runs: list[dict[str, Any]],
) -> None:
    """Write direct, aggregate plots for a multi-checkpoint evaluation run."""
    if len(runs) < 2:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt, sns = _setup_seaborn_plotting()
    palette = sns.color_palette("deep", n_colors=len(runs))

    # Persist a compact machine-readable comparison alongside the plots.
    comparison_rows = []
    for run in runs:
        summary = run["summary"]
        depth = summary["depth_metrics"]
        boundary = summary["boundary_metrics"]
        masked_boundary = summary["boundary_metrics_spatial_mask_plus_1cm"]
        mvc = summary["multiview_consistency"]
        performance = summary["performance"]
        comparison_rows.append(
            {
                "model": run["label"],
                **{name: depth[name] for name in DEPTH_METRIC_NAMES},
                **{
                    f"{region}_{name}": metrics[name]
                    for region, metrics in summary["depth_metrics_by_region"].items()
                    for name in ("valid_pixels", *DEPTH_METRIC_NAMES)
                },
                "boundary_mae_m": boundary["boundary"]["mae_m"],
                "non_boundary_mae_m": boundary["non_boundary"]["mae_m"],
                "boundary_spatial_mask_plus_1cm_mae_m": masked_boundary[
                    "boundary"
                ]["mae_m"],
                "non_boundary_spatial_mask_plus_1cm_mae_m": masked_boundary[
                    "non_boundary"
                ]["mae_m"],
                "mvc_mae_m": mvc["mae_m"],
                "mvc_rmse_m": mvc["rmse_m"],
                "mvc_within_1cm": mvc["within_1cm"],
                "mvc_within_2cm": mvc["within_2cm"],
                "mvc_within_5cm": mvc["within_5cm"],
                "model_inference_ms_per_frame": performance[
                    "model_inference_ms_per_frame"
                ],
                "model_inference_fps": performance["model_inference_fps"],
            }
        )
    _write_csv(output_dir / "comparison_summary.csv", comparison_rows)
    with (output_dir / "comparison_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_finite_or_none(comparison_rows), handle, indent=2)
        handle.write("\n")

    def grouped_bar(
        axis: Any,
        rows: list[dict[str, Any]],
        x: str,
        y: str,
        title: str,
        ylabel: str,
        percentage_values: list[float] | None = None,
    ) -> None:
        sns.barplot(
            data=_column_mapping(rows),
            x=x,
            y=y,
            hue="model",
            hue_order=[run["label"] for run in runs],
            palette=palette,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        if percentage_values is not None:
            _set_inverse_percentage_axis(axis, percentage_values)
        axis.legend(
            title="Model",
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_TITLE_FONT_SIZE,
        )

    # Depth metrics for each evaluation domain. Models are adjacent by metric.
    error_names = ["abs_rel", "sq_rel_m", "mae_m", "rmse_m", "rmse_log"]
    delta_specs = [
        ("delta_1", r"$\delta<1.25$"),
        ("delta_2", r"$\delta<1.25^2$"),
        ("delta_3", r"$\delta<1.25^3$"),
    ]
    for domain_slug, domain_label, _, _ in PER_SEQUENCE_REGION_SPECS:
        region_key = DOMAIN_SUMMARY_REGION_KEYS[domain_slug]
        error_rows = [
            {
                "metric": metric,
                "value": float(
                    run["summary"]["depth_metrics_by_region"][region_key][metric]
                ),
                "model": run["label"],
            }
            for metric in error_names
            for run in runs
        ]
        delta_rows = [
            {
                "threshold": label,
                "failure_percentage": 100.0
                * (
                    1.0
                    - float(
                        run["summary"]["depth_metrics_by_region"][region_key][
                            field
                        ]
                    )
                ),
                "model": run["label"],
            }
            for field, label in delta_specs
            for run in runs
        ]
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.2))
        grouped_bar(
            axes[0],
            error_rows,
            "metric",
            "value",
            f"Depth Errors: {domain_label}",
            "Error",
        )
        axes[0].tick_params(axis="x", rotation=30)
        grouped_bar(
            axes[1],
            delta_rows,
            "threshold",
            "failure_percentage",
            f"Threshold Failure Rate: {domain_label}",
            "Pixels Outside Threshold [%]",
            [row["failure_percentage"] for row in delta_rows],
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"depth_metrics_{domain_slug}.png", dpi=180
        )
        plt.close(figure)

    region_error_rows = [
        {
            "region": DEPTH_REGION_LABELS[region],
            "metric": metric,
            "error_cm": 100.0
            * float(run["summary"]["depth_metrics_by_region"][region][field]),
            "model": run["label"],
        }
        for region in DEPTH_REGION_LABELS
        for metric, field in (("MAE", "mae_m"), ("RMSE", "rmse_m"))
        for run in runs
    ]
    figure, axes = plt.subplots(1, 2, figsize=(16, 5.8), sharey=True)
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        metric_rows = [row for row in region_error_rows if row["metric"] == metric]
        grouped_bar(
            axis,
            metric_rows,
            "region",
            "error_cm",
            f"{metric} by Evaluation Region",
            "Depth Error [cm]",
        )
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(output_dir / "mae_rmse_by_region.png", dpi=180)
    plt.close(figure)

    mae_region_rows = [
        row for row in region_error_rows if row["metric"] == "MAE"
    ]
    figure, axis = plt.subplots(figsize=(13, 6.2))
    sns.barplot(
        data=_column_mapping(mae_region_rows),
        x="region",
        y="error_cm",
        hue="model",
        order=list(DEPTH_REGION_LABELS.values()),
        hue_order=[run["label"] for run in runs],
        ax=axis,
        palette=palette,
    )
    axis.set_xlabel("")
    axis.set_ylabel("MAE [cm]")
    axis.set_title("MAE by Model and Evaluation Region")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(
        title="Model",
        loc="upper left",
        fontsize=DIAGNOSTIC_LEGEND_FONT_SIZE,
        title_fontsize=DIAGNOSTIC_LEGEND_TITLE_FONT_SIZE,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "mae_by_region.png", dpi=180)
    plt.close(figure)

    boundary_rows = []
    mvc_rows = []
    for run in runs:
        label = run["label"]
        boundary = run["summary"]["boundary_metrics"]
        mvc = run["summary"]["multiview_consistency"]
        boundary_rows.extend(
            [
                {
                    "region": "Boundary",
                    "mae_cm": 100.0 * float(boundary["boundary"]["mae_m"]),
                    "model": label,
                },
                {
                    "region": "Non-Boundary",
                    "mae_cm": 100.0 * float(boundary["non_boundary"]["mae_m"]),
                    "model": label,
                },
            ]
        )
        mvc_rows.extend(
            [
                {
                    "threshold": threshold,
                    "percentage": 100.0 * (1.0 - float(mvc[field])),
                    "model": label,
                }
                for field, threshold in (
                    ("within_1cm", "<1 cm"),
                    ("within_2cm", "<2 cm"),
                    ("within_5cm", "<5 cm"),
                )
            ]
        )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    grouped_bar(
        axes[0],
        boundary_rows,
        "region",
        "mae_cm",
        "Depth-Boundary Error",
        "MAE [cm]",
    )
    if any(math.isfinite(row["percentage"]) for row in mvc_rows):
        grouped_bar(
            axes[1],
            mvc_rows,
            "threshold",
            "percentage",
            "Multi-View Inconsistency Thresholds",
            "Inconsistent Correspondences [%]",
            [row["percentage"] for row in mvc_rows],
        )
    else:
        axes[1].set_axis_off()
        axes[1].text(
            0.5,
            0.5,
            "Multi-view consistency disabled\n(--no_mvc)",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(output_dir / "boundary_and_consistency.png", dpi=180)
    plt.close(figure)

    for domain_slug, _, _, _ in PER_SEQUENCE_REGION_SPECS:
        domain_boundary_rows = []
        for run in runs:
            label = run["label"]
            domain_boundary = run["summary"]["boundary_metrics_by_region"][
                domain_slug
            ]
            domain_boundary_rows.extend(
                [
                    {
                        "region": "Boundary",
                        "mae_cm": 100.0
                        * float(domain_boundary["boundary"]["mae_m"]),
                        "model": label,
                    },
                    {
                        "region": "Non-Boundary",
                        "mae_cm": 100.0
                        * float(domain_boundary["non_boundary"]["mae_m"]),
                        "model": label,
                    },
                ]
            )
        figure, axis = plt.subplots(figsize=(max(8.0, len(runs) * 2.5), 5.2))
        grouped_bar(
            axis,
            domain_boundary_rows,
            "region",
            "mae_cm",
            "",
            "MAE [cm]",
        )
        axis.set_xlabel("Region")
        figure.tight_layout()
        figure.savefig(output_dir / f"boundary_error_{domain_slug}.png", dpi=180)
        plt.close(figure)

    # Per-sequence bars use model as hue, with MAE and RMSE in separate panels.
    comparison_sequence_labels = _sequence_display_map(
        [
            str(row["sequence"])
            for run in runs
            for row in run["sequence_rows"]
        ]
    )
    for filename_region, region_label, mae_field, rmse_field in (
        PER_SEQUENCE_REGION_SPECS
    ):
        sequence_plot_rows = []
        for run in runs:
            for row in run["sequence_rows"]:
                for metric, field in (
                    ("MAE", mae_field),
                    ("RMSE", rmse_field),
                ):
                    sequence_plot_rows.append(
                        {
                            "sequence": comparison_sequence_labels[
                                str(row["sequence"])
                            ],
                            "metric": metric,
                            "error_cm": 100.0 * float(row[field]),
                            "model": run["label"],
                        }
                    )
        if not sequence_plot_rows:
            continue
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(max(10, len(runs) * 2.5), 10),
        )
        for axis, metric in zip(axes, ("MAE", "RMSE")):
            metric_rows = [
                row
                for row in sequence_plot_rows
                if row["metric"] == metric
            ]
            grouped_bar(
                axis,
                metric_rows,
                "sequence",
                "error_cm",
                f"{metric} per Sequence: {region_label}",
                "Depth Error [cm]",
            )
            axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"error_per_sequence_{filename_region}.png",
            dpi=180,
        )
        if filename_region == "whole_frame":
            figure.savefig(output_dir / "error_per_sequence.png", dpi=180)
        plt.close(figure)

    if any(run["sequence_rows"] for run in runs):
        figure, axes = plt.subplots(
            len(runs),
            1,
            figsize=(max(10, len(runs) * 2.5), 4.5 * len(runs)),
            squeeze=False,
            sharey=True,
        )
        for run_index, (axis, run) in enumerate(zip(axes[:, 0], runs)):
            all_domain_mae_rows = [
                {
                    "sequence": comparison_sequence_labels[
                        str(row["sequence"])
                    ].removeprefix("Sequence "),
                    "domain": region_label,
                    "mae_cm": 100.0 * float(row[mae_field]),
                }
                for row in run["sequence_rows"]
                for _, region_label, mae_field, _ in PER_SEQUENCE_REGION_SPECS
            ]
            if not all_domain_mae_rows:
                axis.set_axis_off()
                continue
            sns.barplot(
                data=_column_mapping(all_domain_mae_rows),
                x="sequence",
                y="mae_cm",
                hue="domain",
                hue_order=[spec[1] for spec in PER_SEQUENCE_REGION_SPECS],
                ax=axis,
                palette="deep",
            )
            axis.tick_params(axis="x", rotation=0)
            axis.set_xlabel("")
            axis.set_ylabel("MAE [cm]")
            axis.set_title(run["label"])
            if run_index == 0:
                axis.legend(title="Domain")
            elif axis.get_legend() is not None:
                axis.get_legend().remove()
        figure.suptitle("MAE per Sequence Across Evaluation Domains")
        figure.supxlabel("Sequence")
        figure.tight_layout()
        figure.savefig(
            output_dir / "mae_per_sequence_all_domains.png",
            dpi=180,
        )
        plt.close(figure)

    for domain_slug, domain_label, mae_field, _ in PER_SEQUENCE_REGION_SPECS:
        normalized_time_rows = []
        normalized_time_curves = []
        for run in runs:
            centers, _sequence_curves, mean, std, recording_counts = (
                _normalized_time_error_curves(
                    run["frame_rows"], error_field=mae_field
                )
            )
            normalized_time_curves.append((run["label"], centers, mean, std))
            for index in np.flatnonzero(np.isfinite(mean)):
                normalized_time_rows.append(
                    {
                        "model": run["label"],
                        "normalized_time": float(centers[index]),
                        "mean_mae_cm": float(mean[index]),
                        "std_mae_cm": float(std[index]),
                        "recordings": int(recording_counts[index]),
                    }
                )
        if not normalized_time_rows:
            continue
        _write_csv(
            output_dir / f"error_over_normalized_time_{domain_slug}.csv",
            normalized_time_rows,
        )
        figure, axis = plt.subplots(figsize=(12, 7.0))
        for color, (label, centers, mean, std) in zip(
            palette, normalized_time_curves
        ):
            finite = np.isfinite(mean)
            if not finite.any():
                continue
            axis.plot(
                centers[finite],
                mean[finite],
                color=color,
                linewidth=2.4,
                label=label,
            )
            axis.fill_between(
                centers[finite],
                np.maximum(0.0, mean[finite] - std[finite]),
                mean[finite] + std[finite],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Normalized Recording Time")
        axis.set_ylabel("Per-Frame MAE [cm]")
        axis.set_title(
            f"Depth Error Over Normalized Recording Time: {domain_label}"
        )
        axis.legend(
            title="Model",
            fontsize=DIAGNOSTIC_LEGEND_FONT_SIZE,
            title_fontsize=DIAGNOSTIC_LEGEND_TITLE_FONT_SIZE,
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"error_over_normalized_time_{domain_slug}.png",
            dpi=180,
        )
        plt.close(figure)

    def mean_line_plot(
        filename: str,
        panels: list[tuple[str, str, str]],
        bins: int = 12,
        fixed_interval: float | None = None,
        y_scale: float = 100.0,
        y_field: str = "mae_m",
        ylabel: str = "Mean Per-Frame MAE [cm]",
        legend_loc: str = "upper right",
        legend_fontsize: int = LEGEND_FONT_SIZE,
        legend_title_fontsize: int = LEGEND_TITLE_FONT_SIZE,
    ) -> None:
        n_panels = len(panels)
        ncols = min(3, n_panels)
        nrows = int(math.ceil(n_panels / ncols))
        if n_panels == 1:
            figure_size = (12, 7.5)
        elif n_panels == 2:
            figure_size = (16, 6.5)
        else:
            figure_size = (5.2 * ncols, 4.2 * nrows)
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figure_size,
            squeeze=False,
        )
        for axis, (field, xlabel, title) in zip(axes.flat, panels):
            for color, run in zip(palette, runs):
                if fixed_interval is None:
                    x, y = _equal_count_mean_curve(
                        run["frame_rows"],
                        field,
                        y_field=y_field,
                        bins=bins,
                        y_scale=y_scale,
                    )
                else:
                    x, y = _fixed_width_mean_curve_from_rows(
                        run["frame_rows"],
                        field,
                        fixed_interval,
                        y_field=y_field,
                        y_scale=y_scale,
                    )
                if x.size:
                    axis.plot(x, y, linewidth=2.2, color=color, label=run["label"])
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            if n_panels != 1:
                axis.legend(
                    title="Model",
                    loc=legend_loc,
                    fontsize=legend_fontsize,
                    title_fontsize=legend_title_fontsize,
                )
        for axis in axes.flat[n_panels:]:
            axis.set_visible(False)
        if n_panels == 1:
            axes.flat[0].legend(
                title="Model",
                loc=legend_loc,
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                frameon=True,
            )
            figure.tight_layout()
        else:
            figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    # Comparison diagnostics intentionally contain only binned mean curves:
    # no per-frame scatter and no markers on the curves.
    for domain_slug, domain_label, mae_field, _ in PER_SEQUENCE_REGION_SPECS:
        diagnostic_specs = (
            (
                "cube_event_activity_vs_error",
                "cube_event_activity",
                "Mean Event Activity at GT-Depth Pixels Inside Workspace Cube",
                "Workspace-Cube Activity vs. Prediction Error",
                500.0,
                "upper right",
            ),
            (
                "outside_cube_event_activity_vs_error",
                "outside_cube_event_activity",
                (
                    "Mean Event Activity at Valid GT-Depth Pixels Outside "
                    "Workspace Cube"
                ),
                "Outside-Cube Activity vs. Prediction Error",
                500.0,
                "upper right",
            ),
            (
                "arm_speed_vs_error",
                "arm_speed_m_s",
                "End-Effector Translational Velocity [m/s]",
                "Velocity vs. Depth Error",
                0.01,
                "upper right",
            ),
            (
                "target_distance_vs_error",
                "target_distance_m",
                "Camera Distance to Recording Target [m]",
                "Target Distance vs. Depth Error",
                None,
                "upper left",
            ),
        )
        for filename_prefix, x_field, xlabel, title, fixed_interval, legend_loc in (
            diagnostic_specs
        ):
            mean_line_plot(
                f"{filename_prefix}_{domain_slug}.png",
                [(x_field, xlabel, f"{title}: {domain_label}")],
                bins=12,
                fixed_interval=fixed_interval,
                y_field=mae_field,
                ylabel=(
                    "MAE [cm]"
                    if x_field
                    in {
                        "cube_event_activity",
                        "outside_cube_event_activity",
                        "arm_speed_m_s",
                    }
                    else f"Mean {domain_label} Per-Frame MAE [cm]"
                ),
                legend_loc=legend_loc,
                legend_fontsize=DIAGNOSTIC_LEGEND_FONT_SIZE,
                legend_title_fontsize=DIAGNOSTIC_LEGEND_TITLE_FONT_SIZE,
            )

    # Only L1 is included in the spatial-mask sweep, with models side by side.
    offset_rows = []
    for run in runs:
        for row in run["summary"]["spatial_mask_z_offset_metrics"]:
            offset_rows.append(
                {
                    "offset_mm": f"{1000.0 * float(row['z_offset_m']):.1f}",
                    "l1_mm": 1000.0 * float(row["l1_m"]),
                    "model": run["label"],
                }
            )
    if offset_rows:
        figure, axis = plt.subplots(figsize=(max(12, len(runs) * 3), 5.5))
        grouped_bar(
            axis,
            offset_rows,
            "offset_mm",
            "l1_mm",
            "L1 Depth Error Under Recomputed Spatial Masks",
            "L1 Error [mm]",
        )
        axis.set_xlabel("Spatial-Mask Cube-Bottom Z Offset [mm]")
        figure.tight_layout()
        figure.savefig(output_dir / "spatial_mask_z_offset_metrics.png", dpi=180)
        plt.close(figure)


def _plot_worst_frames(
    output_dir: Path,
    samples: list[dict[str, Any]],
    domain_slug: str,
    domain_label: str,
) -> None:
    """Save each sequence's highest-MAE frame for one evaluation domain."""
    if not samples:
        return

    plt, _sns = _setup_seaborn_plotting()

    samples = sorted(samples, key=lambda sample: sample["sequence"])
    sequence_labels = _sequence_display_map(
        [str(sample["sequence"]) for sample in samples]
    )
    figure, axes = plt.subplots(
        len(samples),
        4,
        figsize=(16, 3.5 * len(samples)),
        squeeze=False,
    )
    for column, title in enumerate(
        ("Event Activity (Shared Scale)", "GT Depth", "Predicted Depth", "Absolute Error")
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
            f"{sequence_labels[str(sample['sequence'])]}\nFrame {sample['frame_idx']}",
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

    figure.suptitle(f"Worst Per-Sequence Depth Errors: {domain_label}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(
        output_dir / f"worst_error_overview_{domain_slug}.png", dpi=180
    )
    plt.close(figure)


def _plot_qualitative_depth_results(
    output_dir: Path,
    samples: list[dict[str, Any]],
) -> None:
    """Plot events and workspace-masked depth results for each sequence."""
    if not samples:
        return

    plt, _sns = _setup_seaborn_plotting()

    def sequence_sort_key(sample: dict[str, Any]) -> tuple[int, Any]:
        sequence = str(sample["sequence"])
        return (0, int(sequence)) if sequence.isdigit() else (1, sequence)

    samples = sorted(samples, key=sequence_sort_key)
    figure, axes = plt.subplots(
        4,
        len(samples),
        figsize=(max(3.0 * len(samples), 8.0), 10.0),
        squeeze=False,
        constrained_layout=True,
    )

    event_activity_images = [
        np.abs(sample["events"]).sum(axis=0) for sample in samples
    ]
    active_event_values = [
        activity[activity > 0.0]
        for activity in event_activity_images
        if np.any(activity > 0.0)
    ]
    event_vmax = (
        max(
            float(np.percentile(np.concatenate(active_event_values), 99.0)),
            1e-6,
        )
        if active_event_values
        else 1.0
    )

    displayed_depth_values = []
    for sample in samples:
        gt_valid = sample["lower_cube_valid"].astype(bool, copy=False)
        pred_valid = sample["prediction_workspace_mask"].astype(bool, copy=False)
        displayed_depth_values.extend(
            (
                sample["gt"][gt_valid & np.isfinite(sample["gt"])],
                sample["pred"][pred_valid & np.isfinite(sample["pred"])],
            )
        )
    nonempty_depth_values = [values for values in displayed_depth_values if values.size]
    if nonempty_depth_values:
        all_depth_values = np.concatenate(nonempty_depth_values)
        depth_vmin = float(all_depth_values.min())
        depth_vmax = float(all_depth_values.max())
        if depth_vmax <= depth_vmin:
            depth_vmax = depth_vmin + 1e-3
    else:
        depth_vmin, depth_vmax = DEPTH_MIN, D_MAX

    valid_errors = [
        np.abs(sample["pred"] - sample["gt"])[sample["lower_cube_valid"]]
        for sample in samples
        if np.any(sample["lower_cube_valid"])
    ]
    if valid_errors:
        error_vmax = max(
            float(np.percentile(np.concatenate(valid_errors), 99.0)),
            1e-3,
        )
    else:
        error_vmax = 0.1

    depth_cmap = plt.get_cmap("turbo").copy()
    depth_cmap.set_bad("black")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("black")

    depth_image = None
    error_image = None
    event_image = None
    for column, (sample, event_activity) in enumerate(
        zip(samples, event_activity_images)
    ):
        error_valid = sample["lower_cube_valid"].astype(bool, copy=False)
        prediction_valid = sample["prediction_workspace_mask"].astype(
            bool, copy=False
        )
        gt = np.ma.masked_where(~error_valid, sample["gt"])
        # Prediction membership is computed from predicted depth, independently
        # of GT validity, so valid predictions remain visible across GT holes.
        pred = np.ma.masked_where(~prediction_valid, sample["pred"])
        error = np.ma.masked_where(
            ~error_valid,
            np.abs(sample["pred"] - sample["gt"]),
        )

        event_image = axes[0, column].imshow(
            event_activity,
            cmap="gray",
            vmin=0.0,
            vmax=event_vmax,
        )
        depth_image = axes[1, column].imshow(
            gt,
            cmap=depth_cmap,
            vmin=depth_vmin,
            vmax=depth_vmax,
        )
        axes[2, column].imshow(
            pred,
            cmap=depth_cmap,
            vmin=depth_vmin,
            vmax=depth_vmax,
        )
        error_image = axes[3, column].imshow(
            error,
            cmap=error_cmap,
            vmin=0.0,
            vmax=error_vmax,
        )
        for row in range(4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    if event_image is not None:
        event_colorbar = figure.colorbar(
            event_image,
            ax=axes[0, :].ravel().tolist(),
            fraction=0.012,
            pad=0.01,
            aspect=50,
            shrink=0.49,
        )
        event_colorbar.set_ticks(np.linspace(0.0, event_vmax, 3))
    if depth_image is not None:
        depth_colorbar = figure.colorbar(
            depth_image,
            ax=axes[1:3, :].ravel().tolist(),
            fraction=0.014,
            pad=0.01,
            aspect=45,
            shrink=0.54,
        )
        depth_colorbar.set_ticks(np.linspace(depth_vmin, depth_vmax, 4))
    if error_image is not None:
        error_colorbar = figure.colorbar(
            error_image,
            ax=axes[3, :].ravel().tolist(),
            fraction=0.012,
            pad=0.01,
            aspect=50,
            shrink=0.49,
        )
        error_colorbar.set_ticks(np.linspace(0.0, error_vmax, 3))

    figure.savefig(output_dir / "qualitative_depth_results.png", dpi=220)
    plt.close(figure)


def _plot_selected_frames_overview(
    output_dir: Path,
    samples: list[dict[str, Any]],
    include_error: bool = True,
) -> None:
    """Plot the selected-frame modalities for each requested frame."""
    if not samples:
        return

    plt, _sns = _setup_seaborn_plotting()
    if include_error:
        depth_cmap = plt.get_cmap("turbo").copy()
        depth_cmap.set_bad("black")
        error_cmap = plt.get_cmap("magma").copy()
        error_cmap.set_bad("black")

        displayed_depth_values = []
        for sample in samples:
            gt_valid = sample["lower_cube_valid"].astype(bool, copy=False)
            prediction_valid = sample["prediction_workspace_mask"].astype(
                bool, copy=False
            )
            displayed_depth_values.extend(
                (
                    sample["gt"][gt_valid & np.isfinite(sample["gt"])],
                    sample["pred"][
                        prediction_valid & np.isfinite(sample["pred"])
                    ],
                )
            )
        displayed_depth_values = [
            values for values in displayed_depth_values if values.size
        ]
        if displayed_depth_values:
            all_depth_values_cm = 100.0 * np.concatenate(displayed_depth_values)
            depth_vmin_cm = float(math.floor(all_depth_values_cm.min()))
            depth_vmax_cm = float(math.ceil(all_depth_values_cm.max()))
        else:
            depth_vmin_cm = float(math.floor(100.0 * DEPTH_MIN))
            depth_vmax_cm = float(math.ceil(100.0 * D_MAX))
        if depth_vmax_cm <= depth_vmin_cm:
            depth_vmax_cm = depth_vmin_cm + 1.0

        figure, axes = plt.subplots(
            2 * len(samples),
            2,
            figsize=(10.0, 7.0 * len(samples)),
            squeeze=False,
            constrained_layout=True,
        )

        for sample_index, sample in enumerate(samples):
            top_row = 2 * sample_index
            bottom_row = top_row + 1
            event_axis = axes[top_row, 0]
            gt_axis = axes[top_row, 1]
            prediction_axis = axes[bottom_row, 0]
            error_axis = axes[bottom_row, 1]

            event_activity = np.abs(sample["events"]).sum(axis=0)
            active_values = event_activity[event_activity > 0.0]
            event_vmax = (
                max(float(np.percentile(active_values, 99.0)), 1e-6)
                if active_values.size
                else 1.0
            )
            event_axis.imshow(
                event_activity,
                cmap="gray",
                vmin=0.0,
                vmax=event_vmax,
            )

            gt_valid = sample["lower_cube_valid"].astype(bool, copy=False)
            prediction_valid = sample["prediction_workspace_mask"].astype(
                bool, copy=False
            )
            gt_cm = np.ma.masked_where(~gt_valid, 100.0 * sample["gt"])
            prediction_cm = np.ma.masked_where(
                ~prediction_valid,
                100.0 * sample["pred"],
            )
            depth_image = gt_axis.imshow(
                gt_cm,
                cmap=depth_cmap,
                vmin=depth_vmin_cm,
                vmax=depth_vmax_cm,
            )
            prediction_axis.imshow(
                prediction_cm,
                cmap=depth_cmap,
                vmin=depth_vmin_cm,
                vmax=depth_vmax_cm,
            )

            absolute_error_cm = 100.0 * np.abs(
                sample["pred"] - sample["gt"]
            )
            error_cm = np.ma.masked_where(~gt_valid, absolute_error_cm)
            valid_error_values_cm = absolute_error_cm[gt_valid]
            raw_error_vmax_cm = (
                max(
                    float(np.percentile(valid_error_values_cm, 99.0)),
                    0.1,
                )
                if valid_error_values_cm.size
                else 10.0
            )
            error_vmax_cm = math.ceil(10.0 * raw_error_vmax_cm) / 10.0
            error_image = error_axis.imshow(
                error_cm,
                cmap=error_cmap,
                vmin=0.0,
                vmax=error_vmax_cm,
            )

            for axis in (
                event_axis,
                gt_axis,
                prediction_axis,
                error_axis,
            ):
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_facecolor("black")

            depth_colorbar = figure.colorbar(
                depth_image,
                ax=gt_axis,
                orientation="vertical",
                location="right",
                fraction=0.046,
                pad=0.025,
                aspect=25,
                shrink=0.75,
            )
            depth_colorbar.set_ticks(
                np.linspace(depth_vmin_cm, depth_vmax_cm, 3)
            )

            error_colorbar = figure.colorbar(
                error_image,
                ax=error_axis,
                orientation="vertical",
                location="right",
                fraction=0.046,
                pad=0.025,
                aspect=25,
                shrink=0.75,
            )
            error_colorbar.set_ticks(np.linspace(0.0, error_vmax_cm, 3))

        figure.savefig(
            output_dir / "selected_frames_overview.png",
            dpi=220,
        )
        plt.close(figure)
        return

    column_count = 4 if include_error else 3
    figure, axes = plt.subplots(
        len(samples),
        column_count,
        figsize=((15.5 if include_error else 11.8), 4.4 * len(samples)),
        squeeze=False,
        constrained_layout=True,
    )

    depth_cmap = plt.get_cmap("turbo").copy()
    depth_cmap.set_bad("black")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("black")

    displayed_depth_values = []
    for sample in samples:
        gt_valid = sample["lower_cube_valid"].astype(bool, copy=False)
        prediction_valid = sample["prediction_workspace_mask"].astype(
            bool, copy=False
        )
        displayed_depth_values.extend(
            (
                sample["gt"][gt_valid & np.isfinite(sample["gt"])],
                sample["pred"][prediction_valid & np.isfinite(sample["pred"])],
            )
        )
    displayed_depth_values = [
        values for values in displayed_depth_values if values.size
    ]
    if displayed_depth_values:
        all_depth_values = np.concatenate(displayed_depth_values)
        depth_vmin = float(all_depth_values.min())
        depth_vmax = float(all_depth_values.max())
        if depth_vmax <= depth_vmin:
            depth_vmax = depth_vmin + 1e-3
    else:
        depth_vmin, depth_vmax = DEPTH_MIN, D_MAX

    for row, sample in enumerate(samples):
        event_activity = np.abs(sample["events"]).sum(axis=0)
        active_values = event_activity[event_activity > 0.0]
        event_vmax = (
            max(float(np.percentile(active_values, 99.0)), 1e-6)
            if active_values.size
            else 1.0
        )
        event_image = axes[row, 0].imshow(
            event_activity,
            cmap="gray",
            vmin=0.0,
            vmax=event_vmax,
        )

        gt_valid = sample["lower_cube_valid"].astype(bool, copy=False)
        prediction_valid = sample["prediction_workspace_mask"].astype(
            bool, copy=False
        )
        gt = np.ma.masked_where(~gt_valid, sample["gt"])
        prediction = np.ma.masked_where(~prediction_valid, sample["pred"])

        depth_image = axes[row, 1].imshow(
            gt,
            cmap=depth_cmap,
            vmin=depth_vmin,
            vmax=depth_vmax,
        )
        axes[row, 2].imshow(
            prediction,
            cmap=depth_cmap,
            vmin=depth_vmin,
            vmax=depth_vmax,
        )
        if include_error:
            absolute_error = np.abs(sample["pred"] - sample["gt"])
            error = np.ma.masked_where(~gt_valid, absolute_error)
            valid_error_values = absolute_error[gt_valid]
            error_vmax = (
                max(float(np.percentile(valid_error_values, 99.0)), 1e-3)
                if valid_error_values.size
                else 0.1
            )
            error_image = axes[row, 3].imshow(
                error,
                cmap=error_cmap,
                vmin=0.0,
                vmax=error_vmax,
            )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("black")

        event_colorbar = figure.colorbar(
            event_image,
            ax=axes[row, 0],
            orientation="horizontal",
            location="bottom",
            fraction=0.0675,
            pad=0.04,
            aspect=30,
            shrink=0.75,
        )
        event_colorbar.set_ticks(np.linspace(0.0, event_vmax, 3))
        depth_colorbar = figure.colorbar(
            depth_image,
            ax=axes[row, 1:3].tolist(),
            orientation="horizontal",
            location="bottom",
            fraction=0.0675,
            pad=0.04,
            aspect=55,
            shrink=0.75,
        )
        depth_colorbar.set_ticks(np.linspace(depth_vmin, depth_vmax, 4))
        if include_error:
            error_colorbar = figure.colorbar(
                error_image,
                ax=axes[row, 3],
                orientation="horizontal",
                location="bottom",
                fraction=0.0675,
                pad=0.04,
                aspect=30,
                shrink=0.75,
            )
            error_colorbar.set_ticks(np.linspace(0.0, error_vmax, 3))

    output_name = (
        "selected_frames_overview.png"
        if include_error
        else "selected_frames_only_depth.png"
    )
    figure.savefig(output_dir / output_name, dpi=220)
    plt.close(figure)


def _plot_error_maps_by_region(
    output_dir: Path,
    samples: list[dict[str, Any]],
) -> None:
    """Plot whole-frame, workspace-cube, and raised-cube error maps."""
    if not samples:
        return

    plt, _sns = _setup_seaborn_plotting()

    def sequence_sort_key(sample: dict[str, Any]) -> tuple[int, Any]:
        sequence = str(sample["sequence"])
        return (0, int(sequence)) if sequence.isdigit() else (1, sequence)

    samples = sorted(samples, key=sequence_sort_key)
    sequence_labels = _sequence_display_map(
        [str(sample["sequence"]) for sample in samples]
    )
    region_specs = (
        ("valid", "Whole Frame"),
        ("lower_cube_valid", "Workspace Cube"),
        ("upper_cube_valid", "Raised Cube"),
    )
    figure, axes = plt.subplots(
        len(region_specs),
        len(samples),
        figsize=(max(3.0 * len(samples), 8.0), 8.0),
        squeeze=False,
        constrained_layout=True,
    )

    whole_image_errors = [
        np.abs(sample["pred"] - sample["gt"])[sample["valid"]]
        for sample in samples
        if np.any(sample["valid"])
    ]
    if whole_image_errors:
        error_vmax = max(
            float(np.percentile(np.concatenate(whole_image_errors), 99.0)),
            1e-3,
        )
    else:
        error_vmax = 0.1

    error_image = None
    for column, sample in enumerate(samples):
        absolute_error = np.abs(sample["pred"] - sample["gt"])
        axes[0, column].set_title(
            f"{sequence_labels[str(sample['sequence'])]}\nFrame {sample['frame_idx']}",
            fontsize=9,
        )
        for row, (mask_key, region_label) in enumerate(region_specs):
            axis = axes[row, column]
            region_valid = sample[mask_key].astype(bool, copy=False)
            displayed_error = np.ma.masked_where(~region_valid, absolute_error)
            error_image = axis.imshow(
                displayed_error,
                cmap="magma",
                vmin=0.0,
                vmax=error_vmax,
            )
            axis.set_facecolor("#282828")
            mae_cm = (
                100.0 * float(absolute_error[region_valid].mean())
                if np.any(region_valid)
                else math.nan
            )
            mae_label = f"MAE {mae_cm:.2f} cm" if math.isfinite(mae_cm) else "MAE N/A"
            axis.text(
                0.03,
                0.06,
                mae_label,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.65",
                    "alpha": 0.9,
                },
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(region_label, fontsize=10, fontweight="bold")

    if error_image is not None:
        figure.colorbar(
            error_image,
            ax=axes.ravel().tolist(),
            label="Absolute Error [m] (Shared 99th-Percentile Scale)",
            fraction=0.02,
            pad=0.015,
        )
    figure.savefig(output_dir / "error_maps_by_region.png", dpi=220)
    plt.close(figure)


def _plot_masked_boundary_regions(
    output_dir: Path,
    samples: list[dict[str, Any]],
) -> None:
    """Visualize the exact +1.5 cm masked boundary evaluation regions."""
    if not samples:
        return

    plt, sns = _setup_seaborn_plotting()
    from matplotlib.patches import Patch

    samples = sorted(
        samples,
        key=lambda sample: (sample["sequence"], sample["frame_idx"]),
    )
    sequence_labels = _sequence_display_map(
        [str(sample["sequence"]) for sample in samples]
    )
    figure, axes = plt.subplots(
        len(samples),
        2,
        figsize=(12, max(3.0 * len(samples), 5.0)),
        squeeze=False,
    )
    outside_overlay = np.zeros((*samples[0]["gt"].shape, 4), dtype=np.float32)
    outside_overlay[..., :3] = np.asarray(sns.color_palette("deep")[0])

    region_colors = {
        "outside": np.asarray([35, 39, 47], dtype=np.uint8),
        "inside_excluded": np.asarray([150, 155, 165], dtype=np.uint8),
        "non_boundary": np.asarray([43, 116, 189], dtype=np.uint8),
        "boundary": np.asarray([232, 95, 42], dtype=np.uint8),
    }
    depth_image = None
    for row, sample in enumerate(samples):
        spatial_mask = sample["spatial_mask"].astype(bool, copy=False)
        boundary = sample["boundary"].astype(bool, copy=False)
        non_boundary = sample["non_boundary"].astype(bool, copy=False)

        depth_axis, region_axis = axes[row]
        depth_image = depth_axis.imshow(
            sample["gt"],
            cmap="turbo",
            vmin=DEPTH_MIN,
            vmax=D_MAX,
        )
        overlay = outside_overlay.copy()
        overlay[..., 3] = np.where(spatial_mask, 0.0, 0.72)
        depth_axis.imshow(overlay)
        depth_axis.set_ylabel(
            f"{sequence_labels[str(sample['sequence'])]}\nFrame {sample['frame_idx']}",
            fontsize=9,
        )

        regions = np.empty((*spatial_mask.shape, 3), dtype=np.uint8)
        regions[:] = region_colors["outside"]
        regions[spatial_mask] = region_colors["inside_excluded"]
        regions[non_boundary] = region_colors["non_boundary"]
        regions[boundary] = region_colors["boundary"]
        region_axis.imshow(regions)

        for axis in (depth_axis, region_axis):
            axis.set_xticks([])
            axis.set_yticks([])

    axes[0, 0].set_title("GT Depth Inside +1.5 cm Spatial Mask")
    axes[0, 1].set_title("Pixels Used by Boundary Metric")
    if depth_image is not None:
        figure.colorbar(
            depth_image,
            ax=axes[:, 0].tolist(),
            label="GT Depth [m]",
            fraction=0.025,
            pad=0.02,
        )
    figure.legend(
        handles=[
            Patch(color=region_colors["boundary"] / 255.0, label="Boundary"),
            Patch(
                color=region_colors["non_boundary"] / 255.0,
                label="Non-Boundary",
            ),
            Patch(
                color=region_colors["inside_excluded"] / 255.0,
                label="Inside Mask, Invalid Prediction",
            ),
            Patch(
                color=region_colors["outside"] / 255.0,
                label="Outside +1.5 cm Mask",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=True,
    )
    figure.suptitle(
        "Depth-Boundary Evaluation Regions Inside the +1.5 cm Spatial Mask",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(top=0.94, bottom=0.07, wspace=0.08, hspace=0.12)
    figure.savefig(
        output_dir / "boundary_regions_spatial_mask_plus_1cm.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_spatial_mask_offset_metrics(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    plt, sns = _setup_seaborn_plotting()
    labels = [f"{1000.0 * float(row['z_offset_m']):.1f}" for row in rows]
    plot_rows = [
        {
            "offset_mm": label,
            "error_mm": 1000.0 * float(row["l1_m"]),
        }
        for label, row in zip(labels, rows)
    ]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    sns.barplot(
        data=_column_mapping(plot_rows),
        x="offset_mm",
        y="error_mm",
        ax=axis,
        color=sns.color_palette("deep")[0],
    )
    axis.set_xlabel("Spatial-Mask Cube-Bottom Z Offset [mm]")
    axis.set_ylabel("Depth Error [mm]")
    axis.set_title("L1 Depth Error Under Recomputed Spatial Masks")
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
    sequence_labels = _sequence_display_map(
        [str(sample["sequence"]) for sample in samples]
    )
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
                    f"{sequence_labels[str(sample['sequence'])]}\nFrame {sample['frame_idx']}",
                    fontsize=9,
                )

    figure.suptitle(
        "Spatial Mask Z-Offset Sweep: GT Depth With Recomputed Mask Overlay",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "spatial_mask_z_offset_overview.png", dpi=180)
    plt.close(figure)


def _evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    comparison_label: str,
) -> dict[str, Any]:
    print(f"\nLoading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, metadata = _checkpoint_state(checkpoint)
    expected_transform = INTRINSICS_TRANSFORM
    checkpoint_transform = metadata.get("intrinsics_transform", "")
    if checkpoint_transform != expected_transform:
        raise RuntimeError(
            f"Checkpoint uses {checkpoint_transform!r}, but evaluation requested "
            f"{expected_transform!r}. Use a crop-then-resize checkpoint."
        )
    model = _build_model(metadata, state, device)

    model_arch = str(metadata.get("model_arch", "ModernMVSNet"))
    unet_model = (
        model_arch in ("UNet", "UNet+uncertainty", "RecurrentUNet")
        or "predict_uncertainty" in metadata
    )
    evaluates_confidence = (
        not unet_model
        and bool(metadata.get("uncertainty", False))
        and any(key.startswith("confidence_head.") for key in state)
    )
    recurrent_model = bool(metadata.get("recurrent", False)) or model_arch == "RecurrentUNet"
    num_views = int(metadata.get("num_views", 1)) if unet_model else int(
        _required_metadata(metadata, "num_views", 5)
    )
    model_display_name = _model_display_name(unet_model, num_views)
    pose_channels = bool(metadata.get("pose_channels", False)) if unet_model else False
    recurrent_enrollment_range = int(metadata.get("recurrent_enrollment_range", 0))
    view_interval = int(_required_metadata(metadata, "view_interval", 5))
    pose_view_selection = bool(_required_metadata(metadata, "pose_view_selection", False))
    pose_move_threshold = float(
        _required_metadata(metadata, "pose_move_threshold", 0.01)
    )
    allow_unbalanced_pose_views = bool(
        metadata.get(
            "allow_fewer_pose_views",
            metadata.get("allow_unbalanced_pose_views", False),
        )
    )
    if args.pose_layout_override is not None:
        allow_unbalanced_pose_views = args.pose_layout_override
    if pose_view_selection:
        policy_source = (
            "CLI override" if args.pose_layout_override is not None else "checkpoint"
        )
        print(
            "Pose-view layouts: "
            + (
                "up to four sources per side; missing boundary views masked"
                if allow_unbalanced_pose_views
                else "strictly balanced"
            )
            + f" ({policy_source})",
            flush=True,
        )
    coarse_depths = int(_required_metadata(metadata, "coarse_depths", 32))
    linear_depth_candidates = bool(
        _required_metadata(metadata, "linear_depth_candidates", False)
    )

    evaluation_root = args.data_dir
    sequence_dirs = find_precomputed_sequences(evaluation_root)
    if not sequence_dirs:
        raise RuntimeError(
            f"No valid sequences found at or directly under {evaluation_root}"
        )
    selected_example_requests: list[tuple[str, int]] = []
    if args.example_sequence is not None:
        raw_sequence_names = [sequence_dir.name for sequence_dir in sequence_dirs]
        display_to_raw = {
            display.removeprefix("Sequence "): raw
            for raw, display in _sequence_display_map(raw_sequence_names).items()
        }
        for requested_sequence, requested_frame in zip(
            args.example_sequence, args.example_frame
        ):
            requested_sequence = str(requested_sequence)
            if requested_sequence in raw_sequence_names:
                resolved_sequence = requested_sequence
            elif requested_sequence in display_to_raw:
                resolved_sequence = display_to_raw[requested_sequence]
            else:
                raise ValueError(
                    f"Unknown --example_sequence {requested_sequence!r}; use a raw "
                    f"sequence name or a displayed index from 1 to {len(sequence_dirs)}"
                )
            selected_example_requests.append(
                (resolved_sequence, int(requested_frame))
            )
    selected_example_request_set = set(selected_example_requests)
    selected_frames_by_sequence: dict[str, list[int]] = {}
    for sequence_name, frame_index in selected_example_requests:
        selected_frames_by_sequence.setdefault(sequence_name, []).append(frame_index)

    pose_layout_indicator = _pose_layout_directory_indicator(
        allow_unbalanced_pose_views
    )
    output_dir = args.results_folder / (
        f"{pose_layout_indicator}_{checkpoint_path.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = load_event_calibration(_CAMERA_DATA_DIR)
    depth_total = DepthAccumulator()
    lower_cube_depth_total = DepthAccumulator()
    upper_cube_depth_total = DepthAccumulator()
    boundary_totals_by_region = {
        domain_slug: {
            "boundary": ErrorAccumulator(),
            "non_boundary": ErrorAccumulator(),
        }
        for domain_slug, _, _, _ in PER_SEQUENCE_REGION_SPECS
    }
    mvc_total = MVCAccumulator()
    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    worst_frame_by_domain = {
        domain_slug: {}
        for domain_slug, _, _, _ in PER_SEQUENCE_REGION_SPECS
    }
    qualitative_sample_by_sequence: dict[str, dict[str, Any]] = {}
    qualitative_frames_seen: dict[str, int] = {}
    qualitative_rng = np.random.default_rng(QUALITATIVE_RANDOM_SEED)
    selected_example_samples: dict[tuple[str, int], dict[str, Any]] = {}
    spatial_mask_offset_samples: dict[str, dict[str, Any]] = {}
    uncertainty_samples_by_domain = {
        domain_slug: {"uncertainty": [], "error": [], "seen": 0}
        for domain_slug, _, _, _ in PER_SEQUENCE_REGION_SPECS
    }
    uncertainty_rng = np.random.default_rng(20260812)
    boundary_region_samples: list[dict[str, Any]] = []
    boundary_region_frames_seen = 0
    boundary_region_rng = np.random.default_rng(
        BOUNDARY_VISUALIZATION_RANDOM_SEED
    )
    frame_count = 0
    spatial_mask_z_offsets_m = np.linspace(
        args.spatial_mask_offset_min,
        args.spatial_mask_offset_max,
        args.spatial_mask_offset_steps,
        dtype=np.float64,
    ) + SPATIAL_MASK_VERTICAL_SHIFT_M
    plus_1cm_indices = np.flatnonzero(
        np.isclose(
            spatial_mask_z_offsets_m,
            BOUNDARY_SPATIAL_MASK_OFFSET_M,
            rtol=0.0,
            atol=1e-12,
        )
    )
    if plus_1cm_indices.size:
        boundary_mask_offset_index = int(plus_1cm_indices[0])
        mask_offsets_for_computation = spatial_mask_z_offsets_m
    else:
        boundary_mask_offset_index = len(spatial_mask_z_offsets_m)
        mask_offsets_for_computation = np.concatenate(
            [
                spatial_mask_z_offsets_m,
                np.asarray([BOUNDARY_SPATIAL_MASK_OFFSET_M], dtype=np.float64),
            ]
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
        [
            args.target_x,
            args.target_y,
            args.target_z + SPATIAL_MASK_VERTICAL_SHIFT_M,
        ],
        dtype=np.float64,
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
            allow_unbalanced_pose_views=allow_unbalanced_pose_views,
            coarse_depths=coarse_depths,
            linear_depth_candidates=linear_depth_candidates,
            fill_invalid=args.fill_invalid,
            pose_channels=pose_channels,
            recurrent=recurrent_model,
            recurrent_enrollment_range=recurrent_enrollment_range,
            aug=MultiViewAugConfig(enabled=False),
        )
        available_frames = len(dataset)
        all_valid_indices = dataset.valid_indices.copy()
        if args.fast_mode > 1:
            dataset.valid_indices = dataset.valid_indices[
                dataset.valid_indices % args.fast_mode == 0
            ]
        requested_frames = selected_frames_by_sequence.get(sequence_dir.name, [])
        for requested_frame in requested_frames:
            if requested_frame not in all_valid_indices:
                raise ValueError(
                    f"Requested example frame {requested_frame} is not a valid "
                    f"target for sequence {sequence_dir.name} and this checkpoint"
                )
        if requested_frames:
            dataset.valid_indices = np.unique(
                np.append(dataset.valid_indices, requested_frames)
            )
        if args.fast_mode > 1:
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
        sequence_lower_cube_depth = DepthAccumulator()
        sequence_upper_cube_depth = DepthAccumulator()
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
                view_valid_mask = batch["view_valid_mask"].to(
                    device, non_blocking=True
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timers.add("device_transfer", time.perf_counter() - t_transfer)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_start = time.perf_counter()
                if evaluates_confidence:
                    model_output = model(
                        imgs,
                        cam_mats,
                        K,
                        depth_values,
                        return_uncertainty=True,
                        view_valid_mask=view_valid_mask,
                    )
                    prediction_norm, confidence = model_output
                else:
                    prediction_norm = model(
                        imgs,
                        cam_mats,
                        K,
                        depth_values,
                        view_valid_mask=view_valid_mask,
                    )
                    confidence = None
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
                uncertainty_batch = (
                    (1.0 - confidence).detach().cpu() if confidence is not None else None
                )

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

                    computed_offset_masks = depth_cube_masks_for_bottom_offsets(
                        gt_np,
                        cam_batch[index_in_batch],
                        K_batch[index_in_batch],
                        args.target_x,
                        args.target_y,
                        cube_half_side,
                        mask_offsets_for_computation,
                    )
                    offset_masks = computed_offset_masks[
                        : len(spatial_mask_z_offsets_m)
                    ]
                    boundary_spatial_mask = computed_offset_masks[
                        boundary_mask_offset_index
                    ]
                    cube_mask = depth_cube_mask(
                        gt_np,
                        cam_batch[index_in_batch],
                        K_batch[index_in_batch],
                        cube_center,
                        cube_half_side,
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
                    lower_cube_valid = valid_np & cube_mask
                    upper_cube_valid = valid_np & boundary_spatial_mask
                    lower_cube_depth_total.add(
                        pred_np[lower_cube_valid], gt_np[lower_cube_valid]
                    )
                    upper_cube_depth_total.add(
                        pred_np[upper_cube_valid], gt_np[upper_cube_valid]
                    )
                    sequence_lower_cube_depth.add(
                        pred_np[lower_cube_valid], gt_np[lower_cube_valid]
                    )
                    sequence_upper_cube_depth.add(
                        pred_np[upper_cube_valid], gt_np[upper_cube_valid]
                    )
                    if uncertainty_batch is not None:
                        uncertainty_np = uncertainty_batch[
                            index_in_batch, 0
                        ].numpy()
                        for domain_slug, domain_valid in (
                            ("whole_frame", valid_np),
                            ("workspace_cube", lower_cube_valid),
                            ("raised_cube", upper_cube_valid),
                        ):
                            _add_uncertainty_reservoir_samples(
                                uncertainty_samples_by_domain[domain_slug],
                                uncertainty_np[domain_valid],
                                absolute_error[domain_valid],
                                uncertainty_rng,
                            )
                    frame_metrics = _frame_depth_metrics(pred_np, gt_np, valid_np)
                    lower_cube_frame_metrics = _frame_depth_metrics(
                        pred_np, gt_np, lower_cube_valid
                    )
                    upper_cube_frame_metrics = _frame_depth_metrics(
                        pred_np, gt_np, upper_cube_valid
                    )
                    frame_mae = float(frame_metrics["mae_m"])
                    frame_worst10_mae = _worst_fraction_l1(
                        absolute_error[valid_np]
                    )
                    target_events = (
                        batch["imgs"][index_in_batch, 0, :NUM_BINS]
                        .float()
                        .numpy()
                    )
                    event_activity_image = np.abs(target_events).sum(axis=0)
                    cube_event_activity = float(event_activity_image[cube_mask].sum())
                    # Outside-cube activity is defined over every pixel with a
                    # measured GT depth, independently of the main metric mask.
                    gt_depth_valid = np.isfinite(gt_np) & (gt_np > 0.0)
                    outside_cube_mask = gt_depth_valid & ~cube_mask
                    outside_cube_event_activity = float(
                        event_activity_image[outside_cube_mask].sum()
                    )

                    for domain_slug, domain_mae, domain_valid in (
                        ("whole_frame", frame_mae, valid_np),
                        (
                            "workspace_cube",
                            float(lower_cube_frame_metrics["mae_m"]),
                            lower_cube_valid,
                        ),
                        (
                            "raised_cube",
                            float(upper_cube_frame_metrics["mae_m"]),
                            upper_cube_valid,
                        ),
                    ):
                        domain_worst = worst_frame_by_domain[domain_slug]
                        previous_worst = domain_worst.get(sequence_dir.name)
                        if not math.isfinite(domain_mae) or (
                            previous_worst is not None
                            and domain_mae <= previous_worst["mae_m"]
                        ):
                            continue
                        domain_worst[sequence_dir.name] = {
                            "sequence": sequence_dir.name,
                            "frame_idx": int(ref_indices[index_in_batch]),
                            "mae_m": domain_mae,
                            "events": target_events.copy(),
                            "gt": gt_np.copy(),
                            "pred": pred_np.copy(),
                            "valid": domain_valid.copy(),
                        }

                    error_np = np.abs(pred_np - gt_np)
                    boundary_masks_by_region: dict[
                        str, tuple[np.ndarray, np.ndarray]
                    ] = {}
                    for domain_slug, domain_valid in (
                        ("whole_frame", valid_np),
                        ("workspace_cube", lower_cube_valid),
                        ("raised_cube", upper_cube_valid),
                    ):
                        domain_valid_t = torch.from_numpy(domain_valid)
                        boundary_mask, non_boundary_mask = _boundary_masks(
                            domain_valid_t,
                            domain_valid_t,
                            dilation_px=args.boundary_dilation,
                        )
                        boundary_np = boundary_mask.numpy()
                        non_boundary_np = non_boundary_mask.numpy()
                        boundary_masks_by_region[domain_slug] = (
                            boundary_np,
                            non_boundary_np,
                        )
                        boundary_totals_by_region[domain_slug]["boundary"].add(
                            error_np[boundary_np]
                        )
                        boundary_totals_by_region[domain_slug]["non_boundary"].add(
                            error_np[non_boundary_np]
                        )

                    boundary_np, non_boundary_np = boundary_masks_by_region[
                        "whole_frame"
                    ]
                    sequence_boundary.add(error_np[boundary_np])
                    sequence_non_boundary.add(error_np[non_boundary_np])

                    masked_boundary_np, masked_non_boundary_np = (
                        boundary_masks_by_region["raised_cube"]
                    )

                    boundary_region_frames_seen += 1
                    if (
                        len(boundary_region_samples)
                        < BOUNDARY_VISUALIZATION_SAMPLE_COUNT
                    ):
                        replacement = len(boundary_region_samples)
                    else:
                        replacement = int(
                            boundary_region_rng.integers(
                                0, boundary_region_frames_seen
                            )
                        )
                    if replacement < BOUNDARY_VISUALIZATION_SAMPLE_COUNT:
                        boundary_region_sample = {
                            "sequence": sequence_dir.name,
                            "frame_idx": int(ref_indices[index_in_batch]),
                            "gt": gt_np.copy(),
                            "spatial_mask": boundary_spatial_mask.copy(),
                            "boundary": masked_boundary_np.copy(),
                            "non_boundary": masked_non_boundary_np.copy(),
                        }
                        if replacement == len(boundary_region_samples):
                            boundary_region_samples.append(
                                boundary_region_sample
                            )
                        else:
                            boundary_region_samples[replacement] = (
                                boundary_region_sample
                            )

                    current = FramePrediction(
                        frame_idx=int(ref_indices[index_in_batch]),
                        pred=pred_np,
                        T_cam_from_world=cam_batch[index_in_batch],
                        K=K_batch[index_in_batch],
                    )
                    if not args.no_mvc and len(history) >= args.mvc_frame_offset:
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
                    if not args.no_mvc:
                        history.append(current)

                    pose_idx = current.frame_idx
                    if pose_idx < len(pose_motion["position_m"]):
                        position = pose_motion["position_m"][pose_idx]
                        rotation = pose_motion["rotation_xyz_deg"][pose_idx]
                        arm_speed = float(pose_motion["speed_m_s"][pose_idx])
                        normalized_time = float(
                            pose_motion["normalized_time"][pose_idx]
                        )
                    else:
                        position = np.full(3, np.nan, dtype=np.float64)
                        rotation = np.full(3, np.nan, dtype=np.float64)
                        arm_speed = math.nan
                        normalized_time = (
                            pose_idx / (dataset.n_frames - 1)
                            if dataset.n_frames > 1
                            else 0.0
                        )

                    # Reservoir sampling selects one evaluated frame uniformly
                    # at random for each sequence without retaining all frames.
                    qualitative_frames_seen[sequence_dir.name] = (
                        qualitative_frames_seen.get(sequence_dir.name, 0) + 1
                    )
                    if qualitative_rng.integers(
                        0, qualitative_frames_seen[sequence_dir.name]
                    ) == 0:
                        qualitative_sample_by_sequence[sequence_dir.name] = {
                            "sequence": sequence_dir.name,
                            "frame_idx": current.frame_idx,
                            "normalized_time": normalized_time,
                            "events": target_events.copy(),
                            "gt": gt_np.copy(),
                            "pred": pred_np.copy(),
                            "valid": valid_np.copy(),
                            "prediction_workspace_mask": depth_cube_mask(
                                pred_np,
                                cam_batch[index_in_batch],
                                K_batch[index_in_batch],
                                cube_center,
                                cube_half_side,
                            ),
                            "lower_cube_valid": lower_cube_valid.copy(),
                            "upper_cube_valid": upper_cube_valid.copy(),
                        }
                    selected_example_key = (
                        sequence_dir.name,
                        current.frame_idx,
                    )
                    if selected_example_key in selected_example_request_set:
                        selected_example_samples[selected_example_key] = {
                            "sequence": sequence_dir.name,
                            "frame_idx": current.frame_idx,
                            "events": target_events.copy(),
                            "gt": gt_np.copy(),
                            "pred": pred_np.copy(),
                            "prediction_workspace_mask": depth_cube_mask(
                                pred_np,
                                cam_batch[index_in_batch],
                                K_batch[index_in_batch],
                                cube_center,
                                cube_half_side,
                            ),
                            "lower_cube_valid": lower_cube_valid.copy(),
                        }
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
                            "normalized_time": normalized_time,
                            "recording_frames": dataset.n_frames,
                            **frame_metrics,
                            **{
                                f"lower_cube_{name}": value
                                for name, value in lower_cube_frame_metrics.items()
                            },
                            **{
                                f"upper_cube_{name}": value
                                for name, value in upper_cube_frame_metrics.items()
                            },
                            "worst10_mae_m": frame_worst10_mae,
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
                            "boundary_spatial_mask_plus_1cm_pixels": int(
                                masked_boundary_np.sum()
                            ),
                            "boundary_spatial_mask_plus_1cm_mae_m": (
                                float(error_np[masked_boundary_np].mean())
                                if masked_boundary_np.any()
                                else math.nan
                            ),
                            "non_boundary_spatial_mask_plus_1cm_pixels": int(
                                masked_non_boundary_np.sum()
                            ),
                            "non_boundary_spatial_mask_plus_1cm_mae_m": (
                                float(error_np[masked_non_boundary_np].mean())
                                if masked_non_boundary_np.any()
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
        sequence_lower_cube_metrics = sequence_lower_cube_depth.metrics()
        sequence_upper_cube_metrics = sequence_upper_cube_depth.metrics()
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
                **{
                    f"lower_cube_{name}": value
                    for name, value in sequence_lower_cube_metrics.items()
                },
                **{
                    f"upper_cube_{name}": value
                    for name, value in sequence_upper_cube_metrics.items()
                },
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
            "checkpoint": str(checkpoint_path),
            "comparison_label": comparison_label,
            "architecture_label": model_display_name,
            "model_arch": model_arch,
            "num_views": num_views,
            "view_interval": view_interval,
            "pose_view_selection": pose_view_selection,
            "pose_move_threshold_m": pose_move_threshold,
            "allow_unbalanced_pose_views": allow_unbalanced_pose_views,
            "allow_fewer_pose_views": allow_unbalanced_pose_views,
            "pose_layout_cli_override": args.pose_layout_override,
            "coarse_depths": coarse_depths,
            "linear_depth_candidates": linear_depth_candidates,
            "fine_depths": metadata.get("fine_depths"),
        },
        "evaluation_configuration": {
            "fill_invalid": args.fill_invalid,
            "intrinsics_transform": INTRINSICS_TRANSFORM,
            "frame_step": args.fast_mode,
            "boundary_definition": (
                "inside pixel adjacent (4-connected) to undefined GT or "
                "outside valid measured depth"
            ),
            "boundary_dilation_px": args.boundary_dilation,
            "boundary_spatial_mask_offset_m": (
                BOUNDARY_SPATIAL_MASK_OFFSET_M
            ),
            "mvc_enabled": not args.no_mvc,
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
            "depth_metric_regions": {
                "whole_image": "all valid evaluated depth pixels",
                "lower_cube": {
                    "description": "table and object",
                    "center_world_m": cube_center.tolist(),
                    "bottom_z_m": float(cube_center[2] - cube_half_side),
                    "side_m": args.cube_side,
                },
                "upper_cube": {
                    "description": "object only; raised to exclude the table",
                    "center_world_m": [
                        args.target_x,
                        args.target_y,
                        float(BOUNDARY_SPATIAL_MASK_OFFSET_M + cube_half_side),
                    ],
                    "bottom_z_m": BOUNDARY_SPATIAL_MASK_OFFSET_M,
                    "side_m": args.cube_side,
                },
                "mask_source": "measured GT depth backprojected to world coordinates",
            },
            "spatial_mask_offset_min_m": float(spatial_mask_z_offsets_m[0]),
            "spatial_mask_offset_max_m": float(spatial_mask_z_offsets_m[-1]),
            "spatial_mask_offset_steps": args.spatial_mask_offset_steps,
            "spatial_mask_offset_definition": (
                "offset is the world-frame cube-bottom Z; cube center Z equals "
                "offset + cube_side/2; all configured offsets include an "
                f"additional {SPATIAL_MASK_VERTICAL_SHIFT_M:.3f} m upward shift; "
                "recomputed mask replaces the stored mask"
            ),
            "spatial_mask_vertical_shift_m": SPATIAL_MASK_VERTICAL_SHIFT_M,
        },
        # Keep the historical whole-image key for downstream compatibility.
        "depth_metrics": depth_total.metrics(),
        "depth_metrics_by_region": {
            "whole_image": depth_total.metrics(),
            "lower_cube": lower_cube_depth_total.metrics(),
            "upper_cube": upper_cube_depth_total.metrics(),
        },
        "boundary_metrics_by_region": {
            domain_slug: {
                region: accumulator.metrics()
                for region, accumulator in region_accumulators.items()
            }
            for domain_slug, region_accumulators in (
                boundary_totals_by_region.items()
            )
        },
        "boundary_metrics": {
            region: accumulator.metrics()
            for region, accumulator in boundary_totals_by_region[
                "whole_frame"
            ].items()
        },
        "boundary_metrics_spatial_mask_plus_1cm": {
            "z_offset_m": BOUNDARY_SPATIAL_MASK_OFFSET_M,
            "offset_definition": "world-frame cube-bottom Z",
            "boundary": boundary_totals_by_region["raised_cube"][
                "boundary"
            ].metrics(),
            "non_boundary": boundary_totals_by_region["raised_cube"][
                "non_boundary"
            ].metrics(),
        },
        "multiview_consistency": mvc_total.metrics(),
        "cube_event_activity_vs_error": activity_summary,
        "spatial_mask_z_offset_metrics": spatial_mask_offset_rows,
    }

    t_output_writing = time.perf_counter()
    _write_csv(output_dir / "per_frame_metrics.csv", frame_rows)
    _write_csv(output_dir / "per_sequence_metrics.csv", sequence_rows)
    _write_csv(output_dir / "depth_metrics_by_region.csv", _depth_region_rows(summary))
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
    for domain_slug, domain_label, mae_field, _ in PER_SEQUENCE_REGION_SPECS:
        _plot_normalized_time_error(
            output_dir,
            frame_rows,
            domain_slug,
            domain_label,
            mae_field,
        )
    _plot_spatial_mask_offset_metrics(output_dir, spatial_mask_offset_rows)
    _plot_spatial_mask_offset_overview(
        output_dir,
        list(spatial_mask_offset_samples.values()),
        spatial_mask_z_offsets_m,
    )
    _plot_masked_boundary_regions(output_dir, boundary_region_samples)
    for domain_slug, domain_label, _, _ in PER_SEQUENCE_REGION_SPECS:
        domain_samples = uncertainty_samples_by_domain[domain_slug]
        _plot_uncertainty_vs_prediction_error(
            output_dir,
            domain_samples["uncertainty"],
            domain_samples["error"],
            domain_slug,
            domain_label,
        )
    qualitative_samples = list(qualitative_sample_by_sequence.values())
    _plot_qualitative_depth_results(output_dir, qualitative_samples)
    missing_example_requests = [
        request
        for request in selected_example_requests
        if request not in selected_example_samples
    ]
    if missing_example_requests:
        raise RuntimeError(
            "Selected sequence/frame pairs were not captured during evaluation: "
            f"{missing_example_requests}"
        )
    selected_example_rows = [
        selected_example_samples[request]
        for request in selected_example_requests
        if request in selected_example_samples
    ]
    _plot_selected_frames_overview(output_dir, selected_example_rows)
    _plot_selected_frames_overview(
        output_dir,
        selected_example_rows,
        include_error=False,
    )
    _plot_error_maps_by_region(output_dir, qualitative_samples)
    worst_frame_samples_by_domain = {}
    for domain_slug, domain_label, _, _ in PER_SEQUENCE_REGION_SPECS:
        domain_samples = list(worst_frame_by_domain[domain_slug].values())
        worst_frame_samples_by_domain[domain_slug] = domain_samples
        _plot_worst_frames(
            output_dir,
            domain_samples,
            domain_slug,
            domain_label,
        )
    timers.add("plotting", time.perf_counter() - t_plotting)
    eval_time_rows = timers.rows(frame_count)
    _write_eval_times(output_dir, eval_time_rows, frame_count)

    print((output_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    for domain_slug, domain_label, _, _ in PER_SEQUENCE_REGION_SPECS:
        if worst_frame_samples_by_domain[domain_slug]:
            print(
                f"Worst-frame overview ({domain_label}): "
                f"{(output_dir / f'worst_error_overview_{domain_slug}.png').resolve()}",
                flush=True,
            )
    if qualitative_samples:
        print(
            f"Qualitative depth results: "
            f"{(output_dir / 'qualitative_depth_results.png').resolve()}",
            flush=True,
        )
        print(
            f"Regional error maps: "
            f"{(output_dir / 'error_maps_by_region.png').resolve()}",
            flush=True,
        )
    if selected_example_rows:
        print(
            f"Selected-frames overview: "
            f"{(output_dir / 'selected_frames_overview.png').resolve()}",
            flush=True,
        )
        print(
            f"Selected-frames depth-only overview: "
            f"{(output_dir / 'selected_frames_only_depth.png').resolve()}",
            flush=True,
        )
    print(f"Evaluation timing: {(output_dir / 'eval_times.txt').resolve()}", flush=True)
    print(f"Results written to: {output_dir.resolve()}", flush=True)
    return {
        "label": comparison_label,
        "allow_unbalanced_pose_views": allow_unbalanced_pose_views,
        "summary": summary,
        "frame_rows": frame_rows,
        "sequence_rows": sequence_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net-table or multiview checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more supported depth-model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_label",
        "--checkpoint-label",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional display labels paired positionally with --checkpoint. "
            "By default, each checkpoint's full filename is used, keeping "
            "different MVS pipelines distinct in comparison tables and plots."
        ),
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
        help=(
            "Root output directory. Each checkpoint gets a policy-prefixed "
            "subdirectory."
        ),
    )
    parser.add_argument(
        "--comparison_name",
        type=str,
        default="",
        help=(
            "Optional folder name for multi-checkpoint comparison results. "
            "An empty value uses the policy-derived default name."
        ),
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
        "--example_sequence",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Sequences used for selected_frames_overview.png. Each entry accepts "
            "either a raw sequence-directory name or its displayed 1-based index."
        ),
    )
    parser.add_argument(
        "--example_frame",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Original frame indices paired positionally with --example_sequence. "
            "Selected frames are evaluated even when --fast_mode would omit them."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--fill_invalid",
        action="store_true",
        help="Fill invalid GT pixels using the table-plane prior before evaluation.",
    )
    pose_layout_group = parser.add_mutually_exclusive_group()
    pose_layout_group.add_argument(
        "--allow_fewer_pose_views",
        "--allow-fewer-pose-views",
        "--allow_unbalanced_pose_views",
        "--allow-unbalanced-pose-views",
        dest="pose_layout_override",
        action="store_const",
        const=True,
        help="Override the checkpoint and allow fewer masked source views at boundaries.",
    )
    pose_layout_group.add_argument(
        "--strict_balanced_pose_views",
        "--strict-balanced-pose-views",
        dest="pose_layout_override",
        action="store_const",
        const=False,
        help="Override the checkpoint and require balanced pose-view layouts.",
    )
    parser.set_defaults(pose_layout_override=None)
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=0.02,
        help=(
            "Deprecated compatibility option; boundary detection now uses "
            "defined/undefined and inside/outside spatial-mask transitions."
        ),
    )
    parser.add_argument(
        "--boundary_dilation",
        type=int,
        default=2,
        help="Boundary-region dilation radius in pixels.",
    )
    parser.add_argument(
        "--no_mvc",
        action="store_true",
        help=(
            "Disable multi-view consistency reprojection and metrics. This can "
            "substantially reduce CPU evaluation time."
        ),
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
    if (args.example_sequence is None) != (args.example_frame is None):
        parser.error(
            "--example_sequence and --example_frame must be provided together"
        )
    if args.example_sequence is not None:
        if len(args.example_sequence) != len(args.example_frame):
            parser.error(
                "--example_sequence and --example_frame must contain the same "
                "number of entries"
            )
        if any(frame < 0 for frame in args.example_frame):
            parser.error("all --example_frame values must be >= 0")
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
    if args.checkpoint_label is not None:
        if len(args.checkpoint_label) != len(args.checkpoint):
            parser.error(
                "--checkpoint_label must contain exactly one label per --checkpoint"
            )
        if any(not label.strip() for label in args.checkpoint_label):
            parser.error("--checkpoint_label entries must not be empty")
        if len(set(args.checkpoint_label)) != len(args.checkpoint_label):
            parser.error("--checkpoint_label entries must be unique")
    if args.comparison_name:
        comparison_name = Path(args.comparison_name)
        if comparison_name.is_absolute() or len(comparison_name.parts) != 1:
            parser.error("--comparison_name must be a single folder name")
        if args.comparison_name in {".", ".."}:
            parser.error("--comparison_name must not be '.' or '..'")
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
    comparison_labels = (
        args.checkpoint_label
        if args.checkpoint_label is not None
        else [checkpoint_path.name for checkpoint_path in args.checkpoint]
    )
    if len(set(comparison_labels)) != len(comparison_labels):
        raise RuntimeError(
            "Checkpoint filenames are not unique. Use --checkpoint_label to "
            "supply distinct labels for this comparison."
        )
    runs = [
        _evaluate_checkpoint(checkpoint_path, args, device, comparison_label)
        for checkpoint_path, comparison_label in zip(
            args.checkpoint, comparison_labels
        )
    ]
    if len(runs) > 1:
        comparison_policies = {
            bool(run["allow_unbalanced_pose_views"]) for run in runs
        }
        if len(comparison_policies) == 1:
            comparison_indicator = _pose_layout_directory_indicator(
                comparison_policies.pop()
            )
        else:
            comparison_indicator = "unbalanced_pose_views_mixed"
        comparison_dir = args.results_folder / (
            args.comparison_name or f"{comparison_indicator}_comparison"
        )
        _write_comparison_results(comparison_dir, runs)
        print(f"Comparison written to: {comparison_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
