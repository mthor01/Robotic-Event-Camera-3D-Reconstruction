#!/usr/bin/env python3
"""Evaluate multiview.py checkpoints on held-out sequences.

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
    lines = [
        f"Checkpoint: {summary['checkpoint']}",
        f"Evaluation root: {summary['evaluation_root']}",
        f"Sequences: {summary['sequence_count']}",
        f"Frames: {summary['frame_count']}",
        f"Evaluation frame step: {summary['evaluation_configuration']['frame_step']}",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_results(
    output_dir: Path,
    summary: dict[str, Any],
    frame_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = summary["depth_metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    error_names = ["abs_rel", "sq_rel_m", "mae_m", "rmse_m", "rmse_log"]
    axes[0].bar(error_names, [depth[name] for name in error_names], color="#4472c4")
    axes[0].set_title("Depth errors")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    delta_names = ["delta_1", "delta_2", "delta_3"]
    axes[1].bar(
        [r"$\delta<1.25$", r"$\delta<1.25^2$", r"$\delta<1.25^3$"],
        [100.0 * depth[name] for name in delta_names],
        color="#70ad47",
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].set_title("Threshold accuracy")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "depth_metrics.png", dpi=180)
    plt.close(figure)

    boundary = summary["boundary_metrics"]
    mvc = summary["multiview_consistency"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        ["Boundary MAE", "Non-boundary MAE"],
        [boundary["boundary"]["mae_m"], boundary["non_boundary"]["mae_m"]],
        color=["#c55a11", "#5b9bd5"],
    )
    axes[0].set_ylabel("Error [m]")
    axes[0].set_title("Depth-boundary error")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        ["<1 cm", "<2 cm", "<5 cm"],
        [
            100.0 * mvc["within_1cm"],
            100.0 * mvc["within_2cm"],
            100.0 * mvc["within_5cm"],
        ],
        color="#8064a2",
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Consistent correspondences [%]")
    axes[1].set_title("Multi-view consistency thresholds")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "boundary_and_consistency.png", dpi=180)
    plt.close(figure)

    if frame_rows:
        mae_cm = [100.0 * float(row["mae_m"]) for row in frame_rows]
        rmse_cm = [100.0 * float(row["rmse_m"]) for row in frame_rows]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].hist(mae_cm, bins=40, color="#4472c4", alpha=0.85)
        axes[0].set_xlabel("Per-frame MAE [cm]")
        axes[0].set_ylabel("Frames")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].hist(rmse_cm, bins=40, color="#ed7d31", alpha=0.85)
        axes[1].set_xlabel("Per-frame RMSE [cm]")
        axes[1].set_ylabel("Frames")
        axes[1].grid(axis="y", alpha=0.25)
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
            axes[0].scatter(activity, mae_m, s=9, alpha=0.35, linewidths=0)
            axes[0].set_xlabel("Event activity at GT-depth pixels inside cube")
            axes[0].set_ylabel("Whole-frame MAE [m]")
            axes[0].set_title("Relevant-region activity vs prediction error")
            axes[0].grid(True, alpha=0.25)

            bins = summary["cube_event_activity_vs_error"]["activity_bins"]
            axes[1].plot(
                [row["mean_activity"] for row in bins],
                [row["mean_mae_m"] for row in bins],
                marker="o",
                color="#c55a11",
            )
            axes[1].set_xlabel("Mean cube-masked event activity")
            axes[1].set_ylabel("Mean whole-frame MAE [m]")
            axes[1].set_title("Equal-frame activity bins")
            axes[1].grid(True, alpha=0.25)
            figure.tight_layout()
            figure.savefig(
                output_dir / "cube_event_activity_vs_error.png", dpi=180
            )
            plt.close(figure)


def _plot_worst_frames(
    output_dir: Path,
    samples: list[dict[str, Any]],
) -> None:
    """Save the highest-MAE frame from each evaluated object/sequence."""
    if not samples:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = sorted(samples, key=lambda sample: sample["sequence"])
    figure, axes = plt.subplots(
        len(samples),
        4,
        figsize=(16, 3.5 * len(samples)),
        squeeze=False,
    )
    for column, title in enumerate(
        ("Events (summed)", "GT depth", "Pred depth", "Abs error")
    ):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")

    depth_cmap = plt.get_cmap("turbo")
    error_cmap = plt.get_cmap("turbo")
    invalid_color = np.array([40, 40, 40], dtype=np.uint8)

    def colorize(values: np.ndarray, vmin: float, vmax: float, cmap: Any) -> np.ndarray:
        normalized = np.clip((values - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        return (cmap(normalized)[..., :3] * 255).astype(np.uint8)

    for row, sample in enumerate(samples):
        ax_events, ax_gt, ax_pred, ax_error = axes[row]
        valid = sample["valid"]
        pred_valid = (
            np.isfinite(sample["pred"])
            & (sample["pred"] >= DEPTH_MIN)
            & (sample["pred"] <= D_MAX)
        )

        event_sum = sample["events"].sum(axis=0)
        event_min = float(event_sum.min())
        event_range = float(event_sum.max() - event_min)
        event_vis = np.clip((event_sum - event_min) / max(event_range, 1e-6), 0.0, 1.0)
        ax_events.imshow(event_vis, cmap="gray", vmin=0.0, vmax=1.0)
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


def _evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    print(f"\nLoading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state, metadata = _checkpoint_state(checkpoint)
    model = _build_model(metadata, state, device)

    num_views = int(_required_metadata(metadata, "num_views", 5))
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
    frame_count = 0
    cube_center = np.asarray(
        [args.target_x, args.target_y, args.target_z], dtype=np.float64
    )
    cube_half_side = args.cube_side / 2.0

    for sequence_number, sequence_dir in enumerate(sequence_dirs, start=1):
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
            for batch_number, batch in enumerate(loader, start=1):
                imgs = batch["imgs"].to(device, non_blocking=True)
                cam_mats = batch["cam_mats"].to(device, non_blocking=True)
                K = batch["K"].to(device, non_blocking=True)
                depth_values = batch["depth_values"].to(device, non_blocking=True)
                prediction_norm = model(imgs, cam_mats, K, depth_values)
                prediction = (
                    prediction_norm * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                ).detach().cpu()

                gt_batch = batch["dep_t"].float()
                mask_batch = batch["mask_t"] > 0.5
                ref_indices = batch["ref_idx"].tolist()
                cam_batch = batch["cam_mats"][:, 0].float().numpy()
                K_batch = batch["K"].float().numpy()

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
                        }
                    )
                    frame_count += 1

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
            }
        )

    activity_summary = _activity_error_summary(frame_rows)
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_val_l1_m": metadata.get("val_l1"),
        "evaluation_root": str(evaluation_root.resolve()),
        "sequence_count": len(sequence_dirs),
        "frame_count": frame_count,
        "model_configuration": {
            "num_views": num_views,
            "view_interval": view_interval,
            "pose_view_selection": pose_view_selection,
            "pose_move_threshold_m": pose_move_threshold,
            "num_depths": num_depths,
            "fine_depths": metadata.get("fine_depths"),
            "feature_encoder": metadata.get("feature_encoder"),
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
        },
        "depth_metrics": depth_total.metrics(),
        "boundary_metrics": {
            "boundary": boundary_total.metrics(),
            "non_boundary": non_boundary_total.metrics(),
        },
        "multiview_consistency": mvc_total.metrics(),
        "cube_event_activity_vs_error": activity_summary,
    }

    _write_csv(output_dir / "per_frame_metrics.csv", frame_rows)
    _write_csv(output_dir / "per_sequence_metrics.csv", sequence_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite_or_none(summary), handle, indent=2)
        handle.write("\n")
    _write_summary_text(output_dir / "summary.txt", summary)
    _plot_results(output_dir, summary, frame_rows)
    worst_frame_samples = list(worst_frame_by_sequence.values())
    _plot_worst_frames(output_dir, worst_frame_samples)

    print((output_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    if worst_frame_samples:
        print(
            f"Worst-frame overview: "
            f"{(output_dir / 'worst_error_overview.png').resolve()}",
            flush=True,
        )
    print(f"Results written to: {output_dir.resolve()}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more multiview.py checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more checkpoints produced by multiview.py.",
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
