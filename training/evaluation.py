#!/usr/bin/env python3
"""Evaluate an MVS checkpoint.

Example:
    python3 evaluation.py \\
        --checkpoint checkpoints/mvs/best_model.pth \\
        --data_dir ../data/Event_and_Depth/eval

The data directory may be either a folder containing multiple sequence
folders or one sequence folder. Depth metrics are reported for the whole
frame, the workspace cube, and a raised cube that excludes the table. Results
are written to results/<pose-layout>_<checkpoint-stem>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_DIR = Path(__file__).resolve().parent.parent
_CAMERA_DATA_DIR = _REPO_DIR / "camera_data"

from train_mvs import (
    MultiViewAugConfig,
    MultiViewTableDataset,
    load_checkpoint,
)
from helpers import (
    INTRINSICS_TRANSFORM,
    depth_cube_mask,
    find_precomputed_sequences,
    load_event_calibration,
)
from config import (
    DATA_ROOT,
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
# The workspace cube is shifted up slightly; the raised cube additionally
# starts 1 cm above the table so that it contains only the object.
SPATIAL_MASK_VERTICAL_SHIFT_M = 0.005
RAISED_CUBE_BOTTOM_Z_M = 0.01 + SPATIAL_MASK_VERTICAL_SHIFT_M
QUALITATIVE_RANDOM_SEED = 20260813
LEGEND_FONT_SIZE = 12
LEGEND_TITLE_FONT_SIZE = 13


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


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(val) for val in value]
    return value


def _frame_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    accumulator = DepthAccumulator()
    accumulator.add(pred[valid], gt[valid])
    return accumulator.metrics()


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


def _write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Checkpoint: {summary['checkpoint']}",
        f"Evaluation root: {summary['evaluation_root']}",
        f"Sequences: {summary['sequence_count']}",
        f"Frames: {summary['frame_count']}",
        f"Evaluation frame step: {summary['evaluation_configuration']['frame_step']}",
        "",
        "Depth metrics by spatial region",
    ]
    for row in _depth_region_rows(summary):
        lines.append(f"  {row['label']}")
        for name in DEPTH_METRIC_NAMES:
            lines.append(f"    {name}: {_format_metric(name, row[name])}")
        lines.append(f"    valid_pixels: {row['valid_pixels']}")
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
) -> None:
    """Plot the selected-frame modalities for each requested frame."""
    if not samples:
        return

    plt, _sns = _setup_seaborn_plotting()
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


def _region_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Prefix region metric names as in the per-frame and per-sequence CSVs."""
    if prefix == "whole_image":
        return dict(metrics)
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _evaluate_checkpoint(args: argparse.Namespace, device: torch.device) -> None:
    checkpoint_path = args.checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}", flush=True)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    num_views = int(ckpt["num_views"])
    pose_move_threshold = float(ckpt["pose_move_threshold"])
    allow_unbalanced_pose_views = bool(
        ckpt.get(
            "allow_fewer_pose_views",
            ckpt.get("allow_unbalanced_pose_views", False),
        )
    )
    if args.pose_layout_override is not None:
        allow_unbalanced_pose_views = args.pose_layout_override
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
    coarse_depths = int(ckpt["coarse_depths"])

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

    output_dir = args.results_folder / (
        f"{_pose_layout_directory_indicator(allow_unbalanced_pose_views)}_"
        f"{checkpoint_path.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = load_event_calibration(_CAMERA_DATA_DIR)
    depth_totals = {region: DepthAccumulator() for region in DEPTH_REGION_LABELS}
    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    qualitative_sample_by_sequence: dict[str, dict[str, Any]] = {}
    qualitative_frames_seen: dict[str, int] = {}
    qualitative_rng = np.random.default_rng(QUALITATIVE_RANDOM_SEED)
    selected_example_samples: dict[tuple[str, int], dict[str, Any]] = {}
    cube_half_side = SPATIAL_CUBE_SIDE / 2.0
    cube_center = np.asarray(
        [
            SPATIAL_TARGET_X,
            SPATIAL_TARGET_Y,
            SPATIAL_TARGET_Z + SPATIAL_MASK_VERTICAL_SHIFT_M,
        ],
        dtype=np.float64,
    )
    raised_cube_center = np.asarray(
        [SPATIAL_TARGET_X, SPATIAL_TARGET_Y, RAISED_CUBE_BOTTOM_Z_M + cube_half_side],
        dtype=np.float64,
    )

    for sequence_number, sequence_dir in enumerate(sequence_dirs, start=1):
        dataset = MultiViewTableDataset(
            sequence_dir,
            calib=calib,
            num_views=num_views,
            pose_move_threshold=pose_move_threshold,
            allow_unbalanced_pose_views=allow_unbalanced_pose_views,
            coarse_depths=coarse_depths,
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
        sequence_totals = {region: DepthAccumulator() for region in DEPTH_REGION_LABELS}
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
            for batch in loader:
                prediction_norm = model(
                    batch["imgs"].to(device, non_blocking=True),
                    batch["cam_mats"].to(device, non_blocking=True),
                    batch["K"].to(device, non_blocking=True),
                    batch["depth_values"].to(device, non_blocking=True),
                    view_valid_mask=batch["view_valid_mask"].to(device, non_blocking=True),
                )
                prediction = (
                    prediction_norm * (D_MAX - DEPTH_MIN) + DEPTH_MIN
                ).detach().cpu()
                gt_batch = batch["dep_t"].float()
                mask_batch = batch["mask_t"] > 0.5
                cam_batch = batch["cam_mats"][:, 0].float().numpy()
                K_batch = batch["K"].float().numpy()

                for index_in_batch, frame_idx in enumerate(batch["ref_idx"].tolist()):
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
                    T_cam_from_world = cam_batch[index_in_batch]
                    K = K_batch[index_in_batch]
                    region_valid = {
                        "whole_image": valid_np,
                        "lower_cube": valid_np & depth_cube_mask(
                            gt_np, T_cam_from_world, K, cube_center, cube_half_side
                        ),
                        "upper_cube": valid_np & depth_cube_mask(
                            gt_np, T_cam_from_world, K, raised_cube_center, cube_half_side
                        ),
                    }
                    frame_row: dict[str, Any] = {
                        "sequence": sequence_dir.name,
                        "frame_idx": frame_idx,
                    }
                    for region, valid in region_valid.items():
                        depth_totals[region].add(pred_np[valid], gt_np[valid])
                        sequence_totals[region].add(pred_np[valid], gt_np[valid])
                        frame_row.update(
                            _region_metrics(
                                region, _frame_depth_metrics(pred_np, gt_np, valid)
                            )
                        )
                    frame_rows.append(frame_row)

                    # Reservoir sampling selects one evaluated frame uniformly
                    # at random for each sequence without retaining all frames.
                    qualitative_frames_seen[sequence_dir.name] = (
                        qualitative_frames_seen.get(sequence_dir.name, 0) + 1
                    )
                    keep_qualitative = qualitative_rng.integers(
                        0, qualitative_frames_seen[sequence_dir.name]
                    ) == 0
                    selected_example_key = (sequence_dir.name, frame_idx)
                    keep_example = selected_example_key in selected_example_request_set
                    if keep_qualitative or keep_example:
                        figure_sample = {
                            "sequence": sequence_dir.name,
                            "frame_idx": frame_idx,
                            "events": batch["imgs"][index_in_batch, 0, :NUM_BINS]
                            .float()
                            .numpy()
                            .copy(),
                            "gt": gt_np.copy(),
                            "pred": pred_np.copy(),
                            "prediction_workspace_mask": depth_cube_mask(
                                pred_np, T_cam_from_world, K, cube_center, cube_half_side
                            ),
                            "lower_cube_valid": region_valid["lower_cube"].copy(),
                        }
                        if keep_qualitative:
                            qualitative_sample_by_sequence[sequence_dir.name] = figure_sample
                        if keep_example:
                            selected_example_samples[selected_example_key] = figure_sample

        sequence_rows.append(
            {
                "sequence": sequence_dir.name,
                "frames": len(dataset),
                **{
                    name: value
                    for region, accumulator in sequence_totals.items()
                    for name, value in _region_metrics(region, accumulator.metrics()).items()
                },
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_l1_m": ckpt.get("val_l1"),
        "checkpoint_val_p95_m": ckpt.get("val_p95"),
        "checkpoint_val_l1_worst10_m": ckpt.get("val_l1_worst10"),
        "evaluation_root": str(evaluation_root.resolve()),
        "sequence_count": len(sequence_dirs),
        "frame_count": len(frame_rows),
        "model_configuration": {
            "checkpoint": str(checkpoint_path),
            "model_arch": ckpt.get("model_arch", "ModernMVSNet"),
            "num_views": num_views,
            "pose_move_threshold_m": pose_move_threshold,
            "allow_unbalanced_pose_views": allow_unbalanced_pose_views,
            "pose_layout_cli_override": args.pose_layout_override,
            "coarse_depths": coarse_depths,
            "fine_depths": ckpt.get("fine_depths"),
        },
        "evaluation_configuration": {
            "intrinsics_transform": INTRINSICS_TRANSFORM,
            "frame_step": args.fast_mode,
            "depth_metric_regions": {
                "whole_image": "all valid evaluated depth pixels",
                "lower_cube": {
                    "description": "table and object",
                    "center_world_m": cube_center.tolist(),
                    "bottom_z_m": float(cube_center[2] - cube_half_side),
                    "side_m": SPATIAL_CUBE_SIDE,
                },
                "upper_cube": {
                    "description": "object only; raised to exclude the table",
                    "center_world_m": raised_cube_center.tolist(),
                    "bottom_z_m": RAISED_CUBE_BOTTOM_Z_M,
                    "side_m": SPATIAL_CUBE_SIDE,
                },
                "mask_source": "measured GT depth backprojected to world coordinates",
            },
        },
        # Keep the historical whole-image key for downstream compatibility.
        "depth_metrics": depth_totals["whole_image"].metrics(),
        "depth_metrics_by_region": {
            region: accumulator.metrics() for region, accumulator in depth_totals.items()
        },
    }

    _write_csv(output_dir / "per_frame_metrics.csv", frame_rows)
    _write_csv(output_dir / "per_sequence_metrics.csv", sequence_rows)
    _write_csv(output_dir / "depth_metrics_by_region.csv", _depth_region_rows(summary))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_finite_or_none(summary), handle, indent=2)
        handle.write("\n")
    _write_summary_text(output_dir / "summary.txt", summary)

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
    _plot_selected_frames_overview(
        output_dir,
        [selected_example_samples[request] for request in selected_example_requests],
    )

    print((output_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    print(f"Results written to: {output_dir.resolve()}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MVS checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MVS checkpoint saved by train_mvs.py.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=_REPO_DIR / DATA_ROOT / "eval",
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
            "Root output directory. Results are written to a policy-prefixed "
            "subdirectory named after the checkpoint."
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
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.results_folder.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    _evaluate_checkpoint(args, device)


if __name__ == "__main__":
    main()
