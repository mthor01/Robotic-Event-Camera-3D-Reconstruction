#!/usr/bin/env python3
"""
TensorBoard path and image-visualization helpers.

Each logged panel contains up to `n_samples` rows, one row per sample.
Each row has six sub-images (left to right):

  events | gt depth | mask | pred depth | error | overlay (events + pred depth)

Usage in train_unet.py
----------------------
    from tensorboard_helper import UNetVizLogger

    viz_train = UNetVizLogger(writer, n_samples=4, tag="viz/train")
    viz_val   = UNetVizLogger(writer, n_samples=4, tag="viz/val")

    # Inside run_epoch (or the training loop):
    viz_train.add_batch(voxels, depth, mask, pred)

    # At the end of each epoch:
    viz_train.flush(step=epoch)
    viz_val.flush(step=epoch)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


DEFAULT_TB_ROOT = Path(__file__).resolve().parent / "checkpoints" / "tensorboard"


def tensorboard_run_dir(
    model_name: str,
    run_name: str,
    tb_root: Path | str | None = None,
) -> Path:
    root = Path(tb_root) if tb_root is not None else DEFAULT_TB_ROOT
    return root / model_name / run_name


# ---------------------------------------------------------------------------
# Colourmap helpers
# ---------------------------------------------------------------------------

def _get_cmap(name: str):
    """Return a matplotlib colormap, compatible with old and new matplotlib."""
    try:
        import matplotlib
        return matplotlib.colormaps[name]          # matplotlib >= 3.7
    except AttributeError:
        import matplotlib.cm as cm
        return cm.get_cmap(name)                   # matplotlib < 3.7


_CMAP_TURBO  = None
_CMAP_PLASMA = None
_CMAP_GRAY   = None


def _cmap_turbo():
    global _CMAP_TURBO
    if _CMAP_TURBO is None:
        _CMAP_TURBO = _get_cmap("turbo")
    return _CMAP_TURBO


def _cmap_plasma():
    global _CMAP_PLASMA
    if _CMAP_PLASMA is None:
        _CMAP_PLASMA = _get_cmap("plasma")
    return _CMAP_PLASMA


def _cmap_gray():
    global _CMAP_GRAY
    if _CMAP_GRAY is None:
        _CMAP_GRAY = _get_cmap("gray")
    return _CMAP_GRAY


# ---------------------------------------------------------------------------
# Low-level image building utilities
# ---------------------------------------------------------------------------

_BORDER_PX    = 2          # pixel width of separators
_BORDER_VALUE = 0.45       # separator brightness (mid-gray)
_INVALID_VALUE = 0.10      # color for masked-out pixels

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import DEPTH_MIN as _DEPTH_MIN_CFG, D_MAX as _DEPTH_MAX_CFG

# Depth range used for colormap normalisation
_DEPTH_MIN = _DEPTH_MIN_CFG   # metres  (from config.py)
_DEPTH_MAX = _DEPTH_MAX_CFG   # metres  (from config.py)


def _to_np(t: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, convert to float32 numpy array."""
    return t.detach().float().cpu().numpy()


def _norm01(arr: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Linearly map arr to [0, 1]; clamps to that range."""
    lo = float(arr.min()) if vmin is None else vmin
    hi = float(arr.max()) if vmax is None else vmax
    span = hi - lo
    if span < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / span, 0.0, 1.0)


def _colorize(arr01: np.ndarray, cmap) -> np.ndarray:
    """Apply a matplotlib colormap to a (H, W) float32 in [0,1] → (3, H, W)."""
    rgba = cmap(arr01)           # (H, W, 4)
    return rgba[..., :3].astype(np.float32).transpose(2, 0, 1)  # (3, H, W)


def _events_panel(voxels: np.ndarray) -> np.ndarray:
    """
    voxels : (C, H, W) float32
    Returns (3, H, W) grayscale image of total event activity.
    """
    activity = np.abs(voxels).sum(axis=0)          # (H, W)
    return _colorize(_norm01(activity), _cmap_gray())


def _depth_panel(
    depth: np.ndarray,    # (H, W) metres
    mask:  np.ndarray | None = None,    # (H, W) binary float32
) -> np.ndarray:
    """Turbo-coloured depth map; invalid pixels are dark when mask is given."""
    norm = _norm01(depth, vmin=_DEPTH_MIN, vmax=_DEPTH_MAX)
    rgb  = _colorize(norm, _cmap_turbo())           # (3, H, W)
    if mask is not None:
        rgb[:, mask < 0.5] = _INVALID_VALUE
    return rgb


def _mask_panel(mask: np.ndarray) -> np.ndarray:
    """(H, W) binary float32 → (3, H, W) grayscale."""
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    return np.stack([m, m, m], axis=0)


def _error_panel(
    pred:    np.ndarray,  # (H, W) metres
    gt:      np.ndarray,  # (H, W) metres
    mask:    np.ndarray,  # (H, W) binary float32
) -> np.ndarray:
    """Plasma-coloured absolute error; invalid pixels are dark."""
    err  = np.abs(pred - gt) * mask                 # (H, W)
    norm = _norm01(err, vmin=0.0, vmax=max(float(err.max()), 0.01))
    rgb  = _colorize(norm, _cmap_plasma())          # (3, H, W)
    rgb[:, mask < 0.5] = _INVALID_VALUE
    return rgb


def _overlay_panel(
    voxels: np.ndarray,   # (C, H, W)
    pred:   np.ndarray,   # (H, W) metres
    mask:   np.ndarray,   # (H, W) binary float32
    alpha:  float = 0.55,
) -> np.ndarray:
    """
    Events (grayscale background) blended with predicted depth (turbo)
    wherever the spatial mask is valid.
    """
    # Grayscale event background
    activity = np.abs(voxels).sum(axis=0)           # (H, W)
    ev01     = _norm01(activity)
    bg       = np.stack([ev01, ev01, ev01], axis=0) # (3, H, W)

    # Depth colourmap foreground
    depth_rgb = _colorize(_norm01(pred, _DEPTH_MIN, _DEPTH_MAX), _cmap_turbo())

    valid = mask > 0.5                              # (H, W) bool
    out   = bg.copy()
    out[:, valid] = (1.0 - alpha) * bg[:, valid] + alpha * depth_rgb[:, valid]
    return out


def _build_row(
    voxels:   np.ndarray,  # (C, H, W)
    depth_gt: np.ndarray,  # (H, W) metres
    mask:     np.ndarray,  # (H, W) binary
    pred:     np.ndarray,  # (H, W) metres
    table_depth: np.ndarray | None = None,  # (H, W) normalised [0, 1]
    *,
    show_mask: bool = True,
) -> np.ndarray:
    """
    Build one horizontal strip of panels for a single sample.
    Returns (3, H, N*W + (N-1)*BORDER_PX).
    Panel order: events | gt depth | [table depth] | [mask] | pred | error | overlay
    """
    H = voxels.shape[1]
    sep = np.full((3, H, _BORDER_PX), _BORDER_VALUE, dtype=np.float32)

    panels = [_events_panel(voxels), sep, _depth_panel(depth_gt, mask)]

    if table_depth is not None:
        panels.extend([sep, _colorize(np.clip(table_depth, 0.0, 1.0), _cmap_turbo())])

    if show_mask:
        panels.extend([sep, _mask_panel(mask)])

    panels.extend([
        sep,
        _depth_panel(pred, None),
        sep,
        _error_panel(pred, depth_gt, mask),
        sep,
        _overlay_panel(voxels, pred, mask),
    ])

    return np.concatenate(panels, axis=2)   # (3, H, W_total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VizLogger:
    """
    Accumulates sample frames during a training/validation epoch and writes
    a visualization panel to TensorBoard.

    Each sample row shows (left → right):
        events | gt depth | mask | pred depth | error | overlay

    Parameters
    ----------
    writer : SummaryWriter
        Active TensorBoard writer.
    n_samples : int
        How many sample rows to show in the panel (default 4).
    tag : str
        TensorBoard tag, e.g. ``"viz/train"`` or ``"viz/val"``.
    """

    def __init__(
        self,
        writer:    SummaryWriter,
        n_samples: int = 4,
        tag:       str = "viz",
        show_mask: bool = True,
    ) -> None:
        self.writer     = writer
        self.n_samples  = n_samples
        self.tag        = tag
        self.show_mask  = show_mask
        self._buf: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._seen = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Discard all accumulated samples."""
        self._buf.clear()
        self._seen = 0

    # ------------------------------------------------------------------
    def add_batch(
        self,
        voxels: torch.Tensor,  # (B, C, H, W)
        depth:  torch.Tensor,  # (B, 1, H, W)  ground-truth depth in metres
        mask:   torch.Tensor,  # (B, 1, H, W)  binary validity mask
        pred:   torch.Tensor,  # (B, 1, H, W)  predicted depth in metres
        table_depth: torch.Tensor | None = None,  # (B, 1, H, W) normalised [0, 1]
    ) -> None:
        """
        Store representative frames from this batch using reservoir sampling.

        Instead of always taking the first ``n_samples`` frames in the epoch,
        this gives each seen frame equal probability of ending up in the
        visualization buffer. This greatly improves variety for ordered
        validation loaders.

        Call this once per batch during your epoch loop.
        """
        v_np = _to_np(voxels)          # (B, C, H, W)
        d_np = _to_np(depth[:, 0])     # (B, H, W)
        m_np = _to_np(mask[:, 0])      # (B, H, W)
        p_np = _to_np(pred[:, 0])      # (B, H, W)
        t_np = _to_np(table_depth[:, 0]) if table_depth is not None else None  # (B, H, W) or None

        for b in range(v_np.shape[0]):
            sample = (v_np[b], d_np[b], m_np[b], p_np[b], t_np[b] if t_np is not None else None)
            self._seen += 1

            # Fill the reservoir first, then randomly replace existing samples.
            if len(self._buf) < self.n_samples:
                self._buf.append(sample)
                continue

            j = int(self._rng.integers(0, self._seen))
            if j < self.n_samples:
                self._buf[j] = sample

    # ------------------------------------------------------------------
    def flush(self, step: int) -> None:
        """
        Build the visualization panel from accumulated samples, write it to
        TensorBoard, and reset the internal buffer.

        Call once at the end of each epoch.
        """
        if not self._buf:
            return

        rows = [_build_row(*s, show_mask=self.show_mask) for s in self._buf]

        # Horizontal separator between sample rows
        W_total = rows[0].shape[2]
        h_sep   = np.full((3, _BORDER_PX, W_total), _BORDER_VALUE, dtype=np.float32)

        parts: List[np.ndarray] = []
        for i, row in enumerate(rows):
            if i > 0:
                parts.append(h_sep)
            parts.append(row)

        panel = np.clip(np.concatenate(parts, axis=1), 0.0, 1.0)  # (3, H_total, W_total)

        self.writer.add_image(self.tag, panel, global_step=step)
        self.writer.flush()   # force write to disk so TensorBoard shows images immediately
        self.reset()


class EventActivityAccuracyLogger:
    """
    Tracks how total event activity relates to per-frame prediction error.

    TensorBoard outputs:
        <tag>/scatter_activity_vs_l1
        <tag>/corr_activity_vs_l1
        <tag>/corr_activity_vs_accuracy
        <tag>/mean_l1_by_activity_bin_XX
    """

    def __init__(
        self,
        writer: SummaryWriter,
        tag: str = "event_activity_accuracy",
        max_samples: int = 20000,
        n_bins: int = 8,
    ) -> None:
        self.writer = writer
        self.tag = tag.rstrip("/")
        self.max_samples = max_samples
        self.n_bins = n_bins
        self._activity: List[float] = []
        self._l1: List[float] = []
        self._seen = 0
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        self._activity.clear()
        self._l1.clear()
        self._seen = 0

    def add_batch(
        self,
        voxels: torch.Tensor,  # (B, C, H, W)
        depth: torch.Tensor,   # (B, 1, H, W), metres
        mask: torch.Tensor,    # (B, 1, H, W)
        pred: torch.Tensor,    # (B, 1, H, W), metres
    ) -> None:
        with torch.no_grad():
            activity = voxels.detach().abs().sum(dim=(1, 2, 3)).float().cpu().numpy()
            valid = mask.detach() > 0.5
            err = (pred.detach() - depth.detach()).abs()

            valid_count = valid.flatten(1).sum(dim=1)
            err_sum = (err * valid.float()).flatten(1).sum(dim=1)
            l1 = torch.where(
                valid_count > 0,
                err_sum / valid_count.clamp(min=1),
                torch.full_like(err_sum, float("nan")),
            ).float().cpu().numpy()

        for a, e in zip(activity, l1):
            if not np.isfinite(e):
                continue

            self._seen += 1
            if len(self._activity) < self.max_samples:
                self._activity.append(float(a))
                self._l1.append(float(e))
                continue

            j = int(self._rng.integers(0, self._seen))
            if j < self.max_samples:
                self._activity[j] = float(a)
                self._l1[j] = float(e)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def flush(self, step: int) -> None:
        if not self._activity:
            return

        activity = np.asarray(self._activity, dtype=np.float32)
        l1 = np.asarray(self._l1, dtype=np.float32)
        accuracy = 1.0 / (1.0 + l1)

        self.writer.add_scalar(f"{self.tag}/corr_activity_vs_l1", self._corr(activity, l1), step)
        self.writer.add_scalar(
            f"{self.tag}/corr_activity_vs_accuracy",
            self._corr(activity, accuracy),
            step,
        )
        self.writer.add_scalar(f"{self.tag}/mean_activity", float(activity.mean()), step)
        self.writer.add_scalar(f"{self.tag}/mean_l1", float(l1.mean()), step)

        order = np.argsort(activity)
        chunks = np.array_split(order, self.n_bins)
        for i, idx in enumerate(chunks):
            if idx.size == 0:
                continue
            self.writer.add_scalar(
                f"{self.tag}/mean_l1_by_activity_bin_{i:02d}",
                float(l1[idx].mean()),
                step,
            )

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
            ax.scatter(activity, l1, s=8, alpha=0.35, linewidths=0)
            ax.set_xlabel("Total event activity per frame")
            ax.set_ylabel("Mean absolute depth error [m]")
            ax.set_title("Event activity vs prediction error")
            ax.grid(True, alpha=0.25)
            self.writer.add_figure(f"{self.tag}/scatter_activity_vs_l1", fig, step, close=True)
        except Exception:
            self.writer.add_histogram(f"{self.tag}/activity", activity, step)
            self.writer.add_histogram(f"{self.tag}/l1", l1, step)

        self.writer.flush()
        self.reset()


class UncertaintyErrorLogger:
    """
    Tracks how predicted uncertainty relates to absolute depth error.

    TensorBoard outputs:
        <tag>/scatter_uncertainty_vs_error
        <tag>/corr_uncertainty_vs_error
        <tag>/mean_error_by_uncertainty_bin_XX
    """

    def __init__(
        self,
        writer: SummaryWriter,
        tag: str = "uncertainty_error",
        max_samples: int = 20000,
        max_batch_samples: int = 4096,
        n_bins: int = 8,
        images_only: bool = False,
    ) -> None:
        self.writer = writer
        self.tag = tag.rstrip("/")
        self.max_samples = max_samples
        self.max_batch_samples = max_batch_samples
        self.n_bins = n_bins
        self.images_only = images_only
        self._uncertainty: List[float] = []
        self._error: List[float] = []
        self._seen = 0
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        self._uncertainty.clear()
        self._error.clear()
        self._seen = 0

    def add_batch(
        self,
        uncertainty: torch.Tensor,  # (B, 1, H, W), metres
        pred: torch.Tensor,         # (B, 1, H, W), metres
        depth: torch.Tensor,        # (B, 1, H, W), metres
        mask: torch.Tensor,         # (B, 1, H, W)
    ) -> None:
        with torch.no_grad():
            valid = (mask.detach() > 0.5).flatten()
            unc = uncertainty.detach().flatten()[valid].float().cpu().numpy()
            err = (pred.detach() - depth.detach()).abs().flatten()[valid].float().cpu().numpy()

        if unc.size == 0:
            return

        if unc.size > self.max_batch_samples:
            idx = self._rng.choice(unc.size, size=self.max_batch_samples, replace=False)
            unc = unc[idx]
            err = err[idx]

        for u, e in zip(unc, err):
            if not np.isfinite(u) or not np.isfinite(e):
                continue

            self._seen += 1
            if len(self._uncertainty) < self.max_samples:
                self._uncertainty.append(float(u))
                self._error.append(float(e))
                continue

            j = int(self._rng.integers(0, self._seen))
            if j < self.max_samples:
                self._uncertainty[j] = float(u)
                self._error[j] = float(e)

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def flush(self, step: int) -> None:
        if not self._uncertainty:
            return

        uncertainty = np.asarray(self._uncertainty, dtype=np.float32)
        error = np.asarray(self._error, dtype=np.float32)

        if not self.images_only:
            self.writer.add_scalar(
                f"{self.tag}/corr_uncertainty_vs_error",
                self._corr(uncertainty, error),
                step,
            )
            self.writer.add_scalar(
                f"{self.tag}/mean_uncertainty_m",
                float(uncertainty.mean()),
                step,
            )
            self.writer.add_scalar(f"{self.tag}/mean_error_m", float(error.mean()), step)

            order = np.argsort(uncertainty)
            chunks = np.array_split(order, self.n_bins)
            for i, idx in enumerate(chunks):
                if idx.size == 0:
                    continue
                self.writer.add_scalar(
                    f"{self.tag}/mean_error_by_uncertainty_bin_{i:02d}",
                    float(error[idx].mean()),
                    step,
                )

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
            ax.scatter(uncertainty, error, s=4, alpha=0.25, linewidths=0)
            ax.set_xlabel("Predicted uncertainty [m]")
            ax.set_ylabel("Absolute depth error [m]")
            ax.set_title("Uncertainty vs prediction error")
            ax.grid(True, alpha=0.25)
            self.writer.add_figure(f"{self.tag}/scatter_uncertainty_vs_error", fig, step, close=True)
        except Exception:
            if self.images_only:
                raise
            self.writer.add_histogram(f"{self.tag}/uncertainty_m", uncertainty, step)
            self.writer.add_histogram(f"{self.tag}/error_m", error, step)

        self.writer.flush()
        self.reset()


class ErrorDistributionSpatialLogger:
    """
    Logs absolute depth error distribution and where errors occur in the image.

    TensorBoard outputs:
        <tag>/distribution
        <tag>/spatial_mean_error
    """

    def __init__(
        self,
        writer: SummaryWriter,
        tag: str = "error",
        max_samples: int = 20000,
        max_batch_samples: int = 4096,
        images_only: bool = False,
    ) -> None:
        self.writer = writer
        self.tag = tag.rstrip("/")
        self.max_samples = max_samples
        self.max_batch_samples = max_batch_samples
        self.images_only = images_only
        self._errors: List[float] = []
        self._seen = 0
        self._rng = np.random.default_rng()
        self._sum: np.ndarray | None = None
        self._count: np.ndarray | None = None

    def reset(self) -> None:
        self._errors.clear()
        self._seen = 0
        self._sum = None
        self._count = None

    def add_batch(
        self,
        pred: torch.Tensor,   # (B, 1, H, W), metres
        depth: torch.Tensor,  # (B, 1, H, W), metres
        mask: torch.Tensor,   # (B, 1, H, W)
    ) -> None:
        with torch.no_grad():
            err = (pred.detach() - depth.detach()).abs()
            valid = mask.detach() > 0.5

            err_np = err[:, 0].float().cpu().numpy()
            valid_np = valid[:, 0].float().cpu().numpy()
            vals = err[valid].float().cpu().numpy()

        if vals.size > self.max_batch_samples:
            idx = self._rng.choice(vals.size, size=self.max_batch_samples, replace=False)
            vals = vals[idx]

        if self._sum is None:
            self._sum = np.zeros_like(err_np[0], dtype=np.float64)
            self._count = np.zeros_like(err_np[0], dtype=np.float64)

        self._sum += (err_np * valid_np).sum(axis=0)
        self._count += valid_np.sum(axis=0)

        for e in vals:
            if not np.isfinite(e):
                continue

            self._seen += 1
            if len(self._errors) < self.max_samples:
                self._errors.append(float(e))
                continue

            j = int(self._rng.integers(0, self._seen))
            if j < self.max_samples:
                self._errors[j] = float(e)

    def flush(self, step: int) -> None:
        if not self._errors or self._sum is None or self._count is None:
            return

        errors = np.asarray(self._errors, dtype=np.float32)
        mean_m = float(errors.mean())
        median_m = float(np.median(errors))
        p95_m = float(np.percentile(errors, 95))
        if self.images_only:
            import matplotlib.pyplot as plt

            p99_m = float(np.percentile(errors, 99))
            upper_m = max(p99_m, 1e-4)
            fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
            ax.hist(
                np.clip(errors, 0.0, upper_m),
                bins=60,
                color="#4472c4",
                alpha=0.9,
            )
            ax.axvline(
                min(mean_m, upper_m),
                color="#c55a11",
                linewidth=1.5,
                label=f"mean {mean_m * 100:.2f} cm",
            )
            ax.axvline(
                min(median_m, upper_m),
                color="#70ad47",
                linewidth=1.5,
                label=f"median {median_m * 100:.2f} cm",
            )
            ax.axvline(
                min(p95_m, upper_m),
                color="#8064a2",
                linewidth=1.5,
                label=f"p95 {p95_m * 100:.2f} cm",
            )
            ax.set_xlabel("Absolute depth error [m]")
            ax.set_ylabel("Sampled valid pixels")
            ax.set_title(f"Error distribution at step {step} (clipped at p99)")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            self.writer.add_figure(
                f"{self.tag}/distribution_at_step",
                fig,
                global_step=step,
                close=True,
            )
        else:
            self.writer.add_histogram(f"{self.tag}/distribution", errors, step)
            self.writer.add_scalar(f"{self.tag}/mean_m", mean_m, step)
            self.writer.add_scalar(f"{self.tag}/median_m", median_m, step)
            self.writer.add_scalar(f"{self.tag}/p95_m", p95_m, step)

        mean_error = self._sum / np.maximum(self._count, 1.0)
        valid_pixels = self._count > 0
        vmax = float(np.percentile(mean_error[valid_pixels], 98)) if np.any(valid_pixels) else 0.0
        heat01 = _norm01(mean_error.astype(np.float32), vmin=0.0, vmax=max(vmax, 1e-6))
        heat = _colorize(heat01, _cmap_plasma())
        heat[:, ~valid_pixels] = _INVALID_VALUE

        self.writer.add_image(f"{self.tag}/spatial_mean_error", heat, global_step=step)
        self.writer.flush()
        self.reset()
